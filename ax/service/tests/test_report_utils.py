#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
import logging
from collections import namedtuple
from logging import DEBUG, INFO, WARN
from unittest import mock
from unittest.mock import patch

import pandas as pd
from ax.adapter.registry import Generators
from ax.core.arm import Arm
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.core.types import ComparisonOp
from ax.generation_strategy.generation_node import GenerationStep
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.orchestration.orchestrator import Orchestrator
from ax.service.utils.orchestrator_options import OrchestratorOptions
from ax.service.utils.report_utils import (
    _find_sigfigs,
    _get_cross_validation_plots,
    _get_curve_plot_dropdown,
    _get_metric_name_pairs,
    _get_objective_trace_plot,
    _get_objective_v_param_plots,
    _objective_vs_true_objective_scatter,
    construct_comparison_message,
    exp_to_df,
    Experiment,
    FEASIBLE_COL_NAME,
    get_standard_plots,
    maybe_extract_baseline_comparison_values,
    plot_feature_importance_by_feature_plotly,
    warn_if_unpredictable_metrics,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
    get_branin_experiment_with_timestamp_map_metric,
    get_experiment_with_observations,
    get_high_dimensional_branin_experiment,
    get_multi_type_experiment,
    get_test_map_data_experiment,
)
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import get_generation_strategy
from plotly import graph_objects as go
from pyre_extensions import assert_is_instance, none_throws

OBJECTIVE_NAME = "branin"
PARAMETER_COLUMNS = ["x1", "x2"]
FLOAT_COLUMNS: list[str] = [OBJECTIVE_NAME] + PARAMETER_COLUMNS
EXPECTED_COLUMNS: list[str] = [
    "trial_index",
    "arm_name",
    "trial_status",
    "generation_method",
] + FLOAT_COLUMNS
DUMMY_OBJECTIVE_MEAN = 1.2345
DUMMY_SOURCE = "test_source"
DUMMY_MAP_KEY = "test_map_key"
TRUE_OBJECTIVE_NAME = "other_metric"
TRUE_OBJECTIVE_MEAN = 2.3456
DUMMY_MSG = "test_message"


class ReportUtilsTest(TestCase):
    @patch(
        "ax.service.utils.report_utils._merge_results_if_no_duplicates",
        autospec=True,
        return_value=pd.DataFrame(
            [
                # Trial indexes are out-of-order.
                {"arm_name": "a", "trial_index": 1},
                {"arm_name": "b", "trial_index": 2},
                {"arm_name": "c", "trial_index": 0},
            ]
        ),
    )
    def test_exp_to_df_row_ordering(self, _) -> None:
        """
        This test verifies that the returned data frame indexes are
        in the same order as trial index. It mocks _merge_results_if_no_duplicates
        to verify just the ordering of items in the final data frame.
        """
        exp = get_branin_experiment(with_trial=True)
        df = exp_to_df(exp)
        # Check that all 3 rows are in order
        self.assertEqual(len(df), 3)
        for idx, row in df.iterrows():
            self.assertEqual(row["trial_index"], idx)

    @patch(
        "ax.service.utils.report_utils._merge_results_if_no_duplicates",
        autospec=True,
        return_value=pd.DataFrame(
            [
                # Trial indexes are out-of-order.
                {
                    "col1": 1,
                    "arm_name": "a",
                    "trial_status": "FAILED",
                    "generation_method": "Manual",
                    "trial_index": 1,
                },
                {
                    "col1": 2,
                    "arm_name": "b",
                    "trial_status": "COMPLETED",
                    "generation_method": "BO",
                    "trial_index": 2,
                },
                {
                    "col1": 3,
                    "arm_name": "c",
                    "trial_status": "COMPLETED",
                    "generation_method": "Manual",
                    "trial_index": 0,
                },
            ]
        ),
    )
    def test_exp_to_df_col_ordering(self, _) -> None:
        """
        This test verifies that the returned data frame indexes are
        in the same order as trial index. It mocks _merge_results_if_no_duplicates
        to verify just the ordering of items in the final data frame.
        """
        exp = get_branin_experiment(with_trial=True)
        df = exp_to_df(exp)
        self.assertListEqual(
            list(df.columns),
            ["trial_index", "arm_name", "trial_status", "generation_method", "col1"],
        )

    def test_exp_to_df_trial_timing(self) -> None:
        # 1. test all have started, none have completed
        exp = get_test_map_data_experiment(num_trials=3, num_fetches=5, num_complete=0)
        df = exp_to_df(
            exp=exp,
            trial_attribute_fields=["time_run_started", "time_completed"],
            always_include_field_columns=True,
        )
        self.assertTrue("time_run_started" in list(df.columns))
        self.assertTrue("time_completed" in list(df.columns))
        # since all trials started, all should have values
        self.assertFalse(any(df["time_run_started"].isnull()))
        # since no trials are complete, all should be None
        self.assertTrue(all(df["time_completed"].isnull()))

        # 2. test some trials not started yet
        exp.trials[0]._time_run_started = None
        df = exp_to_df(
            exp=exp, trial_attribute_fields=["time_run_started", "time_completed"]
        )
        # the first trial should have NaN for rel_time_run_started
        self.assertTrue(df["time_run_started"].isnull().iloc[0])

        # 3. test all trials not started yet
        for t in exp.trials.values():
            t._time_run_started = None
        df = exp_to_df(
            exp=exp,
            trial_attribute_fields=["time_run_started", "time_completed"],
            always_include_field_columns=True,
        )
        self.assertTrue(all(df["time_run_started"].isnull()))

        # 4. test some trials are completed
        exp = get_test_map_data_experiment(num_trials=3, num_fetches=5, num_complete=2)
        df = exp_to_df(
            exp=exp, trial_attribute_fields=["time_run_started", "time_completed"]
        )
        # the last trial should have NaN for rel_time_completed
        self.assertTrue(df["time_completed"].isnull().iloc[2])

    def test_exp_to_df_with_failure(self) -> None:
        fail_reason = "test reason"

        # Set up experiment with a failed trial
        exp = get_branin_experiment(with_trial=True)
        exp.trials[0].run()
        exp.trials[0].mark_failed(reason=fail_reason)

        df = exp_to_df(exp)
        self.assertEqual(
            set(EXPECTED_COLUMNS + ["reason"]) - set(df.columns), {OBJECTIVE_NAME}
        )
        self.assertEqual(f"{fail_reason}...", df["reason"].iloc[0])

    def test_exp_to_df(self) -> None:
        # MultiTypeExperiment should fail
        exp = get_multi_type_experiment()
        with self.assertRaisesRegex(ValueError, "MultiTypeExperiment"):
            exp_to_df(exp=exp)

        # exp with no trials should return empty results
        exp = get_branin_experiment()
        df = exp_to_df(exp=exp)
        self.assertEqual(len(df), 0)

        # set up experiment
        exp = get_branin_experiment(with_batch=True)

        # check that pre-run experiment returns all columns except objective
        df = exp_to_df(exp)
        self.assertEqual(set(EXPECTED_COLUMNS) - set(df.columns), {OBJECTIVE_NAME})
        self.assertEqual(len(df.index), len(exp.arms_by_name))

        exp.trials[0].run()
        exp.fetch_data()

        # assert result is df with expected columns and length
        df = exp_to_df(exp=exp)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(sorted(df.columns), sorted(EXPECTED_COLUMNS))
        self.assertEqual(len(df.index), len(exp.arms_by_name))

        # test with run_metadata_fields and trial_properties_fields not empty
        # add source to properties
        for _, trial in exp.trials.items():
            trial._properties["source"] = DUMMY_SOURCE
        df = exp_to_df(
            exp, run_metadata_fields=["name"], trial_properties_fields=["source"]
        )
        self.assertIn("name", df.columns)
        self.assertIn("trial_properties_source", df.columns)

        # test column values or types
        self.assertTrue(all(x == 0 for x in df.trial_index))
        self.assertTrue(all(x == "RUNNING" for x in df.trial_status))
        self.assertTrue(all(x == "Sobol" for x in df.generation_method))
        self.assertTrue(all(x == DUMMY_SOURCE for x in df.trial_properties_source))
        self.assertTrue(all(x == "branin_test_experiment_0" for x in df.name))
        for float_column in FLOAT_COLUMNS:
            self.assertTrue(all(isinstance(x, float) for x in df[float_column]))

        # works correctly for failed trials (will need to mock)
        dummy_struct = namedtuple("dummy_struct", "df")
        mock_results = dummy_struct(
            df=pd.DataFrame(
                {
                    "arm_name": ["0_0", "1_0"],
                    "metric_name": [OBJECTIVE_NAME] * 2,
                    "mean": [DUMMY_OBJECTIVE_MEAN] * 2,
                    "sem": [0] * 2,
                    "trial_index": [0, 1],
                    "n": [123] * 2,
                    "frac_nonnull": [1] * 2,
                    "metric_signature": [OBJECTIVE_NAME] * 2,
                }
            )
        )
        mock_results.df.index = range(len(exp.trials) * 2, len(exp.trials) * 2 + 2)
        with patch.object(Experiment, "lookup_data", lambda self: mock_results):
            df = exp_to_df(exp=exp)
        # all but two rows should have a metric value of NaN
        # pyre-fixme[16]: `bool` has no attribute `sum`.
        self.assertEqual(pd.isna(df[OBJECTIVE_NAME]).sum(), len(df.index) - 2)

        # an experiment with more results than arms raises an error
        with (
            patch.object(Experiment, "lookup_data", lambda self: mock_results),
            self.assertRaisesRegex(ValueError, "inconsistent experimental state"),
        ):
            exp_to_df(exp=get_branin_experiment())

        # custom added trial has a generation_method of Manual
        custom_arm = Arm(name="custom", parameters={"x1": 0, "x2": 0})
        exp.new_trial().add_arm(custom_arm)
        df = exp_to_df(exp)
        self.assertEqual(
            df[df.arm_name == "custom"].iloc[0].generation_method, "Manual"
        )
        # failing feasibility calculation doesn't warns and suppresses error
        observations = [[1.0, 2.0, 3.0], [4.0, 5.0, -6.0], [7.0, 8.0, 9.0]]
        exp = get_experiment_with_observations(
            observations=observations,
            constrained=True,
        )
        with (
            patch(
                f"{exp_to_df.__module__}.is_row_feasible",
                side_effect=KeyError(DUMMY_MSG),
            ),
            self.assertLogs(logger="ax", level=WARN) as log,
        ):
            exp_to_df(exp)
            self.assertIn(
                f"Feasibility calculation failed with error: '{DUMMY_MSG}'",
                log.output[0],
            )

        # infeasible arm has `is_feasible = False`.
        df = exp_to_df(exp)
        self.assertListEqual(list(df[FEASIBLE_COL_NAME]), [True, False, True])

        # all rows infeasible.
        observations = [[1.0, 2.0, -3.0], [4.0, 5.0, -6.0], [7.0, 8.0, -9.0]]
        exp = get_experiment_with_observations(
            observations=observations,
            constrained=True,
        )
        df = exp_to_df(exp)
        self.assertListEqual(list(df[FEASIBLE_COL_NAME]), [False, False, False])

    def test_exp_to_df_relative_metrics(self) -> None:
        # set up experiment
        exp = get_branin_experiment(with_trial=True, with_status_quo=False)

        # no status quo arm
        with self.assertLogs(logger="ax", level=WARN) as log:
            exp_to_df(exp, show_relative_metrics=True)
            self.assertIn(
                "No status quo arm found",
                log.output[0],
            )

        # add status quo arm
        exp._status_quo = exp.arms_by_name["0_0"]
        exp.trials[0].run()
        exp.fetch_data()
        relative_df = exp_to_df(exp=exp, show_relative_metrics=True)
        self.assertTrue(f"{OBJECTIVE_NAME}_%CH" in relative_df.columns.tolist())
        self.assertEqual(relative_df[f"{OBJECTIVE_NAME}_%CH"].values[0], 0.0)

    @TestCase.ax_long_test(
        reason=(
            "get_standard_plots still too slow under @mock_botorch_optimize for this "
            "test. Will be deprecated soon."
        )
    )
    @mock_botorch_optimize
    def test_get_standard_plots(self) -> None:
        exp = get_branin_experiment()
        self.assertEqual(
            len(
                get_standard_plots(
                    experiment=exp, model=get_generation_strategy().adapter
                )
            ),
            0,
        )
        exp = get_branin_experiment(with_batch=True, minimize=True)
        exp.trials[0].run()
        exp.trials[0].mark_completed()
        model = Generators.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data())
        for gsa, true_objective_metric_name in itertools.product(
            [False, True], ["branin", None]
        ):
            with self.subTest(global_sensitivity_analysis=gsa):
                plots = get_standard_plots(
                    experiment=exp,
                    model=model,
                    global_sensitivity_analysis=gsa,
                    true_objective_metric_name=true_objective_metric_name,
                )
                self.assertEqual(len(plots), 8 if true_objective_metric_name else 6)
                self.assertTrue(all(isinstance(plot, go.Figure) for plot in plots))

        # Raise an exception in one plot and make sure we generate the others
        for plot_function, num_expected_plots in [
            [_get_curve_plot_dropdown, 8],  # Not used
            [_get_objective_trace_plot, 6],
            [_objective_vs_true_objective_scatter, 7],
            [_get_objective_v_param_plots, 6],
            [_get_cross_validation_plots, 7],
            [plot_feature_importance_by_feature_plotly, 6],
        ]:
            with mock.patch(
                # pyre-ignore
                f"ax.service.utils.report_utils.{plot_function.__name__}",
                side_effect=Exception(),
            ):
                plots = get_standard_plots(
                    experiment=exp,
                    model=model,
                    global_sensitivity_analysis=True,
                    true_objective_metric_name="branin",
                )
            self.assertEqual(len(plots), num_expected_plots)
            self.assertTrue(all(isinstance(plot, go.Figure) for plot in plots))

    @mock_botorch_optimize
    def test_get_standard_plots_moo(self) -> None:
        exp = get_branin_experiment_with_multi_objective(with_batch=True)
        exp.optimization_config.objective.objectives[0].minimize = False
        exp.optimization_config.objective.objectives[1].minimize = True
        assert_is_instance(
            exp.optimization_config, MultiObjectiveOptimizationConfig
        )._objective_thresholds = [
            ObjectiveThreshold(
                metric=exp.metrics["branin_a"], op=ComparisonOp.GEQ, bound=-100.0
            ),
            ObjectiveThreshold(
                metric=exp.metrics["branin_b"], op=ComparisonOp.LEQ, bound=100.0
            ),
        ]
        exp.trials[0].run()
        logger = logging.getLogger("ax.service.utils.report_utils")
        logger.setLevel(DEBUG)
        with self.assertLogs(logger="ax", level=DEBUG) as log:
            plots = get_standard_plots(
                experiment=exp,
                model=Generators.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data()),
            )
            self.assertTrue(
                any(
                    "Pareto plotting not supported for experiments with relative "
                    "objective thresholds." in msg
                    for msg in log.output
                )
            )

            for metric_suffix in ("a", "b"):
                expected_msg = (
                    "Created contour plots for metric branin_"
                    f"{metric_suffix} and parameters ['x2', 'x1']"
                )
                self.assertTrue(any(expected_msg in msg for msg in log.output[1:]))
        self.assertEqual(len(plots), 6)

    @mock_botorch_optimize
    def _test_get_standard_plots_moo_relative_constraints(
        self, trial_is_complete: bool
    ) -> None:
        exp = get_branin_experiment_with_multi_objective(with_batch=True)
        first_obj = assert_is_instance(
            none_throws(exp.optimization_config).objective, MultiObjective
        ).objectives[0]
        first_obj.minimize = False
        first_obj.metric.lower_is_better = False
        assert_is_instance(
            exp.optimization_config, MultiObjectiveOptimizationConfig
        )._objective_thresholds = [
            ObjectiveThreshold(
                metric=exp.metrics["branin_a"], op=ComparisonOp.GEQ, bound=-100.0
            ),
            ObjectiveThreshold(
                metric=exp.metrics["branin_b"], op=ComparisonOp.LEQ, bound=100.0
            ),
        ]
        exp.trials[0].run()
        if trial_is_complete:
            exp.trials[0].mark_completed()

        for ot in assert_is_instance(
            exp.optimization_config, MultiObjectiveOptimizationConfig
        )._objective_thresholds:
            ot.relative = False
        plots = get_standard_plots(
            experiment=exp,
            model=Generators.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data()),
        )
        self.assertEqual(len(plots), 8)

    def test_get_standard_plots_moo_relative_constraints(self) -> None:
        for trial_is_complete in [False, True]:
            with self.subTest(trial_is_complete=trial_is_complete):
                self._test_get_standard_plots_moo_relative_constraints(
                    trial_is_complete=trial_is_complete
                )

    @mock_botorch_optimize
    def test_get_standard_plots_moo_no_objective_thresholds(self) -> None:
        exp = get_branin_experiment_with_multi_objective(with_batch=True)
        exp.optimization_config.objective.objectives[0].minimize = False
        exp.optimization_config.objective.objectives[1].minimize = True
        exp.trials[0].run()
        plots = get_standard_plots(
            experiment=exp,
            model=Generators.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data()),
        )
        self.assertEqual(len(plots), 7)

    @mock_botorch_optimize
    def test_get_standard_plots_map_data(self) -> None:
        exp = get_branin_experiment_with_timestamp_map_metric(with_status_quo=True)
        exp.new_trial().add_arm(exp.status_quo)
        exp.trials[0].run()
        exp.new_trial(generator_run=Generators.SOBOL(experiment=exp).gen(n=1))
        exp.trials[1].run()
        for t in exp.trials.values():
            t.mark_completed()
        plots = get_standard_plots(
            experiment=exp,
            model=Generators.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data()),
            true_objective_metric_name="branin",
        )

        self.assertEqual(len(plots), 10)
        self.assertTrue(all(isinstance(plot, go.Figure) for plot in plots))
        self.assertIn(
            "Objective branin_map vs. True Objective Metric branin",
            [p.layout.title.text for p in plots],
        )

        with self.assertRaisesRegex(
            ValueError, "Please add a valid true_objective_metric_name"
        ):
            plots = get_standard_plots(
                experiment=exp,
                model=Generators.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data()),
                true_objective_metric_name="not_present",
            )

    @mock_botorch_optimize
    def test_skip_contour_high_dimensional(self) -> None:
        exp = get_high_dimensional_branin_experiment()
        # Initial Sobol points
        sobol = Generators.SOBOL(experiment=exp)
        for _ in range(1):
            exp.new_trial(sobol.gen(1)).run()
        model = Generators.BOTORCH_MODULAR(
            experiment=exp,
            data=exp.fetch_data(),
        )
        logger = logging.getLogger("ax.service.utils.report_utils")
        logger.setLevel(DEBUG)
        with self.assertLogs(logger="ax", level=DEBUG) as log:
            _get_objective_v_param_plots(
                experiment=exp, model=model, max_num_contour_plots=2
            )
            self.assertEqual(len(log.output), 1)
            self.assertIn(
                "Created contour plots for metric objective and parameters",
                log.output[0],
            )
        with self.assertLogs(logger="ax", level=WARN) as log:
            _get_objective_v_param_plots(
                experiment=exp, model=model, max_num_contour_plots=1
            )
            self.assertEqual(len(log.output), 1)
            self.assertIn(
                "Skipping creation of contour plots",
                log.output[0],
            )
        with self.assertLogs(logger="ax", level=WARN) as log:
            _get_objective_v_param_plots(
                experiment=exp,
                model=model,
                max_num_contour_plots=1,
                max_num_slice_plots=10,
            )
            # Creates two warnings, one for slice plots and one for contour plots.
            self.assertEqual(len(log.output), 2)

    def test_get_metric_name_pairs(self) -> None:
        exp = get_branin_experiment(with_trial=True)
        exp._optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(
                objectives=[
                    Objective(metric=Metric("m0"), minimize=False),
                    Objective(metric=Metric("m1"), minimize=False),
                    Objective(metric=Metric("m2"), minimize=False),
                    Objective(metric=Metric("m3"), minimize=False),
                    Objective(metric=Metric("m4"), minimize=False),
                ]
            )
        )
        with self.assertLogs(logger="ax", level=INFO) as log:
            metric_name_pairs = _get_metric_name_pairs(experiment=exp)
            self.assertEqual(len(log.output), 1)
            self.assertIn(
                "Creating pairwise Pareto plots for the first `use_n_metrics",
                log.output[0],
            )
        self.assertListEqual(
            list(metric_name_pairs),
            list(itertools.combinations([f"m{i}" for i in range(4)], 2)),
        )

    def test_warn_if_unpredictable_metrics(self) -> None:
        expected_msg = (
            "The following metric(s) are behaving unpredictably and may be noisy or "
            "misconfigured: ['branin']. Please check that they are measuring the "
            "intended quantity, and are expected to vary reliably as a function of "
            "your parameters."
        )

        # Create Orchestrator and run a few trials.
        exp = get_branin_experiment()
        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    generator=Generators.SOBOL,
                    num_trials=3,
                    min_trials_observed=3,
                    max_parallelism=3,
                ),
                GenerationStep(
                    generator=Generators.BOTORCH_MODULAR,
                    num_trials=-1,
                    max_parallelism=3,
                ),
            ]
        )
        gs.experiment = exp
        orchestrator = Orchestrator(
            generation_strategy=gs, experiment=exp, options=OrchestratorOptions()
        )
        orchestrator.run_n_trials(1)
        msg = warn_if_unpredictable_metrics(
            experiment=exp,
            generation_strategy=gs,
            model_fit_threshold=1.0,
        )
        self.assertIsNone(msg)

        orchestrator.run_n_trials(3)

        # Set fitted model to None to test refitting.
        curr_node = orchestrator.generation_strategy._curr
        curr_node.generator_spec_to_gen_from._fitted_adapter = None

        # Threshold 1.0 (should always generate a warning)
        msg = warn_if_unpredictable_metrics(
            experiment=exp,
            generation_strategy=gs,
            model_fit_threshold=1.0,
        )
        self.assertEqual(msg, expected_msg)

        # Threshold -1.0 (should never generate a warning)
        msg = warn_if_unpredictable_metrics(
            experiment=exp,
            generation_strategy=gs,
            model_fit_threshold=-1.0,
        )
        self.assertIsNone(msg)

        # Test with no optimization config.
        exp._tracking_metrics = exp.metrics
        exp._optimization_config = None
        msg = warn_if_unpredictable_metrics(
            experiment=exp,
            generation_strategy=gs,
            model_fit_threshold=1.0,
        )
        self.assertEqual(msg, expected_msg)

        # Test with manually specified metric_names.
        msg = warn_if_unpredictable_metrics(
            experiment=exp,
            generation_strategy=gs,
            model_fit_threshold=1.0,
            metric_names=["branin"],
        )
        self.assertEqual(msg, expected_msg)

        # Test with metric name that isn't in the experiment.
        with self.assertRaisesRegex(
            ValueError, "Invalid metric names: {'bad_metric_name'}"
        ):
            warn_if_unpredictable_metrics(
                experiment=exp,
                generation_strategy=gs,
                model_fit_threshold=1.0,
                metric_names=["bad_metric_name"],
            )

    def test_find_sigfigs(self) -> None:
        self.assertEqual(_find_sigfigs(0.4, 0.5), 2)
        self.assertEqual(_find_sigfigs(0.49, 0.5), 2)
        self.assertEqual(_find_sigfigs(0.499, 0.5), 3)
        self.assertEqual(_find_sigfigs(0.111122, 0.111100), 5)
        self.assertEqual(_find_sigfigs(50.0, 50.0001), 4)
        self.assertEqual(_find_sigfigs(0.04390, 0.03947), 3)
        self.assertEqual(_find_sigfigs(49.1, 50.00001, 2), 2)

    def test_construct_comparison_message_zero_baseline(self) -> None:
        """Test construct_comparison_message returns None when baseline is 0."""
        result = construct_comparison_message(
            objective_name="metric",
            objective_minimize=True,
            baseline_arm_name="baseline",
            baseline_value=0.0,
            comparison_arm_name="comparison",
            comparison_value=10.0,
        )
        self.assertIsNone(result)

    def test_maybe_extract_baseline_comparison_values_metric_missing_soo(
        self,
    ) -> None:
        """Test returns None when metric column missing for single-objective."""
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run()
        exp.fetch_data()
        arm_names = list(exp.arms_by_name.keys())

        # Use a different metric name in optimization config that doesn't exist in data
        exp._optimization_config.objective._metric = Metric(name="nonexistent_metric")

        result = maybe_extract_baseline_comparison_values(
            experiment=exp,
            optimization_config=exp.optimization_config,
            comparison_arm_names=[arm_names[1]],
            baseline_arm_name=arm_names[0],
        )
        self.assertIsNone(result)

    def test_maybe_extract_baseline_comparison_values_metric_missing_moo(self) -> None:
        """Test returns None when metric column missing for multi-objective."""
        exp = get_branin_experiment_with_multi_objective(with_batch=True)
        exp.trials[0].run()
        exp.fetch_data()
        arm_names = list(exp.arms_by_name.keys())

        # Replace one objective metric with a nonexistent one
        moo = none_throws(exp.optimization_config).objective
        moo.objectives[0]._metric = Metric(name="nonexistent_metric")

        result = maybe_extract_baseline_comparison_values(
            experiment=exp,
            optimization_config=exp.optimization_config,
            comparison_arm_names=[arm_names[1]],
            baseline_arm_name=arm_names[0],
        )
        self.assertIsNone(result)
