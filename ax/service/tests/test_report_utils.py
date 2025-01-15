#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import itertools
from collections import namedtuple
from logging import INFO, WARN
from unittest import mock
from unittest.mock import patch, PropertyMock

import pandas as pd
from ax.core.arm import Arm
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.core.types import ComparisonOp
from ax.modelbridge.generation_node import GenerationStep
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.scheduler import Scheduler
from ax.service.utils.report_utils import (
    _format_comparison_string,
    _get_cross_validation_plots,
    _get_curve_plot_dropdown,
    _get_metric_name_pairs,
    _get_objective_trace_plot,
    _get_objective_v_param_plots,
    _get_shortest_unique_suffix_dict,
    _objective_vs_true_objective_scatter,
    BASELINE_ARM_NAME,
    compare_to_baseline,
    compute_maximum_map_values,
    exp_to_df,
    Experiment,
    FEASIBLE_COL_NAME,
    get_standard_plots,
    plot_feature_importance_by_feature_plotly,
    warn_if_unpredictable_metrics,
)
from ax.service.utils.scheduler_options import SchedulerOptions
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
    get_branin_experiment_with_timestamp_map_metric,
    get_branin_search_space,
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

    def test_exp_to_df_max_map_value(self) -> None:
        exp = get_test_map_data_experiment(num_trials=3, num_fetches=5, num_complete=0)

        def compute_maximum_map_values_timestamp(
            experiment: Experiment,
        ) -> dict[int, float]:
            return compute_maximum_map_values(
                experiment=experiment, map_key="timestamp"
            )

        df = exp_to_df(
            exp=exp,
            additional_fields_callables={  # pyre-ignore
                "timestamp": compute_maximum_map_values_timestamp
            },
        )
        self.assertEqual(df["timestamp"].tolist(), [4.0, 4.0, 4.0])

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
        with patch.object(
            Experiment, "lookup_data", lambda self: mock_results
        ), self.assertRaisesRegex(ValueError, "inconsistent experimental state"):
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
        with patch(
            f"{exp_to_df.__module__}._is_row_feasible", side_effect=KeyError(DUMMY_MSG)
        ), self.assertLogs(logger="ax", level=WARN) as log:
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
        print(relative_df)
        self.assertTrue(f"{OBJECTIVE_NAME}_%CH" in relative_df.columns.tolist())
        self.assertEqual(relative_df[f"{OBJECTIVE_NAME}_%CH"].values[0], 0.0)

    def test_get_shortest_unique_suffix_dict(self) -> None:
        expected_output = {
            "abc.123": "abc.123",
            "asdf.abc.123": "asdf.abc.123",
            "def.123": "def.123",
            "abc.456": "456",
            "": "",
            "no_delimiter": "no_delimiter",
        }
        actual_output = _get_shortest_unique_suffix_dict(
            ["abc.123", "abc.456", "def.123", "asdf.abc.123", "", "no_delimiter"]
        )
        self.assertDictEqual(expected_output, actual_output)

    @mock_botorch_optimize
    def test_get_standard_plots(self) -> None:
        exp = get_branin_experiment()
        self.assertEqual(
            len(
                get_standard_plots(
                    experiment=exp, model=get_generation_strategy().model
                )
            ),
            0,
        )
        exp = get_branin_experiment(with_batch=True, minimize=True)
        exp.trials[0].run()
        exp.trials[0].mark_completed()
        model = Models.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data())
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
            self.assertEqual(len(plots), num_expected_plots)  # TODO: this failed
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
        # NOTE: level set to INFO in this block, because the global sensitivity
        # analysis raises an INFO level log entry here. Leaving level=WARN here
        # actually passes on Python 3.8 because of a language internal bug. See
        # https://bugs.python.org/issue41943 for more information.
        with self.assertLogs(logger="ax", level=INFO) as log:
            plots = get_standard_plots(
                experiment=exp,
                model=Models.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data()),
            )
            self.assertEqual(len(log.output), 3)
            self.assertIn(
                "Pareto plotting not supported for experiments with relative objective "
                "thresholds.",
                log.output[0],
            )
            for metric_suffix in ("a", "b"):
                expected_msg = (
                    "Created contour plots for metric branin_"
                    f"{metric_suffix} and parameters ['x2', 'x1']"
                )
                self.assertTrue(any(expected_msg in msg for msg in log.output[1:]))
        self.assertEqual(len(plots), 6)

    @mock_botorch_optimize
    def test_get_standard_plots_moo_relative_constraints(self) -> None:
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

        for ot in assert_is_instance(
            exp.optimization_config, MultiObjectiveOptimizationConfig
        )._objective_thresholds:
            ot.relative = False
        plots = get_standard_plots(
            experiment=exp,
            model=Models.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data()),
        )
        self.assertEqual(len(plots), 8)

    @mock_botorch_optimize
    def test_get_standard_plots_moo_no_objective_thresholds(self) -> None:
        exp = get_branin_experiment_with_multi_objective(with_batch=True)
        exp.optimization_config.objective.objectives[0].minimize = False
        exp.optimization_config.objective.objectives[1].minimize = True
        exp.trials[0].run()
        plots = get_standard_plots(
            experiment=exp,
            model=Models.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data()),
        )
        self.assertEqual(len(plots), 8)

    @mock_botorch_optimize
    def test_get_standard_plots_map_data(self) -> None:
        exp = get_branin_experiment_with_timestamp_map_metric(with_status_quo=True)
        exp.new_trial().add_arm(exp.status_quo)
        exp.trials[0].run()
        exp.new_trial(
            generator_run=Models.SOBOL(search_space=exp.search_space).gen(n=1)
        )
        exp.trials[1].run()
        for t in exp.trials.values():
            t.mark_completed()
        plots = get_standard_plots(
            experiment=exp,
            model=Models.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data()),
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
                model=Models.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data()),
                true_objective_metric_name="not_present",
            )

    @mock_botorch_optimize
    def test_skip_contour_high_dimensional(self) -> None:
        exp = get_high_dimensional_branin_experiment()
        # Initial Sobol points
        sobol = Models.SOBOL(search_space=exp.search_space)
        for _ in range(1):
            exp.new_trial(sobol.gen(1)).run()
        model = Models.BOTORCH_MODULAR(
            experiment=exp,
            data=exp.fetch_data(),
        )
        with self.assertLogs(logger="ax", level=INFO) as log:
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

    def test_compare_to_baseline(self) -> None:
        """Test that compare to baseline parses arm df properly,
        obtains the objective metric values based
        on the provided OptimizationConfig, and
        produces the intended text
        """
        self.maxDiff = None
        OBJECTIVE_METRIC = "foo"

        data = [
            {"trial_index": 0, "arm_name": BASELINE_ARM_NAME, OBJECTIVE_METRIC: 0.2},
            {"trial_index": 1, "arm_name": "dummy", OBJECTIVE_METRIC: 0.5},
            {"trial_index": 2, "arm_name": "optimal", OBJECTIVE_METRIC: 2.5},
            {"trial_index": 3, "arm_name": "bad_optimal", OBJECTIVE_METRIC: 0.05},
        ]
        arms_df = pd.DataFrame(data)

        arms_by_name_mock = {
            BASELINE_ARM_NAME: Arm(name=BASELINE_ARM_NAME, parameters={}),
            "dummy": Arm(name="dummy", parameters={}),
            "optimal": Arm(name="optimal", parameters={}),
            "bad_optimal": Arm(name="bad_optimal", parameters={}),
        }

        with patch(
            "ax.service.utils.report_utils.exp_to_df",
            return_value=arms_df,
        ), patch.object(
            Experiment,
            "arms_by_name",
            new_callable=PropertyMock,
            return_value=arms_by_name_mock,
        ):
            true_obj_metric = Metric(name=OBJECTIVE_METRIC, lower_is_better=False)
            experiment = Experiment(
                search_space=get_branin_search_space(),
                tracking_metrics=[true_obj_metric],
            )

            optimization_config = OptimizationConfig(
                objective=Objective(metric=true_obj_metric, minimize=False),
                outcome_constraints=[],
            )
            experiment.optimization_config = optimization_config

            comparison_arm_names = ["optimal"]

            result = compare_to_baseline(
                experiment=experiment,
                optimization_config=None,
                comparison_arm_names=comparison_arm_names,
                baseline_arm_name=BASELINE_ARM_NAME,
            )

            output_text = _format_comparison_string(
                comparison_arm_name=comparison_arm_names[0],
                baseline_arm_name=BASELINE_ARM_NAME,
                objective_name=OBJECTIVE_METRIC,
                percent_change=1150.0,
                baseline_value=0.2,
                comparison_value=2.5,
                digits=2,
            )

            self.assertNotEqual(result, None)
            self.assertEqual(result, output_text)

            bad_comparison_arm_names = ["bad_optimal"]
            # because result increased from baseline, no improvement result returned
            bad_result = compare_to_baseline(
                experiment=experiment,
                optimization_config=None,
                comparison_arm_names=bad_comparison_arm_names,
                baseline_arm_name=BASELINE_ARM_NAME,
            )
            self.assertEqual(bad_result, None)

    def test_compare_to_baseline_pass_in_opt(self) -> None:
        """Test that compare to baseline parses arm df properly,
        obtains the objective metric values based
        on the provided OptimizationConfig, and
        produces the intended text
        """
        self.maxDiff = None
        OBJECTIVE_METRIC = "foo"

        data = [
            {"trial_index": 0, "arm_name": BASELINE_ARM_NAME, OBJECTIVE_METRIC: 0.2},
            {"trial_index": 1, "arm_name": "dummy", OBJECTIVE_METRIC: 0.5},
            {"trial_index": 2, "arm_name": "optimal", OBJECTIVE_METRIC: 2.5},
            {"trial_index": 3, "arm_name": "bad_optimal", OBJECTIVE_METRIC: 0.05},
        ]
        arms_df = pd.DataFrame(data)

        arms_by_name_mock = {
            BASELINE_ARM_NAME: Arm(name=BASELINE_ARM_NAME, parameters={}),
            "dummy": Arm(name="dummy", parameters={}),
            "optimal": Arm(name="optimal", parameters={}),
            "bad_optimal": Arm(name="bad_optimal", parameters={}),
        }

        with patch(
            "ax.service.utils.report_utils.exp_to_df",
            return_value=arms_df,
        ), patch.object(
            Experiment,
            "arms_by_name",
            new_callable=PropertyMock,
            return_value=arms_by_name_mock,
        ):
            true_obj_metric = Metric(name=OBJECTIVE_METRIC, lower_is_better=False)
            experiment = Experiment(
                search_space=get_branin_search_space(),
                tracking_metrics=[true_obj_metric],
                optimization_config=None,
            )

            optimization_config = OptimizationConfig(
                objective=Objective(metric=true_obj_metric, minimize=False),
                outcome_constraints=[],
            )
            experiment.optimization_config = optimization_config

            comparison_arm_names = ["optimal"]

            result = compare_to_baseline(
                experiment=experiment,
                optimization_config=optimization_config,
                comparison_arm_names=comparison_arm_names,
                baseline_arm_name=BASELINE_ARM_NAME,
            )

            output_text = _format_comparison_string(
                comparison_arm_name=comparison_arm_names[0],
                baseline_arm_name=BASELINE_ARM_NAME,
                objective_name=OBJECTIVE_METRIC,
                percent_change=1150.0,
                baseline_value=0.2,
                comparison_value=2.5,
                digits=2,
            )

            self.assertNotEqual(result, None)
            self.assertEqual(result, output_text)

    def test_compare_to_baseline_minimize(self) -> None:
        """Test that compare to baseline parses arm df properly,
        obtains the objective metric values based
        on the provided OptimizationConfig, and
        produces the intended text.
        For the minimize case.
        Also will use a custom baseline arm name.
        """
        self.maxDiff = None
        OBJECTIVE_METRIC = "foo"
        custom_baseline_arm_name = "custom_baseline"

        data = [
            {
                "trial_index": 0,
                "arm_name": custom_baseline_arm_name,
                OBJECTIVE_METRIC: 0.2,
            },
            {"trial_index": 1, "arm_name": "dummy", OBJECTIVE_METRIC: 0.5},
            {"trial_index": 2, "arm_name": "optimal", OBJECTIVE_METRIC: 0.1},
            {"trial_index": 3, "arm_name": "bad_optimal", OBJECTIVE_METRIC: 1.0},
        ]
        arms_df = pd.DataFrame(data)

        arms_by_name_mock = {
            custom_baseline_arm_name: Arm(name=custom_baseline_arm_name, parameters={}),
            "dummy": Arm(name="dummy", parameters={}),
            "optimal": Arm(name="optimal", parameters={}),
            "bad_optimal": Arm(name="bad_optimal", parameters={}),
        }

        with patch(
            "ax.service.utils.report_utils.exp_to_df",
            return_value=arms_df,
        ), patch.object(
            Experiment,
            "arms_by_name",
            new_callable=PropertyMock,
            return_value=arms_by_name_mock,
        ):
            true_obj_metric = Metric(name=OBJECTIVE_METRIC, lower_is_better=True)
            experiment = Experiment(
                search_space=get_branin_search_space(),
                tracking_metrics=[true_obj_metric],
            )

            optimization_config = OptimizationConfig(
                objective=Objective(metric=true_obj_metric, minimize=True),
                outcome_constraints=[],
            )
            experiment.optimization_config = optimization_config

            comparison_arm_names = ["optimal"]

            result = compare_to_baseline(
                experiment=experiment,
                optimization_config=None,
                comparison_arm_names=comparison_arm_names,
                baseline_arm_name=custom_baseline_arm_name,
            )

            output_text = _format_comparison_string(
                comparison_arm_name=comparison_arm_names[0],
                baseline_arm_name=custom_baseline_arm_name,
                objective_name=OBJECTIVE_METRIC,
                percent_change=50.0,
                baseline_value=0.2,
                comparison_value=0.1,
                digits=2,
            )

            self.assertNotEqual(result, None)
            self.assertEqual(result, output_text)

            bad_comparison_arm_names = ["bad_optimal"]
            # because result increased from baseline, no improvement result returned
            bad_result = compare_to_baseline(
                experiment=experiment,
                optimization_config=None,
                comparison_arm_names=bad_comparison_arm_names,
                baseline_arm_name=BASELINE_ARM_NAME,
            )
            self.assertEqual(bad_result, None)

    def test_compare_to_baseline_edge_case(self) -> None:
        """Test that compare to baseline parses arm df properly,
        obtains the objective metric values based
        on the provided OptimizationConfig, and
        produces the intended text
        """
        self.maxDiff = None
        OBJECTIVE_METRIC = "foo"

        true_obj_metric = Metric(name=OBJECTIVE_METRIC, lower_is_better=True)
        experiment = Experiment(
            search_space=get_branin_search_space(),
            tracking_metrics=[true_obj_metric],
        )

        optimization_config = OptimizationConfig(
            objective=Objective(metric=true_obj_metric, minimize=True),
            outcome_constraints=[],
        )
        experiment.optimization_config = optimization_config
        comparison_arm_names = ["optimal"]

        # baseline value is 0
        data = [
            {"trial_index": 0, "arm_name": BASELINE_ARM_NAME, OBJECTIVE_METRIC: 0.0},
            {"trial_index": 1, "arm_name": "optimal", OBJECTIVE_METRIC: 1.0},
        ]
        arms_df = pd.DataFrame(data)
        arms_by_name_mock = {
            BASELINE_ARM_NAME: Arm(name=BASELINE_ARM_NAME, parameters={}),
            "dummy": Arm(name="dummy", parameters={}),
            "optimal": Arm(name="optimal", parameters={}),
            "bad_optimal": Arm(name="bad_optimal", parameters={}),
        }

        with patch(
            "ax.service.utils.report_utils.exp_to_df",
            return_value=arms_df,
        ), patch.object(
            Experiment,
            "arms_by_name",
            new_callable=PropertyMock,
            return_value=arms_by_name_mock,
        ):
            with self.assertLogs("ax", level=INFO) as log:
                self.assertEqual(
                    compare_to_baseline(
                        experiment=experiment,
                        optimization_config=None,
                        comparison_arm_names=comparison_arm_names,
                        baseline_arm_name=BASELINE_ARM_NAME,
                    ),
                    None,
                )
                self.assertTrue(
                    any(
                        (
                            "compare_to_baseline: baseline has value of 0"
                            ", can't compute percent change."
                        )
                        in log_str
                        for log_str in log.output
                    ),
                    log.output,
                )

        # no best arm names
        with patch(
            "ax.service.utils.report_utils.exp_to_df",
            return_value=arms_df,
        ):
            with self.assertLogs("ax", level=INFO) as log:
                self.assertEqual(
                    compare_to_baseline(
                        experiment=experiment,
                        optimization_config=None,
                        comparison_arm_names=None,
                        baseline_arm_name=BASELINE_ARM_NAME,
                    ),
                    None,
                )
                self.assertTrue(
                    any(
                        (
                            "compare_to_baseline: comparison_arm_names not provided."
                            " Returning None."
                        )
                        in log_str
                        for log_str in log.output
                    ),
                    log.output,
                )

        # no optimization config
        with patch(
            "ax.service.utils.report_utils.exp_to_df",
            return_value=arms_df,
        ):
            with self.assertLogs("ax", level=INFO) as log:
                exp_no_opt = Experiment(
                    search_space=get_branin_search_space(),
                    tracking_metrics=[true_obj_metric],
                    optimization_config=None,
                )
                self.assertEqual(
                    compare_to_baseline(
                        experiment=exp_no_opt,
                        optimization_config=None,
                        comparison_arm_names=comparison_arm_names,
                        baseline_arm_name=BASELINE_ARM_NAME,
                    ),
                    None,
                )
                self.assertEqual(exp_no_opt.optimization_config, None)
                self.assertTrue(
                    any(
                        (
                            "compare_to_baseline: optimization_config neither provided "
                            "in inputs nor present on experiment."
                        )
                        in log_str
                        for log_str in log.output
                    ),
                    log.output,
                )

    def test_compare_to_baseline_arms_not_found(self) -> None:
        self.maxDiff = None
        OBJECTIVE_METRIC = "foo"

        true_obj_metric = Metric(name=OBJECTIVE_METRIC, lower_is_better=True)
        experiment = Experiment(
            search_space=get_branin_search_space(),
            tracking_metrics=[true_obj_metric],
        )

        optimization_config = OptimizationConfig(
            objective=Objective(metric=true_obj_metric, minimize=True),
            outcome_constraints=[],
        )
        experiment.optimization_config = optimization_config
        comparison_arm_names = ["optimal"]

        # baseline value is 0
        data = [
            {"trial_index": 0, "arm_name": BASELINE_ARM_NAME, OBJECTIVE_METRIC: 0.0},
            {"trial_index": 1, "arm_name": "optimal", OBJECTIVE_METRIC: 1.0},
        ]
        arms_df = pd.DataFrame(data)

        # no arms df
        with patch(
            "ax.service.utils.report_utils.exp_to_df",
            return_value=None,
        ):
            with self.assertLogs("ax", level=INFO) as log:
                self.assertEqual(
                    compare_to_baseline(
                        experiment=experiment,
                        optimization_config=None,
                        comparison_arm_names=comparison_arm_names,
                        baseline_arm_name=BASELINE_ARM_NAME,
                    ),
                    None,
                )
                self.assertTrue(
                    any(
                        ("compare_to_baseline: arms_df is None.") in log_str
                        for log_str in log.output
                    ),
                    log.output,
                )

        # best arms df is none
        with patch(
            "ax.service.utils.report_utils.exp_to_df",
            return_value=arms_df,
        ):
            with self.assertLogs("ax", level=INFO) as log:
                comparison_arm_not_found = ["unknown_arm"]
                self.assertEqual(
                    compare_to_baseline(
                        experiment=experiment,
                        optimization_config=None,
                        comparison_arm_names=comparison_arm_not_found,
                        baseline_arm_name=BASELINE_ARM_NAME,
                    ),
                    None,
                )
                self.assertTrue(
                    any(
                        ("compare_to_baseline: comparison_arm_df has no rows.")
                        in log_str
                        for log_str in log.output
                    ),
                    log.output,
                )

        # baseline not found in arms_df
        with patch(
            "ax.service.utils.report_utils.exp_to_df",
            return_value=arms_df,
        ):
            experiment_with_status_quo = experiment
            experiment_with_status_quo.status_quo = Arm(
                name=BASELINE_ARM_NAME,
                parameters={"x1": 0, "x2": 0},
            )
            baseline_arm_name = "not_baseline_arm_in_dataframe"
            with self.assertLogs("ax", level=INFO) as log:
                self.assertEqual(
                    compare_to_baseline(
                        experiment=experiment_with_status_quo,
                        optimization_config=None,
                        comparison_arm_names=comparison_arm_names,
                        baseline_arm_name=baseline_arm_name,
                    ),
                    None,
                )
                self.assertTrue(
                    any(
                        (f"Arm by name {baseline_arm_name=} not found.") in log_str
                        for log_str in log.output
                    ),
                    log.output,
                )

    def test_compare_to_baseline_moo(self) -> None:
        """Test that compare to baseline errors out correctly
        for multi objective problems

        """
        self.maxDiff = None

        data = [
            {
                "trial_index": 0,
                "arm_name": BASELINE_ARM_NAME,
                "m0": 1.0,
                "m1": 1.0,
                "m2": 1.0,
                "m3": 1.0,
            },
            {
                "trial_index": 1,
                "arm_name": "opt_0",
                "m0": 2.5,
                "m1": 0.2,
                "m2": 1.0,
                "m3": 1.0,
            },
            {
                "trial_index": 2,
                "arm_name": "opt_1_min",
                "m0": 1.2,
                "m1": -0.1,
                "m2": 1.0,
                "m3": 1.0,
            },
            {
                "trial_index": 3,
                "arm_name": "opt_3",
                "m0": 0.5,
                "m1": 0.5,
                "m2": 1.0,
                "m3": 1.01,
            },
        ]
        arms_df = pd.DataFrame(data)

        arms_by_name_mock = {
            BASELINE_ARM_NAME: Arm(name=BASELINE_ARM_NAME, parameters={}),
            "dummy": Arm(name="dummy", parameters={}),
            "optimal": Arm(name="optimal", parameters={}),
            "bad_optimal": Arm(name="bad_optimal", parameters={}),
        }

        with patch(
            "ax.service.utils.report_utils.exp_to_df",
            return_value=arms_df,
        ), patch.object(
            Experiment,
            "arms_by_name",
            new_callable=PropertyMock,
            return_value=arms_by_name_mock,
        ):
            m0 = Metric(name="m0", lower_is_better=False)
            m1 = Metric(name="m1", lower_is_better=True)
            m2 = Metric(name="m2", lower_is_better=False)
            m3 = Metric(name="m3", lower_is_better=False)
            experiment = Experiment(
                search_space=get_branin_search_space(),
                tracking_metrics=[m0, m1, m2, m3],
            )

            optimization_config = MultiObjectiveOptimizationConfig(
                objective=MultiObjective(
                    objectives=[
                        Objective(metric=Metric("m0"), minimize=False),
                        Objective(metric=Metric("m1"), minimize=True),
                        Objective(metric=Metric("m3"), minimize=False),
                    ]
                )
            )
            experiment.optimization_config = optimization_config
            self.assertEqual(True, experiment.is_moo_problem)

            comparison_arm_names = ["opt_0", "opt_1_min", "opt_3"]

            preamble = (
                "Below is the greatest improvement, if any,"
                " achieved for each objective metric \n"
            )
            output_text_0 = _format_comparison_string(
                comparison_arm_name="opt_0",
                baseline_arm_name=BASELINE_ARM_NAME,
                objective_name="m0",
                percent_change=150.0,
                baseline_value=1.0,
                comparison_value=2.5,
                digits=2,
            )
            output_text_1 = _format_comparison_string(
                comparison_arm_name="opt_1_min",
                baseline_arm_name=BASELINE_ARM_NAME,
                objective_name="m1",
                percent_change=110.0,
                baseline_value=1.0,
                comparison_value=-0.1,
                digits=2,
            )
            output_text_3 = _format_comparison_string(
                comparison_arm_name="opt_3",
                baseline_arm_name=BASELINE_ARM_NAME,
                objective_name="m3",
                percent_change=1.0,
                baseline_value=1.0,
                comparison_value=1.01,
                digits=2,
            )

            result = none_throws(
                compare_to_baseline(
                    experiment=experiment,
                    optimization_config=None,
                    comparison_arm_names=comparison_arm_names,
                    baseline_arm_name=BASELINE_ARM_NAME,
                ),
            )

            expected_result = (
                preamble
                + " \n* "
                + output_text_0
                + " \n* "
                + output_text_1
                + " \n* "
                + output_text_3
            )

            self.assertEqual(result, expected_result)

    def test_compare_to_baseline_equal(self) -> None:
        """Test case where baseline value is equal to optimal value"""
        self.maxDiff = None
        OBJECTIVE_METRIC = "foo"
        custom_baseline_arm_name = "custom_baseline"

        data = [
            {
                "trial_index": 0,
                "arm_name": custom_baseline_arm_name,
                OBJECTIVE_METRIC: 0.2,
            },
            {"trial_index": 1, "arm_name": "dummy", OBJECTIVE_METRIC: 0.5},
            {"trial_index": 2, "arm_name": "optimal", OBJECTIVE_METRIC: 0.5},
            {"trial_index": 3, "arm_name": "equal", OBJECTIVE_METRIC: 0.2},
        ]
        arms_df = pd.DataFrame(data)

        true_obj_metric = Metric(name=OBJECTIVE_METRIC, lower_is_better=True)
        experiment = Experiment(
            search_space=get_branin_search_space(),
            tracking_metrics=[true_obj_metric],
        )

        optimization_config = OptimizationConfig(
            objective=Objective(metric=true_obj_metric, minimize=True),
            outcome_constraints=[],
        )
        experiment.optimization_config = optimization_config

        comparison_arm_names = ["equal"]

        with patch(
            "ax.service.utils.report_utils.exp_to_df",
            return_value=arms_df,
        ):
            result = compare_to_baseline(
                experiment=experiment,
                optimization_config=optimization_config,
                comparison_arm_names=comparison_arm_names,
                baseline_arm_name=custom_baseline_arm_name,
            )

            self.assertIsNone(result)

    def test_warn_if_unpredictable_metrics(self) -> None:
        expected_msg = (
            "The following metric(s) are behaving unpredictably and may be noisy or "
            "misconfigured: ['branin']. Please check that they are measuring the "
            "intended quantity, and are expected to vary reliably as a function of "
            "your parameters."
        )

        # Create scheduler and run a few trials.
        exp = get_branin_experiment()
        gs = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=3,
                    min_trials_observed=3,
                    max_parallelism=3,
                ),
                GenerationStep(
                    model=Models.BOTORCH_MODULAR, num_trials=-1, max_parallelism=3
                ),
            ]
        )
        gs.experiment = exp
        scheduler = Scheduler(
            generation_strategy=gs, experiment=exp, options=SchedulerOptions()
        )
        scheduler.run_n_trials(1)
        msg = warn_if_unpredictable_metrics(
            experiment=exp,
            generation_strategy=gs,
            model_fit_threshold=1.0,
        )
        self.assertIsNone(msg)

        scheduler.run_n_trials(3)

        # Set fitted model to None to test refitting.
        scheduler.generation_strategy._curr.model_spec_to_gen_from._fitted_model = None

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
