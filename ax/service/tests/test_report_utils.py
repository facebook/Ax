#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
from logging import INFO, WARN
from unittest import mock
from unittest.mock import patch

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
    _objective_vs_true_objective_scatter,
    BASELINE_ARM_NAME,
    compare_to_baseline,
    Experiment,
    get_standard_plots,
    plot_feature_importance_by_feature_plotly,
    select_baseline_arm,
    warn_if_unpredictable_metrics,
)
from ax.service.utils.scheduler_options import SchedulerOptions
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast, not_none
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
    get_branin_experiment_with_timestamp_map_metric,
    get_branin_search_space,
    get_high_dimensional_branin_experiment,
)
from ax.utils.testing.mock import fast_botorch_optimize
from ax.utils.testing.modeling_stubs import get_generation_strategy
from plotly import graph_objects as go


class ReportUtilsTest(TestCase):
    @fast_botorch_optimize
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
            self.assertEqual(len(plots), num_expected_plots)
            self.assertTrue(all(isinstance(plot, go.Figure) for plot in plots))

    @fast_botorch_optimize
    def test_get_standard_plots_moo(self) -> None:
        exp = get_branin_experiment_with_multi_objective(with_batch=True)
        exp.optimization_config.objective.objectives[0].minimize = False
        exp.optimization_config.objective.objectives[1].minimize = True
        checked_cast(
            MultiObjectiveOptimizationConfig, exp.optimization_config
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
                experiment=exp, model=Models.MOO(experiment=exp, data=exp.fetch_data())
            )
            self.assertEqual(len(log.output), 2)
            self.assertIn(
                "Pareto plotting not supported for experiments with relative objective "
                "thresholds.",
                log.output[0],
            )
            self.assertIn(
                "Failed to compute global feature sensitivities:",
                log.output[1],
            )
        self.assertEqual(len(plots), 6)

    @fast_botorch_optimize
    def test_get_standard_plots_moo_relative_constraints(self) -> None:
        exp = get_branin_experiment_with_multi_objective(with_batch=True)
        exp.optimization_config.objective.objectives[0].minimize = False
        exp.optimization_config.objective.objectives[1].minimize = True
        checked_cast(
            MultiObjectiveOptimizationConfig, exp.optimization_config
        )._objective_thresholds = [
            ObjectiveThreshold(
                metric=exp.metrics["branin_a"], op=ComparisonOp.GEQ, bound=-100.0
            ),
            ObjectiveThreshold(
                metric=exp.metrics["branin_b"], op=ComparisonOp.LEQ, bound=100.0
            ),
        ]
        exp.trials[0].run()

        for ot in checked_cast(
            MultiObjectiveOptimizationConfig, exp.optimization_config
        )._objective_thresholds:
            ot.relative = False
        plots = get_standard_plots(
            experiment=exp, model=Models.MOO(experiment=exp, data=exp.fetch_data())
        )
        self.assertEqual(len(plots), 8)

    @fast_botorch_optimize
    def test_get_standard_plots_moo_no_objective_thresholds(self) -> None:
        exp = get_branin_experiment_with_multi_objective(with_batch=True)
        exp.optimization_config.objective.objectives[0].minimize = False
        exp.optimization_config.objective.objectives[1].minimize = True
        exp.trials[0].run()
        plots = get_standard_plots(
            experiment=exp, model=Models.MOO(experiment=exp, data=exp.fetch_data())
        )
        self.assertEqual(len(plots), 8)

    @fast_botorch_optimize
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
            [p.layout.title.text for p in plots],  # pyre-ignore[16]
        )

        with self.assertRaisesRegex(
            ValueError, "Please add a valid true_objective_metric_name"
        ):
            plots = get_standard_plots(
                experiment=exp,
                model=Models.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data()),
                true_objective_metric_name="not_present",
            )

    @fast_botorch_optimize
    def test_skip_contour_high_dimensional(self) -> None:
        exp = get_high_dimensional_branin_experiment()
        # Initial Sobol points
        sobol = Models.SOBOL(search_space=exp.search_space)
        for _ in range(1):
            exp.new_trial(sobol.gen(1)).run()
        model = Models.GPEI(
            experiment=exp,
            data=exp.fetch_data(),
        )
        with self.assertLogs(logger="ax", level=WARN) as log:
            _get_objective_v_param_plots(experiment=exp, model=model)
            self.assertEqual(len(log.output), 1)
            self.assertIn("Skipping creation of 2450 contour plots", log.output[0])
            _get_objective_v_param_plots(
                experiment=exp, model=model, max_num_slice_plots=10
            )
            # Adds two more warnings.
            self.assertEqual(len(log.output), 3)
            self.assertIn("Skipping creation of 50 slice plots", log.output[1])

    def test_get_metric_name_pairs(self) -> None:
        exp = get_branin_experiment(with_trial=True)
        exp._optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(
                objectives=[
                    Objective(metric=Metric("m0")),
                    Objective(metric=Metric("m1")),
                    Objective(metric=Metric("m2")),
                    Objective(metric=Metric("m3")),
                    Objective(metric=Metric("m4")),
                ]
            )
        )
        with self.assertLogs(logger="ax", level=WARN) as log:
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

        with patch(
            "ax.service.utils.report_utils.exp_to_df",
            return_value=arms_df,
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

        with patch(
            "ax.service.utils.report_utils.exp_to_df",
            return_value=arms_df,
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

        with patch(
            "ax.service.utils.report_utils.exp_to_df",
            return_value=arms_df,
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

        with patch(
            "ax.service.utils.report_utils.exp_to_df",
            return_value=arms_df,
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
                        (
                            f"compare_to_baseline: baseline row: {baseline_arm_name=}"
                            " not found in arms"
                        )
                        in log_str
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

        with patch(
            "ax.service.utils.report_utils.exp_to_df",
            return_value=arms_df,
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
                        Objective(metric=Metric("m0")),
                        Objective(metric=Metric("m1"), minimize=True),
                        Objective(metric=Metric("m3")),
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

            result = not_none(
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
            with self.assertLogs("ax", level=INFO) as log:
                result = compare_to_baseline(
                    experiment=experiment,
                    optimization_config=optimization_config,
                    comparison_arm_names=comparison_arm_names,
                    baseline_arm_name=custom_baseline_arm_name,
                )
                self.assertEqual(result, None)
                self.assertTrue(
                    any(
                        (
                            "compare_to_baseline:"
                            f" comparison arm equal"
                            f" did not beat baseline arm {custom_baseline_arm_name}."
                        )
                        in log_str
                        for log_str in log.output
                    ),
                    log.output,
                )

    def test_compare_to_baseline_select_baseline_arm(self) -> None:
        OBJECTIVE_METRIC = "objective"
        true_obj_metric = Metric(name=OBJECTIVE_METRIC, lower_is_better=True)
        experiment = Experiment(
            search_space=get_branin_search_space(),
            tracking_metrics=[true_obj_metric],
        )

        # specified baseline
        data = [
            {
                "trial_index": 0,
                "arm_name": "m_0",
                OBJECTIVE_METRIC: 0.2,
            },
            {
                "trial_index": 1,
                "arm_name": BASELINE_ARM_NAME,
                OBJECTIVE_METRIC: 0.2,
            },
            {
                "trial_index": 2,
                "arm_name": "status_quo",
                OBJECTIVE_METRIC: 0.2,
            },
        ]
        arms_df = pd.DataFrame(data)
        self.assertEqual(
            select_baseline_arm(
                experiment=experiment,
                arms_df=arms_df,
                baseline_arm_name=BASELINE_ARM_NAME,
            ),
            (BASELINE_ARM_NAME, False),
        )

        # specified baseline arm not in trial
        wrong_baseline_name = "wrong_baseline_name"
        with self.assertRaisesRegex(
            ValueError,
            "compare_to_baseline: baseline row: .*" + " not found in arms",
        ):
            select_baseline_arm(
                experiment=experiment,
                arms_df=arms_df,
                baseline_arm_name=wrong_baseline_name,
            ),

        # status quo baseline arm
        experiment_with_status_quo = copy.deepcopy(experiment)
        experiment_with_status_quo.status_quo = Arm(
            name="status_quo",
            parameters={"x1": 0, "x2": 0},
        )
        self.assertEqual(
            select_baseline_arm(
                experiment=experiment_with_status_quo,
                arms_df=arms_df,
                baseline_arm_name=None,
            ),
            ("status_quo", False),
        )
        # first arm from trials
        custom_arm = Arm(name="m_0", parameters={"x1": 0.1, "x2": 0.2})
        experiment.new_trial().add_arm(custom_arm)
        self.assertEqual(
            select_baseline_arm(
                experiment=experiment,
                arms_df=arms_df,
                baseline_arm_name=None,
            ),
            ("m_0", True),
        )

        # none selected
        experiment_with_no_valid_baseline = Experiment(
            search_space=get_branin_search_space(),
            tracking_metrics=[true_obj_metric],
        )
        experiment_with_no_valid_baseline.status_quo = Arm(
            name="not found",
            parameters={"x1": 0, "x2": 0},
        )
        custom_arm = Arm(name="also not found", parameters={"x1": 0.1, "x2": 0.2})
        experiment_with_no_valid_baseline.new_trial().add_arm(custom_arm)
        with self.assertRaisesRegex(
            ValueError, "compare_to_baseline: could not find valid baseline arm"
        ):
            select_baseline_arm(
                experiment=experiment_with_no_valid_baseline,
                arms_df=arms_df,
                baseline_arm_name=None,
            )

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
                GenerationStep(model=Models.GPEI, num_trials=-1, max_parallelism=3),
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
