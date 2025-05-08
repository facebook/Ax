#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from unittest import mock
from unittest.mock import patch, PropertyMock

import numpy as np

import pandas as pd
import torch
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import ObjectiveThreshold, OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.exceptions.core import DataRequiredError
from ax.generation_strategy.dispatch_utils import choose_generation_strategy_legacy
from ax.modelbridge.cross_validation import AssessModelFitResult
from ax.modelbridge.registry import Generators
from ax.modelbridge.torch import TorchAdapter
from ax.service.ax_client import AxClient
from ax.service.utils.best_point import (
    _extract_best_arm_from_gr,
    _is_row_feasible,
    derelativize_opt_config,
    get_best_by_raw_objective_with_trial_index,
    get_best_parameters_from_model_predictions_with_trial_index,
    get_best_raw_objective_point_with_trial_index,
    get_hypervolume_trace_of_outcomes_multi_objective,
    get_trace_by_arm_pull_from_data,
    get_values_of_outcomes_single_or_scalarized_objective,
    logger as best_point_logger,
)
from ax.service.utils.best_point_utils import select_baseline_name_default_first_trial
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_metric,
    get_branin_search_space,
    get_experiment_with_map_data,
    get_experiment_with_observations,
)
from ax.utils.testing.mock import mock_botorch_optimize
from pyre_extensions import assert_is_instance, none_throws

best_point_module: str = get_best_by_raw_objective_with_trial_index.__module__
DUMMY_OPTIMIZATION_CONFIG = "test_optimization_config"


class TestBestPointUtils(TestCase):
    """Testing the best point utilities functionality that is not tested in
    main `AxClient` testing suite (`TestServiceAPI`)."""

    def test_get_values_of_outcomes_single_or_scalarized_objective(self) -> None:
        m1_value = 37
        m2_value = 49
        df = pd.DataFrame.from_records(
            [
                {"m1": m1_value, "m2": m2_value, "irrelevant": 4, "feasible": True},
                {"m1": 2, "m2": 3, "irrelevant": 4, "feasible": False},
            ]
        )
        metrics = [Metric(name="m1"), Metric(name="m2")]
        with self.subTest("MultiObjective not supported"):
            objective = MultiObjective(
                objectives=[Objective(metric=m, minimize=False) for m in metrics],
            )
            with self.assertRaisesRegex(ValueError, "MultiObjective is not supported"):
                get_values_of_outcomes_single_or_scalarized_objective(
                    df_wide=df, objective=objective
                )

        for minimize in [False, True]:
            objective = Objective(
                metric=Metric(name="m1", lower_is_better=minimize), minimize=minimize
            )
            worst = float("inf") if minimize else float("-inf")
            with self.subTest(f"Single objective, minimize={minimize}"):
                value = get_values_of_outcomes_single_or_scalarized_objective(
                    df_wide=df, objective=objective
                )
                self.assertIsInstance(value, np.ndarray)
                self.assertEqual(value.tolist(), [m1_value, worst])

            objective = ScalarizedObjective(
                metrics=metrics, weights=[0.4, 0.6], minimize=minimize
            )

            with self.subTest(f"ScalarizedObjective, {minimize=}"):
                value = get_values_of_outcomes_single_or_scalarized_objective(
                    df_wide=df, objective=objective
                )
                self.assertEqual(len(value), 2)
                self.assertAlmostEqual(value[0], 0.4 * m1_value + 0.6 * m2_value)
                self.assertEqual(value[1], worst)

            objective = ScalarizedObjective(
                metrics=metrics, weights=[0.4, -0.6], minimize=minimize
            )
            with self.subTest(
                f"ScalarizedObjective, {minimize=}, maximize other metric"
            ):
                value = get_values_of_outcomes_single_or_scalarized_objective(
                    df_wide=df, objective=objective
                )
                self.assertEqual(len(value), 2)
                self.assertEqual(value[0], 0.4 * m1_value - 0.6 * m2_value)
                self.assertEqual(value[1], worst)

    def test_get_hypervolume_trace_of_outcomes_multi_objective(self) -> None:
        objective = MultiObjective(
            objectives=[
                Objective(metric=Metric("m1"), minimize=False),
                Objective(metric=Metric("m2"), minimize=False),
            ],
        )
        # Objective thresholds will be inferred to be [1, 1]
        df_wide = pd.DataFrame.from_records(
            [
                {"m1": 1.0, "m2": 1.0, "feasible": True},
                {"m1": 2.0, "m2": 3.0, "feasible": True},
                {"m1": 4.0, "m2": 4.0, "feasible": False},
                {"m1": 3.0, "m2": 2.0, "feasible": True},
            ]
        )
        with self.subTest("Relative objective thresholds not supported"):
            optimization_config = MultiObjectiveOptimizationConfig(
                objective=objective,
                objective_thresholds=[
                    ObjectiveThreshold(
                        metric=Metric("m1"),
                        bound=1.0,
                        relative=True,
                        op=ComparisonOp.GEQ,
                    )
                ],
            )
            with self.assertRaisesRegex(
                ValueError, "Relative objective thresholds are not supported"
            ):
                get_hypervolume_trace_of_outcomes_multi_objective(
                    df_wide=df_wide,
                    optimization_config=optimization_config,
                    use_cumulative_hv=True,
                )

        optimization_config = MultiObjectiveOptimizationConfig(
            objective=objective,
        )
        with self.subTest("Cumulative HV"):
            hvs = get_hypervolume_trace_of_outcomes_multi_objective(
                df_wide=df_wide,
                optimization_config=optimization_config,
                use_cumulative_hv=True,
            )
            self.assertEqual(hvs, [0.0, 2.0, 2.0, 3.0])

        with self.subTest("Non-cumulative HV"):
            hvs = get_hypervolume_trace_of_outcomes_multi_objective(
                df_wide=df_wide,
                optimization_config=optimization_config,
                use_cumulative_hv=False,
            )
            self.assertEqual(hvs, [0.0, 2.0, 0.0, 2.0])

    def test_get_trace_by_arm_pull_from_data(self) -> None:
        objective = Objective(metric=Metric("m1"), minimize=False)
        optimzation_config = OptimizationConfig(
            objective=objective,
            outcome_constraints=[
                OutcomeConstraint(
                    Metric("m2"), op=ComparisonOp.GEQ, bound=0.0, relative=False
                )
            ],
        )

        with self.subTest("No data"):
            df = pd.DataFrame()
            result = get_trace_by_arm_pull_from_data(
                df=df, optimization_config=optimzation_config
            )
            self.assertTrue(
                result.equals(
                    pd.DataFrame(columns=["trial_index", "arm_name", "value"])
                )
            )

        with self.subTest("Data for wrong metric"):
            df = pd.DataFrame.from_records(
                data=[
                    {
                        "trial_index": 0,
                        "arm_name": "0_0",
                        "metric_name": "wrong",
                        "mean": 1.0,
                        "sem": None,
                    },
                ]
            )
            with self.assertRaisesRegex(
                ValueError, "Some metrics are not present for all trials and arms"
            ):
                result = get_trace_by_arm_pull_from_data(
                    df=df, optimization_config=optimzation_config
                )

        df = pd.DataFrame.from_records(
            data=[
                {"trial_index": 0, "arm_name": "0_0", "metric_name": "m1", "mean": 1.0},
                {"trial_index": 0, "arm_name": "0_0", "metric_name": "m2", "mean": 1.0},
                # infeasible
                {"trial_index": 0, "arm_name": "0_1", "metric_name": "m1", "mean": 2.0},
                {
                    "trial_index": 0,
                    "arm_name": "0_1",
                    "metric_name": "m2",
                    "mean": -1.0,
                },
                {
                    "trial_index": 1,
                    "arm_name": "0_0",
                    "metric_name": "extraneous",
                    "mean": 4.0,
                },
                {"trial_index": 1, "arm_name": "0_0", "metric_name": "m1", "mean": 3.0},
                {"trial_index": 1, "arm_name": "0_0", "metric_name": "m2", "mean": 0.0},
            ]
        ).assign(sem=None)

        with self.subTest("Relative optimization config not supported"):
            rel_optimization_config = OptimizationConfig(
                objective=objective,
                outcome_constraints=[
                    OutcomeConstraint(
                        metric=Metric("m2"),
                        op=ComparisonOp.GEQ,
                        bound=0.0,
                        relative=True,
                    )
                ],
            )
            with self.assertRaisesRegex(
                ValueError, "Relativized optimization config not supported"
            ):
                get_trace_by_arm_pull_from_data(
                    df=df, optimization_config=rel_optimization_config
                )

        with self.subTest("Single objective, cumulative"):
            result = get_trace_by_arm_pull_from_data(
                df=df, optimization_config=optimzation_config, use_cumulative_best=True
            )
            self.assertEqual(len(result), 3)
            self.assertEqual(set(result.columns), {"trial_index", "arm_name", "value"})
            self.assertEqual(result["value"].tolist(), [1.0, 1.0, 3.0])

        with self.subTest("Single objective, non-cumulative"):
            result = get_trace_by_arm_pull_from_data(
                df=df, optimization_config=optimzation_config, use_cumulative_best=False
            )
            self.assertEqual(len(result), 3)
            self.assertEqual(set(result.columns), {"trial_index", "arm_name", "value"})
            self.assertEqual(result["value"].tolist(), [1.0, float("-inf"), 3.0])

        moo_opt_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(
                objectives=[
                    Objective(metric=Metric("m1"), minimize=False),
                    Objective(metric=Metric("m2"), minimize=False),
                ],
            ),
        )
        # reference point inferred to be [1, 0]
        with self.subTest("Multi-objective, cumulative"):
            result = get_trace_by_arm_pull_from_data(
                df=df, optimization_config=moo_opt_config, use_cumulative_best=True
            )
            self.assertEqual(len(result), 3)
            self.assertEqual(set(result.columns), {"trial_index", "arm_name", "value"})
            self.assertEqual(result["value"].tolist(), [0.0, 0.0, 2.0])

        with self.subTest("Multi-objective, non-cumulative"):
            result = get_trace_by_arm_pull_from_data(
                df=df, optimization_config=moo_opt_config, use_cumulative_best=False
            )
            self.assertEqual(len(result), 3)
            self.assertEqual(set(result.columns), {"trial_index", "arm_name", "value"})
            self.assertEqual(result["value"].tolist(), [0.0, 0.0, 2.0])

    @mock_botorch_optimize
    def test_best_from_model_prediction(self) -> None:
        exp = get_branin_experiment()
        gs = choose_generation_strategy_legacy(
            search_space=exp.search_space,
            num_initialization_trials=3,
            suggested_model_override=Generators.BOTORCH_MODULAR,
        )

        for _ in range(3):
            generator_run = gs.gen(experiment=exp, n=1)
            trial = exp.new_trial(generator_run=generator_run)
            trial.run().mark_completed()
            exp.attach_data(exp.fetch_data())

        generator_run = gs.gen(experiment=exp, n=1)
        trial = exp.new_trial(generator_run=generator_run)
        trial.run().mark_completed()

        with patch.object(
            TorchAdapter,
            "model_best_point",
            return_value=(
                (
                    exp.trials[0].arms[0],
                    (
                        {"branin": 34.76260622783635},
                        {"branin": {"branin": 0.00028306433439807734}},
                    ),
                )
            ),
        ) as mock_model_best_point, self.assertLogs(
            logger=best_point_logger, level="WARN"
        ) as lg:
            # Test bad model fit causes function to resort back to raw data
            with patch(
                "ax.service.utils.best_point.assess_model_fit",
                return_value=AssessModelFitResult(
                    good_fit_metrics_to_fisher_score={},
                    bad_fit_metrics_to_fisher_score={
                        "branin": 0,
                    },
                ),
            ):
                self.assertIsNotNone(
                    get_best_parameters_from_model_predictions_with_trial_index(
                        experiment=exp, adapter=gs.model
                    )
                )
                self.assertTrue(
                    any("Model fit is poor" in warning for warning in lg.output),
                    msg=lg.output,
                )
                mock_model_best_point.assert_not_called()

            # Test model best point is used when fit is good
            with patch(
                "ax.service.utils.best_point.assess_model_fit",
                return_value=AssessModelFitResult(
                    good_fit_metrics_to_fisher_score={
                        "branin": 0,
                    },
                    bad_fit_metrics_to_fisher_score={},
                ),
            ):
                self.assertIsNotNone(
                    get_best_parameters_from_model_predictions_with_trial_index(
                        experiment=exp, adapter=gs.model
                    )
                )
                mock_model_best_point.assert_called()

        # Assert the non-mocked method works correctly as well
        res = get_best_parameters_from_model_predictions_with_trial_index(
            experiment=exp, adapter=gs.model
        )
        trial_index, best_params, predict_arm = none_throws(res)
        self.assertIsNotNone(best_params)
        self.assertIsNotNone(trial_index)
        self.assertIsNotNone(predict_arm)
        # It works even when there are no predictions already stored on the
        # GeneratorRun
        for trial in exp.trials.values():
            trial.generator_run._best_arm_predictions = None
        res = get_best_parameters_from_model_predictions_with_trial_index(
            experiment=exp, adapter=gs.model
        )
        trial_index, best_params_no_gr, predict_arm_no_gr = none_throws(res)
        self.assertEqual(best_params, best_params_no_gr)
        self.assertEqual(predict_arm, predict_arm_no_gr)
        self.assertIsNotNone(trial_index)
        self.assertIsNotNone(predict_arm)

    def test_best_raw_objective_point(self) -> None:
        with self.subTest("Only early-stopped trials"):
            exp = get_experiment_with_map_data()
            exp.trials[0].mark_running(no_runner_required=True)
            exp.trials[0].mark_early_stopped(unsafe=True)
            with self.assertRaisesRegex(
                ValueError, "Cannot identify best point if no trials are completed."
            ):
                get_best_raw_objective_point_with_trial_index(experiment=exp)

        exp = get_branin_experiment()
        with self.assertRaisesRegex(
            ValueError, "Cannot identify best point if experiment contains no data."
        ):
            get_best_raw_objective_point_with_trial_index(experiment=exp)
        self.assertIsNone(get_best_by_raw_objective_with_trial_index(exp))
        exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters={"x1": 5.0, "x2": 5.0})])
        ).run().complete()
        exp.fetch_data()

        with self.subTest("Data present but not for needed metrics"):
            opt_conf = OptimizationConfig(
                objective=Objective(metric=get_branin_metric(name="not_branin"))
            )
            with self.assertRaisesRegex(
                ValueError, "Some metrics are not present for all trials and arms"
            ):
                get_best_raw_objective_point_with_trial_index(
                    experiment=exp, optimization_config=opt_conf
                )

        # Test constraints work as expected.
        observations = [[1.0, 2.0], [3.0, 4.0], [-5.0, -6.0]]
        exp = get_experiment_with_observations(
            observations=observations,
            constrained=True,
            minimize=False,
        )
        _, __, best_prediction = none_throws(
            get_best_by_raw_objective_with_trial_index(exp)
        )
        best_metrics = none_throws(best_prediction)[0]
        self.assertDictEqual(best_metrics, {"m1": 3.0, "m2": 4.0})

        # Tensor bounds are accepted.
        constraint = none_throws(exp.optimization_config).all_constraints[0]
        # pyre-fixme[8]: Attribute `bound` declared in class `OutcomeConstraint`
        # has type `float` but is used as type `Tensor`.
        constraint.bound = torch.tensor(constraint.bound)
        _, __, best_prediction = none_throws(
            get_best_by_raw_objective_with_trial_index(exp)
        )
        best_metrics = none_throws(best_prediction)[0]
        self.assertDictEqual(best_metrics, {"m1": 3.0, "m2": 4.0})

    def test_best_raw_objective_point_unsatisfiable(self) -> None:
        exp = get_branin_experiment()
        trial = exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters={"x1": 5.0, "x2": 5.0})])
        ).run()
        trial.mark_completed()
        exp.fetch_data()

        # pyre-fixme[16]: Optional type has no attribute `clone`.
        opt_conf = exp.optimization_config.clone()
        opt_conf.outcome_constraints.append(
            OutcomeConstraint(
                metric=get_branin_metric(), op=ComparisonOp.LEQ, bound=0, relative=False
            )
        )

        with self.assertRaisesRegex(ValueError, "No points satisfied"):
            get_best_raw_objective_point_with_trial_index(
                experiment=exp, optimization_config=opt_conf
            )

    def test_best_raw_objective_point_unsatisfiable_relative(self) -> None:
        exp = get_experiment_with_observations(
            observations=[[-1, 1]],
            constrained=True,
        )

        # Create altered optimization config with unsatisfiable relative constraint.
        opt_conf = none_throws(exp.optimization_config).clone()
        opt_conf.outcome_constraints[0].relative = True
        opt_conf.outcome_constraints[0].bound = 9999

        with self.assertRaisesRegex(
            DataRequiredError,
            "Optimization config has relative constraint, but model was not fit"
            " with status quo.",
        ):
            get_best_raw_objective_point_with_trial_index(
                experiment=exp, optimization_config=opt_conf
            )

        exp.status_quo = exp.trials[0].arms[0]

        with self.assertRaisesRegex(ValueError, "No points satisfied"):
            get_best_raw_objective_point_with_trial_index(exp, opt_conf)

    def test_best_raw_objective_point_scalarized(self) -> None:
        exp = get_branin_experiment()
        gs = choose_generation_strategy_legacy(search_space=exp.search_space)
        exp.optimization_config = OptimizationConfig(
            ScalarizedObjective(metrics=[get_branin_metric()], minimize=True)
        )
        with self.assertRaisesRegex(ValueError, "Cannot identify best "):
            get_best_raw_objective_point_with_trial_index(exp)
        self.assertIsNone(
            get_best_parameters_from_model_predictions_with_trial_index(
                experiment=exp, adapter=gs.model
            )
        )
        self.assertIsNone(get_best_by_raw_objective_with_trial_index(experiment=exp))
        params = {"x1": 5.0, "x2": 5.0}
        exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters=params)])
        ).run().complete()
        exp.fetch_data()
        _, parameterization, __ = get_best_raw_objective_point_with_trial_index(exp)
        self.assertEqual(parameterization, params)

    def test_best_raw_objective_point_scalarized_multi(self) -> None:
        exp = get_branin_experiment()
        gs = choose_generation_strategy_legacy(search_space=exp.search_space)
        exp.optimization_config = OptimizationConfig(
            ScalarizedObjective(
                metrics=[get_branin_metric(), get_branin_metric(lower_is_better=False)],
                weights=[0.1, -0.9],
                minimize=True,
            )
        )
        with self.assertRaisesRegex(ValueError, "Cannot identify best "):
            get_best_raw_objective_point_with_trial_index(experiment=exp)
        self.assertIsNone(
            get_best_parameters_from_model_predictions_with_trial_index(
                experiment=exp, adapter=gs.model
            )
        )
        self.assertIsNone(get_best_by_raw_objective_with_trial_index(experiment=exp))
        params = {"x1": 5.0, "x2": 5.0}
        exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters=params)])
        ).run().complete()
        exp.fetch_data()
        _, parameterization, __ = get_best_raw_objective_point_with_trial_index(exp)
        self.assertEqual(parameterization, params)

    def test_derelativize_opt_config(self) -> None:
        # No change to optimization config without relative constraints/thresholds.
        observations = [[-1, 1, 1], [1, 2, 1], [3, -2, -1], [2, 4, 1], [2, 0, 1]]
        exp = get_experiment_with_observations(
            observations=observations, constrained=True
        )
        input_optimization_config = none_throws(exp.optimization_config)
        with self.subTest("No relative constraints"):
            returned_opt_config = derelativize_opt_config(
                optimization_config=input_optimization_config,
                experiment=exp,
            )
            self.assertEqual(input_optimization_config, returned_opt_config)

        with self.subTest("Provided opt config overrides experiment opt config"):
            opt_config = input_optimization_config.clone_with_args(
                outcome_constraints=[]
            )
            returned_opt_config = derelativize_opt_config(
                optimization_config=opt_config,
                experiment=exp,
            )
            self.assertEqual(opt_config, returned_opt_config)
            self.assertNotEqual(opt_config, exp.optimization_config)

        # Add relative constraints.
        for constraint in input_optimization_config.all_constraints:
            constraint.relative = True

        # Check errors.
        with self.subTest("No fit with status quo"), self.assertRaisesRegex(
            DataRequiredError,
            "Optimization config has relative constraint, but model was not fit"
            " with status quo.",
        ):
            derelativize_opt_config(
                optimization_config=input_optimization_config, experiment=exp
            )

        status_quo_trial_index = 2
        exp.status_quo = exp.trials[status_quo_trial_index].arms[0]
        with self.subTest("Relative constraints and status quo"):
            relativized_opt_config = assert_is_instance(
                derelativize_opt_config(
                    optimization_config=input_optimization_config, experiment=exp
                ),
                MultiObjectiveOptimizationConfig,
            )
            relativized_obj = assert_is_instance(
                relativized_opt_config.objective, MultiObjective
            )
            input_obj = assert_is_instance(
                input_optimization_config.objective, MultiObjective
            )
            status_quo_df = exp.lookup_data_for_trial(
                trial_index=status_quo_trial_index
            )[0].df
            # This is not a real test of `derelativize_opt_config` but rather
            # making sure the values on the experiment have't drifted
            self.assertEqual(status_quo_df["metric_name"].tolist(), ["m1", "m2", "m3"])
            self.assertEqual(status_quo_df["mean"].tolist(), [3.0, -2.0, -1.0])

            status_quo_obs = observations[status_quo_trial_index]
            for i in range(2):
                # Relativization is with respect to the absolute value of the
                # status quo value
                self.assertEqual(
                    relativized_obj.objectives[i].minimize,
                    input_obj.objectives[i].minimize,
                )
                self.assertEqual(
                    relativized_opt_config.objective_thresholds[i].bound,
                    status_quo_obs[i],
                )

            self.assertEqual(
                relativized_opt_config.outcome_constraints[0].bound,
                observations[status_quo_trial_index][2],
            )

    def test_is_row_feasible(self) -> None:
        exp = get_experiment_with_observations(
            observations=[[-1, 1, 1], [1, 2, 1], [3, 3, -1], [2, 4, 1], [2, 0, 1]],
            constrained=True,
        )
        feasible_series = _is_row_feasible(
            df=exp.lookup_data().df,
            optimization_config=none_throws(exp.optimization_config),
        )
        expected_per_arm = [False, True, False, True, True]
        expected_series = _repeat_elements(
            list_to_replicate=expected_per_arm, n_repeats=3
        )
        pd.testing.assert_series_equal(
            feasible_series, expected_series, check_names=False
        )

        exp.optimization_config.outcome_constraints[0].relative = True
        relative_constraint_warning = (
            "WARNING:ax.service.utils.best_point:Ignoring relative constraint "
            "OutcomeConstraint(m3 >= 0.0%). Derelativize OptimizationConfig "
            "before passing to `_is_row_feasible`."
        )
        with self.assertLogs(logger=best_point_logger, level="WARN") as lg:
            # with lookout for warnings(" OutcomeConstraint(m3 >= 0.0%) ignored."):
            feasible_series = _is_row_feasible(
                df=exp.lookup_data().df,
                optimization_config=none_throws(exp.optimization_config),
            )
            self.assertTrue(
                any(relative_constraint_warning in warning for warning in lg.output),
                msg=lg.output,
            )
        expected_per_arm = [False, True, True, True, True]
        expected_series = _repeat_elements(
            list_to_replicate=expected_per_arm, n_repeats=3
        )
        pd.testing.assert_series_equal(
            feasible_series, expected_series, check_names=False
        )
        exp._status_quo = exp.trials[0].arms[0]
        for constraint in none_throws(exp.optimization_config).all_constraints:
            constraint.relative = True
        optimization_config = derelativize_opt_config(
            optimization_config=none_throws(exp.optimization_config),
            experiment=exp,
        )
        with self.assertLogs(logger=best_point_logger, level="WARN") as lg:
            # `assertNoLogs` coming in 3.10 - until then we log a dummy warning and
            # continue.
            best_point_logger.warning("Dummy warning")
            feasible_series = _is_row_feasible(
                df=exp.lookup_data().df, optimization_config=optimization_config
            )
            self.assertFalse(
                any(relative_constraint_warning in warning for warning in lg.output),
                msg=lg.output,
            )
        expected_per_arm = [True, True, False, True, False]
        expected_series = _repeat_elements(
            list_to_replicate=expected_per_arm, n_repeats=3
        )
        pd.testing.assert_series_equal(
            feasible_series, expected_series, check_names=False
        )

        # Check that index is carried over for interfacing appropriately
        # with related dataframes.
        exp = get_experiment_with_observations(
            observations=[[-1, 1], [1, 2], [3, 3], [2, 4], [2, 0]],
            constrained=False,
        )
        df = exp.lookup_data().df
        # Artificially redact some data.
        df = df[df["mean"] > 1]
        feasible_series = _is_row_feasible(
            df=df, optimization_config=optimization_config
        )
        pd.testing.assert_index_equal(
            df.index, feasible_series.index, check_names=False
        )

    def test_compare_to_baseline_select_baseline_name_default_first_trial(self) -> None:
        OBJECTIVE_METRIC = "objective"
        true_obj_metric = Metric(name=OBJECTIVE_METRIC, lower_is_better=True)
        experiment = Experiment(
            search_space=get_branin_search_space(),
            tracking_metrics=[true_obj_metric],
        )

        with patch.object(
            Experiment, "arms_by_name", new_callable=PropertyMock
        ) as mock_arms_by_name:
            mock_arms_by_name.return_value = {"arm1": "value1", "arm2": "value2"}
            self.assertEqual(
                select_baseline_name_default_first_trial(
                    experiment=experiment,
                    baseline_arm_name="arm1",
                ),
                ("arm1", False),
            )

        # specified baseline arm not in trial
        wrong_baseline_name = "wrong_baseline_name"
        with self.assertRaisesRegex(
            ValueError,
            "Arm by name .*" + " not found.",
        ):
            select_baseline_name_default_first_trial(
                experiment=experiment,
                baseline_arm_name=wrong_baseline_name,
            )

        # status quo baseline arm
        experiment_with_status_quo = copy.deepcopy(experiment)
        experiment_with_status_quo.status_quo = Arm(
            name="status_quo",
            parameters={"x1": 0, "x2": 0},
        )
        self.assertEqual(
            select_baseline_name_default_first_trial(
                experiment=experiment_with_status_quo,
                baseline_arm_name=None,
            ),
            ("status_quo", False),
        )
        # first arm from trials
        custom_arm = Arm(name="m_0", parameters={"x1": 0.1, "x2": 0.2})
        experiment.new_trial().add_arm(custom_arm)
        self.assertEqual(
            select_baseline_name_default_first_trial(
                experiment=experiment,
                baseline_arm_name=None,
            ),
            ("m_0", True),
        )

        # none selected
        experiment_with_no_valid_baseline = Experiment(
            search_space=get_branin_search_space(),
            tracking_metrics=[true_obj_metric],
        )

        with self.assertRaisesRegex(
            ValueError,
            "Could not find valid baseline arm.",
        ):
            select_baseline_name_default_first_trial(
                experiment=experiment_with_no_valid_baseline,
                baseline_arm_name=None,
            )

    # NOTE: Can't use mock optimize here, since we rely on model predictions.
    # It is still very cheap even with model fitting.
    def test_get_best_point_with_model_prediction(
        self,
    ) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [1.0, 10.0],
                },
            ],
            objectives={"y": ObjectiveProperties(minimize=True)},
            is_test=True,
            choose_generation_strategy_kwargs={"num_initialization_trials": 2},
        )

        params, idx = ax_client.get_next_trial()
        ax_client.complete_trial(idx, raw_data={"y": 0})

        for i in range(1, 3):
            ax_client.get_next_trial()
            ax_client.complete_trial(i, raw_data={"y": i})

        # Mock with no bad fir metrics ensures that the model is used
        # to extract the best point.
        with patch(
            f"{best_point_module}.assess_model_fit",
            return_value=AssessModelFitResult(
                good_fit_metrics_to_fisher_score={"y": 1},
                bad_fit_metrics_to_fisher_score={},
            ),
        ) as mock_model_fit:
            best_index, best_params, predictions = none_throws(
                ax_client.get_best_trial()
            )
        mock_model_fit.assert_called_once()
        self.assertEqual(best_index, idx)
        self.assertEqual(best_params, params)
        # We should get both mean & covariance predictions.
        self.assertEqual(predictions, ({"y": mock.ANY}, {"y": {"y": mock.ANY}}))

        # Also verify that fallback option from GR produces the right trial index.
        gr = ax_client.experiment.trials[2].generator_runs[0]
        best_index, best_params, predictions = none_throws(
            _extract_best_arm_from_gr(gr=gr, trials=ax_client.experiment.trials)
        )
        self.assertEqual(best_index, idx)
        self.assertEqual(best_params, params)
        self.assertEqual(predictions, ({"y": mock.ANY}, {"y": {"y": mock.ANY}}))


def _repeat_elements(list_to_replicate: list[bool], n_repeats: int) -> pd.Series:
    return pd.Series([item for item in list_to_replicate for _ in range(n_repeats)])
