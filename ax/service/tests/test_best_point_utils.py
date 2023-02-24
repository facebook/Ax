#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import List
from unittest.mock import MagicMock, patch

import pandas as pd
import torch
from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.generator_run import GeneratorRun
from ax.core.objective import ScalarizedObjective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.modelbridge.cross_validation import AssessModelFitResult
from ax.modelbridge.registry import Models
from ax.modelbridge.torch import TorchModelBridge
from ax.plot.pareto_utils import get_tensor_converter_model
from ax.service.utils.best_point import (
    _derel_opt_config_wrapper,
    _is_row_feasible,
    extract_Y_from_data,
    get_best_parameters,
    get_best_raw_objective_point,
    logger as best_point_logger,
)
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import not_none
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_metric,
    get_experiment_with_observations,
    get_sobol,
)
from ax.utils.testing.mock import fast_botorch_optimize

best_point_module: str = _derel_opt_config_wrapper.__module__
DUMMY_OPTIMIZATION_CONFIG = "test_optimization_config"


class TestBestPointUtils(TestCase):
    """Testing the best point utilities functionality that is not tested in
    main `AxClient` testing suite (`TestServiceAPI`)."""

    @fast_botorch_optimize
    def test_best_from_model_prediction(self) -> None:
        exp = get_branin_experiment()

        for _ in range(3):
            sobol = Models.SOBOL(search_space=exp.search_space)
            generator_run = sobol.gen(n=1)
            trial = exp.new_trial(generator_run=generator_run)
            trial.run()
            trial.mark_completed()
            exp.attach_data(exp.fetch_data())

        gpei = Models.BOTORCH(experiment=exp, data=exp.lookup_data())
        generator_run = gpei.gen(n=1)
        trial = exp.new_trial(generator_run=generator_run)
        trial.run()
        trial.mark_completed()

        with patch.object(
            TorchModelBridge,
            "model_best_point",
            return_value=(
                (
                    Arm(
                        name="0_0",
                        parameters={"x1": -4.842811906710267, "x2": 11.887089014053345},
                    ),
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
                self.assertIsNotNone(get_best_parameters(exp, Models))
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
                self.assertIsNotNone(get_best_parameters(exp, Models))
                mock_model_best_point.assert_called()

        # Assert the non-mocked method works correctly as well
        self.assertIsNotNone(get_best_parameters(exp, Models))

    def test_best_raw_objective_point(self) -> None:
        exp = get_branin_experiment()
        with self.assertRaisesRegex(ValueError, "Cannot identify best "):
            get_best_raw_objective_point(exp)
        self.assertEqual(get_best_parameters(exp, Models), None)
        exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters={"x1": 5.0, "x2": 5.0})])
        ).run()
        exp.fetch_data()
        # pyre-fixme[16]: Optional type has no attribute `clone`.
        opt_conf = exp.optimization_config.clone()
        opt_conf.objective.metric._name = "not_branin"
        with self.assertRaisesRegex(ValueError, "No data has been logged"):
            get_best_raw_objective_point(exp, opt_conf)

        # Test constraints work as expected.
        observations = [[1.0, 2.0], [3.0, 4.0], [-5.0, -6.0]]
        exp = get_experiment_with_observations(
            observations=observations,
            constrained=True,
            minimize=False,
        )
        _, best_prediction = not_none(get_best_parameters(exp, Models))
        best_metrics = not_none(best_prediction)[0]
        self.assertDictEqual(best_metrics, {"m1": 3.0, "m2": 4.0})

        # Tensor bounds are accepted.
        constraint = not_none(exp.optimization_config).all_constraints[0]
        # pyre-fixme[8]: Attribute `bound` declared in class `OutcomeConstraint`
        # has type `float` but is used as type `Tensor`.
        constraint.bound = torch.tensor(constraint.bound)
        _, best_prediction = not_none(get_best_parameters(exp, Models))
        best_metrics = not_none(best_prediction)[0]
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
            get_best_raw_objective_point(exp, opt_conf)

    def test_best_raw_objective_point_unsatisfiable_relative(self) -> None:
        exp = get_experiment_with_observations(
            observations=[[-1, 1]],
            constrained=True,
        )

        # Create altered optimization config with unsatisfiable relative constraint.
        opt_conf = not_none(exp.optimization_config).clone()
        opt_conf.outcome_constraints[0].relative = True
        opt_conf.outcome_constraints[0].bound = 9999

        with self.assertLogs(logger=best_point_logger, level="WARN") as lg:
            get_best_raw_objective_point(exp, opt_conf)
            self.assertTrue(
                any("No status quo provided" in warning for warning in lg.output),
                msg=lg.output,
            )

        exp.status_quo = exp.trials[0].arms[0]

        with self.assertRaisesRegex(ValueError, "No points satisfied"):
            get_best_raw_objective_point(exp, opt_conf)

    def test_best_raw_objective_point_scalarized(self) -> None:
        exp = get_branin_experiment()
        exp.optimization_config = OptimizationConfig(
            ScalarizedObjective(metrics=[get_branin_metric()], minimize=False)
        )
        with self.assertRaisesRegex(ValueError, "Cannot identify best "):
            get_best_raw_objective_point(exp)
        self.assertEqual(get_best_parameters(exp, Models), None)
        exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters={"x1": 5.0, "x2": 5.0})])
        ).run()
        exp.fetch_data()
        self.assertEqual(get_best_raw_objective_point(exp)[0], {"x1": 5.0, "x2": 5.0})

    def test_best_raw_objective_point_scalarized_multi(self) -> None:
        exp = get_branin_experiment()
        exp.optimization_config = OptimizationConfig(
            ScalarizedObjective(
                metrics=[get_branin_metric(), get_branin_metric()],
                weights=[0.1, -0.9],
                minimize=False,
            )
        )
        with self.assertRaisesRegex(ValueError, "Cannot identify best "):
            get_best_raw_objective_point(exp)
        self.assertEqual(get_best_parameters(exp, Models), None)
        exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters={"x1": 5.0, "x2": 5.0})])
        ).run()
        exp.fetch_data()
        self.assertEqual(get_best_raw_objective_point(exp)[0], {"x1": 5.0, "x2": 5.0})

    @patch(
        f"{best_point_module}.derelativize_optimization_config_with_raw_status_quo",
        return_value=DUMMY_OPTIMIZATION_CONFIG,
    )
    def test_derel_opt_config_wrapper(self, mock_derelativize: MagicMock) -> None:
        # No change to optimization config without relative constraints/thresholds.
        exp = get_experiment_with_observations(
            observations=[[-1, 1, 1], [1, 2, 1], [3, 3, -1], [2, 4, 1], [2, 0, 1]],
            constrained=True,
        )
        input_optimization_config = not_none(exp.optimization_config)
        optimization_config = _derel_opt_config_wrapper(
            optimization_config=input_optimization_config
        )
        self.assertEqual(input_optimization_config, optimization_config)

        # Add relative constraints.
        for constraint in input_optimization_config.all_constraints:
            constraint.relative = True

        # Check errors.
        with self.assertRaisesRegex(
            ValueError,
            "Must specify ModelBridge or Experiment when calling "
            "`_derel_opt_config_wrapper`.",
        ):
            _derel_opt_config_wrapper(optimization_config=input_optimization_config)
        with self.assertRaisesRegex(
            ValueError,
            "`modelbridge` must have status quo if specified. If `modelbridge` is "
            "unspecified, `experiment` must have a status quo.",
        ):
            _derel_opt_config_wrapper(
                optimization_config=input_optimization_config, experiment=exp
            )

        # Set status quo.
        exp.status_quo = exp.trials[0].arms[0]

        # ModelBridges will have specific addresses and so must be self-same to
        # pass equality checks.
        test_modelbridge_1 = get_tensor_converter_model(
            experiment=not_none(exp),
            data=not_none(exp).lookup_data(),
        )
        test_observations_1 = test_modelbridge_1.get_training_data()
        returned_value = _derel_opt_config_wrapper(
            optimization_config=input_optimization_config,
            modelbridge=test_modelbridge_1,
            observations=test_observations_1,
        )
        mock_derelativize.assert_called_with(
            optimization_config=input_optimization_config,
            modelbridge=test_modelbridge_1,
            observations=test_observations_1,
        )
        with patch(
            f"{best_point_module}.get_tensor_converter_model",
            return_value=test_modelbridge_1,
        ), patch(
            f"{best_point_module}.ModelBridge.get_training_data",
            return_value=test_observations_1,
        ):
            returned_value = _derel_opt_config_wrapper(
                optimization_config=input_optimization_config, experiment=exp
            )
        self.assertEqual(returned_value, DUMMY_OPTIMIZATION_CONFIG)
        mock_derelativize.assert_called_with(
            optimization_config=input_optimization_config,
            modelbridge=test_modelbridge_1,
            observations=test_observations_1,
        )

        # Observations and ModelBridge are not constructed from other inputs when
        # provided.
        test_modelbridge_2 = get_tensor_converter_model(
            experiment=not_none(exp),
            data=not_none(exp).lookup_data(),
        )
        test_observations_2 = test_modelbridge_2.get_training_data()
        with self.assertLogs(logger=best_point_logger, level="WARN") as lg, patch(
            f"{best_point_module}.get_tensor_converter_model",
            return_value=test_modelbridge_2,
        ), patch(
            f"{best_point_module}.ModelBridge.get_training_data",
            return_value=test_observations_2,
        ):
            returned_value = _derel_opt_config_wrapper(
                optimization_config=input_optimization_config,
                experiment=exp,
                modelbridge=test_modelbridge_1,
                observations=test_observations_1,
            )
            self.assertTrue(
                any(
                    "ModelBridge and Experiment provided to "
                    "`_derel_opt_config_wrapper`. Ignoring the latter." in warning
                    for warning in lg.output
                ),
                msg=lg.output,
            )
        self.assertEqual(returned_value, DUMMY_OPTIMIZATION_CONFIG)
        mock_derelativize.assert_called_with(
            optimization_config=input_optimization_config,
            modelbridge=test_modelbridge_1,
            observations=test_observations_1,
        )

    def test_extract_Y_from_data(self) -> None:
        experiment = get_branin_experiment()
        sobol_generator = get_sobol(search_space=experiment.search_space)
        for i in range(20):
            sobol_run = sobol_generator.gen(n=1)
            trial = experiment.new_trial(generator_run=sobol_run).mark_running(
                no_runner_required=True
            )
            if i in [3, 8, 10]:
                trial.mark_early_stopped()
            else:
                trial.mark_completed()

        df_dicts = []
        for trial_idx in range(20):
            for metric_name in ["foo", "bar"]:
                df_dicts.append(
                    {
                        "trial_index": trial_idx,
                        "metric_name": metric_name,
                        "arm_name": f"{trial_idx}_0",
                        "mean": float(trial_idx)
                        if metric_name == "foo"
                        else trial_idx + 5.0,
                        "sem": 0.0,
                    }
                )
        df_0 = df_dicts[:2]
        experiment.attach_data(Data(df=pd.DataFrame.from_records(df_dicts)))

        expected_Y = torch.stack(
            [
                torch.arange(20, dtype=torch.double),
                torch.arange(5, 25, dtype=torch.double),
            ],
            dim=-1,
        )
        Y = extract_Y_from_data(
            experiment=experiment,
            metric_names=["foo", "bar"],
        )
        self.assertTrue(torch.allclose(Y, expected_Y))
        # Check that it respects ordering of metric names.
        Y = extract_Y_from_data(
            experiment=experiment,
            metric_names=["bar", "foo"],
        )
        self.assertTrue(torch.allclose(Y, expected_Y[:, [1, 0]]))
        # Extract partial metrics.
        Y = extract_Y_from_data(experiment=experiment, metric_names=["bar"])
        self.assertTrue(torch.allclose(Y, expected_Y[:, [1]]))
        # Works with messed up ordering of data.
        clone_dicts = df_dicts.copy()
        random.shuffle(clone_dicts)
        experiment._data_by_trial = {}
        experiment.attach_data(Data(df=pd.DataFrame.from_records(clone_dicts)))
        Y = extract_Y_from_data(
            experiment=experiment,
            metric_names=["foo", "bar"],
        )
        self.assertTrue(torch.allclose(Y, expected_Y))

        # Check that it skips trials that are not completed.
        experiment.trials[0].mark_running(no_runner_required=True, unsafe=True)
        experiment.trials[1].mark_abandoned(unsafe=True)
        Y = extract_Y_from_data(
            experiment=experiment,
            metric_names=["foo", "bar"],
        )
        self.assertTrue(torch.allclose(Y, expected_Y[2:]))

        # Error with missing data.
        with self.assertRaisesRegex(
            UserInputError, "single data point for each metric"
        ):
            # Skipping first 5 data points since first two trials are not completed.
            extract_Y_from_data(
                experiment=experiment,
                metric_names=["foo", "bar"],
                data=Data(df=pd.DataFrame.from_records(df_dicts[5:])),
            )

        # Check that it errors with BatchTrial.
        experiment = get_branin_experiment()
        BatchTrial(experiment=experiment, index=0).mark_running(
            no_runner_required=True
        ).mark_completed()
        with self.assertRaisesRegex(UnsupportedError, "BatchTrials are not supported."):
            extract_Y_from_data(
                experiment=experiment,
                metric_names=["foo", "bar"],
                data=Data(df=pd.DataFrame.from_records(df_0)),
            )

    def test_is_row_feasible(self) -> None:
        exp = get_experiment_with_observations(
            observations=[[-1, 1, 1], [1, 2, 1], [3, 3, -1], [2, 4, 1], [2, 0, 1]],
            constrained=True,
        )
        feasible_series = _is_row_feasible(
            df=exp.lookup_data().df,
            optimization_config=not_none(exp.optimization_config),
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
                optimization_config=not_none(exp.optimization_config),
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
        for constraint in not_none(exp.optimization_config).all_constraints:
            constraint.relative = True
        optimization_config = _derel_opt_config_wrapper(
            optimization_config=not_none(exp.optimization_config),
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
            observations=[[-1, 1, 1], [1, 2, 1], [3, 3, -1], [2, 4, 1], [2, 0, 1]],
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


def _repeat_elements(list_to_replicate: List[bool], n_repeats: int) -> pd.Series:
    return pd.Series([item for item in list_to_replicate for _ in range(n_repeats)])
