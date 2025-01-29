#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import random
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import torch
from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.objective import ScalarizedObjective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.exceptions.core import UserInputError
from ax.modelbridge.cross_validation import AssessModelFitResult
from ax.modelbridge.registry import Models
from ax.modelbridge.torch import TorchModelBridge
from ax.plot.pareto_utils import get_tensor_converter_model
from ax.service.ax_client import AxClient
from ax.service.utils.best_point import (
    _derel_opt_config_wrapper,
    _extract_best_arm_from_gr,
    _is_row_feasible,
    extract_Y_from_data,
    get_best_parameters,
    get_best_raw_objective_point,
    logger as best_point_logger,
)
from ax.service.utils.best_point_utils import select_baseline_name_default_first_trial
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_metric,
    get_branin_search_space,
    get_experiment_with_observations,
    get_sobol,
)
from ax.utils.testing.mock import mock_botorch_optimize
from pyre_extensions import none_throws

best_point_module: str = _derel_opt_config_wrapper.__module__
DUMMY_OPTIMIZATION_CONFIG = "test_optimization_config"


class TestBestPointUtils(TestCase):
    """Testing the best point utilities functionality that is not tested in
    main `AxClient` testing suite (`TestServiceAPI`)."""

    @mock_botorch_optimize
    def test_best_from_model_prediction(self) -> None:
        exp = get_branin_experiment()

        for _ in range(3):
            sobol = Models.SOBOL(search_space=exp.search_space)
            generator_run = sobol.gen(n=1)
            trial = exp.new_trial(generator_run=generator_run)
            trial.run()
            trial.mark_completed()
            exp.attach_data(exp.fetch_data())

        model = Models.BOTORCH_MODULAR(experiment=exp, data=exp.lookup_data())
        generator_run = model.gen(n=1)
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
        best_params = get_best_parameters(exp, Models)
        self.assertIsNotNone(best_params)
        # It works even when there are no predictions already stored on the
        # GeneratorRun
        for trial in exp.trials.values():
            trial.generator_run._best_arm_predictions = None
        best_params_no_gr = get_best_parameters(exp, Models)
        self.assertEqual(best_params, best_params_no_gr)

    def test_best_raw_objective_point(self) -> None:
        exp = get_branin_experiment()
        with self.assertRaisesRegex(ValueError, "Cannot identify best "):
            get_best_raw_objective_point(exp)
        self.assertEqual(get_best_parameters(exp, Models), None)
        exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters={"x1": 5.0, "x2": 5.0})])
        ).run().complete()
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
        _, best_prediction = none_throws(get_best_parameters(exp, Models))
        best_metrics = none_throws(best_prediction)[0]
        self.assertDictEqual(best_metrics, {"m1": 3.0, "m2": 4.0})

        # Tensor bounds are accepted.
        constraint = none_throws(exp.optimization_config).all_constraints[0]
        # pyre-fixme[8]: Attribute `bound` declared in class `OutcomeConstraint`
        # has type `float` but is used as type `Tensor`.
        constraint.bound = torch.tensor(constraint.bound)
        _, best_prediction = none_throws(get_best_parameters(exp, Models))
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
            get_best_raw_objective_point(exp, opt_conf)

    def test_best_raw_objective_point_unsatisfiable_relative(self) -> None:
        exp = get_experiment_with_observations(
            observations=[[-1, 1]],
            constrained=True,
        )

        # Create altered optimization config with unsatisfiable relative constraint.
        opt_conf = none_throws(exp.optimization_config).clone()
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
            ScalarizedObjective(metrics=[get_branin_metric()], minimize=True)
        )
        with self.assertRaisesRegex(ValueError, "Cannot identify best "):
            get_best_raw_objective_point(exp)
        self.assertEqual(get_best_parameters(exp, Models), None)
        exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters={"x1": 5.0, "x2": 5.0})])
        ).run().complete()
        exp.fetch_data()
        self.assertEqual(get_best_raw_objective_point(exp)[0], {"x1": 5.0, "x2": 5.0})

    def test_best_raw_objective_point_scalarized_multi(self) -> None:
        exp = get_branin_experiment()
        exp.optimization_config = OptimizationConfig(
            ScalarizedObjective(
                metrics=[get_branin_metric(), get_branin_metric(lower_is_better=False)],
                weights=[0.1, -0.9],
                minimize=True,
            )
        )
        with self.assertRaisesRegex(ValueError, "Cannot identify best "):
            get_best_raw_objective_point(exp)
        self.assertEqual(get_best_parameters(exp, Models), None)
        exp.new_trial(
            generator_run=GeneratorRun(arms=[Arm(parameters={"x1": 5.0, "x2": 5.0})])
        ).run().complete()
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
        input_optimization_config = none_throws(exp.optimization_config)
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
            experiment=none_throws(exp),
            data=none_throws(exp).lookup_data(),
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
            experiment=none_throws(exp),
            data=none_throws(exp).lookup_data(),
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
                        "mean": (
                            float(trial_idx)
                            if metric_name == "foo"
                            else trial_idx + 5.0
                        ),
                        "sem": 0.0,
                    }
                )
        experiment.attach_data(Data(df=pd.DataFrame.from_records(df_dicts)))

        expected_Y = torch.stack(
            [
                torch.arange(20, dtype=torch.double),
                torch.arange(5, 25, dtype=torch.double),
            ],
            dim=-1,
        )
        Y, trial_indices = extract_Y_from_data(
            experiment=experiment,
            metric_names=["foo", "bar"],
        )
        expected_trial_indices = torch.arange(20)
        self.assertTrue(torch.allclose(Y, expected_Y))
        self.assertTrue(torch.equal(trial_indices, expected_trial_indices))
        # Check that it respects ordering of metric names.
        Y, trial_indices = extract_Y_from_data(
            experiment=experiment,
            metric_names=["bar", "foo"],
        )
        self.assertTrue(torch.allclose(Y, expected_Y[:, [1, 0]]))
        self.assertTrue(torch.equal(trial_indices, expected_trial_indices))
        # Extract partial metrics.
        Y, trial_indices = extract_Y_from_data(
            experiment=experiment, metric_names=["bar"]
        )
        self.assertTrue(torch.allclose(Y, expected_Y[:, [1]]))
        self.assertTrue(torch.equal(trial_indices, expected_trial_indices))
        # Works with messed up ordering of data.
        clone_dicts = df_dicts.copy()
        random.shuffle(clone_dicts)
        experiment._data_by_trial = {}
        experiment.attach_data(Data(df=pd.DataFrame.from_records(clone_dicts)))
        Y, trial_indices = extract_Y_from_data(
            experiment=experiment,
            metric_names=["foo", "bar"],
        )
        self.assertTrue(torch.allclose(Y, expected_Y))
        self.assertTrue(torch.equal(trial_indices, expected_trial_indices))

        # Check that it skips trials that are not completed.
        experiment.trials[0].mark_running(no_runner_required=True, unsafe=True)
        experiment.trials[1].mark_abandoned(unsafe=True)
        Y, trial_indices = extract_Y_from_data(
            experiment=experiment,
            metric_names=["foo", "bar"],
        )
        self.assertTrue(torch.allclose(Y, expected_Y[2:]))
        self.assertTrue(torch.equal(trial_indices, expected_trial_indices[2:]))

        # Error with missing data.
        with self.assertRaisesRegex(
            UserInputError, "Trial 2 is missing data on metrics {'foo'}."
        ):
            # Skipping first 5 data points since first two trials are not completed.

            extract_Y_from_data(
                experiment=experiment,
                metric_names=["foo", "bar"],
                data=Data(df=pd.DataFrame.from_records(df_dicts[5:])),
            )

        # Error with extra data.
        with self.assertRaisesRegex(
            UserInputError, "Trial data has more than one row per arm, metric pair. "
        ):
            # Skipping first 5 data points since first two trials are not completed.
            base_df = pd.DataFrame.from_records(df_dicts[5:])
            extract_Y_from_data(
                experiment=experiment,
                metric_names=["foo", "bar"],
                data=Data(df=pd.concat((base_df, base_df))),
            )

        # Check that it works with BatchTrial.
        experiment = get_branin_experiment()
        batch_trial = BatchTrial(experiment=experiment, index=0)
        batch_trial.add_arm(Arm(name="0_0", parameters={"x1": 0.0, "x2": 0.0}))
        batch_trial.add_arm(Arm(name="0_1", parameters={"x1": 1.0, "x2": 0.0}))
        batch_trial.mark_running(no_runner_required=True).mark_completed()
        df_dicts_batch = []
        for i in (0, 1):
            for metric_name in ["foo", "bar"]:
                df_dicts_batch.append(
                    {
                        "trial_index": 0,
                        "metric_name": metric_name,
                        "arm_name": f"0_{i}",
                        "mean": float(i) if metric_name == "foo" else i + 5.0,
                        "sem": 0.0,
                    }
                )
        batch_df = pd.DataFrame.from_records(df_dicts_batch)
        Y, trial_indices = extract_Y_from_data(
            experiment=experiment,
            metric_names=["foo", "bar"],
            data=Data(df=batch_df),
        )
        self.assertTrue(torch.allclose(Y, expected_Y[:2]))
        self.assertTrue(torch.equal(trial_indices, torch.zeros(2, dtype=torch.long)))

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
        optimization_config = _derel_opt_config_wrapper(
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
