#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from contextlib import ExitStack
from typing import Any
from unittest import mock
from unittest.mock import Mock

import numpy as np
import torch
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.observation import (
    Observation,
    ObservationData,
    ObservationFeatures,
    recombine_observations,
)
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import ScalarizedOutcomeConstraint
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace, SearchSpaceDigest
from ax.core.types import ComparisonOp
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.registry import MBM_X_trans
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.standardize_y import StandardizeY
from ax.modelbridge.transforms.unit_x import UnitX
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch_base import TorchGenResults, TorchModel
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast
from ax.utils.testing.core_stubs import (
    get_branin_data,
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
    get_experiment_with_observations,
    get_search_space_for_range_value,
    get_search_space_for_range_values,
)
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import get_observation1, transform_1, transform_2
from botorch.utils.datasets import (
    ContextualDataset,
    MultiTaskDataset,
    SupervisedDataset,
)
from pandas import DataFrame
from pyre_extensions import none_throws


def _get_modelbridge_from_experiment(
    experiment: Experiment,
    transforms: list[type[Transform]] | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> TorchModelBridge:
    return TorchModelBridge(
        experiment=experiment,
        search_space=experiment.search_space,
        data=experiment.lookup_data(),
        model=BoTorchModel(),
        transforms=transforms or [],
        torch_dtype=dtype,
        torch_device=device,
    )


def _get_mock_modelbridge(
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    fit_out_of_design: bool = False,
) -> TorchModelBridge:
    return TorchModelBridge(
        experiment=Mock(),
        search_space=Mock(),
        data=Mock(),
        model=Mock(),
        transforms=[],
        torch_dtype=dtype,
        torch_device=device,
        fit_out_of_design=fit_out_of_design,
    )


class TorchModelBridgeTest(TestCase):
    @mock.patch(
        f"{ModelBridge.__module__}.ModelBridge.__init__",
        autospec=True,
        return_value=None,
    )
    def test_TorchModelBridge(
        self,
        mock_init: Mock,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        ma = _get_mock_modelbridge(dtype=dtype, device=device)
        ma._fit_tracking_metrics = True
        ma._experiment_properties = {}
        dtype = dtype or torch.double
        self.assertEqual(ma.dtype, dtype)
        self.assertEqual(ma.device, device)
        self.assertFalse(mock_init.call_args[-1]["fit_out_of_design"])
        self.assertIsNone(ma._last_observations)
        tkwargs: dict[str, Any] = {"dtype": dtype, "device": device}
        # Test `_fit`.
        feature_names = ["x1", "x2", "x3"]
        model = mock.MagicMock(TorchModel, autospec=True, instance=True)
        search_space = get_search_space_for_range_values(
            min=0.0, max=5.0, parameter_names=feature_names
        )
        X = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], **tkwargs)
        datasets = {
            "y1": SupervisedDataset(
                X=X,
                Y=torch.tensor([[3.0], [1.0]], **tkwargs),
                Yvar=torch.tensor([[4.0], [2.0]], **tkwargs),
                feature_names=feature_names,
                outcome_names=["y1"],
            ),
            "y2": SupervisedDataset(
                X=X,
                Y=torch.tensor([[2.0], [0.0]], **tkwargs),
                Yvar=torch.tensor([[2.0], [1.0]], **tkwargs),
                feature_names=feature_names,
                outcome_names=["y2"],
            ),
        }
        observation_features = [
            ObservationFeatures(parameters=dict(zip(feature_names, Xi.tolist())))
            for Xi in X
        ]
        observation_data = [
            ObservationData(
                metric_names=["y1", "y2"],
                means=np.array(y1 + y2),  # here y is already a list
                covariance=np.diag(yvar1 + yvar2),  # here yvar is already a list
            )
            for y1, y2, yvar1, yvar2 in zip(
                datasets["y1"].Y.tolist(),
                datasets["y2"].Y.tolist(),
                none_throws(datasets["y1"].Yvar).tolist(),
                none_throws(datasets["y2"].Yvar).tolist(),
            )
        ]
        observations = recombine_observations(observation_features, observation_data)
        ssd = SearchSpaceDigest(
            feature_names=feature_names,
            bounds=[(0, 1)] * 3,
        )

        with mock.patch(
            f"{TorchModelBridge.__module__}.extract_search_space_digest",
            return_value=ssd,
        ):
            ma._fit(
                model=model,
                search_space=search_space,
                observations=observations,
            )
        model_fit_args = model.fit.mock_calls[0][2]
        self.assertEqual(model_fit_args["datasets"], list(datasets.values()))
        self.assertEqual(model_fit_args["search_space_digest"], ssd)
        self.assertIsNone(model_fit_args["candidate_metadata"])
        self.assertEqual(ma._last_observations, observations)

        with mock.patch(f"{TorchModelBridge.__module__}.logger.debug") as mock_logger:
            ma._fit(
                model=model,
                search_space=search_space,
                observations=observations,
            )
        mock_logger.assert_called_once_with(
            "The observations are identical to the last set of observations "
            "used to fit the model. Skipping model fitting."
        )

        # Test `_predict`
        model.predict.return_value = (
            torch.tensor([[3.0, 2.0]], **tkwargs),
            torch.tensor([[[4.0, 0.0], [0.0, 3.0]]], **tkwargs),
        )
        pr_obs_data_expected = ObservationData(
            metric_names=["y1", "y2"],
            means=np.array([3.0, 2.0]),
            covariance=np.diag([4.0, 3.0]),
        )
        pr_obs_data = ma._predict(observation_features=observation_features[:1])
        self.assertTrue(torch.equal(model.predict.mock_calls[0][2]["X"], X[:1]))
        self.assertEqual(
            pr_obs_data,
            [pr_obs_data_expected],
        )

        # Test `_gen`

        # Hack in some properties set in the (mocked) `Modelbridge.__init__`
        ma.transforms = {}
        ma.fit_time_since_gen = 0.0
        ma._arms_by_signature = {}
        ma._fit_out_of_design = False
        ma._model_key = None
        ma._model_kwargs = None
        ma._bridge_kwargs = None

        model.gen.return_value = TorchGenResults(
            points=torch.tensor([[1.0, 2.0, 3.0]], **tkwargs),
            weights=torch.tensor([1.0], **tkwargs),
            gen_metadata={"foo": 99},
        )
        model.best_point.return_value = torch.tensor([1.0, 2.0, 3.0], **tkwargs)
        opt_config = OptimizationConfig(
            objective=Objective(metric=Metric("y1"), minimize=False),
        )
        with ExitStack() as es:
            es.enter_context(
                mock.patch(
                    f"{TorchModelBridge.__module__}.extract_search_space_digest",
                    return_value=ssd,
                )
            )
            es.enter_context(
                mock.patch(
                    f"{ModelBridge.__module__}.clamp_observation_features",
                    side_effect=lambda of, ss: of,
                )
            )
            es.enter_context(
                mock.patch(
                    f"{ModelBridge.__module__}.ModelBridge._get_serialized_model_state",
                    return_value={},
                )
            )
            es.enter_context(
                mock.patch(
                    f"{TorchModelBridge.__module__}.TorchModelBridge."
                    "_array_callable_to_tensor_callable",
                    return_value=torch.round,
                )
            )
            gen_run = ma.gen(
                n=3,
                search_space=search_space,
                optimization_config=opt_config,
                pending_observations={
                    "y2": [
                        ObservationFeatures(
                            parameters={"x1": 1.0, "x2": 2.0, "x3": 3.0}
                        )
                    ]
                },
                fixed_features=ObservationFeatures(parameters={"x2": 3.0}),
                model_gen_options={"option": "yes"},
            )
        gen_args = model.gen.mock_calls[0][2]
        self.assertEqual(gen_args["n"], 3)
        self.assertEqual(gen_args["search_space_digest"].bounds, [(0, 1)] * 3)
        gen_opt_config = gen_args["torch_opt_config"]
        self.assertTrue(
            torch.equal(
                gen_opt_config.objective_weights,
                torch.tensor([1.0, 0.0], **tkwargs),
            )
        )
        self.assertIsNone(gen_opt_config.outcome_constraints)
        self.assertIsNone(gen_opt_config.linear_constraints)
        self.assertEqual(gen_opt_config.fixed_features, {1: 3.0})
        X_pending_y1, X_pending_y2 = gen_opt_config.pending_observations
        self.assertTrue(torch.equal(X_pending_y1, torch.tensor([], **tkwargs)))
        self.assertTrue(
            torch.equal(X_pending_y2, torch.tensor([[1.0, 2.0, 3.0]], **tkwargs))
        )
        self.assertEqual(gen_opt_config.model_gen_options, {"option": "yes"})
        self.assertIs(gen_opt_config.rounding_func, torch.round)
        self.assertFalse(gen_opt_config.is_moo)
        self.assertEqual(gen_opt_config.opt_config_metrics, opt_config.metrics)
        self.assertEqual(gen_args["search_space_digest"].target_values, {})
        self.assertEqual(len(gen_run.arms), 1)
        self.assertEqual(gen_run.arms[0].parameters, {"x1": 1.0, "x2": 2.0, "x3": 3.0})
        self.assertEqual(gen_run.weights, [1.0])
        self.assertEqual(gen_run.fit_time, 0.0)
        self.assertEqual(gen_run.gen_metadata, {"foo": 99})

        # Test `_cross_validate`
        model.cross_validate.return_value = (
            torch.tensor([[3.0, 2.0]], **tkwargs),
            torch.tensor([[[4.0, 0.0], [0.0, 3.0]]], **tkwargs),
        )
        cv_obs_data_expected = ObservationData(
            metric_names=["y1", "y2"],
            means=np.array([3.0, 2.0]),
            covariance=np.diag([4.0, 3.0]),
        )
        cv_test_points = [
            ObservationFeatures(parameters={"x1": 1.0, "x2": 3.0, "x3": 2.0})
        ]
        X_test = torch.tensor([[1.0, 3.0, 2.0]], **tkwargs)

        with mock.patch(
            f"{TorchModelBridge.__module__}.extract_search_space_digest",
            return_value=ssd,
        ):
            cv_obs_data = ma._cross_validate(
                search_space=search_space,
                cv_training_data=observations,
                cv_test_points=cv_test_points,
            )
        model_cv_args = model.cross_validate.mock_calls[0][2]
        self.assertEqual(model_cv_args["datasets"], list(datasets.values()))
        self.assertTrue(torch.equal(model_cv_args["X_test"], X_test))
        self.assertEqual(model_cv_args["search_space_digest"], ssd)
        self.assertEqual(cv_obs_data, [cv_obs_data_expected])

        # Transform observations
        # This functionality is likely to be deprecated (T134940274)
        # so this is not a thorough test.
        ma.transform_observations(observations)

        # Transform observation features
        obsf = [ObservationFeatures(parameters={"x": 1.0, "y": 2.0})]
        ma.parameters = ["x", "y"]
        X = ma._transform_observation_features(obsf)
        self.assertTrue(torch.equal(X, torch.tensor([[1.0, 2.0]], **tkwargs)))
        # test fit out of design
        _get_mock_modelbridge(fit_out_of_design=True)
        self.assertTrue(mock_init.call_args[-1]["fit_out_of_design"])

    def test_TorchModelBridge_float(self) -> None:
        self.test_TorchModelBridge(dtype=torch.float)

    def test_TorchModelBridge_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_TorchModelBridge(device=torch.device("cuda"))

    def test_evaluate_acquisition_function(self) -> None:
        experiment = get_branin_experiment(with_completed_trial=True)
        modelbridge = _get_modelbridge_from_experiment(
            experiment=experiment, transforms=[UnitX, StandardizeY]
        )
        obsf = ObservationFeatures(parameters={"x1": 1.0, "x2": 2.0})

        # Check for value error when optimization config is not set.
        with mock.patch.object(
            modelbridge, "_optimization_config", None
        ), self.assertRaisesRegex(ValueError, "optimization_config"):
            modelbridge.evaluate_acquisition_function(observation_features=[obsf])

        mock_acq_val = 5.0
        with mock.patch.object(
            modelbridge, "_evaluate_acquisition_function", return_value=[mock_acq_val]
        ) as mock_eval:
            acqf_vals = modelbridge.evaluate_acquisition_function(
                observation_features=[obsf]
            )
        self.assertEqual(acqf_vals, [mock_acq_val])
        mock_eval.assert_called_once()
        # Check that the private method was called with transformed obsf.
        # Bounds for branin are [-5.0, 10.0] and [0.0, 15.0].
        expected_X = [6.0 / 15.0, 2.0 / 15.0]
        self.assertEqual(
            mock_eval.call_args.kwargs["observation_features"],
            [
                [
                    ObservationFeatures(
                        parameters={"x1": expected_X[0], "x2": expected_X[1]}
                    )
                ]
            ],
        )

        # Check calls down to the acquisition function.
        acqf_path = "botorch.acquisition.logei.qLogNoisyExpectedImprovement.forward"
        with mock.patch(
            acqf_path, return_value=torch.tensor([mock_acq_val], dtype=torch.double)
        ) as mock_acqf:
            acqf_vals = modelbridge.evaluate_acquisition_function(
                observation_features=[obsf]
            )
        self.assertEqual(acqf_vals, [mock_acq_val])
        mock_acqf.assert_called_once()
        expected_tensor = torch.tensor([expected_X], dtype=torch.double)
        self.assertTrue(
            torch.allclose(mock_acqf.call_args.kwargs["X"], expected_tensor)
        )

        # Test evaluating at multiple points.
        # Case 1: List[ObsFeat, ObsFeat], should be 2 x 1 x d.
        with mock.patch(
            acqf_path,
            return_value=torch.tensor([mock_acq_val, mock_acq_val], dtype=torch.double),
        ) as mock_acqf:
            acqf_vals = modelbridge.evaluate_acquisition_function(
                observation_features=[obsf, obsf.clone()]
            )
        mock_acqf.assert_called_once()
        self.assertTrue(
            torch.allclose(
                mock_acqf.call_args.kwargs["X"], expected_tensor.repeat(2, 1, 1)
            )
        )
        # Case 2: List[List[ObsFeat, ObsFeat]], should be 1 x 2 x d.
        with mock.patch(
            acqf_path,
            return_value=torch.tensor([mock_acq_val, mock_acq_val], dtype=torch.double),
        ) as mock_acqf:
            acqf_vals = modelbridge.evaluate_acquisition_function(
                observation_features=[[obsf, obsf.clone()]]
            )
        mock_acqf.assert_called_once()
        self.assertTrue(
            torch.allclose(
                mock_acqf.call_args.kwargs["X"], expected_tensor.repeat(1, 2, 1)
            )
        )

    @mock.patch(
        f"{ModelBridge.__module__}.unwrap_observation_data",
        autospec=True,
        return_value=(2, 2),
    )
    @mock.patch(
        f"{ModelBridge.__module__}.gen_arms",
        autospec=True,
        return_value=([Arm(parameters={})], {}),
    )
    @mock.patch(
        f"{ModelBridge.__module__}.ModelBridge.predict",
        autospec=True,
        return_value=({"m": [1.0]}, {"m": {"m": [2.0]}}),
    )
    @mock.patch(f"{TorchModelBridge.__module__}.TorchModelBridge._fit", autospec=True)
    @mock.patch(
        f"{TorchModel.__module__}.TorchModel.gen",
        return_value=TorchGenResults(
            points=torch.tensor([[1]]),
            weights=torch.tensor([1.0]),
        ),
        autospec=True,
    )
    def test_best_point(
        self,
        _mock_gen,
        _mock_fit,
        _mock_predict,
        _mock_gen_arms,
        _mock_unwrap,
    ) -> None:
        exp = Experiment(search_space=get_search_space_for_range_value(), name="test")
        oc = OptimizationConfig(
            objective=Objective(metric=Metric("a"), minimize=False),
            outcome_constraints=[],
        )
        search_space = get_search_space_for_range_value()
        modelbridge = TorchModelBridge(
            search_space=search_space,
            model=TorchModel(),
            transforms=[transform_1, transform_2],
            experiment=exp,
            data=Data(),
            optimization_config=oc,
        )

        self.assertEqual(
            list(modelbridge.transforms.keys()),
            ["Cast", "transform_1", "transform_2"],
        )

        # _fit is mocked, which sets these
        modelbridge.parameters = list(search_space.parameters.keys())
        modelbridge.outcomes = ["a"]

        with mock.patch(
            f"{TorchModel.__module__}.TorchModel.best_point",
            return_value=torch.tensor([1.0]),
            autospec=True,
        ):
            run = modelbridge.gen(n=1, optimization_config=oc)
            arm, predictions = none_throws(run.best_arm_predictions)
            model_arm, model_predictions = none_throws(modelbridge.model_best_point())
            predictions = none_throws(predictions)
            model_predictions = none_throws(model_predictions)
        self.assertEqual(arm.parameters, {})
        self.assertEqual(predictions[0], {"m": 1.0})
        self.assertEqual(predictions[1], {"m": {"m": 2.0}})
        self.assertEqual(model_predictions[0], {"m": 1.0})
        self.assertEqual(model_predictions[1], {"m": {"m": 2.0}})

        # test optimization config validation - raise error when
        # ScalarizedOutcomeConstraint contains a metric that is not in the outcomes
        with self.assertRaises(ValueError):
            modelbridge.gen(
                n=1,
                optimization_config=OptimizationConfig(
                    objective=Objective(metric=Metric("a"), minimize=False),
                    outcome_constraints=[
                        ScalarizedOutcomeConstraint(
                            metrics=[Metric("wrong_metric_name")],
                            weights=[1.0],
                            op=ComparisonOp.LEQ,
                            bound=0,
                        )
                    ],
                ),
            )

        with mock.patch(
            f"{TorchModel.__module__}.TorchModel.best_point",
            side_effect=NotImplementedError,
            autospec=True,
        ):
            res = modelbridge.model_best_point()
        self.assertIsNone(res)

    def test_importances(self) -> None:
        experiment = get_branin_experiment_with_multi_objective(
            with_completed_trial=True
        )
        modelbridge = _get_modelbridge_from_experiment(experiment=experiment)
        # Model doesn't have enough data for training, so equal importances.
        self.assertEqual(
            modelbridge.feature_importances("branin_a"), {"x1": 0.5, "x2": 0.5}
        )
        self.assertEqual(
            modelbridge.feature_importances("branin_b"), {"x1": 0.5, "x2": 0.5}
        )

    @mock.patch(
        f"{TorchModel.__module__}.TorchModel.gen",
        return_value=TorchGenResults(
            points=torch.tensor([[1, 2], [2, 3]]),
            weights=torch.tensor([1.0, 2.0]),
            candidate_metadata=[
                {"some_key": "some_value_0"},
                {"some_key": "some_value_1"},
            ],
        ),
        autospec=True,
    )
    @mock.patch(f"{TorchModel.__module__}.TorchModel.update", autospec=True)
    @mock.patch(f"{TorchModel.__module__}.TorchModel.fit", autospec=True)
    def test_candidate_metadata_propagation(
        self,
        mock_model_fit: Mock,
        mock_model_update: Mock,
        mock_model_gen: Mock,
    ) -> None:
        exp = get_branin_experiment(with_status_quo=True, with_batch=True)
        # Check that the metadata is correctly re-added to observation
        # features during `fit`.
        # pyre-fixme[16]: `BaseTrial` has no attribute `_generator_run_structs`.
        preexisting_batch_gr = exp.trials[0]._generator_run_structs[0].generator_run
        preexisting_batch_gr._candidate_metadata_by_arm_signature = {
            preexisting_batch_gr.arms[0].signature: {
                "preexisting_batch_cand_metadata": "some_value"
            }
        }
        modelbridge = TorchModelBridge(
            experiment=exp,
            search_space=exp.search_space,
            model=TorchModel(),
            transforms=[],
            data=get_branin_data(),
        )

        datasets = mock_model_fit.call_args[1].get("datasets")
        X_expected = torch.tensor(
            [list(exp.trials[0].arms[0].parameters.values())],
            dtype=torch.double,
        )
        for dataset in datasets:
            self.assertTrue(torch.equal(dataset.X, X_expected))

        self.assertEqual(
            mock_model_fit.call_args[1].get("candidate_metadata"),
            [[{"preexisting_batch_cand_metadata": "some_value"}]],
        )

        # Check that `gen` correctly propagates the metadata to the GR.
        gr = modelbridge.gen(n=1)
        self.assertEqual(
            gr.candidate_metadata_by_arm_signature,
            {
                gr.arms[0].signature: {"some_key": "some_value_0"},
                gr.arms[1].signature: {"some_key": "some_value_1"},
            },
        )

        # Check that `None` candidate metadata is handled correctly.
        mock_model_gen.return_value = TorchGenResults(
            points=torch.tensor([[2, 4], [3, 5]]),
            weights=torch.tensor([1.0, 2.0]),
            candidate_metadata=None,
        )
        gr = modelbridge.gen(n=1)
        self.assertIsNone(gr.candidate_metadata_by_arm_signature)

        # Check that no candidate metadata is handled correctly.
        exp = get_branin_experiment(with_status_quo=True)

        with mock.patch(
            f"{TorchModelBridge.__module__}."
            "TorchModelBridge._validate_observation_data",
            autospec=True,
        ):
            modelbridge = TorchModelBridge(
                search_space=exp.search_space,
                experiment=exp,
                model=TorchModel(),
                data=Data(),
                transforms=[],
            )
            # Hack in outcome names to bypass validation (since we did not pass any
            # to the model so _fit did not populate this)
            metric_name = next(iter(exp.metrics))
            modelbridge.outcomes = [metric_name]
            modelbridge._metric_names = {metric_name}
        gr = modelbridge.gen(n=1)
        self.assertIsNone(mock_model_fit.call_args[1].get("candidate_metadata"))
        self.assertIsNone(gr.candidate_metadata_by_arm_signature)

    @mock.patch(f"{TorchModel.__module__}.TorchModel.fit", autospec=True)
    def test_fit_tracking_metrics(self, mock_model_fit: Mock) -> None:
        exp = get_experiment_with_observations(
            observations=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            with_tracking_metrics=True,
        )
        for fit_tracking_metrics in (True, False):
            mock_model_fit.reset_mock()
            modelbridge = TorchModelBridge(
                experiment=exp,
                search_space=exp.search_space,
                data=exp.lookup_data(),
                model=TorchModel(),
                transforms=[],
                fit_tracking_metrics=fit_tracking_metrics,
            )
            mock_model_fit.assert_called_once()
            call_kwargs = mock_model_fit.call_args.kwargs
            if fit_tracking_metrics:
                expected_outcomes = ["m1", "m2", "m3"]
            else:
                expected_outcomes = ["m1", "m2"]
            self.assertEqual(modelbridge.outcomes, expected_outcomes)
            self.assertEqual(len(call_kwargs["datasets"]), len(expected_outcomes))

    def test_convert_observations(self) -> None:
        experiment = get_branin_experiment(with_completed_trial=True)
        mb = _get_modelbridge_from_experiment(experiment=experiment)
        raw_X = torch.rand(10, 3) * 5
        raw_X[:, -1].round_()  # Make sure last column is integer.
        raw_X[0, -1] = 0  # Make sure task value 0 exists.
        raw_Y = torch.sin(raw_X).sum(-1)
        feature_names = ["x0", "x1", "x2"]
        metric_names = ["y"]
        observation_features = [
            ObservationFeatures(
                parameters={feature_names[i]: x_[i].item() for i in range(3)}
            )
            for x_ in raw_X
        ]
        observation_data = [
            ObservationData(
                metric_names=metric_names,
                means=np.asarray([y]),
                covariance=np.array([[float("nan")]]),
            )
            for y in raw_Y
        ]
        for use_task, expected_class in (
            (True, MultiTaskDataset),
            (False, SupervisedDataset),
        ):
            search_space_digest = SearchSpaceDigest(
                feature_names=feature_names,
                bounds=[(0.0, 5.0)] * 3,
                ordinal_features=[2],
                discrete_choices={2: list(range(0, 11))},
                task_features=[2] if use_task else [],
                target_values={2: 0} if use_task else {},  # pyre-ignore
            )
            converted_datasets, ordered_outcomes, _ = mb._convert_observations(
                observation_data=observation_data,
                observation_features=observation_features,
                outcomes=metric_names,
                parameters=feature_names,
                search_space_digest=search_space_digest,
            )
            self.assertEqual(len(converted_datasets), 1)
            dataset = none_throws(converted_datasets[0])
            self.assertIs(dataset.__class__, expected_class)
            if use_task:
                sort_idx = torch.argsort(raw_X[:, -1])
                expected_X = raw_X[sort_idx]
                expected_Y = raw_Y[sort_idx].unsqueeze(-1)
            else:
                expected_X = raw_X
                expected_Y = raw_Y.unsqueeze(-1)
            self.assertTrue(torch.equal(dataset.X, expected_X))
            self.assertTrue(torch.equal(dataset.Y, expected_Y))
            self.assertIsNone(dataset.Yvar)
            self.assertEqual(dataset.feature_names, feature_names)
            self.assertEqual(dataset.outcome_names, metric_names)
            self.assertEqual(ordered_outcomes, metric_names)

            with self.assertRaisesRegex(ValueError, "was not observed."):
                mb._convert_observations(
                    observation_data=observation_data,
                    observation_features=observation_features,
                    outcomes=metric_names + ["extra"],
                    parameters=feature_names,
                    search_space_digest=search_space_digest,
                )

    def test_convert_contextual_observations(self) -> None:
        raw_X = torch.rand(10, 3) * 5
        raw_X[:, -1].round_()  # Make sure last column is integer.
        raw_X[0, -1] = 0  # Make sure task value 0 exists.
        raw_Y = torch.sin(raw_X).sum(-1)
        feature_names = ["x0", "x1", "x2"]
        metric_names = ["y", "y:c0", "y:c1", "y:c2"]
        parameter_decomposition = {f"c{i}": [f"x{i}"] for i in range(3)}
        metric_decomposition = {f"c{i}": [f"y:c{i}"] for i in range(3)}

        with mock.patch(
            f"{ModelBridge.__module__}.ModelBridge.__init__",
            autospec=True,
        ):
            mb = _get_mock_modelbridge()
            mb._experiment_properties = {
                "parameter_decomposition": parameter_decomposition,
                "metric_decomposition": metric_decomposition,
            }

        observation_features = [
            ObservationFeatures(
                parameters={feature_names[i]: x_[i].item() for i in range(3)}
            )
            for x_ in raw_X
        ]
        num_m = len(metric_names)
        observation_data = [
            ObservationData(
                metric_names=metric_names,
                means=np.asarray([y for _ in range(num_m)]),
                covariance=np.array(
                    [float("nan") for _ in range(num_m * num_m)]
                ).reshape([num_m, num_m]),
            )
            for y in raw_Y
        ]
        converted_datasets, ordered_outcomes, _ = mb._convert_observations(
            observation_data=observation_data,
            observation_features=observation_features,
            outcomes=metric_names,
            parameters=feature_names,
            search_space_digest=SearchSpaceDigest(
                feature_names=feature_names,
                bounds=[(0.0, 5.0)] * 3,
                ordinal_features=[2],
                discrete_choices={2: list(range(0, 11))},
                task_features=[],
                target_values={},
            ),
        )
        self.assertEqual(len(converted_datasets), 2)
        expected_outcomes = list(converted_datasets[0].outcome_names)
        expected_outcomes.extend(list(converted_datasets[1].outcome_names))
        self.assertEqual(ordered_outcomes, expected_outcomes)
        for dataset in converted_datasets:
            self.assertIsInstance(dataset, ContextualDataset)
            self.assertEqual(dataset.feature_names, feature_names)
            self.assertDictEqual(
                checked_cast(ContextualDataset, dataset).parameter_decomposition,
                parameter_decomposition,
            )
            if len(dataset.outcome_names) == 1:
                self.assertListEqual(dataset.outcome_names, ["y"])
                self.assertTrue(torch.equal(dataset.X, raw_X))
                self.assertTrue(torch.equal(dataset.Y, raw_Y.unsqueeze(-1)))
            else:
                self.assertListEqual(dataset.outcome_names, ["y:c0", "y:c1", "y:c2"])
                self.assertListEqual(
                    checked_cast(ContextualDataset, dataset).context_buckets,
                    ["c0", "c1", "c2"],
                )
                self.assertDictEqual(
                    none_throws(
                        checked_cast(ContextualDataset, dataset).metric_decomposition
                    ),
                    metric_decomposition,
                )
                self.assertTrue(torch.equal(dataset.X, raw_X))
                self.assertTrue(
                    torch.equal(
                        dataset.Y,
                        torch.cat([raw_Y.unsqueeze(-1) for _ in range(3)], dim=-1),
                    )
                )
        # Test _get_fit_args handling of outcome names
        mb._fit_tracking_metrics = True
        search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name=f"x{i}",
                    lower=0.0,
                    upper=5.0,
                    parameter_type=ParameterType.FLOAT,
                )
                for i in range(3)
            ]
        )
        observations = []
        for i, od in enumerate(observation_data):
            observations.append(Observation(data=od, features=observation_features[i]))
        converted_datasets2, _, _ = mb._get_fit_args(
            search_space=search_space,
            observations=observations,
            parameters=feature_names,
            update_outcomes_and_parameters=True,
        )
        self.assertEqual(mb.outcomes, expected_outcomes)
        self.assertEqual(converted_datasets, converted_datasets2)
        datasets, _, _ = mb._get_fit_args(
            search_space=search_space,
            observations=observations,
            parameters=feature_names,
            update_outcomes_and_parameters=False,
        )
        self.assertEqual(mb.outcomes, expected_outcomes)

    @mock_botorch_optimize
    def test_gen_metadata_untransform(self) -> None:
        experiment = get_experiment_with_observations(
            observations=[[0.0, 1.0], [2.0, 3.0]]
        )
        model = BoTorchModel()
        mb = TorchModelBridge(
            experiment=experiment,
            search_space=experiment.search_space,
            data=experiment.lookup_data(),
            model=model,
            transforms=[],
        )
        for additional_metadata in (
            {},
            {"objective_thresholds": None},
            {
                "objective_thresholds": checked_cast(
                    MultiObjectiveOptimizationConfig, experiment.optimization_config
                ).objective_thresholds,
            },
        ):
            gen_return_value = TorchGenResults(
                points=torch.tensor([[1.0, 2.0]]),
                weights=torch.tensor([1.0]),
                gen_metadata={Keys.EXPECTED_ACQF_VAL: [1.0], **additional_metadata},
            )
            with mock.patch.object(
                mb, "_untransform_objective_thresholds"
            ) as mock_untransform, mock.patch.object(
                model, "gen", return_value=gen_return_value
            ):
                mb.gen(n=1)
            if additional_metadata.get("objective_thresholds", None) is None:
                mock_untransform.assert_not_called()
            else:
                mock_untransform.assert_called_once()

    @mock_botorch_optimize
    def test_gen_with_expanded_parameter_space(self) -> None:
        # Test that an expanded search space with range and unordered choice
        # parameters can still generate (when using the default transforms).
        search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x1",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=1.0,
                ),
                ChoiceParameter(
                    name="x2",
                    parameter_type=ParameterType.FLOAT,
                    values=[0.0, 1.0, 2.0],
                    is_ordered=False,
                ),
            ]
        )
        experiment = get_experiment_with_observations(
            observations=[[0.0, 1.0], [2.0, 3.0]], search_space=search_space
        )
        # Attach a trial from outside of the search space.
        trial = experiment.new_trial(
            generator_run=GeneratorRun(
                arms=[Arm(parameters={"x1": 1.5, "x2": 0.5}, name="manual")]
            )
        )
        data = Data(
            df=DataFrame.from_records(
                [
                    {
                        "arm_name": "manual",
                        "metric_name": metric,
                        "mean": o,
                        "sem": None,
                        "trial_index": trial.index,
                    }
                    for metric, o in (("m1", 0.2), ("m2", 0.5))
                ]
            )
        )
        experiment.attach_data(data)
        trial.run().complete()
        modelbridge = _get_modelbridge_from_experiment(
            experiment=experiment, transforms=MBM_X_trans
        )
        # Check the expanded model space. Range is expanded, Choice is not.
        model_space = modelbridge._model_space
        self.assertEqual(
            model_space.parameters["x1"],
            RangeParameter(
                name="x1", lower=0.0, upper=1.5, parameter_type=ParameterType.FLOAT
            ),
        )
        self.assertEqual(model_space.parameters["x2"], search_space.parameters["x2"])
        self.assertNotEqual(modelbridge._model_space, modelbridge._search_space)
        # Generate candidates.
        gr = modelbridge.gen(n=3)
        self.assertEqual(len(gr.arms), 3)
