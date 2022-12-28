#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from unittest import mock

import numpy as np
import torch
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.observation import (
    ObservationData,
    ObservationFeatures,
    recombine_observations,
)
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import ScalarizedOutcomeConstraint
from ax.core.search_space import SearchSpace, SearchSpaceDigest
from ax.core.types import ComparisonOp
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.models.torch_base import TorchGenResults, TorchModel
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data,
    get_branin_experiment,
    get_branin_search_space,
    get_search_space_for_range_value,
)
from ax.utils.testing.modeling_stubs import get_observation1, transform_1, transform_2
from botorch.utils.datasets import FixedNoiseDataset


class TorchModelBridgeTest(TestCase):
    @mock.patch(
        f"{ModelBridge.__module__}.ModelBridge.__init__",
        autospec=True,
        return_value=None,
    )
    # pyre-fixme[2]: Parameter must be annotated.
    def test_TorchModelBridge(self, mock_init, dtype=None, device=None) -> None:
        ma = TorchModelBridge(
            # pyre-fixme[6]: For 1st param expected `Experiment` but got `None`.
            experiment=None,
            # pyre-fixme[6]: For 2nd param expected `SearchSpace` but got `None`.
            search_space=None,
            # pyre-fixme[6]: For 3rd param expected `Data` but got `None`.
            data=None,
            # pyre-fixme[6]: For 4th param expected `TorchModel` but got `None`.
            model=None,
            transforms=[],
            torch_dtype=dtype,
            torch_device=device,
        )
        dtype = dtype or torch.double
        self.assertEqual(ma.dtype, dtype)
        self.assertEqual(ma.device, device)
        self.assertFalse(mock_init.call_args[-1]["fit_out_of_design"])
        self.assertIsNone(ma._last_observations)
        tkwargs = {"dtype": dtype, "device": device}
        # Test `_fit`.
        feature_names = ["x1", "x2", "x3"]
        model = mock.MagicMock(TorchModel, autospec=True, instance=True)
        search_space = mock.MagicMock(SearchSpace, autospec=True, instance=True)
        type(search_space).parameters = mock.PropertyMock(
            return_value={fn: None for fn in feature_names}  # only need `.keys()`
        )
        X = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], **tkwargs)
        datasets = {
            "y1": FixedNoiseDataset(
                X=X,
                Y=torch.tensor([[3.0], [1.0]], **tkwargs),
                Yvar=torch.tensor([[4.0], [2.0]], **tkwargs),
            ),
            "y2": FixedNoiseDataset(
                X=X,
                Y=torch.tensor([[2.0], [0.0]], **tkwargs),
                Yvar=torch.tensor([[2.0], [1.0]], **tkwargs),
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
                datasets["y1"].Y().tolist(),
                datasets["y2"].Y().tolist(),
                datasets["y1"].Yvar().tolist(),
                datasets["y2"].Yvar().tolist(),
            )
        ]
        observations = recombine_observations(observation_features, observation_data)
        ssd = SearchSpaceDigest(
            feature_names=feature_names, bounds=[(0, 1)] * 3  # pyre-ignore
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
        self.assertEqual(model_fit_args["metric_names"], ["y1", "y2"])
        self.assertEqual(model_fit_args["search_space_digest"], ssd)
        self.assertIsNone(model_fit_args["candidate_metadata"])
        self.assertEqual(ma._last_observations, observations)

        with mock.patch(f"{TorchModelBridge.__module__}.logger.info") as mock_logger:
            ma._fit(
                model=model,
                search_space=search_space,
                observations=observations,
            )
        mock_logger.assert_called_once_with(
            "The observations are identical to the last set of observations "
            "used to fit the model. Skipping model fitting."
        )

        # Test `_update`
        with mock.patch(
            f"{TorchModelBridge.__module__}.extract_search_space_digest",
            return_value=ssd,
        ):
            ma._update(
                search_space=search_space,
                observations=observations,
            )
        model_update_args = model.update.mock_calls[0][2]
        self.assertEqual(model_update_args["datasets"], list(datasets.values()))
        self.assertEqual(model_update_args["metric_names"], ["y1", "y2"])
        self.assertEqual(model_update_args["search_space_digest"], ssd)
        self.assertIsNone(model_update_args["candidate_metadata"])

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
                optimization_config=OptimizationConfig(
                    objective=Objective(metric=Metric("y1"), minimize=False),
                ),
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
        self.assertEqual(gen_args["search_space_digest"].target_fidelities, {})
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
        ma = TorchModelBridge(
            # pyre-fixme[6]: For 1st param expected `Experiment` but got `None`.
            experiment=None,
            # pyre-fixme[6]: For 2nd param expected `SearchSpace` but got `None`.
            search_space=None,
            # pyre-fixme[6]: For 3rd param expected `Data` but got `None`.
            data=None,
            # pyre-fixme[6]: For 4th param expected `TorchModel` but got `None`.
            model=None,
            transforms=[],
            torch_dtype=dtype,
            torch_device=device,
            fit_out_of_design=True,
        )
        self.assertTrue(mock_init.call_args[-1]["fit_out_of_design"])

    def test_TorchModelBridge_float(self) -> None:
        self.test_TorchModelBridge(dtype=torch.float)

    def test_TorchModelBridge_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_TorchModelBridge(device=torch.device("cuda"))

    @mock.patch(f"{TorchModel.__module__}.TorchModel", autospec=True)
    @mock.patch(f"{ModelBridge.__module__}.ModelBridge.__init__")
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def test_evaluate_acquisition_function(self, _, mock_torch_model):
        ma = TorchModelBridge(
            # pyre-fixme[6]: For 1st param expected `Experiment` but got `None`.
            experiment=None,
            # pyre-fixme[6]: For 2nd param expected `SearchSpace` but got `None`.
            search_space=None,
            # pyre-fixme[6]: For 3rd param expected `Data` but got `None`.
            data=None,
            # pyre-fixme[6]: For 4th param expected `TorchModel` but got `None`.
            model=None,
            transforms=[],
            torch_dtype=torch.float64,
            torch_device=torch.device("cpu"),
        )
        # These attributes would've been set by `ModelBridge` __init__, but it's mocked.
        ma.model = mock_torch_model()
        t = mock.MagicMock(Transform, autospec=True, wraps=Transform(None, None, None))
        ma.transforms = {"ExampleTransform": t}
        ma.parameters = ["x", "y"]
        model_eval_acqf = mock_torch_model.return_value.evaluate_acquisition_function
        model_eval_acqf.return_value = torch.tensor([5.0], dtype=torch.float64)

        ma._model_space = get_branin_search_space()
        ma._optimization_config = None
        ma.outcomes = ["test_metric"]

        with self.assertRaisesRegex(ValueError, "optimization_config"):
            ma.evaluate_acquisition_function(
                observation_features=[
                    ObservationFeatures(parameters={"x": 1.0, "y": 2.0})
                ],
            )

        with mock.patch(
            "ax.modelbridge.torch.extract_search_space_digest",
            return_value=SearchSpaceDigest(feature_names=[], bounds=[]),
        ):
            acqf_vals = ma.evaluate_acquisition_function(
                observation_features=[
                    ObservationFeatures(parameters={"x": 1.0, "y": 2.0})
                ],
                optimization_config=OptimizationConfig(
                    objective=Objective(metric=Metric(name="test_metric"))
                ),
            )

        self.assertEqual(acqf_vals, [5.0])
        t.transform_observation_features.assert_any_call(
            [ObservationFeatures(parameters={"x": 1.0, "y": 2.0})],
        )
        t.transform_observation_features.reset_mock()
        model_eval_acqf.assert_called_once()
        self.assertTrue(
            torch.equal(  # `call_args` is an (args, kwargs) tuple
                model_eval_acqf.call_args[1]["X"],
                torch.tensor([[[1.0, 2.0]]], dtype=torch.float64),
            )
        )

        # Test evaluating at multiple points.
        # Case 1: List[ObsFeat, ObsFeat], should be 2 x 1 x d.
        with mock.patch(
            "ax.modelbridge.torch.extract_search_space_digest",
            return_value=SearchSpaceDigest(feature_names=[], bounds=[]),
        ):
            acqf_vals = ma.evaluate_acquisition_function(
                observation_features=[
                    ObservationFeatures(parameters={"x": 1.0, "y": 2.0}),
                    ObservationFeatures(parameters={"x": 1.0, "y": 2.0}),
                ],
                optimization_config=OptimizationConfig(
                    objective=Objective(metric=Metric(name="test_metric"))
                ),
            )
        t.transform_observation_features.assert_any_call(
            [ObservationFeatures(parameters={"x": 1.0, "y": 2.0})],
        )
        t.transform_observation_features.reset_mock()
        self.assertTrue(
            torch.equal(  # `call_args` is an (args, kwargs) tuple
                model_eval_acqf.call_args[-1]["X"],
                torch.tensor([[[1.0, 2.0]], [[1.0, 2.0]]], dtype=torch.float64),
            )
        )
        # Case 2: List[List[ObsFeat, ObsFeat]], should be 1 x 2 x d.
        with mock.patch(
            "ax.modelbridge.torch.extract_search_space_digest",
            return_value=SearchSpaceDigest(feature_names=[], bounds=[]),
        ):
            acqf_vals = ma.evaluate_acquisition_function(
                observation_features=[
                    [
                        ObservationFeatures(parameters={"x": 1.0, "y": 2.0}),
                        ObservationFeatures(parameters={"x": 1.0, "y": 2.0}),
                    ]
                ],
                optimization_config=OptimizationConfig(
                    objective=Objective(metric=Metric(name="test_metric"))
                ),
            )
        t.transform_observation_features.assert_any_call(
            [
                ObservationFeatures(parameters={"x": 1.0, "y": 2.0}),
                ObservationFeatures(parameters={"x": 1.0, "y": 2.0}),
            ],
        )
        self.assertTrue(
            torch.equal(  # `call_args` is an (args, kwargs) tuple
                model_eval_acqf.call_args[-1]["X"],
                torch.tensor([[[1.0, 2.0], [1.0, 2.0]]], dtype=torch.float64),
            )
        )

    @mock.patch(
        f"{ModelBridge.__module__}.observations_from_data",
        autospec=True,
        return_value=([get_observation1()]),
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
            points=torch.tensor([[1, 2]]),
            weights=torch.tensor([1.0]),
        ),
        autospec=True,
    )
    # pyre-fixme[3]: Return type must be annotated.
    def test_best_point(
        self,
        _mock_gen,
        _mock_fit,
        _mock_predict,
        _mock_gen_arms,
        _mock_unwrap,
        _mock_obs_from_data,
    ):
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
            return_value=torch.tensor([1.0, 2.0]),
            autospec=True,
        ):
            run = modelbridge.gen(n=1, optimization_config=oc)
            # pyre-fixme[23]: Unable to unpack `Optional[Tuple[Arm,
            #  Optional[Tuple[Dict[str, float], Optional[Dict[str, typing.Dict[str,
            #  float]]]]]]]` into 2 values.
            arm, predictions = run.best_arm_predictions
            # pyre-fixme[23]: Unable to unpack `Optional[Tuple[Arm,
            #  Optional[Tuple[Dict[str, float], Optional[Dict[str, typing.Dict[str,
            #  float]]]]]]]` into 2 values.
            model_arm, model_predictions = modelbridge.model_best_point()
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

    @mock.patch(
        f"{ModelBridge.__module__}.observations_from_data",
        autospec=True,
        return_value=([get_observation1()]),
    )
    @mock.patch(
        f"{ModelBridge.__module__}.unwrap_observation_data",
        autospec=True,
        return_value=(2, 2),
    )
    @mock.patch(
        f"{ModelBridge.__module__}.gen_arms",
        autospec=True,
        return_value=[Arm(parameters={})],
    )
    @mock.patch(
        f"{ModelBridge.__module__}.ModelBridge.predict",
        autospec=True,
        return_value=({"m": [1.0]}, {"m": {"m": [2.0]}}),
    )
    @mock.patch(f"{TorchModelBridge.__module__}.TorchModelBridge._fit", autospec=True)
    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `numpy.array([[[1.000000]], [[2.000000]]])` to decorator factory
    #  `unittest.mock.patch`.
    @mock.patch(
        f"{TorchModel.__module__}.TorchModel.feature_importances",
        return_value=np.array([[[1.0]], [[2.0]]]),
        autospec=True,
    )
    # pyre-fixme[3]: Return type must be annotated.
    def test_importances(
        self,
        _mock_feature_importances,
        _mock_fit,
        _mock_predict,
        _mock_gen_arms,
        _mock_unwrap,
        _mock_obs_from_data,
    ):
        exp = Experiment(search_space=get_search_space_for_range_value(), name="test")
        search_space = get_search_space_for_range_value()
        modelbridge = TorchModelBridge(
            search_space=search_space,
            model=TorchModel(),
            transforms=[transform_1, transform_2],
            experiment=exp,
            data=Data(),
        )
        # _fit is mocked, which typically sets these
        modelbridge.parameters = list(search_space.parameters.keys())
        modelbridge.outcomes = ["a", "b"]

        self.assertEqual(modelbridge.feature_importances("a"), {"x": [1.0]})
        self.assertEqual(modelbridge.feature_importances("b"), {"x": [2.0]})

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
    # pyre-fixme[3]: Return type must be annotated.
    def test_candidate_metadata_propagation(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        mock_model_fit,
        # pyre-fixme[2]: Parameter must be annotated.
        mock_model_update,
        # pyre-fixme[2]: Parameter must be annotated.
        mock_model_gen,
    ):
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
            self.assertTrue(torch.equal(dataset.X(), X_expected))

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

        # Check that the metadata is correctly re-added to observation
        # features during `update`.
        batch = exp.new_batch_trial(generator_run=gr)
        modelbridge.update(
            experiment=exp, new_data=get_branin_data(trial_indices=[batch.index])
        )
        datasets = mock_model_update.call_args[1].get("datasets")
        X_expected = torch.cat((X_expected, torch.tensor([[1, 2]])), dim=-2)
        for dataset in datasets:
            self.assertTrue(torch.equal(dataset.X(), X_expected))

        self.assertEqual(
            mock_model_update.call_args[1].get("candidate_metadata"),
            [
                [
                    {"preexisting_batch_cand_metadata": "some_value"},
                    # new data contained data just for arm '1_0', not for '1_1',
                    # so we don't expect to see '{"some_key": "some_value_1"}'
                    # in candidate metadata.
                    {"some_key": "some_value_0"},
                ]
            ],
        )

        # Check that `None` candidate metadata is handled correctly.
        mock_model_gen.return_value = TorchGenResults(
            points=torch.tensor([[2, 4], [3, 5]]),
            weights=torch.tensor([1.0, 2.0]),
            candidate_metadata=None,
        )
        gr = modelbridge.gen(n=1)
        self.assertIsNone(gr.candidate_metadata_by_arm_signature)

        # Check that the metadata is correctly re-added to observation
        # features during `update`.
        batch = exp.new_batch_trial(generator_run=gr)
        modelbridge.update(
            experiment=exp, new_data=get_branin_data(trial_indices=[batch.index])
        )
        datasets = mock_model_update.call_args[1].get("datasets")
        X_expected = torch.cat((X_expected, torch.tensor([[2, 4]])), dim=-2)
        for dataset in datasets:
            self.assertTrue(torch.equal(dataset.X(), X_expected))

        self.assertEqual(
            mock_model_update.call_args[1].get("candidate_metadata"),
            [
                [
                    {"preexisting_batch_cand_metadata": "some_value"},
                    {"some_key": "some_value_0"},
                    {},
                ]
            ],
        )

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
            # pyre-fixme[8]: Attribute has type `List[str]`; used as `str`.
            # pyre-fixme[8]: Attribute has type `Set[str]`; used as `str`.
            modelbridge.outcomes = modelbridge._metric_names = next(iter(exp.metrics))
        gr = modelbridge.gen(n=1)
        self.assertIsNone(mock_model_fit.call_args[1].get("candidate_metadata"))
        self.assertIsNone(gr.candidate_metadata_by_arm_signature)
        batch = exp.new_batch_trial(generator_run=gr)
        modelbridge.update(
            experiment=exp, new_data=get_branin_data(trial_indices=[batch.index])
        )
