#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import numpy as np
import torch
from ax.core.observation import ObservationFeatures
from ax.modelbridge.array import ArrayModelBridge
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.models.torch_base import TorchModel
from ax.utils.common.testutils import TestCase


class TorchModelBridgeTest(TestCase):
    @mock.patch(
        f"{ArrayModelBridge.__module__}.ArrayModelBridge.__init__",
        autospec=True,
        return_value=None,
    )
    def testTorchModelBridge(self, mock_init):
        torch_dtype = torch.float64
        torch_device = torch.device("cpu")
        ma = TorchModelBridge(
            experiment=None,
            search_space=None,
            data=None,
            model=None,
            transforms=[],
            torch_dtype=torch.float64,
            torch_device=torch.device("cpu"),
        )
        self.assertEqual(ma.dtype, torch.float64)
        self.assertEqual(ma.device, torch.device("cpu"))
        self.assertFalse(mock_init.call_args[-1]["fit_out_of_design"])
        # Test `fit`.
        model = mock.MagicMock(TorchModel, autospec=True, instance=True)
        X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        Y = np.array([[3.0], [4.0]])
        var = np.array([[1.0], [2.0]])
        ma._model_fit(
            model=model,
            Xs=[X],
            Ys=[Y],
            Yvars=[var],
            bounds=None,
            feature_names=[],
            metric_names=[],
            task_features=[],
            fidelity_features=[],
            candidate_metadata=[],
        )
        model_fit_args = model.fit.mock_calls[0][2]
        self.assertTrue(
            torch.equal(
                model_fit_args["Xs"][0],
                torch.tensor(X, dtype=torch_dtype, device=torch_device),
            )
        )
        self.assertTrue(
            torch.equal(
                model_fit_args["Ys"][0],
                torch.tensor(Y, dtype=torch_dtype, device=torch_device),
            )
        )
        self.assertTrue(
            torch.equal(
                model_fit_args["Yvars"][0],
                torch.tensor(var, dtype=torch_dtype, device=torch_device),
            )
        )
        # Test `update` (need to fill required fields before call to `_model_update`).
        ma.parameters = []
        ma.outcomes = []
        ma._model_update(
            Xs=[X],
            Ys=[Y],
            Yvars=[var],
            candidate_metadata=[],
            bounds=None,
            feature_names=[],
            metric_names=[],
            task_features=[],
            fidelity_features=[],
            target_fidelities=[],
        )
        model_update_args = model.update.mock_calls[0][2]
        self.assertTrue(
            torch.equal(
                model_update_args["Xs"][0],
                torch.tensor(X, dtype=torch_dtype, device=torch_device),
            )
        )
        self.assertTrue(
            torch.equal(
                model_update_args["Ys"][0],
                torch.tensor(Y, dtype=torch_dtype, device=torch_device),
            )
        )
        self.assertTrue(
            torch.equal(
                model_update_args["Yvars"][0],
                torch.tensor(var, dtype=torch_dtype, device=torch_device),
            )
        )
        # Predict
        model.predict.return_value = (torch.tensor([3.0]), torch.tensor([4.0]))
        f, var = ma._model_predict(X)
        self.assertTrue(
            torch.equal(
                model.predict.mock_calls[0][2]["X"],
                torch.tensor(X, dtype=torch_dtype, device=torch_device),
            )
        )
        self.assertTrue(np.array_equal(f, np.array([3.0])))
        self.assertTrue(np.array_equal(var, np.array([4.0])))
        # Gen
        model.gen.return_value = (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([1.0]),
            {},
            [],
        )
        X, w, _gen_metadata, _candidate_metadata = ma._model_gen(
            n=3,
            bounds=[(0, 1)],
            objective_weights=np.array([1.0, 0.0]),
            outcome_constraints=None,
            linear_constraints=None,
            fixed_features={1: 3.0},
            pending_observations=[np.array([]), np.array([1.0, 2.0, 3.0])],
            model_gen_options={"option": "yes"},
            rounding_func=np.round,
            target_fidelities=None,
        )
        gen_args = model.gen.mock_calls[0][2]
        self.assertEqual(gen_args["n"], 3)
        self.assertEqual(gen_args["bounds"], [(0, 1)])
        self.assertTrue(
            torch.equal(
                gen_args["objective_weights"],
                torch.tensor([1.0, 0.0], dtype=torch_dtype, device=torch_device),
            )
        )
        self.assertIsNone(gen_args["outcome_constraints"])
        self.assertIsNone(gen_args["linear_constraints"])
        self.assertEqual(gen_args["fixed_features"], {1: 3.0})
        self.assertTrue(
            torch.equal(
                gen_args["pending_observations"][0],
                torch.tensor([], dtype=torch_dtype, device=torch_device),
            )
        )
        self.assertTrue(
            torch.equal(
                gen_args["pending_observations"][1],
                torch.tensor([1.0, 2.0, 3.0], dtype=torch_dtype, device=torch_device),
            )
        )
        self.assertEqual(gen_args["model_gen_options"], {"option": "yes"})
        self.assertIsNone(gen_args["target_fidelities"])
        # check rounding function
        t = torch.tensor([0.1, 0.6], dtype=torch_dtype, device=torch_device)
        self.assertTrue(torch.equal(gen_args["rounding_func"](t), torch.round(t)))

        self.assertTrue(np.array_equal(X, np.array([1.0, 2.0, 3.0])))
        self.assertTrue(np.array_equal(w, np.array([1.0])))

        # Cross-validate
        model.cross_validate.return_value = (torch.tensor([3.0]), torch.tensor([4.0]))
        f, var = ma._model_cross_validate(
            Xs_train=[X],
            Ys_train=[Y],
            Yvars_train=[var],
            X_test=X,
            bounds=[(0, 1)],
            task_features=[],
            feature_names=[],
            metric_names=[],
            fidelity_features=[],
        )
        model_cv_args = model.cross_validate.mock_calls[0][2]
        self.assertTrue(
            torch.equal(
                model_cv_args["Xs_train"][0],
                torch.tensor(X, dtype=torch_dtype, device=torch_device),
            )
        )
        self.assertTrue(
            torch.equal(
                model_cv_args["Ys_train"][0],
                torch.tensor(Y, dtype=torch_dtype, device=torch_device),
            )
        )
        self.assertTrue(
            torch.equal(
                model_cv_args["Yvars_train"][0],
                torch.tensor(var, dtype=torch_dtype, device=torch_device),
            )
        )
        self.assertTrue(
            torch.equal(
                model_cv_args["X_test"],
                torch.tensor(X, dtype=torch_dtype, device=torch_device),
            )
        )
        self.assertTrue(np.array_equal(f, np.array([3.0])))
        self.assertTrue(np.array_equal(var, np.array([4.0])))

        # Transform observation features
        obsf = [ObservationFeatures(parameters={"x": 1.0, "y": 2.0})]
        ma.parameters = ["x", "y"]
        X = ma._transform_observation_features(obsf)
        self.assertTrue(
            torch.equal(
                X, torch.tensor([[1.0, 2.0]], dtype=torch_dtype, device=torch_device)
            )
        )
        # test fit out of design
        ma = TorchModelBridge(
            experiment=None,
            search_space=None,
            data=None,
            model=None,
            transforms=[],
            torch_dtype=torch.float64,
            torch_device=torch.device("cpu"),
            fit_out_of_design=True,
        )
        self.assertTrue(mock_init.call_args[-1]["fit_out_of_design"])

    @mock.patch(f"{TorchModel.__module__}.TorchModel", autospec=True)
    @mock.patch(f"{ArrayModelBridge.__module__}.ArrayModelBridge.__init__")
    def test_evaluate_acquisition_function(self, _, mock_torch_model):
        ma = TorchModelBridge(
            experiment=None,
            search_space=None,
            data=None,
            model=None,
            transforms=[],
            torch_dtype=torch.float64,
            torch_device=torch.device("cpu"),
        )
        # These attributes would've been set by `ArrayModelBridge` __init__, but it's
        # mocked.
        ma.model = mock_torch_model()
        t = mock.MagicMock(Transform, autospec=True)
        t.transform_observation_features.return_value = [
            ObservationFeatures(parameters={"x": 3.0, "y": 4.0})
        ]
        ma.transforms = {"ExampleTransform": t}
        ma.parameters = ["x", "y"]
        model_eval_acqf = mock_torch_model.return_value.evaluate_acquisition_function
        model_eval_acqf.return_value = torch.tensor([5.0], dtype=torch.float64)
        acqf_vals = ma.evaluate_acquisition_function(
            observation_features=[ObservationFeatures(parameters={"x": 1.0, "y": 2.0})]
        )
        self.assertEqual(acqf_vals, [5.0])
        t.transform_observation_features.assert_called_with(
            [ObservationFeatures(parameters={"x": 1.0, "y": 2.0})]
        )
        model_eval_acqf.assert_called_once()
        self.assertTrue(
            torch.equal(  # `call_args` is an (args, kwargs) tuple
                model_eval_acqf.call_args[1]["X"],
                torch.tensor([[3.0, 4.0]], dtype=torch.float64),
            )
        )
