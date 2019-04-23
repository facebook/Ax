#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from unittest import mock

import numpy as np
import torch
from ax.modelbridge.torch import TorchModelBridge
from ax.models.torch_base import TorchModel
from ax.utils.common.testutils import TestCase


class TorchModelBridgeTest(TestCase):
    @mock.patch(
        "ax.modelbridge.array.ArrayModelBridge.__init__",
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
        # Fit
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
            task_features=[],
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
        # Update
        ma._model_update(Xs=[X], Ys=[Y], Yvars=[var])
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
        model.gen.return_value = (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0]))
        X, w = ma._model_gen(
            n=3,
            bounds=[(0, 1)],
            objective_weights=np.array([1.0, 0.0]),
            outcome_constraints=None,
            linear_constraints=None,
            fixed_features={1: 3.0},
            pending_observations=[np.array([]), np.array([1.0, 2.0, 3.0])],
            model_gen_options={"option": "yes"},
            rounding_func=np.round,
        )
        gen_args = model.gen.mock_calls[0][2]
        self.assertEqual(gen_args["n"], 3)
        self.assertEqual(gen_args["bounds"], [(0, 1)])
        print(gen_args["objective_weights"])
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
        self.assertTrue(np.array_equal(X, np.array([1.0, 2.0, 3.0])))
        self.assertTrue(np.array_equal(w, np.array([1.0])))
        # Cross-validate
        model.cross_validate.return_value = (torch.tensor([3.0]), torch.tensor([4.0]))
        f, var = ma._model_cross_validate(
            Xs_train=[X], Ys_train=[Y], Yvars_train=[var], X_test=X
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
