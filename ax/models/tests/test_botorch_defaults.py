#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from unittest import mock

import torch
from ax.models.torch.botorch_defaults import _get_model, get_and_fit_model
from ax.utils.common.testutils import TestCase
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP


class BotorchDefaultsTest(TestCase):
    def test_get_model(self):
        x = torch.zeros(2, 2)
        y = torch.zeros(2, 1)
        var = torch.zeros(2, 1)
        partial_var = torch.tensor([0, float("nan")])
        unknown_var = torch.tensor([float("nan"), float("nan")])
        model = _get_model(x, y, unknown_var, None)
        self.assertIsInstance(model, SingleTaskGP)

        model = _get_model(X=x, Y=y, Yvar=var)
        self.assertIsInstance(model, FixedNoiseGP)
        model = _get_model(X=x, Y=y, Yvar=unknown_var, task_feature=1)
        self.assertTrue(type(model) == MultiTaskGP)  # Don't accept subclasses.
        model = _get_model(X=x, Y=y, Yvar=var, task_feature=1)
        self.assertIsInstance(model, FixedNoiseMultiTaskGP)
        with self.assertRaises(ValueError):
            model = _get_model(X=x, Y=y, Yvar=partial_var, task_feature=None)
        model = _get_model(X=x, Y=y, Yvar=var, fidelity_features=[-1])
        self.assertTrue(isinstance(model, SingleTaskMultiFidelityGP))
        with self.assertRaises(NotImplementedError):
            _get_model(X=x, Y=y, Yvar=var, task_feature=1, fidelity_features=[-1])

    @mock.patch("ax.models.torch.botorch_defaults._get_model", autospec=True)
    @mock.patch("ax.models.torch.botorch_defaults.ModelListGP", autospec=True)
    def test_task_feature(self, gp_mock, get_model_mock):
        x = [torch.zeros(2, 2)]
        y = [torch.zeros(2, 1)]
        yvars = [torch.ones(2, 1)]
        get_and_fit_model(
            Xs=x,
            Ys=y,
            Yvars=yvars,
            task_features=[1],
            fidelity_features=[],
            state_dict=[],
            refit_model=False,
        )
        # Check that task feature was correctly passed to _get_model
        self.assertEqual(get_model_mock.mock_calls[0][2]["task_feature"], 1)

        # check error on multiple task features
        with self.assertRaises(NotImplementedError):
            get_and_fit_model(
                Xs=x,
                Ys=y,
                Yvars=yvars,
                task_features=[0, 1],
                fidelity_features=[],
                state_dict=[],
                refit_model=False,
            )

        # check error on multiple fidelity features
        with self.assertRaises(NotImplementedError):
            get_and_fit_model(
                Xs=x,
                Ys=y,
                Yvars=yvars,
                task_features=[],
                fidelity_features=[-1, -2],
                state_dict=[],
                refit_model=False,
            )

        # check error on botch task and fidelity feature
        with self.assertRaises(NotImplementedError):
            get_and_fit_model(
                Xs=x,
                Ys=y,
                Yvars=yvars,
                task_features=[1],
                fidelity_features=[-1],
                state_dict=[],
                refit_model=False,
            )
