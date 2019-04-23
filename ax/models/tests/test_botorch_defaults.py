#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import mock
import torch
from ax.models.torch.botorch_defaults import _get_model, get_and_fit_model
from ax.utils.common.testutils import TestCase
from botorch.models import FixedNoiseGP
from botorch.models.multitask import FixedNoiseMultiTaskGP


class BotorchDefaultsTest(TestCase):
    def test_get_model(self):
        x = torch.zeros(2, 2)
        y = torch.zeros(2)
        se = torch.zeros(2)
        model = _get_model(x, y, se, None)
        self.assertTrue(isinstance(model, FixedNoiseGP))
        model = _get_model(x, y, se, 1)
        self.assertTrue(isinstance(model, FixedNoiseMultiTaskGP))

    @mock.patch("ax.models.torch.botorch_defaults._get_model", autospec=True)
    @mock.patch("ax.models.torch.botorch_defaults.ModelListGP", autospec=True)
    def test_task_feature(self, gp_mock, get_model_mock):
        x = [torch.zeros(2, 2)]
        y = [torch.zeros(2, 1)]
        yvars = [torch.ones(2, 1)]
        get_and_fit_model(Xs=x, Ys=y, Yvars=yvars, task_features=[1], state_dict=[])
        # Check that task feature was correctly passed to _get_model
        self.assertEqual(get_model_mock.mock_calls[0][2]["task_feature"], 1)

        with self.assertRaises(ValueError):
            get_and_fit_model(
                Xs=x, Ys=y, Yvars=yvars, task_features=[0, 1], state_dict=[]
            )
