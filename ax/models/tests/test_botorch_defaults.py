#!/usr/bin/env python3

import torch
from ax.models.torch.botorch_defaults import _get_model
from ax.utils.common.testutils import TestCase
from botorch.models import ConstantNoiseGP


class BotorchDefaultsTest(TestCase):
    def test_get_model(self):
        x = torch.zeros(1, 1)
        y = torch.zeros(1)
        se = torch.zeros(1)
        model = _get_model(x, y, se)
        self.assertTrue(isinstance(model, ConstantNoiseGP))
