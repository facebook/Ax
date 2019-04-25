#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from ax.exceptions.model import ModelError
from ax.models.torch.utils import is_noiseless
from ax.utils.common.testutils import TestCase
from botorch.models import HeteroskedasticSingleTaskGP, ModelListGP, SingleTaskGP


class TorchModelUtilsTest(TestCase):
    def test_is_noiseless(self):
        x = torch.zeros(1, 1)
        y = torch.zeros(1)
        se = torch.zeros(1)
        model = SingleTaskGP(x, y)
        self.assertTrue(is_noiseless(model))
        model = HeteroskedasticSingleTaskGP(x, y, se)
        self.assertFalse(is_noiseless(model))
        with self.assertRaises(ModelError):
            is_noiseless(ModelListGP([]))
