#!/usr/bin/env python3


import torch
from ae.lazarus.ae.exceptions.model import ModelError
from ae.lazarus.ae.models.torch.utils import is_noiseless
from ae.lazarus.ae.utils.common.testutils import TestCase
from botorch.models import (
    FidelityAwareSingleTaskGP,
    HeteroskedasticSingleTaskGP,
    MultiOutputGP,
    SingleTaskGP,
)


class TorchModelUtilsTest(TestCase):
    def test_is_noiseless(self):
        x = torch.zeros(1, 1)
        y = torch.zeros(1)
        se = torch.zeros(1)
        model = SingleTaskGP(x, y)
        self.assertTrue(is_noiseless(model))
        model = HeteroskedasticSingleTaskGP(x, y, se)
        self.assertFalse(is_noiseless(model))
        model = FidelityAwareSingleTaskGP(x, y, se)
        self.assertFalse(is_noiseless(model))
        with self.assertRaises(ModelError):
            is_noiseless(MultiOutputGP([]))
