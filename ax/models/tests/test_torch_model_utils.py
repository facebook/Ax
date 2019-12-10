#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from ax.exceptions.model import ModelError
from ax.models.torch.utils import is_noiseless, normalize_indices, subset_model
from ax.utils.common.testutils import TestCase
from botorch.models import HeteroskedasticSingleTaskGP, ModelListGP, SingleTaskGP


class TorchUtilsTest(TestCase):
    def test_is_noiseless(self):
        x = torch.zeros(1, 1)
        y = torch.zeros(1, 1)
        se = torch.zeros(1, 1)
        model = SingleTaskGP(x, y)
        self.assertTrue(is_noiseless(model))
        model = HeteroskedasticSingleTaskGP(x, y, se)
        self.assertFalse(is_noiseless(model))
        with self.assertRaises(ModelError):
            is_noiseless(ModelListGP())

    def testNormalizeIndices(self):
        indices = [0, 2]
        nlzd_indices = normalize_indices(indices, 3)
        self.assertEqual(nlzd_indices, indices)
        nlzd_indices = normalize_indices(indices, 4)
        self.assertEqual(nlzd_indices, indices)
        indices = [0, -1]
        nlzd_indices = normalize_indices(indices, 3)
        self.assertEqual(nlzd_indices, [0, 2])
        with self.assertRaises(ValueError):
            nlzd_indices = normalize_indices([3], 3)
        with self.assertRaises(ValueError):
            nlzd_indices = normalize_indices([-4], 3)

    def testSubsetModel(self):
        x = torch.zeros(1, 1)
        y = torch.zeros(1, 2)
        model = SingleTaskGP(x, y)
        self.assertEqual(model.num_outputs, 2)
        # basic test, can subset
        obj_weights = torch.tensor([1.0, 0.0])
        model_sub, obj_weights_sub, ocs_sub = subset_model(model, obj_weights)
        self.assertIsNone(ocs_sub)
        self.assertEqual(model_sub.num_outputs, 1)
        self.assertTrue(torch.equal(obj_weights_sub, torch.tensor([1.0])))
        # basic test, cannot subset
        obj_weights = torch.tensor([1.0, 2.0])
        model_sub, obj_weights_sub, ocs_sub = subset_model(model, obj_weights)
        self.assertIsNone(ocs_sub)
        self.assertIs(model_sub, model)  # check identity
        self.assertIs(obj_weights_sub, obj_weights)  # check identity
        # test w/ outcome constraints, can subset
        obj_weights = torch.tensor([1.0, 0.0])
        ocs = (torch.tensor([[1.0, 0.0]]), torch.tensor([1.0]))
        model_sub, obj_weights_sub, ocs_sub = subset_model(model, obj_weights, ocs)
        self.assertEqual(model_sub.num_outputs, 1)
        self.assertTrue(torch.equal(obj_weights_sub, torch.tensor([1.0])))
        self.assertTrue(torch.equal(ocs_sub[0], torch.tensor([[1.0]])))
        self.assertTrue(torch.equal(ocs_sub[1], torch.tensor([1.0])))
        # test w/ outcome constraints, cannot subset
        obj_weights = torch.tensor([1.0, 0.0])
        ocs = (torch.tensor([[0.0, 1.0]]), torch.tensor([1.0]))
        model_sub, obj_weights_sub, ocs_sub = subset_model(model, obj_weights, ocs)
        self.assertIs(model_sub, model)  # check identity
        self.assertIs(obj_weights_sub, obj_weights)  # check identity
        self.assertIs(ocs_sub, ocs)  # check identity
        # test unsupported
        yvar = torch.ones(1, 2)
        model = HeteroskedasticSingleTaskGP(x, y, yvar)
        model_sub, obj_weights_sub, ocs = subset_model(model, obj_weights)
        self.assertIsNone(ocs)
        self.assertIs(model_sub, model)  # check identity
        self.assertIs(obj_weights_sub, obj_weights)  # check identity
        # test error on size inconsistency
        obj_weights = torch.ones(3)
        with self.assertRaises(RuntimeError):
            subset_model(model, obj_weights)
