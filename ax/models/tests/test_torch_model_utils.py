#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from ax.exceptions.model import ModelError
from ax.models.torch.utils import (
    _generate_sobol_points,
    is_noiseless,
    normalize_indices,
    sample_hypersphere_positive_quadrant,
    sample_simplex,
    subset_model,
    tensor_callable_to_array_callable,
)
from ax.utils.common.testutils import TestCase
from botorch.models import HeteroskedasticSingleTaskGP, ModelListGP, SingleTaskGP
from torch import Tensor


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

    def testGenerateSobolPoints(self):
        bounds = [(0.0, 1.0) for _ in range(3)]
        linear_constraints = (
            torch.tensor([[1, -1, 0]], dtype=torch.double),
            torch.tensor([[0]], dtype=torch.double),
        )

        def test_rounding_func(x: Tensor) -> Tensor:
            return x

        gen_sobol = _generate_sobol_points(
            n_sobol=100,
            bounds=bounds,
            device=torch.device("cpu"),
            linear_constraints=linear_constraints,
            rounding_func=test_rounding_func,
        )
        self.assertEqual(len(gen_sobol), 100)
        self.assertIsInstance(gen_sobol, Tensor)

    def testSampleSimplex(self):
        for d in range(1, 10):
            self.assertTrue(
                sample_simplex(d)
                .sum()
                .isclose(torch.tensor([1.0], dtype=torch.double)),
                "sampled simplex point's components do not sum to 1.0",
            )

    def testSampleHyperspherePositiveQuadrant(self):
        for d in range(1, 10):
            self.assertTrue(
                sample_hypersphere_positive_quadrant(d)
                .norm()
                .isclose(torch.tensor([1.0], dtype=torch.double)),
                "sampled hypersphere point's norm is not 1.0",
            )

    def testTensorCallableToArrayCallable(self):
        def tensor_func(x: Tensor) -> Tensor:
            return np.exp(x)

        new_func = tensor_callable_to_array_callable(
            tensor_func=tensor_func, device=torch.device("cpu")
        )
        self.assertTrue(callable(new_func))
        self.assertIsInstance(new_func(np.array([1.0, 2.0])), np.ndarray)
