#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest import mock

import numpy as np
import torch
from ax.models.torch.utils import (
    _generate_sobol_points,
    normalize_indices,
    subset_model,
    tensor_callable_to_array_callable,
)
from ax.utils.common.testutils import TestCase
from botorch.models import SingleTaskGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import ModelList
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP
from pyre_extensions import none_throws
from torch import Tensor


class TorchUtilsTest(TestCase):
    def test_NormalizeIndices(self) -> None:
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

    def test_GenerateSobolPoints(self) -> None:
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

    def test_TensorCallableToArrayCallable(self) -> None:
        def tensor_func(x: Tensor) -> Tensor:
            return torch.pow(x, 2)

        new_func = tensor_callable_to_array_callable(
            tensor_func=tensor_func, device=torch.device("cpu")
        )
        x = np.array([5.0, 2.0])
        self.assertTrue(np.array_equal(new_func(x), np.array([25.0, 4.0])))


class SubsetModelTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.x = torch.zeros(1, 1)
        self.y = torch.rand(1, 2)
        self.obj_t = torch.rand(2)
        self.model = SingleTaskGP(self.x, self.y)
        self.obj_weights = torch.tensor([1.0, 0.0])

    def test_can_subset(self) -> None:
        # basic test, can subset
        subset_model_results = subset_model(self.model, self.obj_weights)
        model_sub = subset_model_results.model
        obj_weights_sub = subset_model_results.objective_weights
        ocs_sub = subset_model_results.outcome_constraints
        obj_t_sub = subset_model_results.objective_thresholds
        self.assertIsNone(ocs_sub)
        self.assertIsNone(obj_t_sub)
        self.assertEqual(model_sub.num_outputs, 1)
        self.assertTrue(torch.equal(obj_weights_sub, torch.tensor([1.0])))

    def test_cannot_subset(self) -> None:
        obj_weights = torch.tensor([1.0, 2.0])
        subset_model_results = subset_model(self.model, obj_weights)
        model_sub = subset_model_results.model
        obj_weights_sub = subset_model_results.objective_weights
        ocs_sub = subset_model_results.outcome_constraints
        obj_t_sub = subset_model_results.objective_thresholds
        self.assertIsNone(ocs_sub)
        self.assertIsNone(obj_t_sub)
        self.assertIs(model_sub, self.model)  # check identity
        self.assertIs(obj_weights_sub, obj_weights)  # check identity
        self.assertTrue(torch.equal(subset_model_results.indices, torch.tensor([0, 1])))

    def test_with_outcome_constraints_can_subset(self) -> None:
        ocs = (torch.tensor([[1.0, 0.0]]), torch.tensor([1.0]))
        subset_model_results = subset_model(self.model, self.obj_weights, ocs)
        model_sub = subset_model_results.model
        obj_weights_sub = subset_model_results.objective_weights
        ocs_sub = subset_model_results.outcome_constraints
        obj_t_sub = subset_model_results.objective_thresholds
        self.assertEqual(model_sub.num_outputs, 1)
        self.assertIsNone(obj_t_sub)
        self.assertTrue(torch.equal(obj_weights_sub, torch.tensor([1.0])))
        # pyre-fixme[16]: Optional type has no attribute `__getitem__`.
        self.assertTrue(torch.equal(ocs_sub[0], torch.tensor([[1.0]])))
        self.assertTrue(torch.equal(ocs_sub[1], torch.tensor([1.0])))
        self.assertTrue(torch.equal(subset_model_results.indices, torch.tensor([0])))

    def test_with_outcome_constraints_cannot_subset(self) -> None:
        ocs = (torch.tensor([[0.0, 1.0]]), torch.tensor([1.0]))
        subset_model_results = subset_model(self.model, self.obj_weights, ocs)
        model_sub = subset_model_results.model
        obj_weights_sub = subset_model_results.objective_weights
        ocs_sub = subset_model_results.outcome_constraints
        obj_t_sub = subset_model_results.objective_thresholds
        self.assertIs(model_sub, self.model)  # check identity
        self.assertIsNone(obj_t_sub)
        self.assertIs(obj_weights_sub, self.obj_weights)  # check identity
        self.assertIs(ocs_sub, ocs)  # check identity
        self.assertTrue(torch.equal(subset_model_results.indices, torch.tensor([0, 1])))

    def test_with_obj_thresholds_cannot_subset(self) -> None:
        # test w/ objective thresholds, cannot subset
        ocs = (torch.tensor([[0.0, 1.0]]), torch.tensor([1.0]))
        subset_model_results = subset_model(
            self.model, self.obj_weights, ocs, self.obj_t
        )
        model_sub = subset_model_results.model
        obj_weights_sub = subset_model_results.objective_weights
        ocs_sub = subset_model_results.outcome_constraints
        obj_t_sub = subset_model_results.objective_thresholds
        self.assertIs(model_sub, self.model)  # check identity
        self.assertIs(self.obj_t, obj_t_sub)
        self.assertIs(obj_weights_sub, self.obj_weights)  # check identity
        self.assertTrue(torch.equal(subset_model_results.indices, torch.tensor([0, 1])))
        self.assertIs(ocs_sub, ocs)  # check identity

    def test_with_obj_thresholds_can_subset(self) -> None:
        # test w/ objective thresholds, can subset
        ocs = (torch.tensor([[1.0, 0.0]]), torch.tensor([1.0]))
        subset_model_results = subset_model(
            self.model, self.obj_weights, ocs, self.obj_t
        )
        model_sub = subset_model_results.model
        obj_weights_sub = subset_model_results.objective_weights
        ocs_sub = none_throws(subset_model_results.outcome_constraints)
        obj_t_sub = subset_model_results.objective_thresholds
        self.assertTrue(torch.equal(subset_model_results.indices, torch.tensor([0])))
        self.assertEqual(model_sub.num_outputs, 1)
        self.assertTrue(torch.equal(obj_weights_sub, torch.tensor([1.0])))
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
        self.assertTrue(torch.equal(obj_t_sub, self.obj_t[:1]))
        self.assertTrue(torch.equal(ocs_sub[0], torch.tensor([[1.0]])))
        self.assertTrue(torch.equal(ocs_sub[1], torch.tensor([1.0])))

    def test_unsupported(self) -> None:
        yvar = torch.ones(1, 2)
        model = SingleTaskGP(train_X=self.x, train_Y=self.y, train_Yvar=yvar)
        with mock.patch.object(model, "subset_output", side_effect=NotImplementedError):
            subset_model_results = subset_model(model, self.obj_weights)
        model_sub = subset_model_results.model
        obj_weights_sub = subset_model_results.objective_weights
        ocs_sub = subset_model_results.outcome_constraints
        self.assertIsNone(ocs_sub)
        self.assertIs(model_sub, model)  # check identity
        self.assertIs(obj_weights_sub, self.obj_weights)  # check identity
        self.assertTrue(torch.equal(subset_model_results.indices, torch.tensor([0, 1])))
        # test error on size inconsistency
        obj_weights = torch.ones(4)
        obj_weights[0] = 0
        with self.assertRaises(RuntimeError):
            subset_model(model, obj_weights)


class SubsetModelTestMultiTask(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.x1 = torch.tensor([[1.0, 2.0, 1.0], [2.0, 3.0, 0.0]])
        self.y1 = torch.tensor([[0.0], [1.0]])
        self.x2 = torch.tensor([[0.0, 3.0, 1.0], [1.0, 4.0, 0.0]])
        self.y2 = torch.tensor([[2.0], [3.0]])

    def test_multitask_modellist(self) -> None:
        m1 = MultiTaskGP(self.x1, self.y1, task_feature=2, output_tasks=[0])
        m2 = MultiTaskGP(self.x2, self.y2, task_feature=2, output_tasks=[0])
        model = ModelListGP(m1, m2)
        # test model is not subset when model.num_outputs >
        # len(obj_weights), but all outcomes are relevant.
        # This test is explicitly tests that model is not
        # subset when subset_model is called because all
        # outcomes are relevant.
        obj_weights = torch.ones(2)
        subset_model_results = subset_model(model, obj_weights)
        self.assertIs(subset_model_results.model, model)
        # test subset
        obj_weights = torch.tensor([1.0, 0.0])
        subset_model_results = subset_model(model, obj_weights)
        # check that the model is m1
        self.assertIs(subset_model_results.model, m1)

    def test_model_list(self) -> None:
        # three output model
        m1 = GenericDeterministicModel(lambda x: x, num_outputs=3)
        # two output model
        m2 = SingleTaskGP(
            train_X=self.x1,
            train_Y=torch.cat([self.y1, self.y2], dim=-1),
        )
        model = ModelList(m1, m2)
        obj_weights = torch.zeros(5)
        obj_weights[:3] = 1
        subset_model_results = subset_model(model, obj_weights)
        self.assertIs(subset_model_results.model, m1)
        # set subset where m2 is subset
        m1 = GenericDeterministicModel(lambda x: x, num_outputs=1)
        model = ModelList(m1, m2)
        obj_weights = torch.ones(3)
        obj_weights[1] = 0
        subset_model_results = subset_model(model, obj_weights)
        models = subset_model_results.model.models
        # pyre-fixme[6]: For 1st argument expected
        #  `pyre_extensions.PyreReadOnly[Sized]` but got `Union[Tensor, Module]`.
        self.assertEqual(len(models), 2)
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.assertIs(models[0], m1)
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.assertIsInstance(models[1], SingleTaskGP)
        # check that second model is the second output of m2
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.assertTrue(torch.equal(models[1].train_targets, m2.train_targets[1]))

    def test_nested_model_list_gp(self) -> None:
        m1 = MultiTaskGP(
            train_X=torch.cat([self.x1, self.x2], dim=0),
            train_Y=torch.cat([self.y1, self.y2], dim=0),
            task_feature=2,
            output_tasks=[0],
        )
        m2a = SingleTaskGP(
            train_X=self.x1,
            train_Y=self.y1,
        )
        m2b = SingleTaskGP(
            train_X=self.x2,
            train_Y=self.y2,
        )
        model = ModelListGP(m1, ModelListGP(m2a, m2b))
        obj_weights = torch.zeros(4)
        obj_weights[0] = 1
        obj_weights[2] = 1
        subset_model_results = subset_model(model, obj_weights)
        models = subset_model_results.model.models
        # pyre-fixme[6]: For 1st argument expected
        #  `pyre_extensions.PyreReadOnly[Sized]` but got `Union[Tensor, Module]`.
        self.assertEqual(len(models), 2)
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.assertIs(models[0], m1)
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.assertIs(models[1], m2b)
