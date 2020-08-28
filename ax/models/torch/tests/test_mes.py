#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import torch
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.mes import (
    MaxValueEntropySearch,
    MultiFidelityMaxValueEntropySearch,
)
from ax.models.torch.botorch_modular.multi_fidelity import MultiFidelityAcquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import TrainingData


ACQUISITION_PATH = f"{Acquisition.__module__}"
MES_PATH = f"{MaxValueEntropySearch.__module__}"
MULTI_FIDELITY_PATH = f"{MultiFidelityAcquisition.__module__}"


class AcquisitionSetUp:
    def setUp(self):
        self.botorch_model_class = SingleTaskGP
        self.surrogate = Surrogate(botorch_model_class=self.botorch_model_class)
        self.Xs = [torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])]
        self.Ys = [torch.tensor([[3.0], [4.0]])]
        self.Yvars = [torch.tensor([[0.0], [2.0]])]
        self.training_data = TrainingData(Xs=self.Xs, Ys=self.Ys, Yvars=self.Yvars)
        self.fidelity_features = [2]
        self.surrogate.construct(
            training_data=self.training_data, fidelity_features=self.fidelity_features
        )

        self.bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
        self.botorch_acqf_class = qMaxValueEntropy
        self.objective_weights = torch.tensor([1.0])
        self.target_fidelities = {2: 1.0}
        self.pending_observations = [
            torch.tensor([[1.0, 3.0, 4.0]]),
            torch.tensor([[2.0, 6.0, 8.0]]),
        ]
        self.outcome_constraints = (torch.tensor([[1.0]]), torch.tensor([[0.5]]))
        self.linear_constraints = None
        self.fixed_features = {1: 2.0}
        self.options = {
            Keys.FIDELITY_WEIGHTS: {2: 1.0},
            Keys.COST_INTERCEPT: 1.0,
            Keys.NUM_TRACE_OBSERVATIONS: 0,
        }

        self.optimizer_options = {
            Keys.NUM_RESTARTS: 40,
            Keys.RAW_SAMPLES: 1024,
            Keys.FRAC_RANDOM: 0.2,
        }
        self.inequality_constraints = [
            (torch.tensor([0, 1]), torch.tensor([-1.0, 1.0]), 1)
        ]


class MaxValueEntropySearchTest(AcquisitionSetUp, TestCase):
    def setUp(self):
        super().setUp()

    @patch(
        f"{ACQUISITION_PATH}.Acquisition.compute_model_dependencies", return_value={}
    )
    def test_compute_model_dependencies(self, mock_Acquisition_compute):
        # `MaxValueEntropySearch.compute_model_dependencies` should call
        # `Acquisition.compute_model_dependencies` once.
        depedencies = MaxValueEntropySearch.compute_model_dependencies(
            surrogate=self.surrogate,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
        )
        mock_Acquisition_compute.assert_called_once()
        self.assertTrue(Keys.CANDIDATE_SET in depedencies)
        self.assertTrue(Keys.MAXIMIZE in depedencies)

    @patch(f"{ACQUISITION_PATH}.Acquisition.optimize")
    def test_optimize(self, mock_Acquisition_optimize):
        self.acquisition = MaxValueEntropySearch(
            surrogate=self.surrogate,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            botorch_acqf_class=self.botorch_acqf_class,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            target_fidelities=self.target_fidelities,
            options=self.options,
        )
        # `MaxValueEntropySearch.optimize()` should call
        # `Acquisition.optimize()` once.
        self.acquisition.optimize(
            bounds=self.bounds,
            n=1,
            optimizer_class=None,
            inequality_constraints=self.inequality_constraints,
            fixed_features=self.fixed_features,
            rounding_func="func",
            optimizer_options=self.optimizer_options,
        )
        mock_Acquisition_optimize.assert_called_once()
        # `sequential` should be set to True
        self.optimizer_options.update({Keys.SEQUENTIAL: True})
        mock_Acquisition_optimize.assert_called_with(
            bounds=self.bounds,
            n=1,
            inequality_constraints=None,
            fixed_features=self.fixed_features,
            rounding_func="func",
            optimizer_options=self.optimizer_options,
        )


class MultiFidelityMaxValueEntropySearchTest(AcquisitionSetUp, TestCase):
    def setUp(self):
        super().setUp()

    @patch(
        f"{MES_PATH}.MaxValueEntropySearch.compute_model_dependencies",
        return_value={Keys.CANDIDATE_SET: None, Keys.MAXIMIZE: True},
    )
    @patch(
        f"{MULTI_FIDELITY_PATH}.MultiFidelityAcquisition.compute_model_dependencies",
        return_value={Keys.CURRENT_VALUE: 0.0},
    )
    def test_compute_model_dependencies(self, mock_MF_compute, mock_MES_compute):
        # `MultiFidelityMaxValueEntropySearch.compute_model_dependencies` should
        # call `MaxValueEntropySearch.compute_model_dependencies` once and
        # call `MultiFidelityAcquisition.compute_model_dependencies` once.
        depedencies = MultiFidelityMaxValueEntropySearch.compute_model_dependencies(
            surrogate=self.surrogate,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            target_fidelities=self.target_fidelities,
        )
        mock_MES_compute.assert_called_once()
        mock_MF_compute.assert_called_once()
        # `dependencies` should be combination of `MaxValueEntropySearch` dependencies
        # and `MultiFidelityAcquisition` dependencies.
        self.assertTrue(Keys.CANDIDATE_SET in depedencies)
        self.assertTrue(Keys.MAXIMIZE in depedencies)
        self.assertTrue(Keys.CURRENT_VALUE in depedencies)

    @patch(f"{MES_PATH}.MaxValueEntropySearch.optimize")
    def test_optimize(self, mock_MES_optimize):
        self.acquisition = MultiFidelityMaxValueEntropySearch(
            surrogate=self.surrogate,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            botorch_acqf_class=self.botorch_acqf_class,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            target_fidelities=self.target_fidelities,
            options=self.options,
        )
        # `MultiFidelityMaxValueEntropySearch.optimize()` should call
        # `MaxValueEntropySearch.optimize()` once.
        self.acquisition.optimize(
            bounds=self.bounds,
            n=1,
            optimizer_class=None,
            inequality_constraints=self.inequality_constraints,
            fixed_features=self.fixed_features,
            rounding_func="func",
            optimizer_options=self.optimizer_options,
        )
        mock_MES_optimize.assert_called_once()
        mock_MES_optimize.assert_called_with(
            bounds=self.bounds,
            n=1,
            optimizer_class=None,
            inequality_constraints=self.inequality_constraints,
            fixed_features=self.fixed_features,
            rounding_func="func",
            optimizer_options=self.optimizer_options,
        )
