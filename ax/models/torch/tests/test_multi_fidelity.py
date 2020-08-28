#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import torch
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.multi_fidelity import MultiFidelityAcquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.testutils import TestCase
from botorch.models.gp_regression import SingleTaskGP


ACQUISITION_PATH = f"{Acquisition.__module__}"
MULTI_FIDELITY_PATH = f"{MultiFidelityAcquisition.__module__}"


class MultiFidelityAcquisitionTest(TestCase):
    def setUp(self):

        self.botorch_model_class = SingleTaskGP
        self.surrogate = Surrogate(botorch_model_class=self.botorch_model_class)

        self.acquisition_options = {"num_fantasies": 64}
        self.bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
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
            "fidelity_weights": {2: 1.0},
            "cost_intercept": 1.0,
            "num_trace_observations": 0,
        }

    @patch(f"{ACQUISITION_PATH}.Acquisition.__init__", return_value=None)
    @patch(f"{ACQUISITION_PATH}.Acquisition.optimize")
    def test_optimize(self, mock_Acquisition_optimize, mock_Acquisition_init):
        # `MultiFidelityAcquisition.optimize()` should call `Acquisition.optimize()`
        # once.
        self.acquisition = MultiFidelityAcquisition(
            surrogate=self.surrogate,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
        )
        self.acquisition.optimize(bounds=self.bounds, n=1)
        mock_Acquisition_optimize.assert_called_once()

    @patch(f"{MULTI_FIDELITY_PATH}.AffineFidelityCostModel", return_value="cost_model")
    @patch(f"{MULTI_FIDELITY_PATH}.InverseCostWeightedUtility", return_value=None)
    def test_compute_model_dependencies(self, mock_inverse_utility, mock_affine_model):
        # Raise Error if `fidelity_weights` and `target_fidelities` do
        # not align.
        with self.assertRaisesRegex(RuntimeError, "Must provide the same indices"):
            MultiFidelityAcquisition.compute_model_dependencies(
                surrogate=self.surrogate,
                bounds=self.bounds,
                objective_weights=self.objective_weights,
                target_fidelities={1: 5.0},
                pending_observations=self.pending_observations,
                outcome_constraints=self.outcome_constraints,
                linear_constraints=self.linear_constraints,
                fixed_features=self.fixed_features,
                options=self.options,
            )

        dependencies = MultiFidelityAcquisition.compute_model_dependencies(
            surrogate=self.surrogate,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            target_fidelities=self.target_fidelities,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            options=self.options,
        )
        mock_affine_model.assert_called_with(
            fidelity_weights=self.options["fidelity_weights"],
            fixed_cost=self.options["cost_intercept"],
        )
        mock_inverse_utility.assert_called_with(cost_model="cost_model")
        self.assertTrue("cost_aware_utility" in dependencies)
        self.assertTrue("project" in dependencies)
        self.assertTrue("expand" in dependencies)
