#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from unittest.mock import Mock, patch

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.multi_fidelity import MultiFidelityAcquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import fast_botorch_optimize
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.datasets import SupervisedDataset


ACQUISITION_PATH: str = Acquisition.__module__
MULTI_FIDELITY_PATH: str = MultiFidelityAcquisition.__module__
MFKG_PATH = (
    f"{qMultiFidelityKnowledgeGradient.__module__}.qMultiFidelityKnowledgeGradient"
)


class MultiFidelityAcquisitionTest(TestCase):
    @fast_botorch_optimize
    def setUp(self) -> None:
        self.botorch_model_class = SingleTaskGP
        self.surrogate = Surrogate(botorch_model_class=self.botorch_model_class)
        self.X = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        self.Y = torch.tensor([[3.0], [4.0]])
        self.Yvar = torch.tensor([[0.0], [2.0]])
        self.feature_names = ["a", "b", "c"]
        self.metric_names = ["metric"]
        self.training_data = [
            SupervisedDataset(
                X=self.X,
                Y=self.Y,
                feature_names=self.feature_names,
                outcome_names=self.metric_names,
            )
        ]
        self.fidelity_features = [2]
        self.search_space_digest = SearchSpaceDigest(
            feature_names=self.feature_names,
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            target_values={2: 1.0},
            fidelity_features=self.fidelity_features,
        )
        self.surrogate.fit(
            datasets=self.training_data,
            metric_names=self.metric_names,
            search_space_digest=self.search_space_digest,
        )
        self.acquisition_options = {Keys.NUM_FANTASIES: 64}

        self.objective_weights = torch.tensor([1.0])
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
        self.torch_opt_config = TorchOptConfig(
            objective_weights=self.objective_weights,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
        )

    @patch(
        f"{ACQUISITION_PATH}.Acquisition.compute_model_dependencies", return_value={}
    )
    @patch(f"{MULTI_FIDELITY_PATH}.AffineFidelityCostModel", return_value="cost_model")
    @patch(f"{MULTI_FIDELITY_PATH}.InverseCostWeightedUtility", return_value=None)
    @patch(f"{MULTI_FIDELITY_PATH}.project_to_target_fidelity", return_value=None)
    @patch(f"{MULTI_FIDELITY_PATH}.expand_trace_observations", return_value=None)
    def test_compute_model_dependencies(
        self,
        mock_expand: Mock,
        mock_project: Mock,
        mock_inverse_utility: Mock,
        mock_affine_model: Mock,
        mock_Acquisition_compute: Mock,
    ) -> None:
        # TODO: Patch only `MFKG_PATH.__init__` once `construct_inputs`
        # implemented for qMFKG.
        with patch(
            f"{MULTI_FIDELITY_PATH}.MultiFidelityAcquisition.__init__",
            return_value=None,
        ):
            # We don't actually need to instantiate the BoTorch acqf in these tests.
            mf_acquisition = MultiFidelityAcquisition(
                surrogates={"regression": self.surrogate},
                search_space_digest=self.search_space_digest,
                torch_opt_config=TorchOptConfig(
                    objective_weights=self.objective_weights
                ),
                botorch_acqf_class=qMultiFidelityKnowledgeGradient,
            )
        # Raise Error if `fidelity_weights` and `target_fidelities` do not align.
        with self.assertRaisesRegex(RuntimeError, "Must provide the same indices"):
            mf_acquisition.compute_model_dependencies(
                surrogates={"regression": self.surrogate},
                search_space_digest=dataclasses.replace(
                    self.search_space_digest,
                    fidelity_features=[1],
                    target_values={1: 5.0},
                ),
                torch_opt_config=self.torch_opt_config,
                options=self.options,
            )
        # Make sure `fidelity_weights` are set when they are not passed in.
        mf_acquisition.compute_model_dependencies(
            surrogates={"regression": self.surrogate},
            search_space_digest=dataclasses.replace(
                self.search_space_digest,
                fidelity_features=[2, 3],
                target_values={2: 5.0, 3: 5.0},
            ),
            torch_opt_config=self.torch_opt_config,
            options={Keys.COST_INTERCEPT: 1.0, Keys.NUM_TRACE_OBSERVATIONS: 0},
        )
        mock_affine_model.assert_called_with(
            fidelity_weights={2: 1.0, 3: 1.0}, fixed_cost=1.0
        )
        # Usual case.
        dependencies = mf_acquisition.compute_model_dependencies(
            surrogates={"regression": self.surrogate},
            search_space_digest=self.search_space_digest,
            torch_opt_config=self.torch_opt_config,
            options=self.options,
        )
        mock_Acquisition_compute.assert_called_with(
            surrogates={"regression": self.surrogate},
            search_space_digest=self.search_space_digest,
            torch_opt_config=self.torch_opt_config,
            options=self.options,
        )
        mock_affine_model.assert_called_with(
            fidelity_weights=self.options[Keys.FIDELITY_WEIGHTS],
            fixed_cost=self.options[Keys.COST_INTERCEPT],
        )
        mock_inverse_utility.assert_called_with(cost_model="cost_model")
        self.assertTrue(Keys.COST_AWARE_UTILITY in dependencies)
        self.assertTrue(Keys.PROJECT in dependencies)
        self.assertTrue(Keys.EXPAND in dependencies)
        # Check that `project` and `expand` are defined correctly.
        project = dependencies.get(Keys.PROJECT)
        project(torch.tensor([1.0]))
        mock_project.assert_called_with(
            X=torch.tensor([1.0]),
            target_fidelities=self.search_space_digest.target_values,
        )
        expand = dependencies.get(Keys.EXPAND)
        expand(torch.tensor([1.0]))
        mock_expand.assert_called_with(
            X=torch.tensor([1.0]),
            fidelity_dims=sorted(self.search_space_digest.target_values),
            num_trace_obs=self.options.get(Keys.NUM_TRACE_OBSERVATIONS),
        )
