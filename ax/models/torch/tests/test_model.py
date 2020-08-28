#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import ANY, patch

import torch
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.kg import KnowledgeGradient
from ax.models.torch.botorch_modular.mes import MaxValueEntropySearch
from ax.models.torch.botorch_modular.model import (
    BoTorchModel,
    construct_acquisition_and_optimizer_options,
)
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.testutils import TestCase
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.containers import TrainingData


CURRENT_PATH = f"{__name__}"
MODEL_PATH = f"{BoTorchModel.__module__}"
SURROGATE_PATH = f"{Surrogate.__module__}"


class BoTorchModelTest(TestCase):
    def setUp(self):
        self.botorch_model_class = SingleTaskGP
        self.surrogate = Surrogate(botorch_model_class=self.botorch_model_class)
        self.acquisition_class = KnowledgeGradient
        self.acquisition_options = {"num_fantasies": 64}
        self.model = BoTorchModel(
            surrogate=self.surrogate,
            acquisition_class=self.acquisition_class,
            acquisition_options=self.acquisition_options,
        )

        self.Xs = [torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])]
        self.Ys = [torch.tensor([[3.0], [4.0]])]
        self.Yvars = [torch.tensor([[0.0], [2.0]])]
        self.training_data = TrainingData(Xs=self.Xs, Ys=self.Ys, Yvars=self.Yvars)
        self.bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
        self.task_features = []
        self.feature_names = ["x1", "x2", "x3"]
        self.metric_names = ["y"]
        self.fidelity_features = [2]
        self.target_fidelities = {1: 1.0}
        self.candidate_metadata = []

        self.optimizer_options = {"num_restarts": 40, "raw_samples": 1024}
        self.model_gen_options = {"optimizer_kwargs": self.optimizer_options}
        self.objective_weights = torch.tensor([1.0])
        self.outcome_constraints = None
        self.linear_constraints = None
        self.fixed_features = None
        self.pending_observations = None
        self.rounding_func = "func"

    def test_init(self):
        # Default model with no specifications.
        model = BoTorchModel()
        self.assertEqual(model.acquisition_class, Acquisition)
        with self.assertRaisesRegex(
            ValueError, "`AcquisitionFunction` has not yet been set."
        ):
            model.botorch_acqf_class
        # Model that specifies `botorch_acqf_class`.
        model = BoTorchModel(botorch_acqf_class=qKnowledgeGradient)
        self.assertEqual(model.acquisition_class, Acquisition)
        self.assertEqual(model.botorch_acqf_class, qKnowledgeGradient)
        # Model with `Acquisition` that specifies a `default_botorch_acqf_class`.
        model = BoTorchModel(acquisition_class=MaxValueEntropySearch)
        self.assertEqual(model.acquisition_class, MaxValueEntropySearch)
        self.assertEqual(model.botorch_acqf_class, qMaxValueEntropy)

    @patch(f"{SURROGATE_PATH}.Surrogate.fit")
    def test_fit(self, mock_fit):
        self.model.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            bounds=self.bounds,
            task_features=self.task_features,
            feature_names=self.feature_names,
            metric_names=self.metric_names,
            fidelity_features=self.fidelity_features,
            target_fidelities=self.target_fidelities,
            candidate_metadata=self.candidate_metadata,
        )
        mock_fit.assert_called_with(
            training_data=self.training_data,
            bounds=self.bounds,
            task_features=self.task_features,
            feature_names=self.feature_names,
            fidelity_features=self.fidelity_features,
            target_fidelities=self.target_fidelities,
            metric_names=self.metric_names,
            candidate_metadata=self.candidate_metadata,
            state_dict=None,
            refit=True,
        )

    @patch(f"{SURROGATE_PATH}.Surrogate.predict")
    def test_predict(self, mock_predict):
        self.model.predict(X=self.Xs[0])
        mock_predict.assert_called_with(X=self.Xs[0])

    @patch(
        f"{MODEL_PATH}.construct_acquisition_and_optimizer_options",
        return_value=({"num_fantasies": 64}, {"num_restarts": 40, "raw_samples": 1024}),
    )
    @patch(f"{CURRENT_PATH}.KnowledgeGradient")
    @patch(f"{MODEL_PATH}.get_rounding_func", return_value="func")
    @patch(f"{MODEL_PATH}._to_inequality_constraints", return_value=[])
    def test_gen(
        self,
        mock_inequality_constraints,
        mock_rounding,
        mock_kg,
        mock_construct_options,
    ):
        mock_kg.return_value.optimize.return_value = (
            torch.tensor([1.0]),
            torch.tensor([2.0]),
        )
        model = BoTorchModel(
            surrogate=self.surrogate,
            acquisition_class=KnowledgeGradient,
            acquisition_options=self.acquisition_options,
        )
        model.surrogate.construct(
            training_data=self.training_data, fidelity_features=self.fidelity_features
        )
        model.gen(
            n=1,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            pending_observations=self.pending_observations,
            model_gen_options=self.model_gen_options,
            rounding_func=self.rounding_func,
            target_fidelities=self.target_fidelities,
        )
        # Assert `construct_acquisition_and_optimizer_options` called with kwargs
        mock_construct_options.assert_called_with(
            acqf_options=self.acquisition_options,
            model_gen_options=self.model_gen_options,
        )
        # Assert `acquisition_class` called with kwargs
        mock_kg.assert_called_with(
            surrogate=self.surrogate,
            botorch_acqf_class=model.botorch_acqf_class,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            target_fidelities=self.target_fidelities,
            options=self.acquisition_options,
        )
        # Assert `optimize` called with kwargs
        mock_kg.return_value.optimize.assert_called_with(
            bounds=ANY,
            n=1,
            inequality_constraints=[],
            fixed_features=self.fixed_features,
            rounding_func="func",
            optimizer_options=self.optimizer_options,
        )

    def test_construct_acquisition_and_optimizer_options(self):
        # two dicts for `Acquisition` should be concatenated
        acqf_options = {"num_fantasies": 64}

        acquisition_function_kwargs = {"current_value": torch.tensor([1.0])}
        optimizer_kwargs = {"num_restarts": 40, "raw_samples": 1024}
        model_gen_options = {
            "acquisition_function_kwargs": acquisition_function_kwargs,
            "optimizer_kwargs": optimizer_kwargs,
        }

        (
            final_acq_options,
            final_opt_options,
        ) = construct_acquisition_and_optimizer_options(
            acqf_options=acqf_options, model_gen_options=model_gen_options
        )
        self.assertEqual(
            final_acq_options,
            {"num_fantasies": 64, "current_value": torch.tensor([1.0])},
        )
        self.assertEqual(final_opt_options, optimizer_kwargs)
