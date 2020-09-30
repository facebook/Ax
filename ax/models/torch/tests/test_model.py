#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import ANY, patch

import torch
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.kg import KnowledgeGradient
from ax.models.torch.botorch_modular.list_surrogate import ListSurrogate
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.utils import choose_model_class
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.torch_stubs import get_torch_test_data
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.containers import TrainingData


CURRENT_PATH = __name__
MODEL_PATH = BoTorchModel.__module__
SURROGATE_PATH = Surrogate.__module__
UTILS_PATH = choose_model_class.__module__
ACQUISITION_PATH = Acquisition.__module__


class BoTorchModelTest(TestCase):
    def setUp(self):
        self.botorch_model_class = SingleTaskGP
        self.surrogate = Surrogate(botorch_model_class=self.botorch_model_class)
        self.acquisition_class = KnowledgeGradient
        self.botorch_acqf_class = qKnowledgeGradient
        self.acquisition_options = {Keys.NUM_FANTASIES: 64}
        self.surrogate_fit_options = {
            Keys.REFIT_ON_UPDATE: True,
            Keys.STATE_DICT: {"non-empty": "non-empty"},
            Keys.WARM_START_REFITTING: False,
        }
        self.model = BoTorchModel(
            surrogate=self.surrogate,
            acquisition_class=self.acquisition_class,
            acquisition_options=self.acquisition_options,
            surrogate_fit_options=self.surrogate_fit_options,
        )

        self.dtype = torch.float
        Xs1, Ys1, Yvars1, self.bounds, _, _, _ = get_torch_test_data(dtype=self.dtype)
        Xs2, Ys2, Yvars2, _, _, _, _ = get_torch_test_data(dtype=self.dtype, offset=1.0)
        self.Xs = Xs1 + Xs2
        self.Ys = Ys1 + Ys2
        self.Yvars = Yvars1 + Yvars2
        self.X = Xs1[0]
        self.Y = Ys1[0]
        self.Yvar = Yvars1[0]
        self.training_data = TrainingData(X=self.X, Y=self.Y, Yvar=self.Yvar)
        self.bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
        self.task_features = []
        self.feature_names = ["x1", "x2", "x3"]
        self.metric_names = ["y"]
        self.metric_names_for_list_surrogate = ["y1", "y2"]
        self.fidelity_features = [2]
        self.target_fidelities = {1: 1.0}
        self.candidate_metadata = []

        self.optimizer_options = {Keys.NUM_RESTARTS: 40, Keys.RAW_SAMPLES: 1024}
        self.model_gen_options = {Keys.OPTIMIZER_KWARGS: self.optimizer_options}
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
        # Model that specifies `botorch_acqf_class`.
        model = BoTorchModel(botorch_acqf_class=qExpectedImprovement)
        self.assertEqual(model.acquisition_class, Acquisition)
        self.assertEqual(model.botorch_acqf_class, qExpectedImprovement)
        # Model with `Acquisition` that specifies a `default_botorch_acqf_class`.
        model = BoTorchModel(acquisition_class=KnowledgeGradient)
        self.assertEqual(model.acquisition_class, KnowledgeGradient)
        self.assertEqual(model.botorch_acqf_class, qKnowledgeGradient)

    def test_surrogate_property(self):
        self.assertEqual(self.surrogate, self.model.surrogate)
        self.model._surrogate = None
        with self.assertRaisesRegex(ValueError, "Surrogate has not yet been set."):
            self.model.surrogate

    def test_botorch_acqf_class_property(self):
        self.assertEqual(self.botorch_acqf_class, self.model.botorch_acqf_class)
        self.model._botorch_acqf_class = None
        with self.assertRaisesRegex(
            ValueError, "`AcquisitionFunction` has not yet been set."
        ):
            self.model.botorch_acqf_class

    @patch(f"{SURROGATE_PATH}.Surrogate.fit")
    @patch(f"{MODEL_PATH}.choose_model_class", return_value=SingleTaskGP)
    def test_fit(self, mock_choose_model_class, mock_fit):
        # If surrogate is not yet set, initialize it with dispatcher functions.
        self.model._surrogate = None
        self.model.fit(
            Xs=[self.X],
            Ys=[self.Y],
            Yvars=[self.Yvar],
            bounds=self.bounds,
            task_features=self.task_features,
            feature_names=self.feature_names,
            metric_names=self.metric_names,
            fidelity_features=self.fidelity_features,
            target_fidelities=self.target_fidelities,
            candidate_metadata=self.candidate_metadata,
        )
        # `choose_model_class` is called.
        mock_choose_model_class.assert_called_with(
            Yvars=[self.Yvar],
            task_features=self.task_features,
            fidelity_features=self.fidelity_features,
        )
        # Since we want to refit on updates but not warm start refit, we clear the
        # state dict.
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
        self.model.predict(X=self.X)
        mock_predict.assert_called_with(X=self.X)

    @patch(
        f"{MODEL_PATH}.construct_acquisition_and_optimizer_options",
        return_value=({"num_fantasies": 64}, {"num_restarts": 40, "raw_samples": 1024}),
    )
    @patch(f"{CURRENT_PATH}.KnowledgeGradient")
    @patch(f"{MODEL_PATH}.get_rounding_func", return_value="func")
    @patch(f"{MODEL_PATH}._to_inequality_constraints", return_value=[])
    @patch(f"{MODEL_PATH}.choose_botorch_acqf_class", return_value=qKnowledgeGradient)
    def test_gen(
        self,
        mock_choose_botorch_acqf_class,
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
        model._botorch_acqf_class = None
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
        # Assert `choose_botorch_acqf_class` is called
        mock_choose_botorch_acqf_class.assert_called_once()
        self.assertEqual(model._botorch_acqf_class, qKnowledgeGradient)
        # Assert `acquisition_class` called with kwargs
        mock_kg.assert_called_with(
            surrogate=self.surrogate,
            botorch_acqf_class=model.botorch_acqf_class,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            pending_observations=self.pending_observations,
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

    def test_best_point(self):
        with self.assertRaises(NotImplementedError):
            self.model.best_point(
                bounds=self.bounds, objective_weights=self.objective_weights
            )

    @patch(
        f"{MODEL_PATH}.construct_acquisition_and_optimizer_options",
        return_value=({"num_fantasies": 64}, {"num_restarts": 40, "raw_samples": 1024}),
    )
    @patch(f"{CURRENT_PATH}.KnowledgeGradient", autospec=True)
    def test_evaluate_acquisition_function(self, _mock_kg, _mock_construct_options):
        model = BoTorchModel(
            surrogate=self.surrogate,
            acquisition_class=KnowledgeGradient,
            acquisition_options=self.acquisition_options,
        )
        model.surrogate.construct(
            training_data=self.training_data, fidelity_features=self.fidelity_features
        )
        model.evaluate_acquisition_function(
            X=self.X,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
            pending_observations=self.pending_observations,
            acq_options=self.acquisition_options,
            target_fidelities=self.target_fidelities,
        )
        # `_mock_kg` is a mock of class, so to check the mock `evaluate` on
        # instance of that class, we use `_mock_kg.return_value.evaluate`
        _mock_kg.return_value.evaluate.assert_called()

    @patch(
        f"{ACQUISITION_PATH}.Acquisition._extract_training_data",
        # Mock to register calls, but still execute the function.
        side_effect=Acquisition._extract_training_data,
    )
    @patch(
        f"{ACQUISITION_PATH}.Acquisition.optimize",
        # Dummy candidates and acquisition function value.
        return_value=(torch.tensor([[2.0]]), torch.tensor([1.0])),
    )
    def test_list_surrogate_choice(self, _, mock_extract_training_data):
        model = BoTorchModel()
        model.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            bounds=self.bounds,
            task_features=self.task_features,
            feature_names=self.feature_names,
            metric_names=self.metric_names_for_list_surrogate,
            fidelity_features=self.fidelity_features,
            target_fidelities=self.target_fidelities,
            candidate_metadata=self.candidate_metadata,
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
        mock_extract_training_data.assert_called_once()
        self.assertIsInstance(
            mock_extract_training_data.call_args[1]["surrogate"], ListSurrogate
        )
