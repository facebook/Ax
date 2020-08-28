#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import torch
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.testutils import TestCase
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model, TrainingData
from botorch.sampling.samplers import SobolQMCNormalSampler
from gpytorch.kernels import Kernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


ACQUISITION_PATH = f"{Acquisition.__module__}"
CURRENT_PATH = f"{__name__}"
SURROGATE_PATH = f"{Surrogate.__module__}"


class SurrogateTest(TestCase):
    def setUp(self):
        self.botorch_model_class = SingleTaskGP
        self.mll_class = ExactMarginalLogLikelihood
        self.device = torch.device("cpu")
        self.dtype = torch.float
        self.Xs = [
            torch.tensor(
                [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=self.dtype, device=self.device
            )
        ]
        self.Ys = [torch.tensor([[3.0], [4.0]], dtype=self.dtype, device=self.device)]
        self.Yvars = [
            torch.tensor([[0.0], [2.0]], dtype=self.dtype, device=self.device)
        ]
        self.training_data = TrainingData(Xs=self.Xs, Ys=self.Ys, Yvars=self.Yvars)
        self.surrogate_kwargs = self.botorch_model_class.construct_inputs(
            self.training_data
        )
        self.surrogate = Surrogate(
            botorch_model_class=self.botorch_model_class, mll_class=self.mll_class
        )
        self.bounds = [(0.0, 1.0), (1.0, 4.0), (2.0, 5.0)]
        self.task_features = []
        self.feature_names = ["x1", "x2", "x3"]
        self.metric_names = ["y"]
        self.fidelity_features = []
        self.target_fidelities = {1: 1.0}
        self.fixed_features = {1: 2.0}
        self.refit = True
        self.objective_weights = torch.tensor(
            [-1.0, 1.0], dtype=self.dtype, device=self.device
        )
        self.outcome_constraints = (torch.tensor([[1.0]]), torch.tensor([[0.5]]))
        self.linear_constraints = (
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[0.5], [1.0]]),
        )
        self.options = {}

    @patch(f"{CURRENT_PATH}.Kernel")
    @patch(f"{CURRENT_PATH}.Likelihood")
    def test_init(self, mock_Likelihood, mock_Kernel):
        self.assertEqual(self.surrogate.botorch_model_class, self.botorch_model_class)
        self.assertEqual(self.surrogate.mll_class, self.mll_class)
        with self.assertRaisesRegex(NotImplementedError, "Customizing likelihood"):
            Surrogate(
                botorch_model_class=self.botorch_model_class, likelihood=Likelihood()
            )
        with self.assertRaisesRegex(NotImplementedError, "Customizing kernel"):
            Surrogate(
                botorch_model_class=self.botorch_model_class, kernel_class=Kernel()
            )

    def test_model_property(self):
        with self.assertRaisesRegex(
            ValueError, "BoTorch `Model` has not yet been constructed."
        ):
            self.surrogate.model

    def test_training_data_property(self):
        with self.assertRaisesRegex(
            ValueError,
            "Underlying BoTorch `Model` has not yet received its training_data.",
        ):
            self.surrogate.training_data

    def test_from_BoTorch(self):
        surrogate = Surrogate.from_BoTorch(
            self.botorch_model_class(**self.surrogate_kwargs)
        )
        self.assertIsInstance(surrogate.model, self.botorch_model_class)
        self.assertFalse(surrogate._should_reconstruct)

    @patch(f"{CURRENT_PATH}.SingleTaskGP.__init__", return_value=None)
    def test_construct(self, mock_GP):
        base_surrogate = Surrogate(botorch_model_class=Model)
        with self.assertRaisesRegex(TypeError, "Cannot construct an abstract model."):
            base_surrogate.construct(
                training_data=self.training_data,
                fidelity_features=self.fidelity_features,
            )
        self.surrogate.construct(
            training_data=self.training_data, fidelity_features=self.fidelity_features
        )
        mock_GP.assert_called_with(train_X=self.Xs[0], train_Y=self.Ys[0])

    @patch(f"{CURRENT_PATH}.SingleTaskGP.load_state_dict", return_value=None)
    @patch(f"{CURRENT_PATH}.ExactMarginalLogLikelihood")
    @patch(f"{SURROGATE_PATH}.fit_gpytorch_model")
    def test_fit(self, mock_fit_gpytorch, mock_MLL, mock_state_dict):
        surrogate = Surrogate(
            botorch_model_class=self.botorch_model_class,
            mll_class=ExactMarginalLogLikelihood,
        )
        # Checking that model is None before `fit` (and `construct`) calls.
        self.assertIsNone(surrogate._model)
        # Should instantiate mll and `fit_gpytorch_model` when `state_dict`
        # is `None`.
        surrogate.fit(
            training_data=self.training_data,
            bounds=self.bounds,
            task_features=self.task_features,
            feature_names=self.feature_names,
            metric_names=self.metric_names,
            fidelity_features=self.fidelity_features,
            target_fidelities=self.target_fidelities,
            refit=self.refit,
        )
        mock_state_dict.assert_not_called()
        mock_MLL.assert_called_once()
        mock_fit_gpytorch.assert_called_once()
        mock_state_dict.reset_mock()
        mock_MLL.reset_mock()
        mock_fit_gpytorch.reset_mock()
        # Should `load_state_dict` when `state_dict` is not `None`
        # and `refit` is `False`.
        state_dict = {}
        surrogate.fit(
            training_data=self.training_data,
            bounds=self.bounds,
            task_features=self.task_features,
            feature_names=self.feature_names,
            metric_names=self.metric_names,
            fidelity_features=self.fidelity_features,
            target_fidelities=self.target_fidelities,
            refit=False,
            state_dict=state_dict,
        )
        mock_state_dict.assert_called_once()
        mock_MLL.assert_not_called()
        mock_fit_gpytorch.assert_not_called()

    @patch(f"{SURROGATE_PATH}.predict_from_model")
    def test_predict(self, mock_predict):
        self.surrogate.construct(
            training_data=self.training_data, fidelity_features=self.fidelity_features
        )
        self.surrogate.predict(X=self.Xs[0])
        mock_predict.assert_called_with(model=self.surrogate.model, X=self.Xs[0])

    def test_best_in_sample_point(self):
        self.surrogate.construct(
            training_data=self.training_data, fidelity_features=self.fidelity_features
        )
        # `best_in_sample_point` requires `objective_weights`
        with patch(
            f"{SURROGATE_PATH}.best_in_sample_point", return_value=None
        ) as mock_best_in_sample:
            with self.assertRaisesRegex(ValueError, "Could not obtain"):
                self.surrogate.best_in_sample_point(
                    bounds=self.bounds, objective_weights=None
                )
        with patch(
            f"{SURROGATE_PATH}.best_in_sample_point", return_value=(self.Xs[0], 0.0)
        ) as mock_best_in_sample:
            best_point, observed_value = self.surrogate.best_in_sample_point(
                bounds=self.bounds,
                objective_weights=self.objective_weights,
                outcome_constraints=self.outcome_constraints,
                linear_constraints=self.linear_constraints,
                fixed_features=self.fixed_features,
                options=self.options,
            )
            mock_best_in_sample.assert_called_with(
                Xs=self.training_data.Xs,
                model=self.surrogate,
                bounds=self.bounds,
                objective_weights=self.objective_weights,
                outcome_constraints=self.outcome_constraints,
                linear_constraints=self.linear_constraints,
                fixed_features=self.fixed_features,
                options=self.options,
            )

    @patch(f"{ACQUISITION_PATH}.Acquisition.__init__", return_value=None)
    @patch(
        f"{ACQUISITION_PATH}.Acquisition.optimize",
        return_value=([torch.tensor([0.0])], [torch.tensor([1.0])]),
    )
    @patch(
        f"{SURROGATE_PATH}.pick_best_out_of_sample_point_acqf_class",
        return_value=(qSimpleRegret, {"sampler": SobolQMCNormalSampler}),
    )
    def test_best_out_of_sample_point(
        self, mock_best_point_util, mock_acqf_optimize, mock_acqf_init
    ):
        self.surrogate.construct(
            training_data=self.training_data, fidelity_features=self.fidelity_features
        )
        # currently cannot use function with fixed features
        with self.assertRaisesRegex(NotImplementedError, "Fixed features"):
            self.surrogate.best_out_of_sample_point(
                bounds=self.bounds,
                objective_weights=self.objective_weights,
                fixed_features=self.fixed_features,
            )
        candidate, acqf_value = self.surrogate.best_out_of_sample_point(
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fidelity_features=self.fidelity_features,
            target_fidelities=self.target_fidelities,
            options=self.options,
        )
        mock_acqf_init.assert_called_with(
            surrogate=self.surrogate,
            botorch_acqf_class=qSimpleRegret,
            bounds=self.bounds,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=None,
            target_fidelities=self.target_fidelities,
            options={"sampler": SobolQMCNormalSampler},
        )
        self.assertTrue(torch.equal(candidate, torch.tensor([0.0])))
        self.assertTrue(torch.equal(acqf_value, torch.tensor([1.0])))

    @patch(f"{SURROGATE_PATH}.Surrogate.fit")
    def test_update(self, mock_fit):
        self.surrogate.construct(
            training_data=self.training_data, fidelity_features=self.fidelity_features
        )
        # Call `fit` by default
        self.surrogate.update(
            training_data=self.training_data,
            bounds=self.bounds,
            task_features=self.task_features,
            feature_names=self.feature_names,
            metric_names=self.metric_names,
            fidelity_features=self.fidelity_features,
            refit=self.refit,
        )
        mock_fit.assert_called_with(
            training_data=self.training_data,
            bounds=self.bounds,
            task_features=self.task_features,
            feature_names=self.feature_names,
            metric_names=self.metric_names,
            fidelity_features=self.fidelity_features,
            candidate_metadata=None,
            state_dict=self.surrogate.model.state_dict,
            refit=self.refit,
        )
        # If should not be reconstructed, raise Error
        self.surrogate._should_reconstruct = False
        with self.assertRaisesRegex(
            NotImplementedError, ".* models that should not be re-constructed"
        ):
            self.surrogate.update(
                training_data=self.training_data,
                bounds=self.bounds,
                task_features=self.task_features,
                feature_names=self.feature_names,
                metric_names=self.metric_names,
                fidelity_features=self.fidelity_features,
                refit=self.refit,
            )
