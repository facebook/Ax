#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock, patch

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import UserInputError
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.torch_stubs import get_torch_test_data
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.models import SaasFullyBayesianSingleTaskGP, SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.containers import TrainingData
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel  # noqa: F401
from gpytorch.likelihoods import (  # noqa: F401
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    Likelihood,
)
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood
from torch import Tensor

ACQUISITION_PATH = f"{Acquisition.__module__}"
CURRENT_PATH = f"{__name__}"
SURROGATE_PATH = f"{Surrogate.__module__}"


class SingleTaskGPWithDifferentConstructor(SingleTaskGP):
    def __init__(self, train_X: Tensor, train_Y: Tensor):
        super().__init__(train_X=train_X, train_Y=train_Y)


class SurrogateTest(TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float
        self.Xs, self.Ys, self.Yvars, self.bounds, _, _, _ = get_torch_test_data(
            dtype=self.dtype
        )
        self.training_data = TrainingData.from_block_design(
            X=self.Xs[0], Y=self.Ys[0], Yvar=self.Yvars[0]
        )
        self.mll_class = ExactMarginalLogLikelihood
        self.search_space_digest = SearchSpaceDigest(
            feature_names=["x1", "x2"],
            bounds=self.bounds,
            target_fidelities={1: 1.0},
        )
        self.metric_names = ["y"]
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

    def _get_surrogate(self, botorch_model_class):
        surrogate = Surrogate(
            botorch_model_class=botorch_model_class, mll_class=self.mll_class
        )
        surrogate_kwargs = botorch_model_class.construct_inputs(self.training_data)
        return surrogate, surrogate_kwargs

    @patch(f"{CURRENT_PATH}.Kernel")
    @patch(f"{CURRENT_PATH}.Likelihood")
    def test_init(self, mock_Likelihood, mock_Kernel):
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            self.assertEqual(surrogate.botorch_model_class, botorch_model_class)
            self.assertEqual(surrogate.mll_class, self.mll_class)

    @patch(f"{SURROGATE_PATH}.fit_gpytorch_model")
    def test_mll_options(self, _):
        mock_mll = MagicMock(self.mll_class)
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
            mll_class=mock_mll,
            mll_options={"some_option": "some_value"},
        )
        surrogate.fit(
            training_data=self.training_data,
            search_space_digest=self.search_space_digest,
            metric_names=self.metric_names,
            refit=self.refit,
        )
        self.assertEqual(mock_mll.call_args[1]["some_option"], "some_value")

    def test_botorch_transforms(self):
        # Successfully passing down the transforms
        input_transform = Normalize(d=self.Xs[0].shape[-1])
        outcome_transform = Standardize(m=self.Ys[0].shape[-1])
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        surrogate.fit(
            training_data=self.training_data,
            search_space_digest=self.search_space_digest,
            metric_names=self.metric_names,
            refit=self.refit,
        )
        botorch_model = surrogate.model
        self.assertIs(botorch_model.input_transform, input_transform)
        self.assertIs(botorch_model.outcome_transform, outcome_transform)

        # Error handling if the model does not support transforms.
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGPWithDifferentConstructor,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        with self.assertRaisesRegex(UserInputError, "BoTorch model"):
            surrogate.fit(
                training_data=self.training_data,
                search_space_digest=self.search_space_digest,
                metric_names=self.metric_names,
                refit=self.refit,
            )

    def test_model_property(self):
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            with self.assertRaisesRegex(
                ValueError, "BoTorch `Model` has not yet been constructed."
            ):
                surrogate.model

    def test_training_data_property(self):
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            with self.assertRaisesRegex(
                ValueError,
                "Underlying BoTorch `Model` has not yet received its training_data.",
            ):
                surrogate.training_data

    def test_dtype_property(self):
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.construct(
                training_data=self.training_data,
                fidelity_features=self.search_space_digest.fidelity_features,
            )
            self.assertEqual(self.dtype, surrogate.dtype)

    def test_device_property(self):
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.construct(
                training_data=self.training_data,
                fidelity_features=self.search_space_digest.fidelity_features,
            )
            self.assertEqual(self.device, surrogate.device)

    def test_from_botorch(self):
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate_kwargs = botorch_model_class.construct_inputs(self.training_data)
            surrogate = Surrogate.from_botorch(botorch_model_class(**surrogate_kwargs))
            self.assertIsInstance(surrogate.model, botorch_model_class)
            self.assertTrue(surrogate._constructed_manually)

    @patch(f"{CURRENT_PATH}.SaasFullyBayesianSingleTaskGP.__init__", return_value=None)
    @patch(f"{CURRENT_PATH}.SingleTaskGP.__init__", return_value=None)
    def test_construct(self, mock_GP, mock_SAAS):
        mock_GPs = [mock_SAAS, mock_GP]
        for i, botorch_model_class in enumerate(
            [SaasFullyBayesianSingleTaskGP, SingleTaskGP]
        ):
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            with self.assertRaises(NotImplementedError):
                # Base `Model` does not implement `construct_inputs`.
                Surrogate(botorch_model_class=Model).construct(
                    training_data=self.training_data,
                    fidelity_features=self.search_space_digest.fidelity_features,
                )
            surrogate.construct(
                training_data=self.training_data,
                fidelity_features=self.search_space_digest.fidelity_features,
            )
            mock_GPs[i].assert_called_once()
            call_kwargs = mock_GPs[i].call_args[1]
            self.assertTrue(torch.equal(call_kwargs["train_X"], self.Xs[0]))
            self.assertTrue(torch.equal(call_kwargs["train_Y"], self.Ys[0]))
            self.assertFalse(surrogate._constructed_manually)

            # Check that `model_options` passed to the `Surrogate` constructor are
            # properly propagated.
            with patch.object(
                botorch_model_class,
                "construct_inputs",
                wraps=botorch_model_class.construct_inputs,
            ) as mock_construct_inputs:
                surrogate = Surrogate(
                    botorch_model_class=botorch_model_class,
                    mll_class=self.mll_class,
                    model_options={"some_option": "some_value"},
                )
                surrogate.construct(self.training_data)
                mock_construct_inputs.assert_called_with(
                    training_data=self.training_data, some_option="some_value"
                )

    def test_construct_custom_model(self):
        # Test error for unsupported covar_module and likelihood.
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGPWithDifferentConstructor,
            mll_class=self.mll_class,
            covar_module_class=RBFKernel,
            likelihood_class=FixedNoiseGaussianLikelihood,
        )
        with self.assertRaisesRegex(UserInputError, "does not support"):
            surrogate.construct(self.training_data)
        # Pass custom options to a SingleTaskGP and make sure they are used
        noise_constraint = Interval(1e-6, 1e-1)
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
            mll_class=LeaveOneOutPseudoLikelihood,
            covar_module_class=RBFKernel,
            covar_module_options={"ard_num_dims": 1},
            likelihood_class=GaussianLikelihood,
            likelihood_options={"noise_constraint": noise_constraint},
        )
        surrogate.construct(self.training_data)
        self.assertEqual(type(surrogate._model.likelihood), GaussianLikelihood)
        self.assertEqual(
            surrogate._model.likelihood.noise_covar.raw_noise_constraint,
            noise_constraint,
        )
        self.assertEqual(surrogate.mll_class, LeaveOneOutPseudoLikelihood)
        self.assertEqual(type(surrogate._model.covar_module), RBFKernel)
        self.assertEqual(surrogate._model.covar_module.ard_num_dims, 1)

    @patch(f"{CURRENT_PATH}.SingleTaskGP.load_state_dict", return_value=None)
    @patch(f"{SURROGATE_PATH}.fit_fully_bayesian_model_nuts")
    @patch(f"{SURROGATE_PATH}.fit_gpytorch_model")
    @patch(f"{CURRENT_PATH}.ExactMarginalLogLikelihood")
    def test_fit(self, mock_MLL, mock_fit_gpytorch, mock_fit_saas, mock_state_dict):
        for mock_fit, botorch_model_class in zip(
            [mock_fit_saas, mock_fit_gpytorch],
            [SaasFullyBayesianSingleTaskGP, SingleTaskGP],
        ):
            surrogate, surrogate_kwargs = self._get_surrogate(
                botorch_model_class=botorch_model_class
            )
            # Checking that model is None before `fit` (and `construct`) calls.
            self.assertIsNone(surrogate._model)
            # Should instantiate mll and `fit_gpytorch_model` when `state_dict`
            # is `None`.
            surrogate.fit(
                training_data=self.training_data,
                search_space_digest=self.search_space_digest,
                metric_names=self.metric_names,
                refit=self.refit,
            )
            # Check that training data is correctly passed through to the
            # BoTorch `Model`.
            self.assertTrue(
                torch.equal(
                    surrogate.model.train_inputs[0],
                    surrogate_kwargs.get("train_X"),
                )
            )
            self.assertTrue(
                torch.equal(
                    surrogate.model.train_targets,
                    surrogate_kwargs.get("train_Y").squeeze(1),
                )
            )
            mock_state_dict.assert_not_called()
            mock_fit.assert_called_once()
            mock_state_dict.reset_mock()
            mock_MLL.reset_mock()
            mock_fit.reset_mock()
            # Should `load_state_dict` when `state_dict` is not `None`
            # and `refit` is `False`.
            state_dict = {"state_attribute": "value"}
            surrogate.fit(
                training_data=self.training_data,
                search_space_digest=self.search_space_digest,
                metric_names=self.metric_names,
                refit=False,
                state_dict=state_dict,
            )
            mock_state_dict.assert_called_once()
            mock_MLL.assert_not_called()
            mock_fit.assert_not_called()
            mock_state_dict.reset_mock()
            mock_MLL.reset_mock()

    @patch(f"{SURROGATE_PATH}.predict_from_model")
    def test_predict(self, mock_predict):
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.construct(
                training_data=self.training_data,
                fidelity_features=self.search_space_digest.fidelity_features,
            )
            surrogate.predict(X=self.Xs[0])
            mock_predict.assert_called_with(model=surrogate.model, X=self.Xs[0])

    def test_best_in_sample_point(self):
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.construct(
                training_data=self.training_data,
                fidelity_features=self.search_space_digest.fidelity_features,
            )
            # `best_in_sample_point` requires `objective_weights`
            with patch(
                f"{SURROGATE_PATH}.best_in_sample_point", return_value=None
            ) as mock_best_in_sample:
                with self.assertRaisesRegex(ValueError, "Could not obtain"):
                    surrogate.best_in_sample_point(
                        search_space_digest=self.search_space_digest,
                        objective_weights=None,
                    )
            with patch(
                f"{SURROGATE_PATH}.best_in_sample_point", return_value=(self.Xs[0], 0.0)
            ) as mock_best_in_sample:
                best_point, observed_value = surrogate.best_in_sample_point(
                    search_space_digest=self.search_space_digest,
                    objective_weights=self.objective_weights,
                    outcome_constraints=self.outcome_constraints,
                    linear_constraints=self.linear_constraints,
                    fixed_features=self.fixed_features,
                    options=self.options,
                )
                mock_best_in_sample.assert_called_with(
                    Xs=[self.training_data.X],
                    model=surrogate,
                    bounds=self.search_space_digest.bounds,
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
        return_value=(qSimpleRegret, {Keys.SAMPLER: SobolQMCNormalSampler}),
    )
    def test_best_out_of_sample_point(
        self, mock_best_point_util, mock_acqf_optimize, mock_acqf_init
    ):
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.construct(
                training_data=self.training_data,
                fidelity_features=self.search_space_digest.fidelity_features,
            )
            # currently cannot use function with fixed features
            with self.assertRaisesRegex(NotImplementedError, "Fixed features"):
                surrogate.best_out_of_sample_point(
                    search_space_digest=self.search_space_digest,
                    objective_weights=self.objective_weights,
                    fixed_features=self.fixed_features,
                )
            candidate, acqf_value = surrogate.best_out_of_sample_point(
                search_space_digest=self.search_space_digest,
                objective_weights=self.objective_weights,
                outcome_constraints=self.outcome_constraints,
                linear_constraints=self.linear_constraints,
                options=self.options,
            )
            mock_acqf_init.assert_called_with(
                surrogate=surrogate,
                botorch_acqf_class=qSimpleRegret,
                search_space_digest=self.search_space_digest,
                objective_weights=self.objective_weights,
                outcome_constraints=self.outcome_constraints,
                linear_constraints=self.linear_constraints,
                fixed_features=None,
                options={Keys.SAMPLER: SobolQMCNormalSampler},
            )
            self.assertTrue(torch.equal(candidate, torch.tensor([0.0])))
            self.assertTrue(torch.equal(acqf_value, torch.tensor([1.0])))

    @patch(f"{CURRENT_PATH}.SingleTaskGP.load_state_dict", return_value=None)
    @patch(f"{SURROGATE_PATH}.fit_fully_bayesian_model_nuts")
    @patch(f"{SURROGATE_PATH}.fit_gpytorch_model")
    @patch(f"{CURRENT_PATH}.ExactMarginalLogLikelihood")
    def test_update(self, mock_MLL, mock_fit_gpytorch, mock_fit_saas, mock_state_dict):
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, surrogate_kwargs = self._get_surrogate(
                botorch_model_class=botorch_model_class
            )
            surrogate.construct(
                training_data=self.training_data,
                fidelity_features=self.search_space_digest.fidelity_features,
            )
            # Check that correct arguments are passed to `fit`.
            with patch(f"{SURROGATE_PATH}.Surrogate.fit") as mock_fit:
                # Call `fit` by default
                surrogate.update(
                    training_data=self.training_data,
                    search_space_digest=self.search_space_digest,
                    metric_names=self.metric_names,
                    refit=self.refit,
                    state_dict={"key": "val"},
                )
                mock_fit.assert_called_with(
                    training_data=self.training_data,
                    search_space_digest=self.search_space_digest,
                    metric_names=self.metric_names,
                    candidate_metadata=None,
                    refit=self.refit,
                    state_dict={"key": "val"},
                )

            # Check that the training data is correctly passed through to the
            # BoTorch `Model`.
            Xs, Ys, Yvars, bounds, _, _, _ = get_torch_test_data(
                dtype=self.dtype, offset=1.0
            )
            training_data = TrainingData.from_block_design(
                X=Xs[0], Y=Ys[0], Yvar=Yvars[0]
            )
            surrogate_kwargs = botorch_model_class.construct_inputs(training_data)
            surrogate.update(
                training_data=training_data,
                search_space_digest=self.search_space_digest,
                metric_names=self.metric_names,
                refit=self.refit,
                state_dict={"key": "val"},
            )
            self.assertTrue(
                torch.equal(
                    surrogate.model.train_inputs[0],
                    surrogate_kwargs.get("train_X"),
                )
            )
            self.assertTrue(
                torch.equal(
                    surrogate.model.train_targets,
                    surrogate_kwargs.get("train_Y").squeeze(1),
                )
            )

            # If should not be reconstructed, check that error is raised.
            surrogate._constructed_manually = True
            with self.assertRaisesRegex(NotImplementedError, ".* constructed manually"):
                surrogate.update(
                    training_data=self.training_data,
                    search_space_digest=self.search_space_digest,
                    metric_names=self.metric_names,
                    refit=self.refit,
                )

    def test_serialize_attributes_as_kwargs(self):
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            expected = surrogate.__dict__
            self.assertEqual(surrogate._serialize_attributes_as_kwargs(), expected)
