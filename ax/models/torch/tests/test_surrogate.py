#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from typing import Any, Dict, Tuple, Type
from unittest.mock import MagicMock, Mock, patch

import numpy as np

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.utils import fit_botorch_model
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast
from ax.utils.testing.torch_stubs import get_torch_test_data
from ax.utils.testing.utils import generic_equals
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.models import SaasFullyBayesianSingleTaskGP, SingleTaskGP
from botorch.models.model import Model
from botorch.models.transforms.input import InputPerturbation, Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.datasets import SupervisedDataset
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel  # noqa: F401
from gpytorch.likelihoods import (  # noqa: F401
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    Likelihood,  # noqa: F401
)
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood
from torch import Tensor


ACQUISITION_PATH = f"{Acquisition.__module__}"
CURRENT_PATH = f"{__name__}"
SURROGATE_PATH = f"{Surrogate.__module__}"
UTILS_PATH = f"{fit_botorch_model.__module__}"


class SingleTaskGPWithDifferentConstructor(SingleTaskGP):
    def __init__(self, train_X: Tensor, train_Y: Tensor) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y)


class SurrogateTest(TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.dtype = torch.float
        self.Xs, self.Ys, self.Yvars, self.bounds, _, _, _ = get_torch_test_data(
            dtype=self.dtype
        )
        self.training_data = [SupervisedDataset(X=self.Xs[0], Y=self.Ys[0])]
        self.mll_class = ExactMarginalLogLikelihood
        self.search_space_digest = SearchSpaceDigest(
            feature_names=["x1", "x2"],
            bounds=self.bounds,
            target_fidelities={1: 1.0},
        )
        self.metric_names = ["x_y"]
        self.original_metric_names = ["x", "y"]
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
        self.torch_opt_config = TorchOptConfig(
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
        )

    def _get_surrogate(
        self, botorch_model_class: Type[Model]
    ) -> Tuple[Surrogate, Dict[str, Any]]:
        surrogate = Surrogate(
            botorch_model_class=botorch_model_class, mll_class=self.mll_class
        )
        surrogate_kwargs = botorch_model_class.construct_inputs(self.training_data[0])
        return surrogate, surrogate_kwargs

    @patch(f"{CURRENT_PATH}.Kernel")
    @patch(f"{CURRENT_PATH}.Likelihood")
    def test_init(self, mock_Likelihood: Mock, mock_Kernel: Mock) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            self.assertEqual(surrogate.botorch_model_class, botorch_model_class)
            self.assertEqual(surrogate.mll_class, self.mll_class)

    @patch(f"{UTILS_PATH}.fit_gpytorch_mll")
    # pyre-fixme[3]: Return type must be annotated.
    def test_mll_options(self, _):
        mock_mll = MagicMock(self.mll_class)
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
            mll_class=mock_mll,
            mll_options={"some_option": "some_value"},
        )
        surrogate.fit(
            datasets=self.training_data,
            metric_names=self.metric_names,
            search_space_digest=self.search_space_digest,
            refit=self.refit,
        )
        self.assertEqual(mock_mll.call_args[1]["some_option"], "some_value")

    def test_botorch_transforms(self) -> None:
        # Successfully passing down the transforms
        input_transform = Normalize(d=self.Xs[0].shape[-1])
        outcome_transform = Standardize(m=self.Ys[0].shape[-1])
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        surrogate.fit(
            datasets=self.training_data,
            metric_names=self.metric_names,
            search_space_digest=self.search_space_digest,
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
                datasets=self.training_data,
                metric_names=self.metric_names,
                search_space_digest=self.search_space_digest,
                refit=self.refit,
            )

    def test_model_property(self) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            with self.assertRaisesRegex(
                ValueError, "BoTorch `Model` has not yet been constructed."
            ):
                surrogate.model

    def test_training_data_property(self) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            with self.assertRaisesRegex(
                ValueError,
                "Underlying BoTorch `Model` has not yet received its training_data.",
            ):
                surrogate.training_data

    def test_dtype_property(self) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.construct(
                datasets=self.training_data,
                fidelity_features=self.search_space_digest.fidelity_features,
            )
            self.assertEqual(self.dtype, surrogate.dtype)

    def test_device_property(self) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.construct(
                datasets=self.training_data,
                fidelity_features=self.search_space_digest.fidelity_features,
            )
            self.assertEqual(self.device, surrogate.device)

    def test_from_botorch(self) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate_kwargs = botorch_model_class.construct_inputs(
                self.training_data[0]
            )
            surrogate = Surrogate.from_botorch(botorch_model_class(**surrogate_kwargs))
            self.assertIsInstance(surrogate.model, botorch_model_class)
            self.assertTrue(surrogate._constructed_manually)

    @patch(f"{CURRENT_PATH}.SaasFullyBayesianSingleTaskGP.__init__", return_value=None)
    @patch(f"{CURRENT_PATH}.SingleTaskGP.__init__", return_value=None)
    def test_construct(self, mock_GP: Mock, mock_SAAS: Mock) -> None:
        mock_GPs = [mock_SAAS, mock_GP]
        for i, botorch_model_class in enumerate(
            [SaasFullyBayesianSingleTaskGP, SingleTaskGP]
        ):
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            with self.assertRaises(TypeError):
                # Base `Model` does not implement `posterior`, so instantiating it here
                # will fail.
                Surrogate(botorch_model_class=Model).construct(
                    datasets=self.training_data,
                    fidelity_features=self.search_space_digest.fidelity_features,
                )
            surrogate.construct(
                datasets=self.training_data,
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
                    training_data=self.training_data[0], some_option="some_value"
                )

    def test_construct_custom_model(self) -> None:
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
            # pyre-fixme[16]: Optional type has no attribute `likelihood`.
            surrogate._model.likelihood.noise_covar.raw_noise_constraint,
            noise_constraint,
        )
        self.assertEqual(surrogate.mll_class, LeaveOneOutPseudoLikelihood)
        self.assertEqual(type(surrogate._model.covar_module), RBFKernel)
        # pyre-fixme[16]: Optional type has no attribute `covar_module`.
        self.assertEqual(surrogate._model.covar_module.ard_num_dims, 1)

    @patch(
        f"{CURRENT_PATH}.SaasFullyBayesianSingleTaskGP.load_state_dict",
        return_value=None,
    )
    @patch(f"{CURRENT_PATH}.SingleTaskGP.load_state_dict", return_value=None)
    @patch(f"{UTILS_PATH}.fit_fully_bayesian_model_nuts")
    @patch(f"{UTILS_PATH}.fit_gpytorch_mll")
    @patch(f"{CURRENT_PATH}.ExactMarginalLogLikelihood")
    def test_fit(
        self,
        mock_MLL: Mock,
        mock_fit_gpytorch: Mock,
        mock_fit_saas: Mock,
        mock_state_dict_gpytorch: Mock,
        mock_state_dict_saas: Mock,
    ) -> None:
        for mock_fit, mock_state_dict, botorch_model_class in zip(
            [mock_fit_saas, mock_fit_gpytorch],
            [mock_state_dict_saas, mock_state_dict_gpytorch],
            [SaasFullyBayesianSingleTaskGP, SingleTaskGP],
        ):
            surrogate, surrogate_kwargs = self._get_surrogate(
                botorch_model_class=botorch_model_class
            )
            # Checking that model is None before `fit` (and `construct`) calls.
            self.assertIsNone(surrogate._model)
            # Should instantiate mll and `fit_gpytorch_mll` when `state_dict`
            # is `None`.
            surrogate.fit(
                datasets=self.training_data,
                metric_names=self.metric_names,
                search_space_digest=self.search_space_digest,
                refit=self.refit,
            )
            # Check that training data is correctly passed through to the
            # BoTorch `Model`.
            self.assertTrue(
                torch.equal(
                    surrogate.model.train_inputs[0],  # pyre-ignore
                    surrogate_kwargs.get("train_X"),
                )
            )
            self.assertTrue(
                torch.equal(
                    checked_cast(Tensor, surrogate.model.train_targets),
                    surrogate_kwargs.get("train_Y").squeeze(1),
                )
            )
            mock_state_dict.assert_not_called()
            mock_fit.assert_called_once()
            mock_state_dict.reset_mock()
            mock_MLL.reset_mock()
            mock_fit.reset_mock()
            # Check that the optional original_metric_names arg propagates
            # through surrogate._outcomes.
            surrogate.fit(
                datasets=self.training_data,
                metric_names=self.metric_names,
                search_space_digest=self.search_space_digest,
                refit=self.refit,
                original_metric_names=self.original_metric_names,
            )
            self.assertEqual(surrogate._outcomes, self.original_metric_names)
            mock_state_dict.reset_mock()
            mock_MLL.reset_mock()
            mock_fit.reset_mock()
            # Should `load_state_dict` when `state_dict` is not `None`
            # and `refit` is `False`.
            state_dict = {"state_attribute": torch.zeros(1)}
            surrogate.fit(
                datasets=self.training_data,
                metric_names=self.metric_names,
                search_space_digest=self.search_space_digest,
                refit=False,
                state_dict=state_dict,
            )
            mock_state_dict.assert_called_once()
            mock_MLL.assert_not_called()
            mock_fit.assert_not_called()
            mock_state_dict.reset_mock()
            mock_MLL.reset_mock()

    @patch(f"{SURROGATE_PATH}.predict_from_model")
    def test_predict(self, mock_predict: Mock) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.construct(
                datasets=self.training_data,
                fidelity_features=self.search_space_digest.fidelity_features,
            )
            surrogate.predict(X=self.Xs[0])
            mock_predict.assert_called_with(model=surrogate.model, X=self.Xs[0])

    def test_best_in_sample_point(self) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.construct(
                datasets=self.training_data,
                fidelity_features=self.search_space_digest.fidelity_features,
            )
            # `best_in_sample_point` requires `objective_weights`
            with patch(
                f"{SURROGATE_PATH}.best_in_sample_point", return_value=None
            ) as mock_best_in_sample:
                with self.assertRaisesRegex(ValueError, "Could not obtain"):
                    surrogate.best_in_sample_point(
                        search_space_digest=self.search_space_digest,
                        torch_opt_config=dataclasses.replace(
                            self.torch_opt_config,
                            objective_weights=None,
                        ),
                    )
            with patch(
                f"{SURROGATE_PATH}.best_in_sample_point", return_value=(self.Xs[0], 0.0)
            ) as mock_best_in_sample:
                best_point, observed_value = surrogate.best_in_sample_point(
                    search_space_digest=self.search_space_digest,
                    torch_opt_config=self.torch_opt_config,
                    options=self.options,
                )
                mock_best_in_sample.assert_called_once()
                _, ckwargs = mock_best_in_sample.call_args
                for X, dataset in zip(ckwargs["Xs"], self.training_data):
                    self.assertTrue(torch.equal(X, dataset.X()))
                self.assertIs(ckwargs["model"], surrogate)
                self.assertIs(ckwargs["bounds"], self.search_space_digest.bounds)
                self.assertIs(ckwargs["options"], self.options)
                for attr in (
                    "objective_weights",
                    "outcome_constraints",
                    "linear_constraints",
                    "fixed_features",
                ):
                    self.assertTrue(generic_equals(ckwargs[attr], getattr(self, attr)))

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
        self,
        mock_best_point_util: Mock,
        mock_acqf_optimize: Mock,
        mock_acqf_init: Mock,
    ) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.construct(
                datasets=self.training_data,
                fidelity_features=self.search_space_digest.fidelity_features,
            )
            # currently cannot use function with fixed features
            with self.assertRaisesRegex(NotImplementedError, "Fixed features"):
                surrogate.best_out_of_sample_point(
                    search_space_digest=self.search_space_digest,
                    torch_opt_config=self.torch_opt_config,
                )
            torch_opt_config = dataclasses.replace(
                self.torch_opt_config,
                fixed_features=None,
            )
            candidate, acqf_value = surrogate.best_out_of_sample_point(
                search_space_digest=self.search_space_digest,
                torch_opt_config=torch_opt_config,
                options=self.options,
            )
            mock_acqf_init.assert_called_with(
                surrogate=surrogate,
                botorch_acqf_class=qSimpleRegret,
                search_space_digest=self.search_space_digest,
                torch_opt_config=torch_opt_config,
                options={Keys.SAMPLER: SobolQMCNormalSampler},
            )
            self.assertTrue(torch.equal(candidate, torch.tensor([0.0])))
            self.assertTrue(torch.equal(acqf_value, torch.tensor([1.0])))

    @patch(
        f"{CURRENT_PATH}.SaasFullyBayesianSingleTaskGP.load_state_dict",
        return_value=None,
    )
    @patch(f"{CURRENT_PATH}.SingleTaskGP.load_state_dict", return_value=None)
    @patch(f"{UTILS_PATH}.fit_fully_bayesian_model_nuts")
    @patch(f"{UTILS_PATH}.fit_gpytorch_mll")
    @patch(f"{CURRENT_PATH}.ExactMarginalLogLikelihood")
    def test_update(
        self,
        mock_MLL: Mock,
        mock_fit_gpytorch: Mock,
        mock_fit_saas: Mock,
        mock_state_dict_gpytorch: Mock,
        mock_state_dict_saas: Mock,
    ) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, surrogate_kwargs = self._get_surrogate(
                botorch_model_class=botorch_model_class
            )
            surrogate.construct(
                datasets=self.training_data,
                fidelity_features=self.search_space_digest.fidelity_features,
            )
            # Check that correct arguments are passed to `fit`.
            with patch(f"{SURROGATE_PATH}.Surrogate.fit") as mock_fit:
                # Call `fit` by default
                surrogate.update(
                    datasets=self.training_data,
                    metric_names=self.metric_names,
                    search_space_digest=self.search_space_digest,
                    refit=self.refit,
                    state_dict={"key": torch.zeros(1)},
                )
                mock_fit.assert_called_with(
                    datasets=self.training_data,
                    metric_names=self.metric_names,
                    search_space_digest=self.search_space_digest,
                    candidate_metadata=None,
                    refit=self.refit,
                    state_dict={"key": torch.zeros(1)},
                )

            # Check that the training data is correctly passed through to the
            # BoTorch `Model`.
            Xs, Ys, Yvars, bounds, _, _, _ = get_torch_test_data(
                dtype=self.dtype, offset=1.0
            )
            surrogate_kwargs = botorch_model_class.construct_inputs(
                self.training_data[0]
            )
            surrogate.update(
                datasets=self.training_data,
                metric_names=self.metric_names,
                search_space_digest=self.search_space_digest,
                refit=self.refit,
                state_dict={"key": torch.zeros(1)},
            )
            self.assertTrue(
                torch.equal(
                    surrogate.model.train_inputs[0],  # pyre-ignore
                    surrogate_kwargs.get("train_X"),
                )
            )
            self.assertTrue(
                torch.equal(
                    checked_cast(Tensor, surrogate.model.train_targets),
                    surrogate_kwargs.get("train_Y").squeeze(1),
                )
            )

            # If should not be reconstructed, check that error is raised.
            surrogate._constructed_manually = True
            with self.assertRaisesRegex(NotImplementedError, ".* constructed manually"):
                surrogate.update(
                    datasets=self.training_data,
                    metric_names=self.metric_names,
                    search_space_digest=self.search_space_digest,
                    refit=self.refit,
                )

    def test_serialize_attributes_as_kwargs(self) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            expected = surrogate.__dict__
            self.assertEqual(surrogate._serialize_attributes_as_kwargs(), expected)

        with self.assertRaisesRegex(
            UnsupportedError, "Surrogates constructed manually"
        ):
            surrogate, _ = self._get_surrogate(botorch_model_class=SingleTaskGP)
            surrogate._constructed_manually = True
            surrogate._serialize_attributes_as_kwargs()

    def test_w_robust_digest(self) -> None:
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
        )
        # Error handling.
        with self.assertRaisesRegex(NotImplementedError, "Environmental variable"):
            surrogate.construct(
                datasets=self.training_data,
                robust_digest={"environmental_variables": ["a"]},
            )
        robust_digest = {
            "sample_param_perturbations": lambda: np.zeros((2, 2)),
            "environmental_variables": [],
            "multiplicative": False,
        }
        surrogate.input_transform = Normalize(d=2)
        with self.assertRaisesRegex(NotImplementedError, "input transforms"):
            surrogate.construct(
                datasets=self.training_data,
                robust_digest=robust_digest,
            )
        # Input perturbation is constructed.
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
        )
        surrogate.construct(
            datasets=self.training_data,
            robust_digest=robust_digest,
        )
        intf = checked_cast(InputPerturbation, surrogate.model.input_transform)
        self.assertIsInstance(intf, InputPerturbation)
        self.assertTrue(torch.equal(intf.perturbation_set, torch.zeros(2, 2)))
