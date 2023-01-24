#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import math
from typing import Any, Dict, Tuple, Type
from unittest.mock import MagicMock, Mock, patch

import numpy as np

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.botorch_modular.utils import choose_model_class, fit_botorch_model
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast, not_none
from ax.utils.testing.torch_stubs import get_torch_test_data
from ax.utils.testing.utils import generic_equals
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.models import SaasFullyBayesianSingleTaskGP, SingleTaskGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from botorch.models.model import Model, ModelList  # noqa: F401
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import InputPerturbation, Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.datasets import FixedNoiseDataset, SupervisedDataset
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel  # noqa: F401
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

RANK = "rank"


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
        self.metric_names = ["metric"]
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

    def test_clone_reset(self) -> None:
        surrogate = self._get_surrogate(botorch_model_class=SingleTaskGP)[0]
        self.assertEqual(surrogate, surrogate.clone_reset())

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
                metric_names=self.metric_names,
                fidelity_features=self.search_space_digest.fidelity_features,
            )
            self.assertEqual(self.dtype, surrogate.dtype)

    def test_device_property(self) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.construct(
                datasets=self.training_data,
                metric_names=self.metric_names,
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
                    metric_names=self.metric_names,
                    fidelity_features=self.search_space_digest.fidelity_features,
                )
            surrogate.construct(
                datasets=self.training_data,
                metric_names=self.metric_names,
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
                surrogate.construct(
                    self.training_data,
                    metric_names=self.metric_names,
                )
                mock_construct_inputs.assert_called_with(
                    training_data=self.training_data[0], some_option="some_value"
                )

            # seach_space_digest may not be None if no model_class provided
            with self.assertRaisesRegex(
                UserInputError, "seach_space_digest may not be None"
            ):
                surrogate = Surrogate()
                surrogate.construct(
                    datasets=self.training_data,
                    metric_names=self.metric_names,
                )

            # botorch_model_class must be set to construct single model Surrogate
            with self.assertRaisesRegex(ValueError, "botorch_model_class must be set"):
                surrogate = Surrogate()
                surrogate._construct_model(dataset=self.training_data[0])

    def test_construct_custom_model(self) -> None:
        # Test error for unsupported covar_module and likelihood.
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGPWithDifferentConstructor,
            mll_class=self.mll_class,
            covar_module_class=RBFKernel,
            likelihood_class=FixedNoiseGaussianLikelihood,
        )
        with self.assertRaisesRegex(UserInputError, "does not support"):
            surrogate.construct(
                self.training_data,
                metric_names=self.metric_names,
            )
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
        surrogate.construct(
            self.training_data,
            metric_names=self.metric_names,
        )
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
                metric_names=self.metric_names,
                fidelity_features=self.search_space_digest.fidelity_features,
            )
            surrogate.predict(X=self.Xs[0])
            mock_predict.assert_called_with(model=surrogate.model, X=self.Xs[0])

    def test_best_in_sample_point(self) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.construct(
                datasets=self.training_data,
                metric_names=self.metric_names,
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
                metric_names=self.metric_names,
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
                surrogates={"self": surrogate},
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
                metric_names=self.metric_names,
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
                metric_names=self.metric_names,
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
                metric_names=self.metric_names,
                robust_digest=robust_digest,
            )
        # Input perturbation is constructed.
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
        )
        surrogate.construct(
            datasets=self.training_data,
            metric_names=self.metric_names,
            robust_digest=robust_digest,
        )
        intf = checked_cast(InputPerturbation, surrogate.model.input_transform)
        self.assertIsInstance(intf, InputPerturbation)
        self.assertTrue(torch.equal(intf.perturbation_set, torch.zeros(2, 2)))


class SurrogateWithModelListTest(TestCase):
    def setUp(self) -> None:
        self.outcomes = ["outcome_1", "outcome_2"]
        self.mll_class = ExactMarginalLogLikelihood
        self.dtype = torch.float
        self.search_space_digest = SearchSpaceDigest(
            feature_names=[], bounds=[], task_features=[0]
        )
        self.task_features = [0]
        Xs1, Ys1, Yvars1, bounds, _, _, _ = get_torch_test_data(
            dtype=self.dtype, task_features=self.search_space_digest.task_features
        )
        # Change the inputs/outputs a bit so the data isn't identical
        Xs1[0] *= 2
        Ys1[0] += 1
        Xs2, Ys2, Yvars2, _, _, _, _ = get_torch_test_data(
            dtype=self.dtype, task_features=self.search_space_digest.task_features
        )
        self.botorch_submodel_class_per_outcome = {
            self.outcomes[0]: choose_model_class(
                datasets=[
                    FixedNoiseDataset(X=X, Y=Y, Yvar=Yvar)
                    for X, Y, Yvar in zip(Xs1, Ys1, Yvars1)
                ],
                search_space_digest=self.search_space_digest,
            ),
            self.outcomes[1]: choose_model_class(
                datasets=[
                    FixedNoiseDataset(X=X, Y=Y, Yvar=Yvar)
                    for X, Y, Yvar in zip(Xs2, Ys2, Yvars2)
                ],
                search_space_digest=self.search_space_digest,
            ),
        }
        self.botorch_model_class = FixedNoiseMultiTaskGP
        for submodel_cls in self.botorch_submodel_class_per_outcome.values():
            self.assertEqual(submodel_cls, FixedNoiseMultiTaskGP)
        self.Xs = Xs1 + Xs2
        self.Ys = Ys1 + Ys2
        self.Yvars = Yvars1 + Yvars2
        self.fixed_noise_training_data = [
            FixedNoiseDataset(X=X, Y=Y, Yvar=Yvar)
            for X, Y, Yvar in zip(self.Xs, self.Ys, self.Yvars)
        ]
        self.supervised_training_data = [
            SupervisedDataset(X=X, Y=Y) for X, Y in zip(self.Xs, self.Ys)
        ]
        self.submodel_options_per_outcome = {
            RANK: 1,
        }
        self.surrogate = Surrogate(
            botorch_model_class=FixedNoiseMultiTaskGP,
            mll_class=self.mll_class,
            model_options=self.submodel_options_per_outcome,
        )
        self.bounds = [(0.0, 1.0), (1.0, 4.0)]
        self.feature_names = ["x1", "x2"]

    def test_init(self) -> None:
        self.assertEqual(
            [self.surrogate.botorch_model_class] * 2,
            [*self.botorch_submodel_class_per_outcome.values()],
        )
        self.assertEqual(self.surrogate.mll_class, self.mll_class)
        with self.assertRaisesRegex(
            ValueError, "BoTorch `Model` has not yet been constructed"
        ):
            self.surrogate.model

    @patch.object(
        FixedNoiseMultiTaskGP,
        "construct_inputs",
        wraps=FixedNoiseMultiTaskGP.construct_inputs,
    )
    def test_construct_per_outcome_options(
        self, mock_MTGP_construct_inputs: Mock
    ) -> None:
        self.surrogate.construct(
            datasets=self.fixed_noise_training_data,
            metric_names=self.outcomes,
            task_features=self.task_features,
        )
        # Should construct inputs for MTGP twice.
        self.assertEqual(len(mock_MTGP_construct_inputs.call_args_list), 2)
        # First construct inputs should be called for MTGP with training data #0.
        for idx in range(len(mock_MTGP_construct_inputs.call_args_list)):
            self.assertEqual(
                # `call_args` is a tuple of (args, kwargs), and we check kwargs.
                mock_MTGP_construct_inputs.call_args_list[idx][1],
                {
                    "fidelity_features": [],
                    "task_feature": self.task_features[0],
                    # TODO: Figure out how to handle Multitask GPs and construct-inputs.
                    # I believe this functionality with modlular botorch model is
                    # currently broken as MultiTaskGP.construct_inputs expects a dict
                    # mapping string keys (outcomes) to input datasets
                    "training_data": FixedNoiseDataset(
                        X=self.Xs[idx], Y=self.Ys[idx], Yvar=self.Yvars[idx]
                    ),
                    "rank": 1,
                },
            )

    @patch.object(
        MultiTaskGP,
        "construct_inputs",
        wraps=MultiTaskGP.construct_inputs,
    )
    def test_construct_per_outcome_options_no_Yvar(self, _) -> None:
        surrogate = Surrogate(
            botorch_model_class=MultiTaskGP,
            mll_class=self.mll_class,
            model_options=self.submodel_options_per_outcome,
        )
        # Test that splitting the training data works correctly when Yvar is None.
        surrogate.construct(
            datasets=self.supervised_training_data,
            task_features=self.task_features,
            metric_names=self.outcomes,
        )
        for ds in not_none(surrogate._training_data):
            self.assertTrue(isinstance(ds, SupervisedDataset))
            self.assertFalse(isinstance(ds, FixedNoiseDataset))
        self.assertEqual(len(not_none(surrogate._training_data)), 2)

    @patch.object(
        FixedNoiseMultiTaskGP,
        "construct_inputs",
        wraps=FixedNoiseMultiTaskGP.construct_inputs,
    )
    def test_construct_per_outcome_error_raises(
        self, mock_MTGP_construct_inputs: Mock
    ) -> None:
        surrogate = Surrogate(
            botorch_model_class=self.botorch_model_class,
            mll_class=self.mll_class,
            model_options=self.submodel_options_per_outcome,
        )

        with self.assertRaisesRegex(
            NotImplementedError,
            "Multi-Fidelity GP models with task_features are "
            "currently not supported.",
        ):
            surrogate.construct(
                datasets=self.fixed_noise_training_data,
                metric_names=self.outcomes,
                task_features=self.task_features,
                fidelity_features=[1],
            )
        with self.assertRaisesRegex(
            NotImplementedError,
            "This model only supports 1 task feature!",
        ):
            surrogate.construct(
                datasets=self.fixed_noise_training_data,
                metric_names=self.outcomes,
                task_features=[0, 1],
            )

        # must either provide `botorch_submodel_class` or `search_space_digest`
        with self.assertRaisesRegex(
            UserInputError, "Must either provide `botorch_submodel_class` or"
        ):
            surrogate = Surrogate()
            surrogate._construct_model_list(
                datasets=self.fixed_noise_training_data,
                metric_names=self.outcomes,
            )

    @patch(f"{CURRENT_PATH}.ModelList.load_state_dict", return_value=None)
    @patch(f"{CURRENT_PATH}.ExactMarginalLogLikelihood")
    @patch(f"{UTILS_PATH}.fit_gpytorch_mll")
    @patch(f"{UTILS_PATH}.fit_fully_bayesian_model_nuts")
    def test_fit(
        self,
        mock_fit_nuts: Mock,
        mock_fit_gpytorch: Mock,
        mock_MLL: Mock,
        mock_state_dict: Mock,
    ) -> None:
        default_class = self.botorch_model_class
        surrogates = [
            Surrogate(
                botorch_model_class=default_class,
                mll_class=ExactMarginalLogLikelihood,
            ),
            Surrogate(botorch_model_class=SaasFullyBayesianSingleTaskGP),
            Surrogate(botorch_model_class=SaasFullyBayesianMultiTaskGP),
        ]

        for i, surrogate in enumerate(surrogates):
            # Checking that model is None before `fit` (and `construct`) calls.
            self.assertIsNone(surrogate._model)
            # Should instantiate mll and `fit_gpytorch_mll` when `state_dict`
            # is `None`.
            surrogate.fit(
                datasets=self.fixed_noise_training_data,
                metric_names=self.outcomes,
                search_space_digest=SearchSpaceDigest(
                    feature_names=self.feature_names,
                    bounds=self.bounds,
                    task_features=self.task_features,
                ),
            )
            mock_state_dict.assert_not_called()
            if i == 0:
                self.assertEqual(mock_MLL.call_count, 2)
                self.assertEqual(mock_fit_gpytorch.call_count, 2)
                mock_state_dict.reset_mock()
                mock_MLL.reset_mock()
                mock_fit_gpytorch.reset_mock()
            else:
                self.assertEqual(mock_MLL.call_count, 0)
                self.assertEqual(mock_fit_nuts.call_count, 2)
                mock_fit_nuts.reset_mock()
            # Should `load_state_dict` when `state_dict` is not `None`
            # and `refit` is `False`.
            state_dict = {"state_attribute": "value"}
            surrogate.fit(
                datasets=self.fixed_noise_training_data,
                metric_names=self.outcomes,
                search_space_digest=SearchSpaceDigest(
                    feature_names=self.feature_names,
                    bounds=self.bounds,
                    task_features=self.task_features,
                ),
                refit=False,
                # pyre-fixme[6]: For 5th param expected `Optional[Dict[str,
                #  Tensor]]` but got `Dict[str, str]`.
                state_dict=state_dict,
            )
            mock_state_dict.assert_called_once()
            mock_MLL.assert_not_called()
            mock_fit_gpytorch.assert_not_called()
            mock_fit_nuts.assert_not_called()
            mock_state_dict.reset_mock()

        # Fitting with PairwiseGP should be ok
        fit_botorch_model(
            model=PairwiseGP(
                datapoints=torch.rand(2, 2), comparisons=torch.tensor([[0, 1]])
            ),
            mll_class=PairwiseLaplaceMarginalLogLikelihood,
        )
        # Fitting with unknown model should raise
        with self.assertRaisesRegex(
            NotImplementedError,
            "Model of type GenericDeterministicModel is currently not supported.",
        ):
            fit_botorch_model(
                model=GenericDeterministicModel(f=lambda x: x),
                mll_class=self.mll_class,
            )

    def test_with_botorch_transforms(self) -> None:
        input_transforms = Normalize(d=3)
        outcome_transforms = Standardize(m=1)
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGPWithDifferentConstructor,
            mll_class=ExactMarginalLogLikelihood,
            outcome_transform=outcome_transforms,
            input_transform=input_transforms,
        )
        with self.assertRaisesRegex(UserInputError, "The BoTorch model class"):
            surrogate.construct(
                datasets=self.supervised_training_data,
                metric_names=self.outcomes,
            )
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
            mll_class=ExactMarginalLogLikelihood,
            outcome_transform=outcome_transforms,
            input_transform=input_transforms,
        )
        surrogate.construct(
            datasets=self.supervised_training_data,
            metric_names=self.outcomes,
        )
        # pyre-ignore [9]
        models: torch.nn.modules.container.ModuleList = surrogate.model.models
        for i in range(2):
            self.assertIsInstance(models[i].outcome_transform, Standardize)
            self.assertIsInstance(models[i].input_transform, Normalize)
        self.assertEqual(models[0].outcome_transform.means.item(), 4.5)
        self.assertEqual(models[1].outcome_transform.means.item(), 3.5)
        self.assertAlmostEqual(
            models[0].outcome_transform.stdvs.item(), 1 / math.sqrt(2)
        )
        self.assertAlmostEqual(
            models[1].outcome_transform.stdvs.item(), 1 / math.sqrt(2)
        )
        self.assertTrue(
            torch.all(
                torch.isclose(
                    models[0].input_transform.bounds,
                    2 * models[1].input_transform.bounds,  # pyre-ignore
                )
            )
        )

    def test_serialize_attributes_as_kwargs(self) -> None:
        # TODO[mpolson64] Reimplement this when serialization has been sorted out
        pass
        # expected = self.surrogate.__dict__
        # # The two attributes below don't need to be saved as part of state,
        # # so we remove them from the expected dict.
        # for attr_name in (
        #     "botorch_model_class",
        #     "model_options",
        #     "covar_module_class",
        #     "covar_module_options",
        #     "likelihood_class",
        #     "likelihood_options",
        #     "outcome_transform",
        #     "input_transform",
        # ):
        #     expected.pop(attr_name)
        # self.assertEqual(self.surrogate._serialize_attributes_as_kwargs(), expected)

    def test_construct_custom_model(self) -> None:
        noise_constraint = Interval(1e-4, 10.0)
        for submodel_covar_module_options, submodel_likelihood_options in [
            [{"ard_num_dims": 3}, {"noise_constraint": noise_constraint}],
            [{}, {}],
        ]:
            surrogate = Surrogate(
                botorch_model_class=SingleTaskGP,
                mll_class=ExactMarginalLogLikelihood,
                covar_module_class=MaternKernel,
                covar_module_options=submodel_covar_module_options,
                likelihood_class=GaussianLikelihood,
                likelihood_options=submodel_likelihood_options,
                input_transform=Normalize(d=3),
                outcome_transform=Standardize(m=1),
            )
            surrogate.construct(
                datasets=self.supervised_training_data,
                metric_names=self.outcomes,
            )
            # pyre-fixme[16]: Optional type has no attribute `models`.
            self.assertEqual(len(surrogate._model.models), 2)
            self.assertEqual(surrogate.mll_class, ExactMarginalLogLikelihood)
            # Make sure we properly copied the transforms
            self.assertNotEqual(
                id(surrogate._model.models[0].input_transform),
                id(surrogate._model.models[1].input_transform),
            )
            self.assertNotEqual(
                id(surrogate._model.models[0].outcome_transform),
                id(surrogate._model.models[1].outcome_transform),
            )

            for m in surrogate._model.models:
                self.assertEqual(type(m.likelihood), GaussianLikelihood)
                self.assertEqual(type(m.covar_module), MaternKernel)
                if submodel_covar_module_options:
                    self.assertEqual(m.covar_module.ard_num_dims, 3)
                else:
                    self.assertEqual(m.covar_module.ard_num_dims, None)
                if submodel_likelihood_options:
                    self.assertEqual(
                        type(m.likelihood.noise_covar.raw_noise_constraint), Interval
                    )
                    self.assertEqual(
                        m.likelihood.noise_covar.raw_noise_constraint.lower_bound,
                        noise_constraint.lower_bound,
                    )
                    self.assertEqual(
                        m.likelihood.noise_covar.raw_noise_constraint.upper_bound,
                        noise_constraint.upper_bound,
                    )
                else:
                    self.assertEqual(
                        type(m.likelihood.noise_covar.raw_noise_constraint), GreaterThan
                    )
                    self.assertEqual(
                        m.likelihood.noise_covar.raw_noise_constraint.lower_bound, 1e-4
                    )

    def test_w_robust_digest(self) -> None:
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
        )
        # Error handling.
        with self.assertRaisesRegex(NotImplementedError, "Environmental variable"):
            surrogate.construct(
                datasets=self.supervised_training_data,
                metric_names=self.outcomes,
                robust_digest={"environmental_variables": ["a"]},
            )
        robust_digest = {
            "sample_param_perturbations": lambda: np.zeros((2, 2)),
            "environmental_variables": [],
            "multiplicative": False,
        }
        surrogate.input_transform = Normalize(d=1)
        with self.assertRaisesRegex(NotImplementedError, "input transforms"):
            surrogate.construct(
                datasets=self.supervised_training_data,
                metric_names=self.outcomes,
                robust_digest=robust_digest,
            )
        # Input perturbation is constructed.
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
        )
        surrogate.construct(
            datasets=self.supervised_training_data,
            metric_names=self.outcomes,
            robust_digest=robust_digest,
        )
        for m in surrogate.model.models:  # pyre-ignore
            intf = checked_cast(InputPerturbation, m.input_transform)
            self.assertIsInstance(intf, InputPerturbation)
            self.assertTrue(torch.equal(intf.perturbation_set, torch.zeros(2, 2)))
