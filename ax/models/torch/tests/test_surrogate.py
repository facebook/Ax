#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
import math
from collections import OrderedDict
from typing import Any, Dict, Tuple, Type
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
from ax.core.search_space import RobustSearchSpaceDigest, SearchSpaceDigest
from ax.exceptions.core import UserInputError
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import _extract_model_kwargs, Surrogate
from ax.models.torch.botorch_modular.utils import choose_model_class, fit_botorch_model
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast, not_none
from ax.utils.testing.mock import fast_botorch_optimize
from ax.utils.testing.torch_stubs import get_torch_test_data
from ax.utils.testing.utils import generic_equals
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.models import ModelListGP, SaasFullyBayesianSingleTaskGP, SingleTaskGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.model import Model, ModelList  # noqa: F401 -- used in Mocks.
from botorch.models.multitask import MultiTaskGP
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import InputPerturbation, Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.datasets import SupervisedDataset
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood
from torch import Tensor
from torch.nn import ModuleList  # @manual -- autodeps can't figure it out.


ACQUISITION_PATH = f"{Acquisition.__module__}"
CURRENT_PATH = f"{__name__}"
SURROGATE_PATH = f"{Surrogate.__module__}"
UTILS_PATH = f"{fit_botorch_model.__module__}"

RANK = "rank"


class SingleTaskGPWithDifferentConstructor(SingleTaskGP):
    def __init__(self, train_X: Tensor, train_Y: Tensor) -> None:
        super().__init__(train_X=train_X, train_Y=train_Y)


class ExtractModelKwargsTest(TestCase):
    def test__extract_model_kwargs(self) -> None:
        feature_names = ["a", "b"]
        bounds = [(0.0, 1.0), (0.0, 1.0)]

        with self.subTest("Multi-fidelity with task features not supported"):
            search_space_digest = SearchSpaceDigest(
                feature_names=feature_names,
                bounds=bounds,
                task_features=[0],
                fidelity_features=[0],
            )
            with self.assertRaisesRegex(
                NotImplementedError, "Multi-Fidelity GP models with task_features"
            ):
                _extract_model_kwargs(
                    search_space_digest=search_space_digest,
                )

        with self.subTest("Multiple task features not supported"):
            search_space_digest = SearchSpaceDigest(
                feature_names=feature_names,
                bounds=bounds,
                task_features=[0, 1],
            )
            with self.assertRaisesRegex(
                NotImplementedError, "Multiple task features are not supported"
            ):
                _extract_model_kwargs(
                    search_space_digest=search_space_digest,
                )

        with self.subTest("Task feature provided, fidelity and categorical not"):
            search_space_digest = SearchSpaceDigest(
                feature_names=feature_names,
                bounds=bounds,
                task_features=[1],
            )
            model_kwargs = _extract_model_kwargs(
                search_space_digest=search_space_digest,
            )
            self.assertSetEqual(set(model_kwargs.keys()), {"task_feature"})
            self.assertEqual(model_kwargs["task_feature"], 1)

        with self.subTest("No feature info provided"):
            search_space_digest = SearchSpaceDigest(
                feature_names=feature_names,
                bounds=bounds,
            )
            model_kwargs = _extract_model_kwargs(
                search_space_digest=search_space_digest,
            )
            self.assertEqual(len(model_kwargs.keys()), 0)

        with self.subTest("Fidelity and categorical features provided"):
            search_space_digest = SearchSpaceDigest(
                feature_names=feature_names,
                bounds=bounds,
                fidelity_features=[0],
                categorical_features=[1],
            )
            model_kwargs = _extract_model_kwargs(
                search_space_digest=search_space_digest,
            )
            self.assertSetEqual(
                set(model_kwargs.keys()), {"fidelity_features", "categorical_features"}
            )
            self.assertEqual(model_kwargs["fidelity_features"], [0])
            self.assertEqual(model_kwargs["categorical_features"], [1])


class SurrogateTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.device = torch.device("cpu")
        self.dtype = torch.float
        self.tkwargs = {"device": self.device, "dtype": self.dtype}
        (
            self.Xs,
            self.Ys,
            self.Yvars,
            self.bounds,
            _,
            self.feature_names,
            _,
        ) = get_torch_test_data(dtype=self.dtype)
        self.metric_names = ["metric"]
        self.training_data = [
            SupervisedDataset(
                X=self.Xs[0],
                Y=self.Ys[0],
                feature_names=self.feature_names,
                outcome_names=self.metric_names,
            )
        ]
        self.mll_class = ExactMarginalLogLikelihood
        self.search_space_digest = SearchSpaceDigest(
            feature_names=self.feature_names,
            bounds=self.bounds,
            target_values={1: 1.0},
        )
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
        self.ds2 = SupervisedDataset(
            X=2 * self.Xs[0],
            Y=2 * self.Ys[0],
            feature_names=self.feature_names,
            outcome_names=["m2"],
        )

    def _get_surrogate(
        self, botorch_model_class: Type[Model]
    ) -> Tuple[Surrogate, Dict[str, Any]]:
        if botorch_model_class is SaasFullyBayesianSingleTaskGP:
            mll_options = {"jit_compile": True}
        else:
            mll_options = None
        surrogate = Surrogate(
            botorch_model_class=botorch_model_class,
            mll_class=self.mll_class,
            mll_options=mll_options,
        )
        surrogate_kwargs = botorch_model_class.construct_inputs(self.training_data[0])
        return surrogate, surrogate_kwargs

    def test_init(self) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            self.assertEqual(surrogate.botorch_model_class, botorch_model_class)
            self.assertEqual(surrogate.mll_class, self.mll_class)
            self.assertTrue(surrogate.allow_batched_models)  # True by default

    def test_clone_reset(self) -> None:
        surrogate = self._get_surrogate(botorch_model_class=SingleTaskGP)[0]
        self.assertEqual(surrogate, surrogate.clone_reset())

    @patch(f"{UTILS_PATH}.fit_gpytorch_mll")
    def test_mll_options(self, _) -> None:
        mock_mll = MagicMock(self.mll_class)
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
            mll_class=mock_mll,
            mll_options={"some_option": "some_value"},
        )
        surrogate.fit(
            datasets=self.training_data,
            search_space_digest=self.search_space_digest,
            refit=self.refit,
        )
        self.assertEqual(mock_mll.call_args[1]["some_option"], "some_value")

    @fast_botorch_optimize
    def test_copy_options(self) -> None:
        training_data = self.training_data + [self.ds2]
        d = self.Xs[0].shape[-1]
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
            likelihood_class=GaussianLikelihood,
            likelihood_options={"noise_constraint": GreaterThan(1e-3)},
            mll_class=ExactMarginalLogLikelihood,
            covar_module_class=ScaleKernel,
            covar_module_options={"base_kernel": MaternKernel(ard_num_dims=d)},
            input_transform_classes=[Normalize],
            outcome_transform_classes=[Standardize],
            outcome_transform_options={"Standardize": {"m": 1}},
            allow_batched_models=False,
        )
        surrogate.fit(
            datasets=training_data,
            search_space_digest=self.search_space_digest,
            refit=True,
        )
        models = checked_cast(ModuleList, surrogate.model.models)

        model1_old_lengtscale = (
            models[1].covar_module.base_kernel.lengthscale.detach().clone()
        )
        # Change the lengthscales of one model and make sure the other isn't changed
        models[0].covar_module.base_kernel.lengthscale += 1
        self.assertTrue(
            torch.allclose(
                model1_old_lengtscale,
                models[1].covar_module.base_kernel.lengthscale,
            )
        )
        # Test the same thing with the likelihood noise constraint
        models[0].likelihood.noise_covar.raw_noise_constraint.lower_bound.fill_(1e-4)
        self.assertEqual(
            models[0].likelihood.noise_covar.raw_noise_constraint.lower_bound, 1e-4
        )
        self.assertEqual(
            models[1].likelihood.noise_covar.raw_noise_constraint.lower_bound, 1e-3
        )
        # Check input transform

        # bounds will be taken from the search space digest
        self.assertTrue(
            torch.allclose(
                models[0].input_transform.offset,
                torch.tensor([0, 1, 2], **self.tkwargs),
            )
        )
        self.assertTrue(
            torch.allclose(
                models[1].input_transform.offset,
                torch.tensor([0, 1, 2], **self.tkwargs),
            )
        )
        # Check outcome transform
        self.assertTrue(
            torch.allclose(
                models[0].outcome_transform.means, torch.tensor([3.5], **self.tkwargs)
            )
        )
        self.assertTrue(
            torch.allclose(
                models[1].outcome_transform.means, torch.tensor([7], **self.tkwargs)
            )
        )

    def test_botorch_transforms(self) -> None:
        # Successfully passing down the transforms
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
            outcome_transform_classes=[Standardize],
            input_transform_classes=[Normalize],
        )
        surrogate.fit(
            datasets=self.training_data,
            search_space_digest=self.search_space_digest,
            refit=self.refit,
        )
        botorch_model = surrogate.model
        self.assertIsInstance(botorch_model.input_transform, Normalize)
        self.assertIsInstance(botorch_model.outcome_transform, Standardize)
        self.assertEqual(botorch_model.outcome_transform._m, self.Ys[0].shape[-1])

        # Error handling if the model does not support transforms.
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGPWithDifferentConstructor,
            outcome_transform_classes=[Standardize],
            outcome_transform_options={"Standardize": {"m": self.Ys[0].shape[-1]}},
            input_transform_classes=[Normalize],
        )
        with self.assertRaisesRegex(UserInputError, "BoTorch model"):
            surrogate.fit(
                datasets=self.training_data,
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

    @fast_botorch_optimize
    def test_dtype_and_device_properties(self) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.fit(
                datasets=self.training_data,
                search_space_digest=self.search_space_digest,
            )
            self.assertEqual(self.dtype, surrogate.dtype)
            self.assertEqual(self.device, surrogate.device)

    @patch.object(SingleTaskGP, "__init__", return_value=None)
    @patch(f"{SURROGATE_PATH}.fit_botorch_model")
    def test_fit_model_reuse(self, mock_fit: Mock, mock_init: Mock) -> None:
        surrogate, _ = self._get_surrogate(botorch_model_class=SingleTaskGP)
        search_space_digest = SearchSpaceDigest(
            feature_names=self.feature_names,
            bounds=self.bounds,
        )
        surrogate.fit(
            datasets=self.training_data,
            search_space_digest=search_space_digest,
        )
        mock_fit.assert_called_once()
        mock_init.assert_called_once()
        key = tuple(self.training_data[0].outcome_names)
        submodel = surrogate._submodels[key]
        self.assertIs(surrogate._last_datasets[key], self.training_data[0])
        self.assertIs(surrogate._last_search_space_digest, search_space_digest)

        # Refit with same arguments.
        surrogate.fit(
            datasets=self.training_data,
            search_space_digest=search_space_digest,
        )
        # Still only called once -- i.e. not fitted again:
        mock_fit.assert_called_once()
        mock_init.assert_called_once()
        # Model is still the same object.
        self.assertIs(submodel, surrogate._submodels[key])

        # Change the search space digest.
        bounds = self.bounds.copy()
        bounds[0] = (999.0, 9999.0)
        search_space_digest = SearchSpaceDigest(
            feature_names=self.feature_names,
            bounds=bounds,
        )
        with patch(f"{SURROGATE_PATH}.logger.info") as mock_log:
            surrogate.fit(
                datasets=self.training_data,
                search_space_digest=search_space_digest,
            )
        mock_log.assert_called_once()
        self.assertIn(
            "Discarding all previously trained models", mock_log.call_args[0][0]
        )
        self.assertIsNot(submodel, surrogate._submodels[key])
        self.assertIs(surrogate._last_search_space_digest, search_space_digest)

    def test_construct_model(self) -> None:
        for botorch_model_class in (SaasFullyBayesianSingleTaskGP, SingleTaskGP):
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            with self.assertRaisesRegex(TypeError, "posterior"):
                # Base `Model` does not implement `posterior`, so instantiating it here
                # will fail.
                Surrogate()._construct_model(
                    dataset=self.training_data[0],
                    search_space_digest=self.search_space_digest,
                    botorch_model_class=Model,
                    state_dict=None,
                    refit=True,
                )
            with patch.object(
                botorch_model_class,
                "construct_inputs",
                wraps=botorch_model_class.construct_inputs,
            ) as mock_construct_inputs, patch.object(
                botorch_model_class, "__init__", return_value=None
            ) as mock_init, patch(
                f"{SURROGATE_PATH}.fit_botorch_model"
            ) as mock_fit:
                model = surrogate._construct_model(
                    dataset=self.training_data[0],
                    search_space_digest=self.search_space_digest,
                    botorch_model_class=botorch_model_class,
                    state_dict=None,
                    refit=True,
                )
            mock_init.assert_called_once()
            mock_fit.assert_called_once()
            call_kwargs = mock_init.call_args.kwargs
            self.assertTrue(torch.equal(call_kwargs["train_X"], self.Xs[0]))
            self.assertTrue(torch.equal(call_kwargs["train_Y"], self.Ys[0]))
            self.assertEqual(len(call_kwargs), 2)

            mock_construct_inputs.assert_called_with(
                training_data=self.training_data[0],
            )

            # Check that the model & dataset are cached.
            key = tuple(self.training_data[0].outcome_names)
            self.assertIs(model, surrogate._submodels[key])
            self.assertIs(self.training_data[0], surrogate._last_datasets[key])

            # Attempt to re-fit the same model with the same data.
            with patch(f"{SURROGATE_PATH}.fit_botorch_model") as mock_fit:
                new_model = surrogate._construct_model(
                    dataset=self.training_data[0],
                    search_space_digest=self.search_space_digest,
                    botorch_model_class=botorch_model_class,
                    state_dict=None,
                    refit=True,
                )
            mock_fit.assert_not_called()
            self.assertIs(new_model, model)

            # Model is re-fit if we change the model class.
            with patch(f"{SURROGATE_PATH}.fit_botorch_model") as mock_fit, patch(
                f"{SURROGATE_PATH}.logger.info"
            ) as mock_log:
                surrogate._construct_model(
                    dataset=self.training_data[0],
                    search_space_digest=self.search_space_digest,
                    botorch_model_class=SingleTaskGPWithDifferentConstructor,
                    state_dict=None,
                    refit=True,
                )
            mock_fit.assert_called_once()
            self.assertIn("model class for outcome(s)", mock_log.call_args[0][0])
            self.assertIsNot(surrogate._submodels[key], model)
            self.assertIsInstance(
                surrogate._submodels[key], SingleTaskGPWithDifferentConstructor
            )

    @fast_botorch_optimize
    def test_construct_custom_model(self) -> None:
        # Test error for unsupported covar_module and likelihood.
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGPWithDifferentConstructor,
            mll_class=self.mll_class,
            covar_module_class=RBFKernel,
            likelihood_class=FixedNoiseGaussianLikelihood,
        )
        with self.assertRaisesRegex(UserInputError, "does not support"):
            surrogate.fit(
                self.training_data,
                search_space_digest=self.search_space_digest,
            )
        # Pass custom options to a SingleTaskGP and make sure they are used
        noise_constraint = Interval(1e-6, 1e-1)
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
            mll_class=LeaveOneOutPseudoLikelihood,
            covar_module_class=RBFKernel,
            covar_module_options={"ard_num_dims": 3},
            likelihood_class=GaussianLikelihood,
            likelihood_options={"noise_constraint": noise_constraint},
        )
        surrogate.fit(
            self.training_data,
            search_space_digest=self.search_space_digest,
        )
        model = not_none(surrogate._model)
        self.assertEqual(type(model.likelihood), GaussianLikelihood)
        noise_constraint.eval()  # For the equality check.
        self.assertEqual(
            # Checking equality of __dict__'s since Interval does not define __eq__.
            model.likelihood.noise_covar.raw_noise_constraint.__dict__,
            noise_constraint.__dict__,
        )
        self.assertEqual(surrogate.mll_class, LeaveOneOutPseudoLikelihood)
        self.assertEqual(type(model.covar_module), RBFKernel)
        self.assertEqual(model.covar_module.ard_num_dims, 3)

    @fast_botorch_optimize
    @patch(f"{SURROGATE_PATH}.predict_from_model")
    def test_predict(self, mock_predict: Mock) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.fit(
                datasets=self.training_data,
                search_space_digest=self.search_space_digest,
            )
            surrogate.predict(X=self.Xs[0])
            mock_predict.assert_called_with(model=surrogate.model, X=self.Xs[0])

    @fast_botorch_optimize
    def test_best_in_sample_point(self) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.fit(
                datasets=self.training_data,
                search_space_digest=self.search_space_digest,
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
                    self.assertTrue(torch.equal(X, dataset.X))
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

    @fast_botorch_optimize
    @patch(f"{ACQUISITION_PATH}.Acquisition.__init__", return_value=None)
    @patch(
        f"{ACQUISITION_PATH}.Acquisition.optimize",
        return_value=(
            torch.tensor([[0.0]]),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
        ),
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
            surrogate.fit(
                datasets=self.training_data,
                search_space_digest=self.search_space_digest,
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
            self.assertTrue(torch.equal(acqf_value, torch.tensor(1.0)))

    def test_serialize_attributes_as_kwargs(self) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            expected = {
                k: v for k, v in surrogate.__dict__.items() if not k.startswith("_")
            }
            self.assertEqual(surrogate._serialize_attributes_as_kwargs(), expected)

    @fast_botorch_optimize
    def test_w_robust_digest(self) -> None:
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
        )
        # Error handling.
        with self.assertRaisesRegex(NotImplementedError, "Environmental variable"):
            robust_digest = RobustSearchSpaceDigest(
                environmental_variables=["a"],
                sample_param_perturbations=lambda: np.zeros((2, 2)),
            )
            surrogate.fit(
                datasets=self.training_data,
                search_space_digest=SearchSpaceDigest(
                    feature_names=self.search_space_digest.feature_names,
                    bounds=self.bounds,
                    task_features=self.search_space_digest.task_features,
                    robust_digest=robust_digest,
                ),
            )

        robust_digest = RobustSearchSpaceDigest(
            sample_param_perturbations=lambda: np.zeros((2, 2)),
            environmental_variables=[],
            multiplicative=False,
        )
        surrogate.input_transform_classes = [Normalize]
        with self.assertRaisesRegex(NotImplementedError, "input transforms"):
            surrogate.fit(
                datasets=self.training_data,
                search_space_digest=SearchSpaceDigest(
                    feature_names=self.search_space_digest.feature_names,
                    bounds=self.bounds,
                    task_features=self.search_space_digest.task_features,
                    robust_digest=robust_digest,
                ),
            )
        # Input perturbation is constructed.
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
        )
        surrogate.fit(
            datasets=self.training_data,
            search_space_digest=SearchSpaceDigest(
                feature_names=self.search_space_digest.feature_names,
                bounds=self.bounds,
                task_features=self.search_space_digest.task_features,
                robust_digest=robust_digest,
            ),
        )
        intf = checked_cast(InputPerturbation, surrogate.model.input_transform)
        self.assertIsInstance(intf, InputPerturbation)
        self.assertTrue(torch.equal(intf.perturbation_set, torch.zeros(2, 2)))

    def test_fit_mixed(self) -> None:
        # Test model construction with categorical variables.
        surrogate = Surrogate()
        search_space_digest = dataclasses.replace(
            self.search_space_digest,
            categorical_features=[0],
        )
        surrogate.fit(
            datasets=self.training_data,
            search_space_digest=search_space_digest,
        )
        self.assertIsInstance(surrogate.model, MixedSingleTaskGP)
        # _ignore_X_dims_scaling_check is the easiest way to check cat dims.
        self.assertEqual(surrogate.model._ignore_X_dims_scaling_check, [0])
        covar_module = checked_cast(Kernel, surrogate.model.covar_module)
        self.assertEqual(
            covar_module.kernels[0].base_kernel.kernels[1].active_dims.tolist(),
            [0],
        )
        self.assertEqual(
            covar_module.kernels[0].base_kernel.kernels[0].active_dims.tolist(),
            [1, 2],
        )
        self.assertEqual(
            covar_module.kernels[1].base_kernel.kernels[1].active_dims.tolist(),
            [0],
        )
        self.assertEqual(
            covar_module.kernels[1].base_kernel.kernels[0].active_dims.tolist(),
            [1, 2],
        )
        # With modellist.
        training_data = self.training_data + [self.ds2]
        surrogate = Surrogate(allow_batched_models=False)
        surrogate.fit(
            datasets=training_data,
            search_space_digest=search_space_digest,
        )
        self.assertIsInstance(surrogate.model, ModelListGP)
        self.assertTrue(
            all(
                isinstance(m, MixedSingleTaskGP)
                for m in checked_cast(ModelListGP, surrogate.model).models
            )
        )


class SurrogateWithModelListTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.outcomes = ["outcome_1", "outcome_2"]
        self.mll_class = ExactMarginalLogLikelihood
        self.dtype = torch.double
        self.task_features = [0]
        Xs1, Ys1, Yvars1, self.bounds, _, self.feature_names, _ = get_torch_test_data(
            dtype=self.dtype, task_features=self.task_features, offset=1.0
        )
        self.single_task_search_space_digest = SearchSpaceDigest(
            feature_names=self.feature_names,
            bounds=self.bounds,
        )
        self.multi_task_search_space_digest = SearchSpaceDigest(
            feature_names=self.feature_names,
            bounds=self.bounds,
            task_features=self.task_features,
        )
        self.ds1 = SupervisedDataset(
            X=Xs1[0],
            Y=Ys1[0],
            Yvar=Yvars1[0],
            feature_names=self.feature_names,
            outcome_names=self.outcomes[:1],
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = get_torch_test_data(
            dtype=self.dtype, task_features=self.task_features
        )
        ds2 = SupervisedDataset(
            X=Xs2[0],
            Y=Ys2[0],
            Yvar=Yvars2[0],
            feature_names=self.feature_names,
            outcome_names=self.outcomes[1:],
        )
        self.botorch_submodel_class_per_outcome = {
            self.outcomes[0]: choose_model_class(
                datasets=[self.ds1],
                search_space_digest=self.multi_task_search_space_digest,
            ),
            self.outcomes[1]: choose_model_class(
                datasets=[ds2], search_space_digest=self.multi_task_search_space_digest
            ),
        }
        self.botorch_model_class = MultiTaskGP
        for submodel_cls in self.botorch_submodel_class_per_outcome.values():
            self.assertEqual(submodel_cls, MultiTaskGP)
        self.ds3 = SupervisedDataset(
            X=Xs1[0],
            Y=Ys2[0],
            Yvar=Yvars2[0],
            feature_names=self.feature_names,
            outcome_names=self.outcomes[1:],
        )
        self.Xs = Xs1 + Xs2
        self.Ys = Ys1 + Ys2
        self.Yvars = Yvars1 + Yvars2
        self.fixed_noise_training_data = [self.ds1, ds2]
        self.supervised_training_data = [
            SupervisedDataset(
                X=ds.X,
                Y=ds.Y,
                feature_names=ds.feature_names,
                outcome_names=ds.outcome_names,
            )
            for ds in self.fixed_noise_training_data
        ]
        self.submodel_options_per_outcome = {
            RANK: 1,
        }
        self.surrogate = Surrogate(
            botorch_model_class=MultiTaskGP,
            mll_class=self.mll_class,
            model_options=self.submodel_options_per_outcome,
        )

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

    @patch(f"{SURROGATE_PATH}.fit_botorch_model")
    @patch.object(
        MultiTaskGP,
        "construct_inputs",
        wraps=MultiTaskGP.construct_inputs,
    )
    def test_construct_per_outcome_options(
        self, mock_MTGP_construct_inputs: Mock, mock_fit: Mock
    ) -> None:
        self.surrogate.model_options.update({"output_tasks": [2]})
        for fixed_noise in (False, True):
            mock_fit.reset_mock()
            mock_MTGP_construct_inputs.reset_mock()
            self.surrogate.fit(
                datasets=(
                    self.fixed_noise_training_data
                    if fixed_noise
                    else self.supervised_training_data
                ),
                search_space_digest=dataclasses.replace(
                    self.multi_task_search_space_digest,
                    task_features=self.task_features,
                ),
            )
            # Should construct inputs for MTGP twice.
            self.assertEqual(len(mock_MTGP_construct_inputs.call_args_list), 2)
            self.assertEqual(mock_fit.call_count, 2)
            # First construct inputs should be called for MTGP with training data #0.
            for idx in range(len(mock_MTGP_construct_inputs.call_args_list)):
                expected_training_data = SupervisedDataset(
                    X=self.Xs[idx],
                    Y=self.Ys[idx],
                    Yvar=self.Yvars[idx] if fixed_noise else None,
                    feature_names=["x1", "x2", "x3"],
                    outcome_names=[self.outcomes[idx]],
                )
                self.assertEqual(
                    # `call_args` is a tuple of (args, kwargs), and we check kwargs.
                    mock_MTGP_construct_inputs.call_args_list[idx][1],
                    {
                        "task_feature": self.task_features[0],
                        "training_data": expected_training_data,
                        "rank": 1,
                        "output_tasks": [2],
                    },
                )

    @patch(
        f"{CURRENT_PATH}.SaasFullyBayesianMultiTaskGP.load_state_dict",
        return_value=None,
    )
    @patch(
        f"{CURRENT_PATH}.SaasFullyBayesianSingleTaskGP.load_state_dict",
        return_value=None,
    )
    @patch(f"{CURRENT_PATH}.Model.load_state_dict", return_value=None)
    @patch(f"{CURRENT_PATH}.ExactMarginalLogLikelihood")
    @patch(f"{UTILS_PATH}.fit_gpytorch_mll")
    @patch(f"{UTILS_PATH}.fit_fully_bayesian_model_nuts")
    def test_fit(
        self,
        mock_fit_nuts: Mock,
        mock_fit_gpytorch: Mock,
        mock_MLL: Mock,
        mock_state_dict: Mock,
        mock_state_dict_saas: Mock,
        mock_state_dict_saas_mtgp: Mock,
    ) -> None:
        default_class = self.botorch_model_class
        surrogates = [
            Surrogate(
                botorch_model_class=default_class,
                mll_class=ExactMarginalLogLikelihood,
                # Check that empty lists also work fine.
                outcome_transform_classes=[],
                input_transform_classes=[],
            ),
            Surrogate(botorch_model_class=SaasFullyBayesianSingleTaskGP),
            Surrogate(botorch_model_class=SaasFullyBayesianMultiTaskGP),
            Surrogate(  # Batch model
                botorch_model_class=SingleTaskGP, mll_class=ExactMarginalLogLikelihood
            ),
            Surrogate(  # ModelListGP
                botorch_model_class=SingleTaskGP,
                mll_class=ExactMarginalLogLikelihood,
                allow_batched_models=False,
            ),
        ]

        for i, surrogate in enumerate(surrogates):
            # Reset mocks
            mock_state_dict.reset_mock()
            mock_MLL.reset_mock()
            mock_fit_gpytorch.reset_mock()
            mock_fit_nuts.reset_mock()

            # Checking that model is None before `fit` (and `construct`) calls.
            self.assertIsNone(surrogate._model)
            # Should instantiate mll and `fit_gpytorch_mll` when `state_dict`
            # is `None`.
            search_space_digest = (
                self.multi_task_search_space_digest
                # pyre-ignore[6]: Incompatible parameter type: In call
                # `issubclass`, for 1st positional argument, expected
                # `Type[typing.Any]` but got `Optional[Type[Model]]`.
                if issubclass(surrogate.botorch_model_class, MultiTaskGP)
                else self.single_task_search_space_digest
            )
            surrogate.fit(
                datasets=[self.ds1, self.ds3],
                search_space_digest=search_space_digest,
            )
            mock_state_dict.assert_not_called()
            if i == 0:
                self.assertEqual(mock_MLL.call_count, 2)
                self.assertEqual(mock_fit_gpytorch.call_count, 2)
                self.assertTrue(isinstance(surrogate.model, ModelListGP))
            elif i in [1, 2]:
                self.assertEqual(mock_MLL.call_count, 0)
                self.assertEqual(mock_fit_nuts.call_count, 2)
                self.assertTrue(isinstance(surrogate.model, ModelListGP))
            elif i == 3:
                self.assertEqual(mock_MLL.call_count, 1)
                self.assertEqual(mock_fit_gpytorch.call_count, 1)
                self.assertTrue(isinstance(surrogate.model, SingleTaskGP))
            elif i == 4:
                self.assertEqual(mock_MLL.call_count, 2)
                self.assertEqual(mock_fit_gpytorch.call_count, 2)
                self.assertTrue(isinstance(surrogate.model, ModelListGP))
            mock_MLL.reset_mock()
            mock_fit_gpytorch.reset_mock()
            mock_fit_nuts.reset_mock()

            # Should `load_state_dict` when `state_dict` is not `None`
            # and `refit` is `False`.
            state_dict = OrderedDict({"state_attribute": torch.ones(2)})
            surrogate._submodels = {}  # Prevent re-use of fitted model.
            surrogate.fit(
                datasets=[self.ds1, self.ds3],
                search_space_digest=search_space_digest,
                refit=False,
                state_dict=state_dict,
            )
            if i == 1:
                self.assertEqual(mock_state_dict_saas.call_count, 2)
                mock_state_dict_saas.reset_mock()
            elif i == 2:
                self.assertEqual(mock_state_dict_saas_mtgp.call_count, 2)
                mock_state_dict_saas_mtgp.reset_mock()
            elif i == 3:
                mock_state_dict.assert_called_once()
            else:
                self.assertEqual(mock_state_dict.call_count, 2)
            mock_state_dict.reset_mock()
            mock_MLL.assert_not_called()
            mock_fit_gpytorch.assert_not_called()
            mock_fit_nuts.assert_not_called()

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

    @fast_botorch_optimize
    def test_with_botorch_transforms(self) -> None:
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGPWithDifferentConstructor,
            mll_class=ExactMarginalLogLikelihood,
            input_transform_classes=[Normalize],
            input_transform_options={
                "Normalize": {"d": 3, "bounds": None, "indices": None}
            },
            outcome_transform_classes=[Standardize],
            outcome_transform_options={"Standardize": {"m": 1}},
        )
        with self.assertRaisesRegex(UserInputError, "The BoTorch model class"):
            surrogate.fit(
                datasets=self.supervised_training_data,
                search_space_digest=SearchSpaceDigest(
                    feature_names=self.feature_names,
                    bounds=self.bounds,
                    task_features=[],
                ),
            )
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
            mll_class=ExactMarginalLogLikelihood,
            input_transform_classes=[Normalize],
            input_transform_options={
                "Normalize": {"d": 3, "bounds": None, "indices": None}
            },
            outcome_transform_classes=[Standardize],
            outcome_transform_options={"Standardize": {"m": 1}},
        )
        surrogate.fit(
            datasets=self.supervised_training_data,
            search_space_digest=SearchSpaceDigest(
                feature_names=self.feature_names,
                bounds=self.bounds,
                task_features=[],
            ),
        )
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
            torch.allclose(
                models[0].input_transform.bounds,
                models[1].input_transform.bounds + 1.0,  # pyre-ignore
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

    @fast_botorch_optimize
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
                input_transform_classes=[Normalize],
                outcome_transform_classes=[Standardize],
                outcome_transform_options={"Standardize": {"m": 1}},
            )
            surrogate.fit(
                datasets=self.supervised_training_data,
                search_space_digest=SearchSpaceDigest(
                    feature_names=self.feature_names,
                    bounds=self.bounds,
                    task_features=[],
                ),
            )
            models = checked_cast(ModelListGP, surrogate._model).models
            self.assertEqual(len(models), 2)
            self.assertEqual(surrogate.mll_class, ExactMarginalLogLikelihood)
            # Make sure we properly copied the transforms.
            self.assertNotEqual(
                id(models[0].input_transform), id(models[1].input_transform)
            )
            self.assertNotEqual(
                id(models[0].outcome_transform), id(models[1].outcome_transform)
            )

            for m in models:
                self.assertEqual(type(m.likelihood), GaussianLikelihood)
                self.assertEqual(type(m.covar_module), MaternKernel)
                if submodel_covar_module_options:
                    self.assertEqual(m.covar_module.ard_num_dims, 3)
                else:
                    self.assertEqual(m.covar_module.ard_num_dims, None)
                m_noise_constraint = m.likelihood.noise_covar.raw_noise_constraint
                if submodel_likelihood_options:
                    self.assertEqual(type(m_noise_constraint), Interval)
                    self.assertEqual(
                        m_noise_constraint.lower_bound, noise_constraint.lower_bound
                    )
                    self.assertEqual(
                        m_noise_constraint.upper_bound, noise_constraint.upper_bound
                    )
                else:
                    self.assertEqual(type(m_noise_constraint), GreaterThan)
                    self.assertAlmostEqual(m_noise_constraint.lower_bound.item(), 1e-4)

    @fast_botorch_optimize
    def test_w_robust_digest(self) -> None:
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
        )
        # Error handling.
        with self.assertRaisesRegex(NotImplementedError, "Environmental variable"):
            surrogate.fit(
                datasets=self.supervised_training_data,
                search_space_digest=SearchSpaceDigest(
                    feature_names=self.feature_names,
                    bounds=self.bounds,
                    task_features=[],
                    robust_digest=RobustSearchSpaceDigest(
                        sample_param_perturbations=lambda: np.zeros((2, 2)),
                        environmental_variables=["a"],
                    ),
                ),
            )
        robust_digest = RobustSearchSpaceDigest(
            sample_param_perturbations=lambda: np.zeros((2, 2)),
            environmental_variables=[],
            multiplicative=False,
        )
        surrogate.input_transform_classes = [Normalize]
        with self.assertRaisesRegex(NotImplementedError, "input transforms"):
            surrogate.fit(
                datasets=self.supervised_training_data,
                search_space_digest=SearchSpaceDigest(
                    feature_names=self.feature_names,
                    bounds=self.bounds,
                    task_features=self.task_features,
                    robust_digest=robust_digest,
                ),
            )
        # Input perturbation is constructed.
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
        )
        surrogate.fit(
            datasets=self.supervised_training_data,
            search_space_digest=SearchSpaceDigest(
                feature_names=self.feature_names,
                bounds=self.bounds,
                task_features=[],
                robust_digest=robust_digest,
            ),
        )
        for m in surrogate.model.models:
            intf = checked_cast(InputPerturbation, m.input_transform)
            self.assertIsInstance(intf, InputPerturbation)
            self.assertTrue(torch.equal(intf.perturbation_set, torch.zeros(2, 2)))
