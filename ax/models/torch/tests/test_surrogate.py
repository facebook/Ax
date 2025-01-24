#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
import math
import warnings
from contextlib import ExitStack
from copy import copy
from itertools import product
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
from ax.core.search_space import RobustSearchSpaceDigest, SearchSpaceDigest
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.models.model_utils import best_in_sample_point
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.kernels import ScaleMaternKernel
from ax.models.torch.botorch_modular.surrogate import (
    _construct_default_input_transforms,
    _construct_specified_input_transforms,
    _extract_model_kwargs,
    _make_botorch_input_transform,
    submodel_input_constructor,
    Surrogate,
    SurrogateSpec,
)
from ax.models.torch.botorch_modular.utils import (
    choose_model_class,
    fit_botorch_model,
    ModelConfig,
)
from ax.models.torch.utils import predict_from_model
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.testutils import TestCase
from ax.utils.stats.model_fit_stats import DIAGNOSTIC_FNS
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.torch_stubs import get_torch_test_data
from ax.utils.testing.utils import generic_equals
from botorch.models import ModelListGP, SaasFullyBayesianSingleTaskGP, SingleTaskGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.model import Model, ModelList  # noqa: F401 -- used in Mocks.
from botorch.models.multitask import MultiTaskGP
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputPerturbation,
    Log10,
    Normalize,
)
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.utils.datasets import MultiTaskDataset, SupervisedDataset
from botorch.utils.transforms import standardize
from botorch.utils.types import DEFAULT
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import Kernel, LinearKernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood
from pyre_extensions import assert_is_instance, none_throws
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


class SurrogateInputConstructorsTest(TestCase):
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

    def test_surrogate_spec_from_deprecated_args(self) -> None:
        botorch_model_class = MultiTaskGP
        botorch_model_kwargs = {"task_feature": 1}
        outcome_transform_classes: list[type[OutcomeTransform]] = [Standardize]
        outcome_transform_options = {"Standardize": {"m": 1}}
        covar_module_class = ScaleKernel
        covar_module_options = {"base_kernel": MaternKernel(ard_num_dims=2)}
        likelihood_class = GaussianLikelihood
        likelihood_options = {"noise_constraint": None}
        with self.assertWarnsRegex(DeprecationWarning, "model_config"):
            spec = SurrogateSpec(
                botorch_model_class=botorch_model_class,
                botorch_model_kwargs=botorch_model_kwargs,
                outcome_transform_classes=outcome_transform_classes,
                outcome_transform_options=outcome_transform_options,
                covar_module_class=covar_module_class,
                covar_module_kwargs=covar_module_options,
                likelihood_class=likelihood_class,
                likelihood_kwargs=likelihood_options,
            )
        expected_spec = SurrogateSpec(
            model_configs=[
                ModelConfig(
                    botorch_model_class=botorch_model_class,
                    model_options=botorch_model_kwargs,
                    outcome_transform_classes=outcome_transform_classes,
                    outcome_transform_options=outcome_transform_options,
                    covar_module_class=covar_module_class,
                    covar_module_options=covar_module_options,
                    likelihood_class=likelihood_class,
                    likelihood_options=likelihood_options,
                    input_transform_classes=DEFAULT,
                    name="from deprecated args",
                )
            ],
        )
        self.assertEqual(spec, expected_spec)
        # With input transforms as None to match legacy default.
        spec = SurrogateSpec(input_transform_classes=None)
        expected_spec = SurrogateSpec(
            model_configs=[
                ModelConfig(input_transform_classes=None, name="from deprecated args")
            ]
        )
        self.assertEqual(spec, expected_spec)

    def test__make_botorch_input_transform(self) -> None:
        feature_names = ["a", "b"]
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        search_space_digest = SearchSpaceDigest(
            feature_names=feature_names,
            bounds=bounds,
        )
        dataset = SupervisedDataset(
            X=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            Y=torch.tensor([[1.0], [2.0]]),
            feature_names=feature_names,
            outcome_names=["metric"],
        )

        with self.subTest("Empty list of specified input transforms"):
            with patch(
                f"{SURROGATE_PATH}._construct_specified_input_transforms",
                wraps=_construct_specified_input_transforms,
            ) as mock_construct_specified_input_transforms:
                transform = _make_botorch_input_transform(
                    input_transform_classes=[],
                    input_transform_options={},
                    search_space_digest=search_space_digest,
                    dataset=dataset,
                )
            mock_construct_specified_input_transforms.assert_called_once_with(
                input_transform_classes=[],
                input_transform_options={},
                search_space_digest=search_space_digest,
                dataset=dataset,
            )
            self.assertIsNone(transform)

        with self.subTest("Empty set of default transforms"):
            with patch(
                f"{SURROGATE_PATH}._construct_default_input_transforms",
                wraps=_construct_default_input_transforms,
            ) as mock_construct_default_input_transforms:
                transform = _make_botorch_input_transform(
                    input_transform_classes=DEFAULT,
                    input_transform_options={},
                    search_space_digest=search_space_digest,
                    dataset=dataset,
                )
            mock_construct_default_input_transforms.assert_called_once_with(
                search_space_digest=search_space_digest,
                dataset=dataset,
            )
            self.assertIsNone(transform)

        with self.subTest("Multiple specified transforms"):
            transform = _make_botorch_input_transform(
                input_transform_classes=[Normalize, Log10],
                input_transform_options={"Log10": {"indices": [0]}},
                search_space_digest=search_space_digest,
                dataset=dataset,
            )
            transform = assert_is_instance(transform, ChainedInputTransform)
            tf_values = list(transform.values())
            self.assertEqual(len(tf_values), 2)
            self.assertIsInstance(tf_values[0], Normalize)
            self.assertIsInstance(tf_values[1], Log10)
            self.assertEqual(tf_values[1].indices.tolist(), [0])

        bounds = [(1.0, 5.0), (2.0, 10.0)]
        search_space_digest = SearchSpaceDigest(
            feature_names=feature_names,
            bounds=bounds,
            task_features=[1],
        )
        with self.subTest("Default Normalize transform"):
            transform = _make_botorch_input_transform(
                input_transform_classes=DEFAULT,
                input_transform_options={},
                search_space_digest=search_space_digest,
                dataset=dataset,
            )
            transform = assert_is_instance(transform, Normalize)
            self.assertEqual(transform.indices.tolist(), [0])
            self.assertEqual(transform.bounds.tolist(), [[1.0], [5.0]])


class SurrogateTest(TestCase):
    def setUp(self, cuda: bool = False) -> None:
        super().setUp()
        self.device = torch.device("cuda" if cuda else "cpu")
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
        ) = get_torch_test_data(dtype=self.dtype, cuda=cuda)
        self.metric_names = ["metric"]
        self.training_data = [
            SupervisedDataset(
                X=self.Xs[0],
                # Note: using 1d Y does not match the 2d TorchOptConfig
                Y=self.Ys[0],
                feature_names=self.feature_names,
                outcome_names=self.metric_names,
            )
        ]
        self.Ys_standardized = [standardize(Y) for Y in self.Ys]
        self.training_data_standardized = [
            SupervisedDataset(
                X=self.Xs[0],
                Y=self.Ys_standardized[0],
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
        self.objective_weights = torch.tensor([-1.0, 1.0], **self.tkwargs)
        # Note: these only work with 1 outcome
        self.outcome_constraints = (
            torch.tensor([[1.0]], **self.tkwargs),
            torch.tensor([[0.5]], **self.tkwargs),
        )
        self.linear_constraints = (
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], **self.tkwargs),
            torch.tensor([[0.5], [1.0]], **self.tkwargs),
        )
        self.options = {}
        self.torch_opt_config = TorchOptConfig(
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
        )
        self.ds2 = SupervisedDataset(
            # pyre-fixme[6]: For 1st argument expected `Union[BotorchContainer,
            #  Tensor]` but got `int`.
            X=2 * self.Xs[0],
            # pyre-fixme[6]: For 2nd argument expected `Union[BotorchContainer,
            #  Tensor]` but got `int`.
            Y=2 * self.Ys[0],
            feature_names=self.feature_names,
            outcome_names=["m2"],
        )

    def _get_surrogate(
        self,
        botorch_model_class: type[Model],
        use_outcome_transform: bool = True,
        n_outcomes: int = 1,
    ) -> tuple[Surrogate, dict[str, Any]]:
        if botorch_model_class is SaasFullyBayesianSingleTaskGP:
            mll_options = {"jit_compile": True}
        else:
            mll_options = None

        if use_outcome_transform:
            outcome_transform_classes: list[type[OutcomeTransform]] = [Standardize]
            outcome_transform_options = {"Standardize": {"m": n_outcomes}}
        else:
            outcome_transform_classes = None
            outcome_transform_options = None

        surrogate = Surrogate(
            botorch_model_class=botorch_model_class,
            mll_class=self.mll_class,
            mll_options=mll_options,
            outcome_transform_classes=outcome_transform_classes,
            outcome_transform_options=outcome_transform_options,
        )
        surrogate_kwargs = botorch_model_class.construct_inputs(self.training_data[0])
        return surrogate, surrogate_kwargs

    def test_init(self) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            self.assertEqual(
                surrogate.surrogate_spec.model_configs[0].botorch_model_class,
                botorch_model_class,
            )
            self.assertEqual(
                surrogate.surrogate_spec.model_configs[0].mll_class, self.mll_class
            )
            self.assertTrue(
                surrogate.surrogate_spec.allow_batched_models
            )  # True by default

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

    @mock_botorch_optimize
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
        models = assert_is_instance(surrogate.model.models, ModuleList)

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
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `_m`.
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

    @mock_botorch_optimize
    def test_dtype_and_device_properties(self) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.fit(
                datasets=self.training_data,
                search_space_digest=self.search_space_digest,
            )
            self.assertEqual(self.dtype, surrogate.dtype)
            self.assertEqual(self.device, surrogate.device)

    @mock_botorch_optimize
    @patch(
        f"{SURROGATE_PATH}.submodel_input_constructor",
        wraps=submodel_input_constructor,
    )
    @patch(f"{SURROGATE_PATH}.fit_botorch_model", wraps=fit_botorch_model)
    def test_fit_model_reuse(self, mock_fit: Mock, mock_constructor: Mock) -> None:
        surrogate, _ = self._get_surrogate(
            botorch_model_class=SingleTaskGP, use_outcome_transform=False
        )
        search_space_digest = SearchSpaceDigest(
            feature_names=self.feature_names,
            bounds=self.bounds,
        )
        surrogate.fit(
            datasets=self.training_data,
            search_space_digest=search_space_digest,
        )
        mock_fit.assert_called_once()
        mock_constructor.assert_called_once()
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
        mock_constructor.assert_called_once()
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
            # Don't use an outcome transform here because the
            # botorch_model_class will change to one that is not compatible with
            # outcome transforms below
            surrogate, _ = self._get_surrogate(
                botorch_model_class=botorch_model_class, use_outcome_transform=False
            )
            with patch.object(
                botorch_model_class,
                "construct_inputs",
                wraps=botorch_model_class.construct_inputs,
            ) as mock_construct_inputs, patch.object(
                botorch_model_class, "__init__", return_value=None, autospec=True
            ) as mock_init, patch(f"{SURROGATE_PATH}.fit_botorch_model") as mock_fit:
                model = surrogate._construct_model(
                    dataset=self.training_data[0],
                    search_space_digest=self.search_space_digest,
                    model_config=surrogate.surrogate_spec.model_configs[0],
                    default_botorch_model_class=botorch_model_class,
                    state_dict=None,
                    refit=True,
                )
            mock_init.assert_called_once()
            mock_fit.assert_called_once()
            call_kwargs = mock_init.call_args.kwargs
            self.assertTrue(torch.equal(call_kwargs["train_X"], self.Xs[0]))
            self.assertTrue(torch.equal(call_kwargs["train_Y"], self.Ys[0]))
            self.assertIsInstance(call_kwargs["input_transform"], Normalize)
            self.assertIsNone(call_kwargs["outcome_transform"])
            self.assertEqual(len(call_kwargs), 4)

            mock_construct_inputs.assert_called_with(
                training_data=self.training_data[0],
            )

            # Cache the model & dataset as we would in `Surrogate.fit``.
            outcomes = self.training_data[0].outcome_names
            key = tuple(outcomes)
            surrogate._submodels[key] = model
            surrogate._last_datasets[key] = self.training_data[0]
            surrogate.metric_to_best_model_config[outcomes[0]] = (
                surrogate.surrogate_spec.model_configs[0]
            )

            # Attempt to re-fit the same model with the same data.
            with patch(f"{SURROGATE_PATH}.fit_botorch_model") as mock_fit:
                new_model = surrogate._construct_model(
                    dataset=self.training_data[0],
                    search_space_digest=self.search_space_digest,
                    model_config=surrogate.surrogate_spec.model_configs[0],
                    default_botorch_model_class=botorch_model_class,
                    state_dict=None,
                    refit=True,
                )
            mock_fit.assert_not_called()
            self.assertIs(new_model, model)

            # Model is not re-fit if we change the model config.
            # The reason is that we cache the best model config.
            # We only reset the best model config and cached models
            # if the search space digest changes
            with patch(f"{SURROGATE_PATH}.fit_botorch_model") as mock_fit:
                model = surrogate._construct_model(
                    dataset=self.training_data[0],
                    search_space_digest=self.search_space_digest,
                    model_config=ModelConfig(
                        botorch_model_class=SingleTaskGPWithDifferentConstructor
                    ),
                    default_botorch_model_class=SingleTaskGP,
                    state_dict=None,
                    refit=True,
                )
            mock_fit.assert_not_called()

            # Model is not re-fit if we change the model class.
            with patch(f"{SURROGATE_PATH}.fit_botorch_model") as mock_fit:
                model = surrogate._construct_model(
                    dataset=self.training_data[0],
                    search_space_digest=SearchSpaceDigest(
                        feature_names=self.feature_names,
                        bounds=self.bounds,
                        target_values={1: 2.0},
                    ),
                    model_config=ModelConfig(),
                    default_botorch_model_class=SingleTaskGP,
                    state_dict=None,
                    refit=True,
                )
            mock_fit.assert_not_called()

    @mock_botorch_optimize
    def test_construct_custom_model(self, use_model_config: bool = False) -> None:
        # Test error for unsupported covar_module and likelihood.
        model_config_kwargs: dict[str, Any] = {
            "botorch_model_class": SingleTaskGPWithDifferentConstructor,
            "mll_class": self.mll_class,
            "covar_module_class": RBFKernel,
            "likelihood_class": FixedNoiseGaussianLikelihood,
        }
        if use_model_config:
            surrogate = Surrogate(
                surrogate_spec=SurrogateSpec(
                    model_configs=[ModelConfig(**model_config_kwargs)]
                )
            )
        else:
            surrogate = Surrogate(**model_config_kwargs)
        with self.assertRaisesRegex(UserInputError, "does not support"):
            surrogate.fit(
                self.training_data,
                search_space_digest=self.search_space_digest,
            )
        # Pass custom options to a SingleTaskGP and make sure they are used
        noise_constraint = Interval(1e-6, 1e-1)
        model_config_kwargs = {
            "botorch_model_class": SingleTaskGP,
            "mll_class": LeaveOneOutPseudoLikelihood,
            "covar_module_class": RBFKernel,
            "covar_module_options": {"ard_num_dims": 3},
            "likelihood_class": GaussianLikelihood,
            "likelihood_options": {"noise_constraint": noise_constraint},
        }
        if use_model_config:
            surrogate = Surrogate(
                surrogate_spec=SurrogateSpec(
                    model_configs=[ModelConfig(**model_config_kwargs)]
                )
            )
        else:
            surrogate = Surrogate(**model_config_kwargs)
        surrogate.fit(
            self.training_data,
            search_space_digest=self.search_space_digest,
        )
        model = none_throws(surrogate._model)
        self.assertEqual(type(model.likelihood), GaussianLikelihood)
        noise_constraint.eval()  # For the equality check.
        self.assertEqual(
            # Checking equality of __dict__'s since Interval does not define __eq__.
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `noise_covar`.
            model.likelihood.noise_covar.raw_noise_constraint.__dict__,
            noise_constraint.__dict__,
        )
        self.assertEqual(
            surrogate.surrogate_spec.model_configs[0].mll_class,
            LeaveOneOutPseudoLikelihood,
        )
        self.assertEqual(type(model.covar_module), RBFKernel)
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
        #  `ard_num_dims`.
        self.assertEqual(model.covar_module.ard_num_dims, 3)

    def test_construct_custom_model_with_config(self) -> None:
        self.test_construct_custom_model(use_model_config=True)

    def test_construct_model_with_metric_to_model_configs(self) -> None:
        surrogate = Surrogate(
            surrogate_spec=SurrogateSpec(
                metric_to_model_configs={
                    "metric": [ModelConfig()],
                    "metric2": [ModelConfig(covar_module_class=ScaleMaternKernel)],
                },
                model_configs=[ModelConfig(covar_module_class=LinearKernel)],
            )
        )
        training_data = self.training_data + [
            SupervisedDataset(
                X=self.Xs[0],
                # Note: using 1d Y does not match the 2d TorchOptConfig
                Y=self.Ys[0],
                feature_names=self.feature_names,
                outcome_names=[f"metric{i}"],
            )
            for i in range(2, 5)
        ]
        surrogate.fit(
            datasets=training_data, search_space_digest=self.search_space_digest
        )
        # test model follows metric_to_model_configs for
        # first two metrics
        self.assertIsInstance(surrogate.model, ModelListGP)
        submodels = surrogate.model.models
        # pyre-fixme[6]: For 1st argument expected
        #  `pyre_extensions.PyreReadOnly[Sized]` but got `Union[Tensor, Module]`.
        self.assertEqual(len(submodels), 4)
        # pyre-fixme[29]: `Union[(self: Tensor) -> Any, Tensor, Module]` is not a
        #  function.
        for m in submodels:
            self.assertIsInstance(m, SingleTaskGP)
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.assertIsInstance(surrogate.model.models[1].covar_module, ScaleKernel)
        self.assertIsInstance(
            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[An...
            surrogate.model.models[1].covar_module.base_kernel,
            MaternKernel,
        )
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.assertIsInstance(surrogate.model.models[0].covar_module, RBFKernel)
        # test model use model_configs for the third metric
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.assertIsInstance(surrogate.model.models[2].covar_module, LinearKernel)

    @mock_botorch_optimize
    @patch(
        "ax.models.torch.botorch_modular.surrogate.DIAGNOSTIC_FNS", wraps=DIAGNOSTIC_FNS
    )
    def test_fit_multiple_model_configs(self, mock_diag_dict: Mock) -> None:
        # These mocks are used because we control which kernel is selected by
        # changing the values of model diagnostics
        mse_side_effect = [0.2, 0.1]
        ll_side_effect = [0.3, 0.05]
        mock_mse = Mock()  # this should select linear kernel
        mock_ll = Mock()  # this should select rbf kernel
        d = {"MSE": mock_mse, "Log likelihood": mock_ll}
        mock_diag_dict.__getitem__.side_effect = d.__getitem__
        base_model_configs = [
            ModelConfig(),
            ModelConfig(covar_module_class=LinearKernel),
        ]
        for eval_criterion, use_per_metric_overrides, multitask in product(
            ("MSE", "Log likelihood"), (False, True), (False, True)
        ):
            if eval_criterion == "MSE":
                mock_diag_fn = mock_mse
                mock_mse.side_effect = mse_side_effect
            else:
                mock_diag_fn = mock_ll
                mock_ll.side_effect = ll_side_effect
            mock_diag_fn.reset_mock()
            with self.subTest(
                eval_criterion=eval_criterion,
                use_per_metric_model_overrides=use_per_metric_overrides,
            ):
                # this will do model selection over the two model configs
                # that are either specified via model_configs or
                # metric_to_model_configs
                if use_per_metric_overrides:
                    metric_to_model_configs = {"metric": base_model_configs}
                    model_configs = [
                        ModelConfig(covar_module_class=MaternKernel)
                    ]  # this should be overridden
                else:
                    model_configs = base_model_configs
                    metric_to_model_configs = {}
                surrogate = Surrogate(
                    surrogate_spec=SurrogateSpec(
                        model_configs=model_configs,
                        metric_to_model_configs=metric_to_model_configs,
                        eval_criterion=eval_criterion,
                    )
                )
                if multitask:
                    dataset = MultiTaskDataset(
                        datasets=[self.training_data_standardized[0], self.ds2],
                        target_outcome_name="metric",
                    )
                    search_space_digest = dataclasses.replace(
                        self.search_space_digest,
                        target_values={-1: 0.0},
                        task_features=[-1],
                    )
                else:
                    dataset = self.training_data_standardized[0]
                    search_space_digest = self.search_space_digest
                with patch.object(
                    surrogate, "model_selection", wraps=surrogate.model_selection
                ) as mock_model_selection, patch.object(
                    surrogate, "cross_validate", wraps=surrogate.cross_validate
                ) as mock_cross_validate, patch.object(
                    surrogate, "_construct_model", wraps=surrogate._construct_model
                ) as mock_construct_model, warnings.catch_warnings(record=True) as ws:
                    # Disable default warning filtering in tests.
                    warnings.filterwarnings("always")
                    surrogate.fit(
                        [dataset],
                        search_space_digest=search_space_digest,
                    )

                # Make sure that no input data standardization warnings are raised.
                self.assertFalse(any("standardized" in str(w) for w in ws))

                # Check that the methods were called with the correct args.
                mock_model_selection.assert_called_once_with(
                    dataset=dataset,
                    model_configs=base_model_configs,
                    default_botorch_model_class=MultiTaskGP
                    if multitask
                    else SingleTaskGP,
                    search_space_digest=search_space_digest,
                    candidate_metadata=None,
                )
                self.assertEqual(mock_cross_validate.call_count, 2)
                expected_call_kwargs: dict[
                    str,
                    SupervisedDataset
                    | type[Model]
                    | SearchSpaceDigest
                    | bool
                    | ModelConfig,
                ] = {
                    "dataset": dataset,
                    "default_botorch_model_class": MultiTaskGP
                    if multitask
                    else SingleTaskGP,
                    "search_space_digest": search_space_digest,
                }
                # check that each call to cross_validate uses the correct
                # model config.
                for i in (0, 1):
                    expected_call_kwargs["model_config"] = base_model_configs[i]
                    call_kwargs = mock_cross_validate.mock_calls[i].kwargs
                    for k, v in expected_call_kwargs.items():
                        self.assertEqual(call_kwargs[k], v)
                    self.assertIsNotNone(call_kwargs["state_dict"])
                # each of two model configs should be fit once to all data, then
                # construct data should be called twice for each in cross_validate
                self.assertEqual(mock_construct_model.call_count, 6)
                if multitask:
                    target_dataset = self.training_data[0]
                    calls = mock_construct_model.mock_calls
                    expected_X = torch.cat(
                        [
                            torch.cat(
                                [
                                    target_dataset.X,
                                    torch.zeros(2, 1, **self.tkwargs),
                                ],
                                dim=-1,
                            ),
                            torch.cat(
                                [self.ds2.X, torch.ones(2, 1, **self.tkwargs)],
                                dim=-1,
                            ),
                        ],
                        dim=0,
                    )
                    # check that only target data is used for evaluation
                    mask = torch.ones(4, dtype=torch.bool, device=self.device)
                    loo_idx = 0
                    for i in range(6):
                        # If i in (0,3) then all data is used.
                        # If i in (1,4) then the first data point from the
                        # target data is excluded.
                        # Otherwise the second data point from the target
                        # data is excluded.
                        if i not in (0, 3):
                            loo_idx = (i - (4 if i > 3 else 1)) % 2
                            mask[loo_idx] = 0
                        self.assertTrue(
                            torch.equal(
                                calls[i].kwargs["dataset"].X,
                                expected_X[mask],
                            )
                        )
                        if i not in (0, 3):
                            mask[loo_idx] = 1

                    self.assertEqual(mock_diag_fn.call_count, 2)
                    model = none_throws(surrogate._model)
                    self.assertIsInstance(
                        model.covar_module,
                        LinearKernel if eval_criterion == "MSE" else RBFKernel,
                    )

    def test_fit_multiple_model_configs_cuda(self) -> None:
        if torch.cuda.is_available():
            self.setUp(cuda=True)
            self.test_fit_multiple_model_configs()

    def test_cross_validate_error_for_heterogeneous_datasets(self) -> None:
        # self.ds2.outcome_names[0] = "metric"
        new_feature_names = copy(self.ds2.feature_names)
        new_feature_names[-1] = "new_feature"
        self.ds2.feature_names = new_feature_names
        dataset = MultiTaskDataset(
            datasets=[self.training_data[0], self.ds2], target_outcome_name="metric"
        )
        surrogate = Surrogate(
            surrogate_spec=SurrogateSpec(
                model_configs=[
                    ModelConfig(),
                    ModelConfig(covar_module_class=ScaleMaternKernel),
                ],
            )
        )
        feature_names = self.feature_names + ["new_feature"]
        ssd = SearchSpaceDigest(
            feature_names=feature_names,
            bounds=self.bounds + self.bounds[-1:],
            target_values={1: 1.0},
        )
        with self.assertRaisesRegex(
            UnsupportedError,
            "Model selection is not supported for datasets with heterogeneous "
            "features.",
        ):
            surrogate.fit(datasets=[dataset], search_space_digest=ssd)

    @mock_botorch_optimize
    @patch(
        "ax.models.torch.botorch_modular.surrogate.DIAGNOSTIC_FNS", wraps=DIAGNOSTIC_FNS
    )
    def test_fit_model_selection_metric_to_model_configs_multiple_metrics(
        self, mock_diag_dict: Mock
    ) -> None:
        # test that the correct model configs are used for each metric.
        # For the first metric (named "metric") the model configs from
        # metric_to_model_configs should be used. For the second metric,
        # the model configs from model_configs should be used.

        # The rank correlation here will lead to an RBFKernel being
        # selected for metric "m2" and a MaternKernel being selected
        # for metric "metric"
        mock_rc = Mock(side_effect=[0.1, 0.2, 0.2, 0.1])
        d = {"Rank correlation": mock_rc}
        mock_diag_dict.__getitem__.side_effect = d.__getitem__

        model_configs = [
            ModelConfig(),
            ModelConfig(covar_module_class=LinearKernel),
        ]
        metric_to_model_configs = {
            "metric": [
                ModelConfig(covar_module_class=ScaleMaternKernel),
                ModelConfig(covar_module_class=MaternKernel),
            ]
        }
        surrogate = Surrogate(
            surrogate_spec=SurrogateSpec(
                model_configs=model_configs,
                metric_to_model_configs=metric_to_model_configs,
            )
        )
        training_data = self.training_data + [self.ds2]
        with patch.object(
            surrogate, "model_selection", wraps=surrogate.model_selection
        ) as mock_model_selection, patch.object(
            surrogate, "cross_validate", wraps=surrogate.cross_validate
        ) as mock_cross_validate:
            surrogate.fit(
                datasets=training_data,
                search_space_digest=self.search_space_digest,
            )
            self.assertEqual(mock_model_selection.call_count, 2)
            expected_model_selection_kwargs: dict[
                str,
                type[SingleTaskGP]
                | SearchSpaceDigest
                | SupervisedDataset
                | list[ModelConfig]
                | None,
            ] = {
                "default_botorch_model_class": SingleTaskGP,
                "search_space_digest": self.search_space_digest,
                "candidate_metadata": None,
            }
            self.assertEqual(mock_cross_validate.call_count, 4)
            expected_cross_validate_kwargs: dict[
                str,
                type[SingleTaskGP]
                | SearchSpaceDigest
                | bool
                | SupervisedDataset
                | list[ModelConfig],
            ] = {
                "default_botorch_model_class": SingleTaskGP,
                "search_space_digest": self.search_space_digest,
            }
            for i in (0, 1):
                expected_model_selection_kwargs["dataset"] = training_data[i]
                model_configs_for_metric = (
                    metric_to_model_configs["metric"] if i == 0 else model_configs
                )
                expected_model_selection_kwargs["model_configs"] = (
                    model_configs_for_metric
                )

                call_kwargs = mock_model_selection.mock_calls[i].kwargs
                for k, v in expected_model_selection_kwargs.items():
                    self.assertEqual(call_kwargs[k], v)
                expected_cross_validate_kwargs["dataset"] = training_data[i]
                # check that each call to cross_validate uses the correct
                # model config.
                for j in (0, 1):
                    expected_cross_validate_kwargs["model_config"] = (
                        # pyre-ignore [6]
                        model_configs_for_metric[j]
                    )
                    call_kwargs = mock_cross_validate.mock_calls[2 * i + j].kwargs
                    for k, v in expected_cross_validate_kwargs.items():
                        self.assertEqual(call_kwargs[k], v)
                    self.assertIsNotNone(call_kwargs["state_dict"])
        self.assertEqual(mock_rc.call_count, 4)
        model = none_throws(surrogate._model)
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.assertIsInstance(model.models[0].covar_module, MaternKernel)
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.assertIsInstance(model.models[1].covar_module, RBFKernel)

    def test_exception_for_multiple_model_configs_and_multioutcome_dataset(
        self,
    ) -> None:
        surrogate = Surrogate(
            surrogate_spec=SurrogateSpec(
                model_configs=[
                    ModelConfig(),
                    ModelConfig(covar_module_class=LinearKernel),
                ]
            )
        )
        td = self.training_data[0]
        dataset = SupervisedDataset(
            X=torch.cat([td.X, self.ds2.X], dim=-1),
            Y=torch.cat([td.Y, self.ds2.Y], dim=-1),
            outcome_names=td.outcome_names + self.ds2.outcome_names,
            feature_names=td.feature_names + self.ds2.feature_names,
        )
        msg = (
            "Multiple model configs are not supported with datasets that contain "
            "multiple outcomes. Each dataset must contain only one outcome."
        )
        with self.assertRaisesRegex(UnsupportedError, msg):
            surrogate.fit(
                datasets=[dataset], search_space_digest=self.search_space_digest
            )

    @mock_botorch_optimize
    def test_predict(self) -> None:
        for botorch_model_class, use_posterior_predictive in product(
            (SaasFullyBayesianSingleTaskGP, SingleTaskGP), (True, False)
        ):
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.fit(
                datasets=self.training_data,
                search_space_digest=self.search_space_digest,
            )
            with patch(
                f"{SURROGATE_PATH}.predict_from_model", wraps=predict_from_model
            ) as mock_predict:
                surrogate.predict(
                    X=self.Xs[0], use_posterior_predictive=use_posterior_predictive
                )
            mock_predict.assert_called_with(
                model=surrogate.model,
                X=self.Xs[0],
                use_posterior_predictive=use_posterior_predictive,
            )

    @mock_botorch_optimize
    def test_best_in_sample_point(self) -> None:
        #  Set up data with 2 outcomes
        training_data = [
            SupervisedDataset(
                X=self.Xs[0],
                Y=self.Ys[0] + i,
                feature_names=self.feature_names,
                outcome_names=self.metric_names,
            )
            for i in range(2)
        ]

        # There are two points, and each have a nonzero probability of obeying
        # the constraints, with the first more likely
        outcome_constraints = (
            torch.tensor([[1.0, 0.0]], **self.tkwargs),
            torch.tensor([[4.0]], **self.tkwargs),
        )
        objective_weights = -torch.ones(2, **self.tkwargs)

        self.options = {}

        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(
                botorch_model_class=botorch_model_class,
                # When `SaasFullyBayesianSingleTaskGP` is used, there will be
                # multiple BoTorch `Model`s with 1d data
                n_outcomes=2 if botorch_model_class == SingleTaskGP else 1,
            )

            surrogate.fit(
                datasets=training_data,
                search_space_digest=self.search_space_digest,
            )

            # `best_in_sample_point` requires `objective_weights`
            with self.subTest("no objective weights"):
                with patch(
                    f"{SURROGATE_PATH}.best_in_sample_point", return_value=None
                ) as mock_best_in_sample, self.assertRaisesRegex(
                    ValueError, "Could not obtain"
                ):
                    surrogate.best_in_sample_point(
                        search_space_digest=self.search_space_digest,
                        torch_opt_config=dataclasses.replace(
                            self.torch_opt_config,
                            objective_weights=None,
                        ),
                    )
            with self.subTest("All points are in the search space"):
                # No linear constraints
                torch_opt_config = TorchOptConfig(
                    objective_weights=objective_weights,
                    outcome_constraints=outcome_constraints,
                )
                # Expand bounds so they aren't violated
                ssd = dataclasses.replace(
                    self.search_space_digest, bounds=[(-100, 100)] * 3
                )

                best_point, value = surrogate.best_in_sample_point(
                    search_space_digest=ssd,
                    torch_opt_config=torch_opt_config,
                    options=self.options,
                )
                # The first point has a higher probability of feasibility and a
                # better predicted objective value (because it had a better
                # observed objective valu)e, so it should be the best point and
                # have positive utility
                self.assertTrue(torch.equal(best_point, self.Xs[0][0]))
                self.assertGreater(value, 0)

            with self.subTest("One point is in the search space"):
                # use linear constraints that won't be violated
                linear_constraints = (
                    torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], **self.tkwargs),
                    torch.tensor([[0.5], [4.0]], **self.tkwargs),
                )

                # the second point is out of bounds and doesn't obey the fixed
                # features
                torch_opt_config = TorchOptConfig(
                    objective_weights=self.objective_weights,
                    outcome_constraints=outcome_constraints,
                    linear_constraints=linear_constraints,
                    fixed_features=self.fixed_features,
                )

                best_point, value = surrogate.best_in_sample_point(
                    search_space_digest=self.search_space_digest,
                    torch_opt_config=torch_opt_config,
                    options=self.options,
                )
                self.assertTrue(torch.equal(best_point, self.Xs[0][0]))
                # The objective value is `obj[best] - min(obj)`, so when there
                # is only one feasible point, it is zero
                self.assertEqual(value, 0)

            # Case 3: Errors when all points are infeasible
            with self.subTest("No points are in the search space"):
                with patch(
                    f"{SURROGATE_PATH}.best_in_sample_point",
                    wraps=best_in_sample_point,
                ) as mock_best_in_sample, self.assertRaisesRegex(
                    ValueError, "Could not obtain best in-sample point."
                ):
                    surrogate.best_in_sample_point(
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

    @mock_botorch_optimize
    def test_best_out_of_sample_point(self) -> None:
        torch.manual_seed(0)
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

            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            surrogate.fit(
                datasets=self.training_data,
                search_space_digest=self.search_space_digest,
            )
            torch_opt_config = TorchOptConfig(objective_weights=torch.tensor([1.0]))
            candidate, acqf_value = surrogate.best_out_of_sample_point(
                search_space_digest=self.search_space_digest,
                torch_opt_config=torch_opt_config,
                options=self.options,
            )
            candidate_in_bounds = all(
                ((x >= b[0]) & (x <= b[1]) for x, b in zip(candidate, self.bounds))
            )
            self.assertTrue(candidate_in_bounds)
            self.assertEqual(candidate.shape, torch.Size([3]))

            # self.training_data has length 1
            sample_mean = self.training_data[0].Y.mean().item()
            self.assertEqual(acqf_value.shape, torch.Size([]))
            # In realistic cases the maximum posterior mean would exceed the
            # sample mean (because the data is standardized), but that might not
            # be true when using `mock_botorch_optimize`
            eps = 1
            self.assertGreaterEqual(
                acqf_value.item(), assert_is_instance(sample_mean, float) - eps
            )

    def test_serialize_attributes_as_kwargs(self) -> None:
        for botorch_model_class in [SaasFullyBayesianSingleTaskGP, SingleTaskGP]:
            surrogate, _ = self._get_surrogate(botorch_model_class=botorch_model_class)
            expected = {
                "surrogate_spec": surrogate.surrogate_spec,
                "refit_on_cv": surrogate.refit_on_cv,
                "metric_to_best_model_config": surrogate.metric_to_best_model_config,
            }
            self.assertEqual(surrogate._serialize_attributes_as_kwargs(), expected)

    @mock_botorch_optimize
    def test_w_robust_digest(self) -> None:
        surrogate = Surrogate(
            botorch_model_class=SingleTaskGP,
        )
        # Error handling.
        robust_digest = RobustSearchSpaceDigest(
            environmental_variables=["a"],
            sample_param_perturbations=lambda: np.zeros((2, 2)),
        )
        with self.assertRaisesRegex(NotImplementedError, "Environmental variable"):
            surrogate.fit(
                datasets=self.training_data,
                search_space_digest=SearchSpaceDigest(
                    feature_names=self.search_space_digest.feature_names,
                    bounds=self.bounds,
                    task_features=self.search_space_digest.task_features,
                    robust_digest=robust_digest,
                ),
            )
        # Mixed with other transforms.
        robust_digest = RobustSearchSpaceDigest(
            sample_param_perturbations=lambda: np.zeros((2, 2)),
            environmental_variables=[],
            multiplicative=False,
        )
        surrogate.surrogate_spec.model_configs[0].input_transform_classes = [Normalize]
        surrogate.fit(
            datasets=self.training_data,
            search_space_digest=SearchSpaceDigest(
                feature_names=self.search_space_digest.feature_names,
                bounds=self.bounds,
                task_features=self.search_space_digest.task_features,
                robust_digest=robust_digest,
            ),
        )
        self.assertIsInstance(surrogate.model.input_transform, ChainedInputTransform)
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
        intf = assert_is_instance(
            surrogate.model.input_transform, ChainedInputTransform
        )
        intf_values = list(intf.values())
        self.assertIsInstance(intf_values[0], InputPerturbation)
        self.assertIsInstance(intf_values[1], Normalize)
        self.assertTrue(torch.equal(intf_values[0].perturbation_set, torch.zeros(2, 2)))

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
        covar_module = assert_is_instance(surrogate.model.covar_module, Kernel)
        self.assertEqual(
            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[An...
            covar_module.kernels[0].base_kernel.kernels[1].active_dims.tolist(),
            [0],
        )
        self.assertEqual(
            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[An...
            covar_module.kernels[0].base_kernel.kernels[0].active_dims.tolist(),
            [1, 2],
        )
        self.assertEqual(
            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[An...
            covar_module.kernels[1].base_kernel.kernels[1].active_dims.tolist(),
            [0],
        )
        self.assertEqual(
            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[An...
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
                for m in assert_is_instance(surrogate.model, ModelListGP).models
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
        model_config = self.surrogate.surrogate_spec.model_configs[0]
        self.assertEqual(
            [model_config.botorch_model_class] * 2,
            [*self.botorch_submodel_class_per_outcome.values()],
        )
        self.assertEqual(model_config.mll_class, self.mll_class)
        with self.assertRaisesRegex(
            ValueError, "BoTorch `Model` has not yet been constructed"
        ):
            self.surrogate.model

    @mock_botorch_optimize
    def test_construct_per_outcome_options(self) -> None:
        self.surrogate.surrogate_spec.model_configs[0].model_options.update(
            {"output_tasks": [2]}
        )
        for fixed_noise in (False, True):
            datasets = (
                self.fixed_noise_training_data
                if fixed_noise
                else self.supervised_training_data
            )
            search_space_digest = dataclasses.replace(
                self.multi_task_search_space_digest,
                task_features=self.task_features,
            )
            with ExitStack() as es:
                mock_fit = es.enter_context(
                    patch(
                        f"{SURROGATE_PATH}.fit_botorch_model", wraps=fit_botorch_model
                    )
                )
                mock_MTGP_construct_inputs = es.enter_context(
                    patch.object(
                        MultiTaskGP,
                        "construct_inputs",
                        wraps=MultiTaskGP.construct_inputs,
                    )
                )
                self.surrogate.fit(
                    datasets=datasets, search_space_digest=search_space_digest
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

    @mock_botorch_optimize
    def test_fit(self) -> None:
        default_class = self.botorch_model_class
        cases = {
            "default": Surrogate(
                botorch_model_class=default_class,
                mll_class=ExactMarginalLogLikelihood,
                # Check that empty lists also work fine.
                outcome_transform_classes=[],
                input_transform_classes=[],
            ),
            "bayesian_stgp": Surrogate(
                botorch_model_class=SaasFullyBayesianSingleTaskGP
            ),
            "bayesian_mtgp": Surrogate(
                botorch_model_class=SaasFullyBayesianMultiTaskGP
            ),
            "batch": Surrogate(
                botorch_model_class=SingleTaskGP, mll_class=ExactMarginalLogLikelihood
            ),
            "ModelListGP": Surrogate(
                botorch_model_class=SingleTaskGP,
                mll_class=ExactMarginalLogLikelihood,
                allow_batched_models=False,
            ),
        }

        # offset makes task feature point to valid outcome indices
        Xs, Ys, Yvars, _, _, _, _ = get_torch_test_data(dtype=self.dtype, offset=-1)
        ds1 = SupervisedDataset(
            X=Xs[0],
            Y=Ys[0],
            Yvar=Yvars[0],
            feature_names=self.feature_names,
            outcome_names=self.outcomes[:1],
        )
        datasets = [ds1, ds1]

        for case, surrogate in cases.items():
            # Reset mocks

            # Checking that model is None before `fit` (and `construct`) calls.
            self.assertIsNone(surrogate._model)
            # Should instantiate mll and `fit_gpytorch_mll` when `state_dict`
            # is `None`.

            is_mtgp = case in ("default", "bayesian_mtgp")

            search_space_digest = (
                self.multi_task_search_space_digest
                if is_mtgp
                else self.single_task_search_space_digest
            )
            if is_mtgp:
                # test error is raised without output_tasks or target_values
                msg = (
                    "output_tasks or target task value must be provided for"
                    " MultiTaskGP."
                )
                with self.assertRaisesRegex(UserInputError, msg):
                    surrogate.fit(
                        datasets=datasets,
                        search_space_digest=search_space_digest,
                    )
                # add target values
                search_space_digest = dataclasses.replace(
                    search_space_digest, target_values={0: 1}
                )

            surrogate.fit(
                datasets=datasets,
                search_space_digest=search_space_digest,
            )

            self.assertIsInstance(
                surrogate.model, SingleTaskGP if case == "batch" else ModelListGP
            )

            # Should `load_state_dict` when `state_dict` is not `None`
            # and `refit` is `False`.
            state_dict = surrogate.model.state_dict()
            surrogate._submodels = {}  # Prevent re-use of fitted model.
            surrogate.fit(
                datasets=datasets,
                search_space_digest=search_space_digest,
                refit=False,
                # pyre-fixme: Incompatible parameter type [6]: In call
                # `Surrogate.fit`, for argument `state_dict`, expected
                # `Optional[OrderedDict[str, Tensor]]` but got `Dict[str,
                # typing.Any]`
                state_dict=state_dict,
            )

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

    @mock_botorch_optimize
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
        with self.assertRaisesRegex(
            UserInputError,
            "The BoTorch model class SingleTaskGPWithDifferentConstructor",
        ):
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
        # pyre-fixme[9]: models has type `ModuleList`; used as `Union[Tensor, Module]`.
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

    @mock_botorch_optimize
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
            models = assert_is_instance(surrogate._model, ModelListGP).models
            self.assertEqual(len(models), 2)
            self.assertEqual(
                surrogate.surrogate_spec.model_configs[0].mll_class,
                ExactMarginalLogLikelihood,
            )
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

    @mock_botorch_optimize
    def test_w_robust_digest(self) -> None:
        surrogate = Surrogate(
            surrogate_spec=SurrogateSpec(
                botorch_model_class=SingleTaskGP,
                input_transform_classes=[],
            )
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
        # Input perturbation is constructed.
        surrogate.fit(
            datasets=self.supervised_training_data,
            search_space_digest=SearchSpaceDigest(
                feature_names=self.feature_names,
                bounds=self.bounds,
                task_features=[],
                robust_digest=robust_digest,
            ),
        )
        # pyre-fixme[29]: `Union[(self: Tensor) -> Any, Tensor, Module]` is not a
        #  function.
        for m in surrogate.model.models:
            intf = assert_is_instance(m.input_transform, InputPerturbation)
            self.assertIsInstance(intf, InputPerturbation)
            self.assertTrue(torch.equal(intf.perturbation_set, torch.zeros(2, 2)))
