#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from unittest.mock import Mock, patch

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import UserInputError
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.list_surrogate import ListSurrogate
from ax.models.torch.botorch_modular.surrogate import fit_botorch_model, Surrogate
from ax.models.torch.botorch_modular.utils import choose_model_class
from ax.models.torch.tests.test_surrogate import SingleTaskGPWithDifferentConstructor
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast
from ax.utils.testing.torch_stubs import get_torch_test_data
from botorch.models import (
    SaasFullyBayesianMultiTaskGP,
    SaasFullyBayesianSingleTaskGP,
    SingleTaskGP,
)
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import FixedNoiseMultiTaskGP, MultiTaskGP
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import InputPerturbation, Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.utils.datasets import FixedNoiseDataset, SupervisedDataset
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel  # noqa: F401
from gpytorch.likelihoods import (  # noqa: F401
    GaussianLikelihood,
    Likelihood,  # noqa: F401
)
from gpytorch.mlls import ExactMarginalLogLikelihood


SURROGATE_PATH = f"{Surrogate.__module__}"
UTILS_PATH = f"{choose_model_class.__module__}"
CURRENT_PATH = f"{__name__}"
ACQUISITION_PATH = f"{Acquisition.__module__}"
RANK = "rank"


class ListSurrogateTest(TestCase):
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
        self.expected_submodel_type = FixedNoiseMultiTaskGP
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
            self.outcomes[0]: {RANK: 1},
            self.outcomes[1]: {RANK: 2},
        }
        self.surrogate = ListSurrogate(
            botorch_submodel_class_per_outcome=self.botorch_submodel_class_per_outcome,
            mll_class=self.mll_class,
            submodel_options_per_outcome=self.submodel_options_per_outcome,
        )
        self.bounds = [(0.0, 1.0), (1.0, 4.0)]
        self.feature_names = ["x1", "x2"]

    def check_ranks(self, c: ListSurrogate) -> None:
        self.assertIsInstance(c, ListSurrogate)
        self.assertIsInstance(c.model, ModelListGP)
        # pyre-fixme[6]: For 1st param expected `Iterable[Variable[_T]]` but got
        #  `Union[Tensor, Module]`.
        for idx, submodel in enumerate(c.model.models):
            self.assertIsInstance(submodel, self.expected_submodel_type)
            self.assertEqual(
                submodel._rank,
                self.submodel_options_per_outcome[self.outcomes[idx]][RANK],
            )

    def test_init(self) -> None:
        self.assertEqual(
            self.surrogate.botorch_submodel_class_per_outcome,
            self.botorch_submodel_class_per_outcome,
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
        with self.assertRaisesRegex(ValueError, "No model class specified for"):
            self.surrogate.construct(
                datasets=self.fixed_noise_training_data, metric_names=["new_metric"]
            )
        self.surrogate.construct(
            datasets=self.fixed_noise_training_data,
            metric_names=self.outcomes,
            task_features=self.task_features,
        )
        self.check_ranks(self.surrogate)
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
                    "rank": self.submodel_options_per_outcome[self.outcomes[idx]][
                        "rank"
                    ],
                },
            )

    @patch.object(
        MultiTaskGP,
        "construct_inputs",
        wraps=MultiTaskGP.construct_inputs,
    )
    def test_construct_per_outcome_options_no_Yvar(self, _) -> None:
        surrogate = ListSurrogate(
            botorch_submodel_class=MultiTaskGP,
            mll_class=self.mll_class,
            submodel_options_per_outcome=self.submodel_options_per_outcome,
        )
        # Test that splitting the training data works correctly when Yvar is None.
        surrogate.construct(
            datasets=self.supervised_training_data,
            task_features=self.task_features,
            metric_names=self.outcomes,
        )
        # pyre-fixme[16]: Optional type has no attribute `__iter__`.
        for ds in surrogate._training_data:
            self.assertTrue(isinstance(ds, SupervisedDataset))
            self.assertFalse(isinstance(ds, FixedNoiseDataset))
        # pyre-fixme[6]: For 1st param expected `Sized` but got
        #  `Optional[List[SupervisedDataset]]`.
        self.assertEqual(len(surrogate._training_data), 2)

    @patch.object(
        FixedNoiseMultiTaskGP,
        "construct_inputs",
        wraps=FixedNoiseMultiTaskGP.construct_inputs,
    )
    def test_construct_shared_shortcut_options(
        self, mock_construct_inputs: Mock
    ) -> None:
        surrogate = ListSurrogate(
            botorch_submodel_class=self.botorch_submodel_class_per_outcome[
                self.outcomes[0]
            ],
            submodel_options={"shared_option": True},
            submodel_options_per_outcome={
                outcome: {"individual_option": f"val_{idx}"}
                for idx, outcome in enumerate(self.outcomes)
            },
        )
        surrogate.construct(
            datasets=self.fixed_noise_training_data,
            metric_names=self.outcomes,
            task_features=self.task_features,
        )
        # 2 submodels should've been constructed, both of type `botorch_submodel_class`.
        self.assertEqual(len(mock_construct_inputs.call_args_list), 2)
        first_call_args, second_call_args = mock_construct_inputs.call_args_list
        for idx in range(len(mock_construct_inputs.call_args_list)):
            self.assertEqual(
                mock_construct_inputs.call_args_list[idx][1],
                {
                    "fidelity_features": [],
                    "individual_option": f"val_{idx}",
                    "shared_option": True,
                    "task_feature": 0,
                    "training_data": FixedNoiseDataset(
                        X=self.Xs[idx], Y=self.Ys[idx], Yvar=self.Yvars[idx]
                    ),
                },
            )

    @patch.object(
        FixedNoiseMultiTaskGP,
        "construct_inputs",
        wraps=FixedNoiseMultiTaskGP.construct_inputs,
    )
    def test_construct_per_outcome_error_raises(
        self, mock_MTGP_construct_inputs: Mock
    ) -> None:
        surrogate = ListSurrogate(
            botorch_submodel_class=self.botorch_submodel_class_per_outcome,
            mll_class=self.mll_class,
            submodel_options_per_outcome=self.submodel_options_per_outcome,
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

    @patch(f"{CURRENT_PATH}.ModelListGP.load_state_dict", return_value=None)
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
        default_class = self.botorch_submodel_class_per_outcome
        surrogates = [
            ListSurrogate(
                botorch_submodel_class_per_outcome=default_class,
                mll_class=ExactMarginalLogLikelihood,
            ),
            ListSurrogate(botorch_submodel_class=SaasFullyBayesianSingleTaskGP),
            ListSurrogate(botorch_submodel_class=SaasFullyBayesianMultiTaskGP),
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
        surrogate = ListSurrogate(
            botorch_submodel_class=SingleTaskGPWithDifferentConstructor,
            mll_class=ExactMarginalLogLikelihood,
            submodel_outcome_transforms=outcome_transforms,
            submodel_input_transforms=input_transforms,
        )
        with self.assertRaisesRegex(UserInputError, "The BoTorch model class"):
            surrogate.construct(
                datasets=self.supervised_training_data,
                metric_names=self.outcomes,
            )
        surrogate = ListSurrogate(
            botorch_submodel_class=SingleTaskGP,
            mll_class=ExactMarginalLogLikelihood,
            submodel_outcome_transforms=outcome_transforms,
            submodel_input_transforms=input_transforms,
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
        expected = self.surrogate.__dict__
        # The two attributes below don't need to be saved as part of state,
        # so we remove them from the expected dict.
        for attr_name in (
            "botorch_model_class",
            "model_options",
            "covar_module_class",
            "covar_module_options",
            "likelihood_class",
            "likelihood_options",
            "outcome_transform",
            "input_transform",
        ):
            expected.pop(attr_name)
        self.assertEqual(self.surrogate._serialize_attributes_as_kwargs(), expected)

    def test_construct_custom_model(self) -> None:
        noise_constraint = Interval(1e-4, 10.0)
        for submodel_covar_module_options, submodel_likelihood_options in [
            [{"ard_num_dims": 3}, {"noise_constraint": noise_constraint}],
            [{}, {}],
        ]:
            surrogate = ListSurrogate(
                botorch_submodel_class=SingleTaskGP,
                mll_class=ExactMarginalLogLikelihood,
                submodel_covar_module_class=MaternKernel,
                submodel_covar_module_options=submodel_covar_module_options,
                submodel_likelihood_class=GaussianLikelihood,
                submodel_likelihood_options=submodel_likelihood_options,
                submodel_input_transforms=Normalize(d=3),
                submodel_outcome_transforms=Standardize(m=1),
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
        surrogate = ListSurrogate(
            botorch_submodel_class=SingleTaskGP,
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
        surrogate.submodel_input_transforms = Normalize(d=1)
        with self.assertRaisesRegex(NotImplementedError, "input transforms"):
            surrogate.construct(
                datasets=self.supervised_training_data,
                metric_names=self.outcomes,
                robust_digest=robust_digest,
            )
        # Input perturbation is constructed.
        surrogate = ListSurrogate(
            botorch_submodel_class=SingleTaskGP,
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
