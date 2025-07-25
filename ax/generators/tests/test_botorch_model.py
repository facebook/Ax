#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import dataclasses
from itertools import chain, product
from typing import Any, cast
from unittest import mock

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import DataRequiredError
from ax.generators.torch.botorch import (
    get_feature_importances_from_botorch_model,
    get_rounding_func,
    LegacyBoTorchGenerator,
)
from ax.generators.torch.botorch_defaults import (
    get_and_fit_model,
    get_chebyshev_scalarization,
)
from ax.generators.torch.utils import sample_simplex
from ax.generators.torch_base import TorchOptConfig
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.torch_stubs import get_torch_test_data
from botorch.acquisition.utils import get_infeasible_cost
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import InputTransform, Warp
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.objective import get_objective_weights_transform
from gpytorch.kernels.constant_kernel import ConstantKernel
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.priors.lkj_prior import LKJCovariancePrior
from pyre_extensions import assert_is_instance, none_throws


FIT_MODEL_MO_PATH = f"{get_and_fit_model.__module__}.fit_gpytorch_mll"
SAMPLE_SIMPLEX_UTIL_PATH = f"{sample_simplex.__module__}.sample_simplex"
SAMPLE_HYPERSPHERE_UTIL_PATH = f"{sample_simplex.__module__}.sample_hypersphere"
CHEBYSHEV_SCALARIZATION_PATH = (
    f"{get_chebyshev_scalarization.__module__}.get_chebyshev_scalarization"
)


def dummy_func(X: torch.Tensor) -> torch.Tensor:
    return X


class LegacyBoTorchGeneratorTest(TestCase):
    @mock_botorch_optimize
    def test_fixed_rank_LegacyBoTorchGenerator(
        self, dtype: torch.dtype = torch.float, cuda: bool = False
    ) -> None:
        Xs1, Ys1, Yvars1, bounds, _, feature_names, __ = get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        model = LegacyBoTorchGenerator(multitask_gp_ranks={"y": 2, "w": 1})
        datasets = [
            SupervisedDataset(
                X=Xs1,
                Y=Ys1,
                Yvar=Yvars1,
                feature_names=feature_names,
                outcome_names=["y"],
            ),
            SupervisedDataset(
                X=Xs2,
                Y=Ys2,
                Yvar=Yvars2,
                feature_names=feature_names,
                outcome_names=["w"],
            ),
        ]
        with self.assertRaisesRegex(RuntimeError, "Please fit the model first"):
            model.model

        with self.assertRaisesRegex(RuntimeError, "Please fit the model first"):
            model.search_space_digest

        search_space_digest = SearchSpaceDigest(
            feature_names=feature_names,
            bounds=bounds,
            task_features=[0],
        )
        with self.assertRaisesRegex(RuntimeError, "manually is disallowed"):
            model.search_space_digest = search_space_digest

        with mock.patch(FIT_MODEL_MO_PATH, wraps=fit_gpytorch_mll) as _mock_fit_model:
            model.fit(
                datasets=datasets,
                search_space_digest=search_space_digest,
            )
        self.assertTrue(isinstance(model.search_space_digest, SearchSpaceDigest))
        self.assertEqual(model.search_space_digest, search_space_digest)
        _mock_fit_model.assert_called_once()

        model.model = model.model  # property assignment isn't blocked
        # Check ranks
        model_list = cast(ModelListGP, model.model).models
        self.assertEqual(model_list[0]._rank, 2)
        self.assertEqual(model_list[1]._rank, 1)

    @mock_botorch_optimize
    def test_fixed_prior_LegacyBoTorchGenerator(
        self, dtype: torch.dtype = torch.float, cuda: bool = False
    ) -> None:
        Xs1, Ys1, Yvars1, bounds, _, feature_names, metric_names = get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        kwargs: dict[str, Any] = {
            "prior": {
                "covar_module_prior": {
                    "lengthscale_prior": GammaPrior(6.0, 3.0),
                    "outputscale_prior": GammaPrior(3.0, 12.0),
                },
                "type": LKJCovariancePrior,
                "sd_prior": GammaPrior(2.0, 0.44),
                "eta": 0.6,
            }
        }
        model = LegacyBoTorchGenerator(**kwargs)
        datasets = [
            SupervisedDataset(
                X=Xs1,
                Y=Ys1,
                Yvar=Yvars1,
                feature_names=feature_names,
                outcome_names=metric_names,
            ),
            SupervisedDataset(
                X=Xs2,
                Y=Ys2,
                Yvar=Yvars2,
                feature_names=feature_names,
                outcome_names=metric_names,
            ),
        ]

        search_space_digest = SearchSpaceDigest(
            feature_names=feature_names, bounds=bounds, task_features=[0]
        )
        with mock.patch(FIT_MODEL_MO_PATH, wraps=fit_gpytorch_mll) as _mock_fit_model:
            model.fit(datasets=datasets, search_space_digest=search_space_digest)
        _mock_fit_model.assert_called_once()

        # Check ranks
        model_list = cast(ModelListGP, model.model).models
        for i in range(1):
            data_covar_module, task_covar_module = model_list[i].covar_module.kernels
            self.assertEqual(
                data_covar_module.base_kernel.lengthscale_prior.concentration,
                6.0,
            )
            self.assertEqual(
                data_covar_module.base_kernel.lengthscale_prior.rate,
                3.0,
            )
            self.assertEqual(
                data_covar_module.outputscale_prior.concentration,
                3.0,
            )
            self.assertEqual(
                data_covar_module.outputscale_prior.rate,
                12.0,
            )
            self.assertIsInstance(
                task_covar_module.IndexKernelPrior, LKJCovariancePrior
            )
            self.assertEqual(
                task_covar_module.IndexKernelPrior.sd_prior.concentration,
                2.0,
            )
            self.assertEqual(task_covar_module.IndexKernelPrior.sd_prior.rate, 0.44)
            self.assertEqual(
                task_covar_module.IndexKernelPrior.correlation_prior.eta,
                0.6,
            )

    @mock_botorch_optimize
    def test_LegacyBoTorchGenerator(
        self, dtype: torch.dtype = torch.float, cuda: bool = False
    ) -> None:
        (
            Xs1,
            Ys1,
            Yvars1,
            bounds,
            tfs,
            feature_names,
            metric_names,
        ) = get_torch_test_data(dtype=dtype, cuda=cuda, constant_noise=True)
        Xs2, Ys2, Yvars2, _, _, _, _ = get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        for use_input_warping in (True, False):
            for use_loocv_pseudo_likelihood in (True, False):
                model = LegacyBoTorchGenerator(
                    use_input_warping=use_input_warping,
                    use_loocv_pseudo_likelihood=use_loocv_pseudo_likelihood,
                )

                # Test ModelListGP

                # make training data different for each output
                Xs2_diff = [Xs2 + 0.1]
                datasets = [
                    SupervisedDataset(
                        X=Xs1,
                        Y=Ys1,
                        Yvar=Yvars1,
                        feature_names=feature_names,
                        outcome_names=metric_names,
                    ),
                    SupervisedDataset(
                        X=Xs2_diff[0],
                        Y=Ys2,
                        Yvar=Yvars2,
                        feature_names=feature_names,
                        outcome_names=metric_names,
                    ),
                ]
                search_space_digest = SearchSpaceDigest(
                    feature_names=feature_names, bounds=bounds, task_features=tfs
                )

                with mock.patch(
                    FIT_MODEL_MO_PATH, wraps=fit_gpytorch_mll
                ) as _mock_fit_model:
                    model.fit(
                        datasets=datasets, search_space_digest=search_space_digest
                    )
                _mock_fit_model.assert_called_once()
                if use_loocv_pseudo_likelihood:
                    mll_cls = LeaveOneOutPseudoLikelihood
                else:
                    mll_cls = ExactMarginalLogLikelihood
                mlls = _mock_fit_model.mock_calls[0][1][0].mlls
                self.assertEqual(len(mlls), 2)
                for mll in mlls:
                    self.assertIsInstance(mll, mll_cls)

                # Check attributes
                self.assertTrue(torch.equal(model.Xs[0], Xs1))
                self.assertTrue(torch.equal(model.Xs[1], Xs2_diff[0]))
                self.assertEqual(model.dtype, Xs1.dtype)
                self.assertEqual(model.device, Xs1.device)
                self.assertIsInstance(model.model, ModelListGP)

                # Check fitting
                model_list = cast(ModelListGP, model.model).models
                untransformed_inputs = [Xs1, Xs2_diff[0]]

                if use_input_warping:
                    transformed_inputs = [
                        assert_is_instance(
                            model.input_transform, InputTransform
                        ).preprocess_transform(x)
                        for model, x in zip(model_list, untransformed_inputs)
                    ]
                else:
                    transformed_inputs = untransformed_inputs

                for i in range(2):
                    self.assertTrue(
                        torch.equal(
                            model_list[i].train_inputs[0], transformed_inputs[i]
                        )
                    )

                    self.assertIsInstance(
                        model_list[i].likelihood, _GaussianLikelihoodBase
                    )

                self.assertTrue(torch.equal(model_list[0].train_targets, Ys1.view(-1)))
                self.assertTrue(torch.equal(model_list[1].train_targets, Ys2.view(-1)))
                if use_input_warping:
                    self.assertTrue(model.use_input_warping)
                for m in model_list:
                    if use_input_warping:
                        self.assertTrue(hasattr(m, "input_transform"))
                        self.assertIsInstance(m.input_transform, Warp)
                    else:
                        self.assertFalse(hasattr(m, "input_transform"))

            # Test batched multi-output SingleTaskGP
            datasets_block = [
                SupervisedDataset(
                    X=Xs1,
                    Y=Ys1,
                    Yvar=Yvars1,
                    feature_names=feature_names,
                    outcome_names=metric_names,
                ),
                SupervisedDataset(
                    X=Xs2,
                    Y=Ys2,
                    Yvar=Yvars2,
                    feature_names=feature_names,
                    outcome_names=metric_names,
                ),
            ]
            with mock.patch(
                FIT_MODEL_MO_PATH, wraps=fit_gpytorch_mll
            ) as _mock_fit_model:
                model.fit(
                    datasets=datasets_block,
                    search_space_digest=SearchSpaceDigest(
                        feature_names=feature_names,
                        bounds=bounds,
                        task_features=tfs,
                    ),
                )
            _mock_fit_model.assert_called_once()

            # Check attributes
            self.assertTrue(torch.equal(model.Xs[0], Xs1))
            self.assertTrue(torch.equal(model.Xs[1], Xs2))
            self.assertEqual(model.dtype, Xs1.dtype)
            self.assertEqual(model.device, Xs1.device)
            if use_input_warping:
                self.assertIsInstance(model.model, ModelListGP)
                models = model.model.models
            else:
                models = [model.model]
            Ys = [Ys1, Ys2]
            for i, m in enumerate(models):
                self.assertIsInstance(m, SingleTaskGP)
                self.assertIsInstance(m.likelihood, FixedNoiseGaussianLikelihood)

                if not use_input_warping:
                    expected_train_inputs = Xs1.unsqueeze(0).expand(2, *Xs1.shape)
                    expected_train_targets = torch.cat([Ys1, Ys2], dim=-1).permute(1, 0)
                else:
                    expected_train_inputs = m.input_transform.preprocess_transform(Xs1)
                    expected_train_targets = Ys[i].squeeze(-1)
                # Check fitting
                # train inputs should be `o x n x 1`
                self.assertTrue(torch.equal(m.train_inputs[0], expected_train_inputs))
                # train targets should be `o x n`
                self.assertTrue(torch.equal(m.train_targets, expected_train_targets))
                self.assertIsInstance(m.likelihood, _GaussianLikelihoodBase)

            # Check infeasible cost can be computed on the model
            tkwargs: dict[str, Any] = {
                "device": torch.device("cuda" if cuda else "cpu"),
                "dtype": dtype,
            }
            objective_weights = torch.tensor([1.0, 0.0], **tkwargs)
            objective_transform = get_objective_weights_transform(objective_weights)
            infeasible_cost = (
                get_infeasible_cost(
                    X=Xs1, model=model.model, objective=objective_transform
                )
                .detach()
                .clone()
            )
            expected_infeasible_cost = -1 * torch.min(
                # pyre-fixme[20]: Argument `1` expected.
                objective_transform(
                    model.model.posterior(Xs1).mean
                    - 6 * model.model.posterior(Xs1).variance.sqrt()
                ).min(),
                torch.tensor(0.0, **tkwargs),
            )
            self.assertLess(
                torch.abs(infeasible_cost - expected_infeasible_cost).item(), 1e-5
            )

            # Check prediction
            X = torch.tensor([[6.0, 7.0, 8.0]], **tkwargs)
            f_mean, f_cov = model.predict(X)
            self.assertEqual(f_mean.shape, torch.Size([1, 2]))
            self.assertEqual(f_cov.shape, torch.Size([1, 2, 2]))

            # Check generation
            objective_weights = torch.tensor([1.0, 0.0], **tkwargs)
            outcome_constraints = (
                torch.tensor([[0.0, 1.0]], **tkwargs),
                torch.tensor([[5.0]], **tkwargs),
            )
            linear_constraints = (
                torch.tensor([[0.0, 1.0, 1.0]], **tkwargs),
                torch.tensor([[100.0]], **tkwargs),
            )
            fixed_features = None
            pending_observations = [
                torch.tensor([[1.0, 3.0, 4.0]], **tkwargs),
                torch.tensor([[2.0, 6.0, 8.0]], **tkwargs),
            ]
            n = 3

            X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)
            acqfv_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)
            model_gen_options = {"subset_model": False}
            # test sequential optimize
            search_space_digest = SearchSpaceDigest(
                feature_names=[],
                bounds=bounds,
            )
            torch_opt_config = TorchOptConfig(
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                linear_constraints=linear_constraints,
                fixed_features=fixed_features,
                pending_observations=pending_observations,
                # pyre-fixme[6]: For 6th param expected `Dict[str,
                #  Union[None, Dict[str, typing.Any], OptimizationConfig,
                #  AcquisitionFunction, float, int, str]]` but got `Dict[str,
                #  bool]`.
                model_gen_options=model_gen_options,
                rounding_func=dummy_func,
            )
            with mock.patch(
                "ax.generators.torch.botorch_defaults.optimize_acqf",
                return_value=(X_dummy, acqfv_dummy),
            ) as mock_optimize_acqf:
                gen_results = model.gen(
                    n=n,
                    search_space_digest=search_space_digest,
                    torch_opt_config=torch_opt_config,
                )
            # note: gen() always returns CPU tensors
            self.assertTrue(torch.equal(gen_results.points, X_dummy.cpu()))
            self.assertTrue(
                torch.equal(gen_results.weights, torch.ones(n, dtype=dtype))
            )
            self.assertEqual(
                mock_optimize_acqf.call_args.kwargs["options"]["init_batch_limit"], 32
            )
            self.assertEqual(
                mock_optimize_acqf.call_args.kwargs["options"]["batch_limit"], 5
            )

            # Repeat without mocking optimize_acqf to make sure it runs
            gen_results = model.gen(
                n=n,
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )
            self.assertTrue(
                torch.equal(gen_results.weights, torch.ones(n, dtype=dtype))
            )

            torch_opt_config = TorchOptConfig(
                objective_weights=objective_weights,
                fixed_features=fixed_features,
                pending_observations=pending_observations,
                model_gen_options={"optimizer_kwargs": {"joint_optimization": True}},
            )
            # test joint optimize
            with mock.patch(
                "ax.generators.torch.botorch_defaults.optimize_acqf",
                return_value=(X_dummy, acqfv_dummy),
            ) as mock_optimize_acqf:
                gen_results = model.gen(
                    n=n,
                    search_space_digest=search_space_digest,
                    torch_opt_config=torch_opt_config,
                )
            # note: gen() always returns CPU tensors
            self.assertTrue(torch.equal(gen_results.points, X_dummy.cpu()))
            self.assertTrue(
                torch.equal(gen_results.weights, torch.ones(n, dtype=dtype))
            )
            mock_optimize_acqf.assert_called_once()

            # test without mocking optimize_acqf to make sure it runs
            gen_results = model.gen(
                n=n,
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )
            self.assertTrue(
                torch.equal(gen_results.weights, torch.ones(n, dtype=dtype))
            )

            # test that fidelity features are unsupported
            with self.assertRaises(NotImplementedError):
                gen_results = model.gen(
                    n=n,
                    search_space_digest=dataclasses.replace(
                        search_space_digest,
                        fidelity_features=[0],
                        target_values={0: 3.0},
                    ),
                    torch_opt_config=torch_opt_config,
                )

            # test get_rounding_func
            dummy_rounding = none_throws(get_rounding_func(rounding_func=dummy_func))
            X_temp = torch.rand(1, 2, 3, 4)
            self.assertTrue(torch.equal(X_temp, dummy_rounding(X_temp)))

            # Check best point selection
            xbest = model.best_point(
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )
            torch_opt_config = dataclasses.replace(
                torch_opt_config,
                fixed_features={0: 100.0},
            )
            xbest = model.best_point(
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )
            self.assertIsNone(xbest)

            # test that fidelity features are unsupported
            with self.assertRaises(NotImplementedError):
                xbest = model.best_point(
                    search_space_digest=dataclasses.replace(
                        search_space_digest,
                        fidelity_features=[0],
                        target_values={0: 3.0},
                    ),
                    torch_opt_config=torch_opt_config,
                )

            # Test cross-validation
            combined_datasets = [
                SupervisedDataset(
                    Xs1,
                    Y=Ys1,
                    Yvar=Yvars1,
                    feature_names=feature_names,
                    outcome_names=metric_names,
                ),
                SupervisedDataset(
                    Xs2,
                    Y=Ys2,
                    Yvar=Yvars2,
                    feature_names=feature_names,
                    outcome_names=metric_names,
                ),
            ]
            mean, variance = model.cross_validate(
                datasets=combined_datasets,
                search_space_digest=search_space_digest,
                X_test=torch.tensor([[1.2, 3.2, 4.2], [2.4, 5.2, 3.2]], **tkwargs),
            )
            self.assertEqual(mean.shape, torch.Size([2, 2]))
            self.assertEqual(variance.shape, torch.Size([2, 2, 2]))

            # Test cross-validation with refit_on_cv
            model.refit_on_cv = True
            mean, variance = model.cross_validate(
                datasets=combined_datasets,
                search_space_digest=search_space_digest,
                X_test=torch.tensor([[1.2, 3.2, 4.2], [2.4, 5.2, 3.2]], **tkwargs),
            )
            self.assertEqual(mean.shape, torch.Size([2, 2]))
            self.assertEqual(variance.shape, torch.Size([2, 2, 2]))

            # Test feature_importances
            importances = model.feature_importances()
            self.assertEqual(importances.shape, torch.Size([2, 1, 3]))

            # test unfit model CV and feature_importances
            unfit_model = LegacyBoTorchGenerator()
            with self.assertRaisesRegex(
                RuntimeError, r"Cannot cross-validate model that has not been fitted"
            ):
                unfit_model.cross_validate(
                    datasets=combined_datasets,
                    search_space_digest=search_space_digest,
                    X_test=Xs1,
                )
            with self.assertRaisesRegex(
                RuntimeError,
                r"Cannot calculate feature_importances without a fitted model",
            ):
                unfit_model.feature_importances()

            # Test loading state dict
            true_state_dict = {
                "mean_module.raw_constant": 1.0,
                "covar_module.raw_lengthscale": [[0.3548, 0.3548, 0.3548]],
                "covar_module.lengthscale_prior._transformed_loc": 1.9635,
                "covar_module.lengthscale_prior._transformed_scale": 1.7321,
                "covar_module.raw_lengthscale_constraint.lower_bound": 0.0250,
                "covar_module.raw_lengthscale_constraint.upper_bound": float("inf"),
            }
            true_state_dict = {
                key: torch.tensor(val, **tkwargs)
                for key, val in true_state_dict.items()
            }
            model = get_and_fit_model(
                Xs=[Xs1],
                Ys=[Ys1],
                Yvars=[Yvars1],
                task_features=[],
                fidelity_features=[],
                metric_names=[metric_names[0]],
                state_dict=true_state_dict,
                refit_model=False,
            )
            for k, v in chain(model.named_parameters(), model.named_buffers()):
                self.assertTrue(torch.equal(true_state_dict[k], v))

            # Test for some change in model parameters & buffer for refit_model=True
            true_state_dict["mean_module.raw_constant"] += 0.1
            true_state_dict["covar_module.raw_lengthscale"] += 0.1
            model = get_and_fit_model(
                Xs=[Xs1],
                Ys=[Ys1],
                Yvars=[Yvars1],
                task_features=[],
                fidelity_features=[],
                metric_names=[metric_names[0]],
                state_dict=true_state_dict,
                refit_model=True,
            )
            self.assertTrue(
                any(
                    not torch.equal(true_state_dict[k], v)
                    for k, v in chain(model.named_parameters(), model.named_buffers())
                )
            )

    def test_LegacyBoTorchGenerator_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_LegacyBoTorchGenerator(cuda=True)

    def test_LegacyBoTorchGenerator_double(self) -> None:
        self.test_LegacyBoTorchGenerator(dtype=torch.double)

    def test_LegacyBoTorchGenerator_double_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_LegacyBoTorchGenerator(dtype=torch.double, cuda=True)

    def test_LegacyBoTorchGeneratorOneOutcome(self) -> None:
        (
            Xs1,
            Ys1,
            Yvars1,
            bounds,
            tfs,
            feature_names,
            metric_names,
        ) = get_torch_test_data(dtype=torch.float, cuda=False, constant_noise=True)
        for use_input_warping, use_loocv_pseudo_likelihood in product(
            (True, False), (True, False)
        ):
            model = LegacyBoTorchGenerator(
                use_input_warping=use_input_warping,
                use_loocv_pseudo_likelihood=use_loocv_pseudo_likelihood,
            )
            with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
                model.fit(
                    datasets=[
                        SupervisedDataset(
                            X=Xs1,
                            Y=Ys1,
                            Yvar=Yvars1,
                            feature_names=feature_names,
                            outcome_names=metric_names,
                        )
                    ],
                    search_space_digest=SearchSpaceDigest(
                        feature_names=feature_names,
                        bounds=bounds,
                        task_features=tfs,
                    ),
                )
                _mock_fit_model.assert_called_once()
                if use_loocv_pseudo_likelihood:
                    mll_cls = LeaveOneOutPseudoLikelihood
                else:
                    mll_cls = ExactMarginalLogLikelihood
                self.assertIsInstance(
                    _mock_fit_model.mock_calls[0][1][0],
                    mll_cls,
                )
            X = torch.rand(2, 3, dtype=torch.float)
            f_mean, f_cov = model.predict(X)
            self.assertTrue(f_mean.shape == torch.Size([2, 1]))
            self.assertTrue(f_cov.shape == torch.Size([2, 1, 1]))
            if use_input_warping:
                self.assertTrue(hasattr(model.model, "input_transform"))
                self.assertIsInstance(model.model.input_transform, Warp)
            else:
                self.assertFalse(hasattr(model.model, "input_transform"))

    def test_LegacyBoTorchGeneratorConstraints(self) -> None:
        (
            Xs1,
            Ys1,
            Yvars1,
            bounds,
            tfs,
            feature_names,
            metric_names,
        ) = get_torch_test_data(dtype=torch.float, cuda=False, constant_noise=True)
        Xs2, Ys2, Yvars2, _, _, _, _ = get_torch_test_data(
            dtype=torch.float, cuda=False, constant_noise=True
        )
        # make infeasible
        Xs2 = -1 * Xs2
        objective_weights = torch.tensor(
            [-1.0, 1.0], dtype=torch.float, device=torch.device("cpu")
        )
        n = 3
        model = LegacyBoTorchGenerator()
        search_space_digest = SearchSpaceDigest(
            feature_names=feature_names,
            bounds=bounds,
            task_features=tfs,
        )
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                datasets=[
                    SupervisedDataset(
                        X=Xs1,
                        Y=Ys1,
                        Yvar=Yvars1,
                        feature_names=feature_names,
                        outcome_names=metric_names,
                    ),
                    SupervisedDataset(
                        X=Xs2,
                        Y=Ys2,
                        Yvar=Yvars2,
                        feature_names=feature_names,
                        outcome_names=metric_names,
                    ),
                ],
                search_space_digest=search_space_digest,
            )
            _mock_fit_model.assert_called_once()

        # because there are no feasible points:
        with self.assertRaises(ValueError):
            model.gen(
                n,
                search_space_digest=search_space_digest,
                torch_opt_config=TorchOptConfig(objective_weights),
            )

    def test_botorchmodel_raises_when_no_data(self) -> None:
        _, _, _, bounds, tfs, feature_names, metric_names = get_torch_test_data(
            dtype=torch.float, cuda=False, constant_noise=True
        )
        search_space_digest = SearchSpaceDigest(
            feature_names=feature_names,
            bounds=bounds,
            task_features=tfs,
        )
        model = LegacyBoTorchGenerator()
        with self.assertRaisesRegex(
            DataRequiredError,
            "LegacyBoTorchGenerator.fit requires non-empty data sets.",
        ):
            model.fit(
                datasets=[],
                search_space_digest=search_space_digest,
            )

    def test_get_feature_importances_from_botorch_model(self) -> None:
        tkwargs: dict[str, Any] = {"dtype": torch.double}
        train_X = torch.rand(5, 3, **tkwargs)
        train_Y = train_X.sum(dim=-1, keepdim=True)
        simple_gp = SingleTaskGP(train_X=train_X, train_Y=train_Y)
        # pyre-fixme[16]: `Module` has no attribute `lengthscale`.
        simple_gp.covar_module.lengthscale = torch.tensor([1, 3, 5], **tkwargs)
        importances = get_feature_importances_from_botorch_model(simple_gp)
        self.assertTrue(np.allclose(importances, np.array([15 / 23, 5 / 23, 3 / 23])))
        self.assertEqual(importances.shape, (1, 1, 3))
        # Model with kernel that has no lengthscales
        simple_gp.covar_module = ConstantKernel()
        with self.assertRaisesRegex(
            NotImplementedError,
            "Failed to extract lengthscales from `m.covar_module` and "
            "`m.covar_module.base_kernel`",
        ):
            get_feature_importances_from_botorch_model(simple_gp)

        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot calculate feature_importances without a fitted model",
        ):
            get_feature_importances_from_botorch_model(None)
