#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from itertools import chain, product
from typing import Any, Dict
from unittest import mock

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import DataRequiredError
from ax.models.torch.botorch import BotorchModel, get_rounding_func
from ax.models.torch.botorch_defaults import (
    get_and_fit_model,
    get_chebyshev_scalarization,
    recommend_best_out_of_sample_point,
)
from ax.models.torch.utils import sample_simplex
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import fast_botorch_optimize
from ax.utils.testing.torch_stubs import get_torch_test_data
from botorch.acquisition.utils import get_infeasible_cost
from botorch.models import FixedNoiseGP, ModelListGP
from botorch.models.transforms.input import Warp
from botorch.utils.datasets import FixedNoiseDataset
from botorch.utils.objective import get_objective_weights_transform
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.mlls import ExactMarginalLogLikelihood, LeaveOneOutPseudoLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.priors.lkj_prior import LKJCovariancePrior


FIT_MODEL_MO_PATH = f"{get_and_fit_model.__module__}.fit_gpytorch_mll"
SAMPLE_SIMPLEX_UTIL_PATH = f"{sample_simplex.__module__}.sample_simplex"
SAMPLE_HYPERSPHERE_UTIL_PATH = f"{sample_simplex.__module__}.sample_hypersphere"
CHEBYSHEV_SCALARIZATION_PATH = (
    f"{get_chebyshev_scalarization.__module__}.get_chebyshev_scalarization"
)


def dummy_func(X: torch.Tensor) -> torch.Tensor:
    return X


class BotorchModelTest(TestCase):
    def test_fixed_rank_BotorchModel(
        self, dtype: torch.dtype = torch.float, cuda: bool = False
    ) -> None:
        Xs1, Ys1, Yvars1, bounds, _, fns, _ = get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        model = BotorchModel(multitask_gp_ranks={"y": 2, "w": 1})
        datasets = [
            FixedNoiseDataset(X=Xs1[0], Y=Ys1[0], Yvar=Yvars1[0]),
            FixedNoiseDataset(X=Xs2[0], Y=Ys2[0], Yvar=Yvars2[0]),
        ]

        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                datasets=datasets,
                metric_names=["y", "w"],
                search_space_digest=SearchSpaceDigest(
                    feature_names=fns,
                    bounds=bounds,
                    task_features=[0],
                ),
            )
            _mock_fit_model.assert_called_once()

        # Check ranks
        # pyre-fixme[16]: Optional type has no attribute `models`.
        model_list = model.model.models
        self.assertEqual(model_list[0]._rank, 2)
        self.assertEqual(model_list[1]._rank, 1)

    def test_fixed_prior_BotorchModel(
        self, dtype: torch.dtype = torch.float, cuda: bool = False
    ) -> None:
        Xs1, Ys1, Yvars1, bounds, _, fns, _ = get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        kwargs = {
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
        model = BotorchModel(**kwargs)  # pyre-ignore [6]
        datasets = [
            FixedNoiseDataset(X=Xs1[0], Y=Ys1[0], Yvar=Yvars1[0]),
            FixedNoiseDataset(X=Xs2[0], Y=Ys2[0], Yvar=Yvars2[0]),
        ]

        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                datasets=datasets,
                metric_names=["y", "w"],
                search_space_digest=SearchSpaceDigest(
                    feature_names=fns,
                    bounds=bounds,
                    task_features=[0],
                ),
            )
            _mock_fit_model.assert_called_once()

        # Check ranks
        # pyre-fixme[16]: Optional type has no attribute `models`.
        model_list = model.model.models
        for i in range(1):
            self.assertEqual(
                model_list[i].covar_module.base_kernel.lengthscale_prior.concentration,
                6.0,
            )
            self.assertEqual(
                model_list[i].covar_module.base_kernel.lengthscale_prior.rate,
                3.0,
            )
            self.assertEqual(
                model_list[i].covar_module.outputscale_prior.concentration,
                3.0,
            )
            self.assertEqual(
                model_list[i].covar_module.outputscale_prior.rate,
                12.0,
            )
            self.assertIsInstance(
                model_list[i].task_covar_module.IndexKernelPrior, LKJCovariancePrior
            )
            self.assertEqual(
                model_list[i].task_covar_module.IndexKernelPrior.sd_prior.concentration,
                2.0,
            )
            self.assertEqual(
                model_list[i].task_covar_module.IndexKernelPrior.sd_prior.rate, 0.44
            )
            self.assertEqual(
                model_list[i].task_covar_module.IndexKernelPrior.correlation_prior.eta,
                0.6,
            )

    @fast_botorch_optimize
    def test_BotorchModel(
        self, dtype: torch.dtype = torch.float, cuda: bool = False
    ) -> None:
        Xs1, Ys1, Yvars1, bounds, tfs, fns, mns = get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        for use_input_warping in (True, False):
            for use_loocv_pseudo_likelihood in (True, False):
                model = BotorchModel(
                    use_input_warping=use_input_warping,
                    use_loocv_pseudo_likelihood=use_loocv_pseudo_likelihood,
                )

                # Test ModelListGP

                # make training data different for each output
                Xs2_diff = [Xs2[0] + 0.1]
                datasets = [
                    FixedNoiseDataset(X=Xs1[0], Y=Ys1[0], Yvar=Yvars1[0]),
                    FixedNoiseDataset(X=Xs2_diff[0], Y=Ys2[0], Yvar=Yvars2[0]),
                ]
                with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
                    model.fit(
                        datasets=datasets,
                        metric_names=mns,
                        search_space_digest=SearchSpaceDigest(
                            feature_names=fns,
                            bounds=bounds,
                            task_features=tfs,
                        ),
                    )
                    _mock_fit_model.assert_called_once()
                    if use_loocv_pseudo_likelihood:
                        mll_cls = LeaveOneOutPseudoLikelihood
                    else:
                        mll_cls = ExactMarginalLogLikelihood
                    mlls = _mock_fit_model.mock_calls[0][1][0].mlls
                    self.assertTrue(len(mlls) == 2)
                    for mll in mlls:
                        self.assertIsInstance(mll, mll_cls)
                # Check attributes
                self.assertTrue(torch.equal(model.Xs[0], Xs1[0]))
                self.assertTrue(torch.equal(model.Xs[1], Xs2_diff[0]))
                self.assertEqual(model.dtype, Xs1[0].dtype)
                self.assertEqual(model.device, Xs1[0].device)
                self.assertIsInstance(model.model, ModelListGP)

                # Check fitting
                # pyre-fixme[16]: Optional type has no attribute `models`.
                model_list = model.model.models
                self.assertTrue(torch.equal(model_list[0].train_inputs[0], Xs1[0]))
                self.assertTrue(torch.equal(model_list[1].train_inputs[0], Xs2_diff[0]))
                self.assertTrue(
                    torch.equal(model_list[0].train_targets, Ys1[0].view(-1))
                )
                self.assertTrue(
                    torch.equal(model_list[1].train_targets, Ys2[0].view(-1))
                )
                self.assertIsInstance(model_list[0].likelihood, _GaussianLikelihoodBase)
                self.assertIsInstance(model_list[1].likelihood, _GaussianLikelihoodBase)
                if use_input_warping:
                    self.assertTrue(model.use_input_warping)
                for m in model_list:
                    if use_input_warping:
                        self.assertTrue(hasattr(m, "input_transform"))
                        self.assertIsInstance(m.input_transform, Warp)
                    else:
                        self.assertFalse(hasattr(m, "input_transform"))

            # Test batched multi-output FixedNoiseGP
            datasets_block = [
                FixedNoiseDataset(X=Xs1[0], Y=Ys1[0], Yvar=Yvars1[0]),
                FixedNoiseDataset(X=Xs2[0], Y=Ys2[0], Yvar=Yvars2[0]),
            ]
            with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
                model.fit(
                    datasets=datasets_block,
                    metric_names=["y1", "y2"],
                    search_space_digest=SearchSpaceDigest(
                        feature_names=fns,
                        bounds=bounds,
                        task_features=tfs,
                    ),
                )
                _mock_fit_model.assert_called_once()

            # Check attributes
            self.assertTrue(torch.equal(model.Xs[0], Xs1[0]))
            self.assertTrue(torch.equal(model.Xs[1], Xs2[0]))
            self.assertEqual(model.dtype, Xs1[0].dtype)
            self.assertEqual(model.device, Xs1[0].device)
            if use_input_warping:
                self.assertIsInstance(model.model, ModelListGP)
                models = model.model.models
            else:
                models = [model.model]
            Ys = [Ys1[0], Ys2[0]]
            for i, m in enumerate(models):
                self.assertIsInstance(m, FixedNoiseGP)
                expected_train_inputs = Xs1[0]

                if not use_input_warping:
                    expected_train_inputs = expected_train_inputs.unsqueeze(0).expand(
                        torch.Size([2])
                        + Xs1[0].shape  # pyre-fixme[58]: Unsupported operand
                    )
                    expected_train_targets = torch.cat(Ys1 + Ys2, dim=-1).permute(1, 0)
                else:
                    expected_train_targets = Ys[i].squeeze(-1)
                # Check fitting
                # train inputs should be `o x n x 1`
                self.assertTrue(
                    torch.equal(
                        m.train_inputs[0],
                        expected_train_inputs,
                    )
                )
                # train targets should be `o x n`
                self.assertTrue(
                    torch.equal(
                        m.train_targets,
                        expected_train_targets,
                    )
                )
                self.assertIsInstance(m.likelihood, _GaussianLikelihoodBase)

            # Check infeasible cost can be computed on the model
            tkwargs: Dict[str, Any] = {
                "device": torch.device("cuda" if cuda else "cpu"),
                "dtype": dtype,
            }
            objective_weights = torch.tensor([1.0, 0.0], **tkwargs)
            objective_transform = get_objective_weights_transform(objective_weights)
            infeasible_cost = torch.tensor(
                get_infeasible_cost(
                    X=Xs1[0], model=model.model, objective=objective_transform
                )
            )
            expected_infeasible_cost = -1 * torch.min(
                # pyre-fixme[20]: Argument `1` expected.
                objective_transform(
                    model.model.posterior(Xs1[0]).mean
                    - 6 * model.model.posterior(Xs1[0]).variance.sqrt()
                ).min(),
                torch.tensor(0.0, **tkwargs),
            )
            self.assertTrue(
                torch.abs(infeasible_cost - expected_infeasible_cost) < 1e-5
            )

            # Check prediction
            X = torch.tensor([[6.0, 7.0, 8.0]], **tkwargs)
            f_mean, f_cov = model.predict(X)
            self.assertTrue(f_mean.shape == torch.Size([1, 2]))
            self.assertTrue(f_cov.shape == torch.Size([1, 2, 2]))

            # Check generation
            objective_weights = torch.tensor([1.0, 0.0], **tkwargs)
            outcome_constraints = (
                torch.tensor([[0.0, 1.0]], **tkwargs),
                torch.tensor([[5.0]], **tkwargs),
            )
            linear_constraints = (
                torch.tensor([[0.0, 1.0, 1.0]]),
                torch.tensor([[100.0]]),
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
            with mock.patch(
                "ax.models.torch.botorch_defaults.optimize_acqf",
                return_value=(X_dummy, acqfv_dummy),
            ) as mock_optimize_acqf:
                gen_results = model.gen(
                    n=n,
                    search_space_digest=search_space_digest,
                    torch_opt_config=TorchOptConfig(
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
                    ),
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
            torch_opt_config = TorchOptConfig(
                objective_weights=objective_weights,
                fixed_features=fixed_features,
                pending_observations=pending_observations,
                model_gen_options={"optimizer_kwargs": {"joint_optimization": True}},
            )
            # test joint optimize
            with mock.patch(
                "ax.models.torch.botorch_defaults.optimize_acqf",
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

            # test that fidelity features are unsupported
            with self.assertRaises(NotImplementedError):
                gen_results = model.gen(
                    n=n,
                    search_space_digest=dataclasses.replace(
                        search_space_digest,
                        target_fidelities={0: 3.0},
                    ),
                    torch_opt_config=torch_opt_config,
                )

            # test get_rounding_func
            dummy_rounding = get_rounding_func(rounding_func=dummy_func)
            X_temp = torch.rand(1, 2, 3, 4)
            # pyre-fixme[29]: `Optional[typing.Callable[[torch._tensor.Tensor],
            #  torch._tensor.Tensor]]` is not a function.
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
                        target_fidelities={0: 3.0},
                    ),
                    torch_opt_config=torch_opt_config,
                )

            # Test cross-validation
            combined_datasets = [
                FixedNoiseDataset(Xs1[0], Y=Ys1[0], Yvar=Yvars1[0]),
                FixedNoiseDataset(Xs2[0], Y=Ys2[0], Yvar=Yvars2[0]),
            ]
            mean, variance = model.cross_validate(
                datasets=combined_datasets,
                metric_names=["y1", "y2"],
                X_test=torch.tensor([[1.2, 3.2, 4.2], [2.4, 5.2, 3.2]], **tkwargs),
            )
            self.assertTrue(mean.shape == torch.Size([2, 2]))
            self.assertTrue(variance.shape == torch.Size([2, 2, 2]))

            # Test cross-validation with refit_on_cv
            model.refit_on_cv = True
            mean, variance = model.cross_validate(
                datasets=combined_datasets,
                metric_names=["y1", "y2"],
                X_test=torch.tensor([[1.2, 3.2, 4.2], [2.4, 5.2, 3.2]], **tkwargs),
            )
            self.assertTrue(mean.shape == torch.Size([2, 2]))
            self.assertTrue(variance.shape == torch.Size([2, 2, 2]))

            # Test update
            model.refit_on_update = False
            model.update(
                datasets=[FixedNoiseDataset(Xs2[0], Y=Ys2[0], Yvar=Yvars2[0])] * 2,
                metric_names=["y1", "y2"],
            )

            # Test feature_importances
            importances = model.feature_importances()
            self.assertEqual(importances.shape, torch.Size([2, 1, 3]))

            # When calling update directly, the data is completely overwritten.
            self.assertTrue(torch.equal(model.Xs[0], Xs2[0]))
            self.assertTrue(torch.equal(model.Xs[1], Xs2[0]))
            self.assertTrue(torch.equal(model.Ys[0], Ys2[0]))
            self.assertTrue(torch.equal(model.Yvars[0], Yvars2[0]))

            model.refit_on_update = True
            with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
                model.update(
                    datasets=[FixedNoiseDataset(Xs2[0], Y=Ys2[0], Yvar=Yvars2[0])] * 2,
                    metric_names=["y1", "y2"],
                )

            # test unfit model CV, update, and feature_importances
            unfit_model = BotorchModel()
            with self.assertRaises(RuntimeError):
                unfit_model.cross_validate(
                    datasets=combined_datasets,
                    metric_names=["y1", "y2"],
                    X_test=Xs1[0],
                )
            with self.assertRaises(RuntimeError):
                unfit_model.update(
                    datasets=combined_datasets, metric_names=["y1", "y2"]
                )
            with self.assertRaises(RuntimeError):
                unfit_model.feature_importances()

            # Test loading state dict
            true_state_dict = {
                "mean_module.raw_constant": 3.5004,
                "covar_module.raw_outputscale": 2.2438,
                "covar_module.base_kernel.raw_lengthscale": [
                    [-0.9274, -0.9274, -0.9274]
                ],
                "covar_module.base_kernel.raw_lengthscale_constraint.lower_bound": 0.1,
                "covar_module.base_kernel.raw_lengthscale_constraint.upper_bound": 2.5,
                "covar_module.base_kernel.lengthscale_prior.concentration": 3.0,
                "covar_module.base_kernel.lengthscale_prior.rate": 6.0,
                "covar_module.raw_outputscale_constraint.lower_bound": 0.2,
                "covar_module.raw_outputscale_constraint.upper_bound": 2.6,
                "covar_module.outputscale_prior.concentration": 2.0,
                "covar_module.outputscale_prior.rate": 0.15,
            }
            true_state_dict = {
                key: torch.tensor(val, **tkwargs)
                for key, val in true_state_dict.items()
            }
            model = get_and_fit_model(
                Xs=Xs1,
                Ys=Ys1,
                Yvars=Yvars1,
                task_features=[],
                fidelity_features=[],
                metric_names=[mns[0]],
                state_dict=true_state_dict,
                refit_model=False,
            )
            for k, v in chain(model.named_parameters(), model.named_buffers()):
                self.assertTrue(torch.equal(true_state_dict[k], v))

            # Test for some change in model parameters & buffer for refit_model=True
            true_state_dict["mean_module.raw_constant"] += 0.1
            true_state_dict["covar_module.raw_outputscale"] += 0.1
            true_state_dict["covar_module.base_kernel.raw_lengthscale"] += 0.1
            model = get_and_fit_model(
                Xs=Xs1,
                Ys=Ys1,
                Yvars=Yvars1,
                task_features=[],
                fidelity_features=[],
                metric_names=[mns[0]],
                state_dict=true_state_dict,
                refit_model=True,
            )
            self.assertTrue(
                any(
                    not torch.equal(true_state_dict[k], v)
                    for k, v in chain(model.named_parameters(), model.named_buffers())
                )
            )

        # Test that recommend_best_out_of_sample_point errors w/o _get_best_point_acqf
        model = BotorchModel(best_point_recommender=recommend_best_out_of_sample_point)
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                # pyre-fixme[61]: `datasets` is undefined, or not always defined.
                datasets=datasets,
                metric_names=mns,
                search_space_digest=SearchSpaceDigest(
                    feature_names=fns,
                    bounds=bounds,
                    task_features=tfs,
                ),
            )
        with self.assertRaises(RuntimeError):
            xbest = model.best_point(
                # pyre-fixme[61]: `search_space_digest` is undefined, or not always
                #  defined.
                search_space_digest=search_space_digest,
                # pyre-fixme[61]: `torch_opt_config` is undefined, or not always
                #  defined.
                torch_opt_config=torch_opt_config,
            )

    def test_BotorchModel_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_BotorchModel(cuda=True)

    def test_BotorchModel_double(self) -> None:
        self.test_BotorchModel(dtype=torch.double)

    def test_BotorchModel_double_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_BotorchModel(dtype=torch.double, cuda=True)

    def test_BotorchModelOneOutcome(self) -> None:
        Xs1, Ys1, Yvars1, bounds, tfs, fns, mns = get_torch_test_data(
            dtype=torch.float, cuda=False, constant_noise=True
        )
        for use_input_warping, use_loocv_pseudo_likelihood in product(
            (True, False), (True, False)
        ):
            model = BotorchModel(
                use_input_warping=use_input_warping,
                use_loocv_pseudo_likelihood=use_loocv_pseudo_likelihood,
            )
            with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
                model.fit(
                    datasets=[FixedNoiseDataset(X=Xs1[0], Y=Ys1[0], Yvar=Yvars1[0])],
                    metric_names=mns[:1],
                    search_space_digest=SearchSpaceDigest(
                        feature_names=fns,
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
                # pyre-fixme[16]: Optional type has no attribute `input_transform`.
                self.assertIsInstance(model.model.input_transform, Warp)
            else:
                self.assertFalse(hasattr(model.model, "input_transform"))

    def test_BotorchModelConstraints(self) -> None:
        Xs1, Ys1, Yvars1, bounds, tfs, fns, mns = get_torch_test_data(
            dtype=torch.float, cuda=False, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = get_torch_test_data(
            dtype=torch.float, cuda=False, constant_noise=True
        )
        # make infeasible
        Xs2[0] = -1 * Xs2[0]
        objective_weights = torch.tensor(
            [-1.0, 1.0], dtype=torch.float, device=torch.device("cpu")
        )
        n = 3
        model = BotorchModel()
        search_space_digest = SearchSpaceDigest(
            feature_names=fns,
            bounds=bounds,
            task_features=tfs,
        )
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                datasets=[
                    FixedNoiseDataset(X=Xs1[0], Y=Ys1[0], Yvar=Yvars1[0]),
                    FixedNoiseDataset(X=Xs2[0], Y=Ys2[0], Yvar=Yvars2[0]),
                ],
                metric_names=mns,
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
        _, _, _, bounds, tfs, fns, mns = get_torch_test_data(
            dtype=torch.float, cuda=False, constant_noise=True
        )
        search_space_digest = SearchSpaceDigest(
            feature_names=fns,
            bounds=bounds,
            task_features=tfs,
        )
        model = BotorchModel()
        with self.assertRaisesRegex(
            DataRequiredError, "BotorchModel.fit requires non-empty data sets."
        ):
            model.fit(
                datasets=[], metric_names=mns, search_space_digest=search_space_digest
            )
