#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict
from unittest import mock

import torch
from ax.models.torch.botorch_defaults import get_NEI
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.botorch_moo_defaults import get_EHVI
from ax.models.torch.utils import HYPERSPHERE
from ax.utils.common.testutils import TestCase
from botorch.acquisition.multi_objective import monte_carlo as moo_monte_carlo
from botorch.models import ModelListGP
from botorch.models.transforms.input import Warp
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization


FIT_MODEL_MO_PATH = "ax.models.torch.botorch_defaults.fit_gpytorch_model"
SAMPLE_SIMPLEX_UTIL_PATH = "ax.models.torch.utils.sample_simplex"
SAMPLE_HYPERSPHERE_UTIL_PATH = "ax.models.torch.utils.sample_hypersphere"
CHEBYSHEV_SCALARIZATION_PATH = (
    "ax.models.torch.botorch_defaults.get_chebyshev_scalarization"
)
EHVI_ACQF_PATH = (
    "botorch.acquisition.utils.moo_monte_carlo.qExpectedHypervolumeImprovement"
)
PARTITIONING_PATH = "botorch.acquisition.utils.NondominatedPartitioning"


def dummy_func(X: torch.Tensor) -> torch.Tensor:
    return X


def _get_optimizer_kwargs() -> Dict[str, int]:
    return {"num_restarts": 2, "raw_samples": 2, "maxiter": 2, "batch_limit": 1}


def _get_torch_test_data(
    dtype=torch.float, cuda=False, constant_noise=True, task_features=None
):
    device = torch.device("cuda") if cuda else torch.device("cpu")
    Xs = [torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=dtype, device=device)]
    Ys = [torch.tensor([[3.0], [4.0]], dtype=dtype, device=device)]
    Yvars = [torch.tensor([[0.0], [2.0]], dtype=dtype, device=device)]
    if constant_noise:
        Yvars[0].fill_(1.0)
    bounds = [(0.0, 1.0), (1.0, 4.0), (2.0, 5.0)]
    feature_names = ["x1", "x2", "x3"]
    task_features = [] if task_features is None else task_features
    metric_names = ["y", "r"]
    return Xs, Ys, Yvars, bounds, task_features, feature_names, metric_names


class BotorchMOOModelTest(TestCase):
    def test_BotorchMOOModel_cuda(self):
        if torch.cuda.is_available():
            self.test_BotorchMOOModel_with_random_scalarization(cuda=True)

    def test_BotorchMOOModel_double(self):
        self.test_BotorchMOOModel_with_random_scalarization(dtype=torch.double)

    def test_BotorchMOOModel_double_cuda(self):
        if torch.cuda.is_available():
            self.test_BotorchMOOModel_with_random_scalarization(
                dtype=torch.double, cuda=True
            )

    def test_BotorchMOOModel_with_random_scalarization(
        self, dtype=torch.float, cuda=False
    ):
        tkwargs = {
            "device": torch.device("cuda") if cuda else torch.device("cpu"),
            "dtype": dtype,
        }
        Xs1, Ys1, Yvars1, bounds, tfs, fns, mns = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        n = 3
        objective_weights = torch.tensor([1.0, 1.0], **tkwargs)

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)
        acqfv_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)

        model = MultiObjectiveBotorchModel(acqf_constructor=get_NEI)
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1 + Xs2,
                Ys=Ys1 + Ys2,
                Yvars=Yvars1 + Yvars2,
                bounds=bounds,
                task_features=tfs,
                feature_names=fns,
                metric_names=mns,
                fidelity_features=[],
            )
            _mock_fit_model.assert_called_once()

        with mock.patch(
            SAMPLE_SIMPLEX_UTIL_PATH,
            autospec=True,
            return_value=torch.tensor([0.7, 0.3], **tkwargs),
        ) as _mock_sample_simplex, mock.patch(
            "ax.models.torch.botorch_defaults.optimize_acqf",
            return_value=(X_dummy, acqfv_dummy),
        ) as _:
            model.gen(
                n,
                bounds,
                objective_weights,
                model_gen_options={
                    "acquisition_function_kwargs": {"random_scalarization": True},
                    "optimizer_kwargs": _get_optimizer_kwargs(),
                },
            )
            # Sample_simplex should be called once for generated candidate.
            self.assertEqual(n, _mock_sample_simplex.call_count)

        with mock.patch(
            SAMPLE_HYPERSPHERE_UTIL_PATH,
            autospec=True,
            return_value=torch.tensor([0.6, 0.8], **tkwargs),
        ) as _mock_sample_hypersphere, mock.patch(
            "ax.models.torch.botorch_defaults.optimize_acqf",
            return_value=(X_dummy, acqfv_dummy),
        ) as _:
            model.gen(
                n,
                bounds,
                objective_weights,
                model_gen_options={
                    "acquisition_function_kwargs": {
                        "random_scalarization": True,
                        "random_scalarization_distribution": HYPERSPHERE,
                    },
                    "optimizer_kwargs": _get_optimizer_kwargs(),
                },
            )
            # Sample_simplex should be called once per generated candidate.
            self.assertEqual(n, _mock_sample_hypersphere.call_count)

        # test input warping
        self.assertFalse(model.use_input_warping)
        model = MultiObjectiveBotorchModel(
            acqf_constructor=get_NEI, use_input_warping=True
        )
        model.fit(
            Xs=Xs1 + Xs2,
            Ys=Ys1 + Ys2,
            Yvars=Yvars1 + Yvars2,
            bounds=bounds,
            task_features=tfs,
            feature_names=fns,
            metric_names=mns,
            fidelity_features=[],
        )
        self.assertTrue(model.use_input_warping)
        self.assertIsInstance(model.model, ModelListGP)
        for m in model.model.models:
            self.assertTrue(hasattr(m, "input_transform"))
            self.assertIsInstance(m.input_transform, Warp)
        self.assertFalse(hasattr(model.model, "input_transform"))

        # test loocv pseudo likelihood
        self.assertFalse(model.use_loocv_pseudo_likelihood)
        model = MultiObjectiveBotorchModel(
            acqf_constructor=get_NEI, use_loocv_pseudo_likelihood=True
        )
        model.fit(
            Xs=Xs1 + Xs2,
            Ys=Ys1 + Ys2,
            Yvars=Yvars1 + Yvars2,
            bounds=bounds,
            task_features=tfs,
            feature_names=fns,
            metric_names=mns,
            fidelity_features=[],
        )
        self.assertTrue(model.use_loocv_pseudo_likelihood)

    def test_BotorchMOOModel_with_chebyshev_scalarization(
        self, dtype=torch.float, cuda=False
    ):
        tkwargs = {
            "device": torch.device("cuda") if cuda else torch.device("cpu"),
            "dtype": dtype,
        }
        Xs1, Ys1, Yvars1, bounds, tfs, fns, mns = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        n = 3
        objective_weights = torch.tensor([1.0, 1.0], **tkwargs)

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)
        acqfv_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)

        model = MultiObjectiveBotorchModel(acqf_constructor=get_NEI)
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1 + Xs2,
                Ys=Ys1 + Ys2,
                Yvars=Yvars1 + Yvars2,
                bounds=bounds,
                task_features=tfs,
                feature_names=fns,
                metric_names=mns,
                fidelity_features=[],
            )
            _mock_fit_model.assert_called_once()

        with mock.patch(
            CHEBYSHEV_SCALARIZATION_PATH, wraps=get_chebyshev_scalarization
        ) as _mock_chebyshev_scalarization, mock.patch(
            "ax.models.torch.botorch_defaults.optimize_acqf",
            return_value=(X_dummy, acqfv_dummy),
        ) as _:
            model.gen(
                n,
                bounds,
                objective_weights,
                model_gen_options={
                    "acquisition_function_kwargs": {"chebyshev_scalarization": True},
                    "optimizer_kwargs": _get_optimizer_kwargs(),
                },
            )
            # get_chebyshev_scalarization should be called once for generated candidate.
            self.assertEqual(n, _mock_chebyshev_scalarization.call_count)

    def test_BotorchMOOModel_with_ehvi(self, dtype=torch.float, cuda=False):
        tkwargs = {
            "device": torch.device("cuda") if cuda else torch.device("cpu"),
            "dtype": dtype,
        }
        Xs1, Ys1, Yvars1, bounds, tfs, fns, mns = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        n = 3
        objective_weights = torch.tensor([1.0, 1.0], **tkwargs)
        model = MultiObjectiveBotorchModel(acqf_constructor=get_EHVI)

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)
        acqfv_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)

        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1 + Xs2,
                Ys=Ys1 + Ys2,
                Yvars=Yvars1 + Yvars2,
                bounds=bounds,
                task_features=tfs,
                feature_names=fns,
                metric_names=mns,
                fidelity_features=[],
            )
            _mock_fit_model.assert_called_once()

        with mock.patch(
            EHVI_ACQF_PATH, wraps=moo_monte_carlo.qExpectedHypervolumeImprovement
        ) as _mock_ehvi_acqf, mock.patch(
            "ax.models.torch.botorch_defaults.optimize_acqf",
            return_value=(X_dummy, acqfv_dummy),
        ) as _, mock.patch(
            PARTITIONING_PATH, wraps=moo_monte_carlo.NondominatedPartitioning
        ) as _mock_partitioning:
            model.gen(
                n,
                bounds,
                objective_weights,
                model_gen_options={"optimizer_kwargs": _get_optimizer_kwargs()},
                objective_thresholds=torch.tensor([1.0, 1.0]),
            )
            # the EHVI acquisition function should be created only once.
            self.assertEqual(1, _mock_ehvi_acqf.call_count)
            # check partitioning strategy
            self.assertEqual(_mock_partitioning.call_args[1]["alpha"], 0.0)

        # 3 objective
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1 + Xs2 + Xs2,
                Ys=Ys1 + Ys2 + Ys2,
                Yvars=Yvars1 + Yvars2 + Yvars2,
                bounds=bounds,
                task_features=tfs,
                feature_names=fns,
                metric_names=mns,
                fidelity_features=[],
            )

        with mock.patch(
            EHVI_ACQF_PATH, wraps=moo_monte_carlo.qExpectedHypervolumeImprovement
        ) as _mock_ehvi_acqf, mock.patch(
            "ax.models.torch.botorch_defaults.optimize_acqf",
            return_value=(X_dummy, acqfv_dummy),
        ) as _, mock.patch(
            PARTITIONING_PATH, wraps=moo_monte_carlo.NondominatedPartitioning
        ) as _mock_partitioning:
            model.gen(
                n,
                bounds,
                torch.tensor([1.0, 1.0, 1.0], **tkwargs),
                model_gen_options={"optimizer_kwargs": _get_optimizer_kwargs()},
                objective_thresholds=torch.tensor([1.0, 1.0, 1.0]),
            )
            # check partitioning strategy
            self.assertEqual(_mock_partitioning.call_args[1]["alpha"], 1e-5)

    def test_BotorchMOOModel_with_random_scalarization_and_outcome_constraints(
        self, dtype=torch.float, cuda=False
    ):
        tkwargs = {
            "device": torch.device("cuda") if cuda else torch.device("cpu"),
            "dtype": dtype,
        }
        Xs1, Ys1, Yvars1, bounds, tfs, fns, mns = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        n = 2
        objective_weights = torch.tensor([1.0, 1.0], **tkwargs)
        model = MultiObjectiveBotorchModel(acqf_constructor=get_NEI)

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)
        acqfv_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)

        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1 + Xs2,
                Ys=Ys1 + Ys2,
                Yvars=Yvars1 + Yvars2,
                bounds=bounds,
                task_features=tfs,
                feature_names=fns,
                metric_names=mns,
                fidelity_features=[],
            )
            _mock_fit_model.assert_called_once()

        with mock.patch(
            SAMPLE_SIMPLEX_UTIL_PATH,
            autospec=True,
            return_value=torch.tensor([0.7, 0.3], **tkwargs),
        ) as _mock_sample_simplex, mock.patch(
            "ax.models.torch.botorch_defaults.optimize_acqf",
            return_value=(X_dummy, acqfv_dummy),
        ) as _:
            model.gen(
                n,
                bounds,
                objective_weights,
                outcome_constraints=(
                    torch.tensor([[1.0, 1.0]], **tkwargs),
                    torch.tensor([[10.0]], **tkwargs),
                ),
                model_gen_options={
                    "acquisition_function_kwargs": {"random_scalarization": True},
                    "optimizer_kwargs": _get_optimizer_kwargs(),
                },
            )
            self.assertEqual(n, _mock_sample_simplex.call_count)

    def test_BotorchMOOModel_with_chebyshev_scalarization_and_outcome_constraints(
        self, dtype=torch.float, cuda=False
    ):
        tkwargs = {
            "device": torch.device("cuda") if cuda else torch.device("cpu"),
            "dtype": torch.float,
        }
        Xs1, Ys1, Yvars1, bounds, tfs, fns, mns = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        n = 2
        objective_weights = torch.tensor([1.0, 1.0], **tkwargs)
        model = MultiObjectiveBotorchModel(acqf_constructor=get_NEI)

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)
        acqfv_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)

        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1 + Xs2,
                Ys=Ys1 + Ys2,
                Yvars=Yvars1 + Yvars2,
                bounds=bounds,
                task_features=tfs,
                feature_names=fns,
                metric_names=mns,
                fidelity_features=[],
            )
            _mock_fit_model.assert_called_once()

        with mock.patch(
            CHEBYSHEV_SCALARIZATION_PATH, wraps=get_chebyshev_scalarization
        ) as _mock_chebyshev_scalarization, mock.patch(
            "ax.models.torch.botorch_defaults.optimize_acqf",
            return_value=(X_dummy, acqfv_dummy),
        ) as _:
            model.gen(
                n,
                bounds,
                objective_weights,
                outcome_constraints=(
                    torch.tensor([[1.0, 1.0]], **tkwargs),
                    torch.tensor([[10.0]], **tkwargs),
                ),
                model_gen_options={
                    "acquisition_function_kwargs": {"chebyshev_scalarization": True},
                    "optimizer_kwargs": _get_optimizer_kwargs(),
                },
            )
            # get_chebyshev_scalarization should be called once for generated candidate.
            self.assertEqual(n, _mock_chebyshev_scalarization.call_count)

    def test_BotorchMOOModel_with_ehvi_and_outcome_constraints(
        self, dtype=torch.float, cuda=False
    ):
        tkwargs = {
            "device": torch.device("cuda") if cuda else torch.device("cpu"),
            "dtype": dtype,
        }
        Xs1, Ys1, Yvars1, bounds, tfs, fns, mns = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        n = 3
        objective_weights = torch.tensor([1.0, 1.0], **tkwargs)
        model = MultiObjectiveBotorchModel(acqf_constructor=get_EHVI)

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)
        acqfv_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)

        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1 + Xs2,
                Ys=Ys1 + Ys2,
                Yvars=Yvars1 + Yvars2,
                bounds=bounds,
                task_features=tfs,
                feature_names=fns,
                metric_names=mns,
                fidelity_features=[],
            )
            _mock_fit_model.assert_called_once()

        with mock.patch(
            EHVI_ACQF_PATH, wraps=moo_monte_carlo.qExpectedHypervolumeImprovement
        ) as _mock_ehvi_acqf, mock.patch(
            "ax.models.torch.botorch_defaults.optimize_acqf",
            return_value=(X_dummy, acqfv_dummy),
        ) as _:
            model.gen(
                n,
                bounds,
                objective_weights,
                outcome_constraints=(
                    torch.tensor([[1.0, 1.0]], **tkwargs),
                    torch.tensor([[10.0]], **tkwargs),
                ),
                model_gen_options={"optimizer_kwargs": _get_optimizer_kwargs()},
                objective_thresholds=torch.tensor([1.0, 1.0]),
            )
            # the EHVI acquisition function should be created only once.
            self.assertEqual(1, _mock_ehvi_acqf.call_count)
