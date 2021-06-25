#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from typing import Dict
from unittest import mock

import ax.models.torch.botorch_moo as botorch_moo
import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxError
from ax.models.torch.botorch_defaults import get_NEI
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.utils import HYPERSPHERE, _get_X_pending_and_observed
from ax.utils.common.testutils import TestCase
from botorch.acquisition.multi_objective import monte_carlo as moo_monte_carlo
from botorch.models import ModelListGP, FixedNoiseGP
from botorch.models.transforms.input import Warp
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.testing import MockPosterior

FIT_MODEL_MO_PATH = "ax.models.torch.botorch_defaults.fit_gpytorch_model"
SAMPLE_SIMPLEX_UTIL_PATH = "ax.models.torch.utils.sample_simplex"
SAMPLE_HYPERSPHERE_UTIL_PATH = "ax.models.torch.utils.sample_hypersphere"
CHEBYSHEV_SCALARIZATION_PATH = (
    "ax.models.torch.botorch_defaults.get_chebyshev_scalarization"
)
EHVI_ACQF_PATH = (
    "botorch.acquisition.utils.moo_monte_carlo.qExpectedHypervolumeImprovement"
)
PARTITIONING_PATH = "botorch.acquisition.utils.FastNondominatedPartitioning"


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
        obj_t = torch.tensor([1.0, 1.0], **tkwargs)

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)
        acqfv_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)

        model = MultiObjectiveBotorchModel(acqf_constructor=get_NEI)
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1 + Xs2,
                Ys=Ys1 + Ys2,
                Yvars=Yvars1 + Yvars2,
                search_space_digest=SearchSpaceDigest(
                    feature_names=fns,
                    bounds=bounds,
                    task_features=tfs,
                ),
                metric_names=mns,
            )
            _mock_fit_model.assert_called_once()

        with mock.patch(
            SAMPLE_SIMPLEX_UTIL_PATH,
            autospec=True,
            return_value=torch.tensor([0.7, 0.3], **tkwargs),
        ) as _mock_sample_simplex, mock.patch(
            "ax.models.torch.botorch_moo_defaults.optimize_acqf_list",
            return_value=(X_dummy, acqfv_dummy),
        ) as _:
            model.gen(
                n,
                bounds,
                objective_weights,
                objective_thresholds=obj_t,
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
            "ax.models.torch.botorch_moo_defaults.optimize_acqf_list",
            return_value=(X_dummy, acqfv_dummy),
        ) as _:
            model.gen(
                n,
                bounds,
                objective_weights,
                objective_thresholds=obj_t,
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
            search_space_digest=SearchSpaceDigest(
                feature_names=fns,
                bounds=bounds,
                task_features=tfs,
            ),
            metric_names=mns,
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
            search_space_digest=SearchSpaceDigest(
                feature_names=fns,
                bounds=bounds,
                task_features=tfs,
            ),
            metric_names=mns,
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
        obj_t = torch.tensor([1.0, 1.0], **tkwargs)

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)
        acqfv_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)

        model = MultiObjectiveBotorchModel(acqf_constructor=get_NEI)
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1 + Xs2,
                Ys=Ys1 + Ys2,
                Yvars=Yvars1 + Yvars2,
                search_space_digest=SearchSpaceDigest(
                    feature_names=fns,
                    bounds=bounds,
                    task_features=tfs,
                ),
                metric_names=mns,
            )
            _mock_fit_model.assert_called_once()

        with mock.patch(
            CHEBYSHEV_SCALARIZATION_PATH, wraps=get_chebyshev_scalarization
        ) as _mock_chebyshev_scalarization, mock.patch(
            "ax.models.torch.botorch_moo_defaults.optimize_acqf_list",
            return_value=(X_dummy, acqfv_dummy),
        ) as _:
            model.gen(
                n,
                bounds,
                objective_weights,
                objective_thresholds=obj_t,
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
        obj_t = torch.tensor([1.0, 1.0], **tkwargs)
        model = MultiObjectiveBotorchModel()

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)
        acqfv_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)

        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1 + Xs2,
                Ys=Ys1 + Ys2,
                Yvars=Yvars1 + Yvars2,
                search_space_digest=SearchSpaceDigest(
                    feature_names=fns,
                    bounds=bounds,
                    task_features=tfs,
                ),
                metric_names=mns,
            )
            _mock_fit_model.assert_called_once()

        with mock.patch(
            EHVI_ACQF_PATH, wraps=moo_monte_carlo.qExpectedHypervolumeImprovement
        ) as _mock_ehvi_acqf, mock.patch(
            "ax.models.torch.botorch_defaults.optimize_acqf",
            return_value=(X_dummy, acqfv_dummy),
        ) as _, mock.patch(
            PARTITIONING_PATH, wraps=moo_monte_carlo.FastNondominatedPartitioning
        ) as _mock_partitioning:
            _, _, gen_metadata, _ = model.gen(
                n,
                bounds,
                objective_weights,
                objective_thresholds=obj_t,
                model_gen_options={"optimizer_kwargs": _get_optimizer_kwargs()},
            )
            # the EHVI acquisition function should be created only once.
            self.assertEqual(1, _mock_ehvi_acqf.call_count)
            # check partitioning strategy
            _mock_partitioning.assert_called_once()
            self.assertTrue(torch.equal(gen_metadata["objective_thresholds"], obj_t))

        # 3 objective
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1 + Xs2 + Xs2,
                Ys=Ys1 + Ys2 + Ys2,
                Yvars=Yvars1 + Yvars2 + Yvars2,
                search_space_digest=SearchSpaceDigest(
                    feature_names=fns,
                    bounds=bounds,
                    task_features=tfs,
                ),
                metric_names=mns,
            )

        with mock.patch(
            EHVI_ACQF_PATH, wraps=moo_monte_carlo.qExpectedHypervolumeImprovement
        ) as _mock_ehvi_acqf, mock.patch(
            "ax.models.torch.botorch_defaults.optimize_acqf",
            return_value=(X_dummy, acqfv_dummy),
        ) as _, mock.patch(
            PARTITIONING_PATH, wraps=moo_monte_carlo.FastNondominatedPartitioning
        ) as _mock_partitioning:
            model.gen(
                n,
                bounds,
                torch.tensor([1.0, 1.0, 1.0], **tkwargs),
                model_gen_options={"optimizer_kwargs": _get_optimizer_kwargs()},
                objective_thresholds=torch.tensor([1.0, 1.0, 1.0], **tkwargs),
            )
            # check partitioning strategy
            _mock_partitioning.assert_called_once()

        # test inferred objective thresholds in gen()
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            # create several data points
            Xs1 = [torch.cat([Xs1[0], Xs1[0] - 0.1], dim=0)]
            Ys1 = [torch.cat([Ys1[0], Ys1[0] - 0.5], dim=0)]
            Ys2 = [torch.cat([Ys2[0], Ys2[0] + 0.5], dim=0)]
            Yvars1 = [torch.cat([Yvars1[0], Yvars1[0] + 0.2], dim=0)]
            Yvars2 = [torch.cat([Yvars2[0], Yvars2[0] + 0.1], dim=0)]
            model.fit(
                Xs=Xs1 + Xs1,
                Ys=Ys1 + Ys2,
                Yvars=Yvars1 + Yvars2,
                search_space_digest=SearchSpaceDigest(
                    feature_names=fns,
                    bounds=bounds,
                    task_features=tfs,
                ),
                metric_names=mns + ["dummy_metric"],
            )
        with ExitStack() as es:
            _mock_ehvi_acqf = es.enter_context(
                mock.patch(
                    EHVI_ACQF_PATH,
                    wraps=moo_monte_carlo.qExpectedHypervolumeImprovement,
                )
            )
            es.enter_context(
                mock.patch(
                    "ax.models.torch.botorch_defaults.optimize_acqf",
                    return_value=(X_dummy, acqfv_dummy),
                )
            )
            _mock_partitioning = es.enter_context(
                mock.patch(
                    PARTITIONING_PATH,
                    wraps=moo_monte_carlo.FastNondominatedPartitioning,
                )
            )
            _mock_model_infer_objective_thresholds = es.enter_context(
                mock.patch.object(
                    model,
                    "infer_objective_thresholds",
                    wraps=model.infer_objective_thresholds,
                )
            )
            _mock_infer_reference_point = es.enter_context(
                mock.patch(
                    "ax.models.torch.botorch_moo.infer_reference_point",
                    wraps=infer_reference_point,
                )
            )
            es.enter_context(
                mock.patch.object(
                    model.model,
                    "posterior",
                    return_value=MockPosterior(
                        mean=torch.tensor(
                            [
                                [11.0, 2.0, 0.0],
                                [9.0, 3.0, 0.0],
                            ]
                        )
                    ),
                )
            )
            outcome_constraints = (
                torch.tensor([[1.0, 0.0, 0.0]]),
                torch.tensor([[10.0]]),
            )
            _, _, gen_metadata, _ = model.gen(
                n,
                bounds,
                objective_weights=torch.tensor([-1.0, -1.0, 0.0]),
                outcome_constraints=outcome_constraints,
                model_gen_options={"optimizer_kwargs": _get_optimizer_kwargs()},
            )
            # the EHVI acquisition function should be created only once.
            self.assertEqual(1, _mock_ehvi_acqf.call_count)
            ckwargs = _mock_model_infer_objective_thresholds.call_args[1]
            X_observed = ckwargs["X_observed"]
            sorted_idcs = X_observed[:, 0].argsort()
            expected_X_observed = torch.tensor([[1.0, 2.0, 3.0], [0.9, 1.9, 2.9]])
            sorted_idcs2 = expected_X_observed[:, 0].argsort()
            self.assertTrue(
                torch.equal(
                    X_observed[sorted_idcs],
                    expected_X_observed[sorted_idcs2],
                )
            )
            self.assertTrue(
                torch.equal(
                    ckwargs["objective_weights"], torch.tensor([-1.0, -1.0, 0.0])
                )
            )
            oc = ckwargs["outcome_constraints"]
            self.assertTrue(torch.equal(oc[0], outcome_constraints[0]))
            self.assertTrue(torch.equal(oc[1], outcome_constraints[1]))
            self.assertIsInstance(ckwargs["model"], FixedNoiseGP)
            self.assertTrue(torch.equal(ckwargs["subset_idcs"], torch.tensor([0, 1])))
            _mock_infer_reference_point.assert_called_once()
            ckwargs = _mock_infer_reference_point.call_args[1]
            self.assertEqual(ckwargs["scale"], 0.1)
            self.assertTrue(
                torch.equal(ckwargs["pareto_Y"], torch.tensor([[-9.0, -3.0]]))
            )
            self.assertIn("objective_thresholds", gen_metadata)
            obj_t = gen_metadata["objective_thresholds"]
            self.assertTrue(torch.equal(obj_t[:2], torch.tensor([9.9, 3.3])))
            self.assertTrue(np.isnan(obj_t[2]))

        # test infer objective thresholds alone
        # include an extra 3rd outcome
        outcome_constraints = (torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[10.0]]))
        with ExitStack() as es:
            _mock_infer_reference_point = es.enter_context(
                mock.patch(
                    "ax.models.torch.botorch_moo.infer_reference_point",
                    wraps=infer_reference_point,
                )
            )
            _mock_get_X_pending_and_observed = es.enter_context(
                mock.patch(
                    "ax.models.torch.botorch_moo._get_X_pending_and_observed",
                    wraps=_get_X_pending_and_observed,
                )
            )
            es.enter_context(
                mock.patch.object(
                    model.model,
                    "posterior",
                    return_value=MockPosterior(
                        mean=torch.tensor(
                            [
                                [11.0, 2.0, 0.0],
                                [9.0, 3.0, 0.0],
                            ]
                        )
                    ),
                )
            )
            linear_constraints = (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([2.0]))
            objective_weights = torch.tensor([-1.0, -1.0, 0.0])
            obj_thresholds = model.infer_objective_thresholds(
                bounds=bounds,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                fixed_features={},
                linear_constraints=linear_constraints,
            )
            _mock_get_X_pending_and_observed.assert_called_once()
            ckwargs = _mock_get_X_pending_and_observed.call_args[1]
            actual_Xs = ckwargs["Xs"]
            for X in actual_Xs:
                self.assertTrue(torch.equal(X, Xs1[0]))
            self.assertEqual(ckwargs["bounds"], bounds)
            self.assertTrue(
                torch.equal(ckwargs["objective_weights"], objective_weights)
            )
            oc = ckwargs["outcome_constraints"]
            self.assertTrue(torch.equal(oc[0], outcome_constraints[0]))
            self.assertTrue(torch.equal(oc[1], outcome_constraints[1]))
            self.assertEqual(ckwargs["fixed_features"], {})
            lc = ckwargs["linear_constraints"]
            self.assertTrue(torch.equal(lc[0], linear_constraints[0]))
            self.assertTrue(torch.equal(lc[1], linear_constraints[1]))
            _mock_infer_reference_point.assert_called_once()
            ckwargs = _mock_infer_reference_point.call_args[1]
            self.assertEqual(ckwargs["scale"], 0.1)
            self.assertTrue(
                torch.equal(ckwargs["pareto_Y"], torch.tensor([[-9.0, -3.0]]))
            )
            self.assertTrue(torch.equal(obj_thresholds[:2], torch.tensor([9.9, 3.3])))
            self.assertTrue(np.isnan(obj_thresholds[2].item()))

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
        obj_t = torch.tensor([1.0, 1.0], **tkwargs)
        model = MultiObjectiveBotorchModel(acqf_constructor=get_NEI)

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)
        acqfv_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)

        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1 + Xs2,
                Ys=Ys1 + Ys2,
                Yvars=Yvars1 + Yvars2,
                search_space_digest=SearchSpaceDigest(
                    feature_names=fns,
                    bounds=bounds,
                    task_features=tfs,
                ),
                metric_names=mns,
            )
            _mock_fit_model.assert_called_once()

        with mock.patch(
            SAMPLE_SIMPLEX_UTIL_PATH,
            autospec=True,
            return_value=torch.tensor([0.7, 0.3], **tkwargs),
        ) as _mock_sample_simplex, mock.patch(
            "ax.models.torch.botorch_moo_defaults.optimize_acqf_list",
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
                objective_thresholds=obj_t,
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
        obj_t = torch.tensor([1.0, 1.0], **tkwargs)
        model = MultiObjectiveBotorchModel(acqf_constructor=get_NEI)

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)
        acqfv_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)

        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1 + Xs2,
                Ys=Ys1 + Ys2,
                Yvars=Yvars1 + Yvars2,
                search_space_digest=SearchSpaceDigest(
                    feature_names=fns,
                    bounds=bounds,
                    task_features=tfs,
                ),
                metric_names=mns,
            )
            _mock_fit_model.assert_called_once()

        with mock.patch(
            CHEBYSHEV_SCALARIZATION_PATH, wraps=get_chebyshev_scalarization
        ) as _mock_chebyshev_scalarization, mock.patch(
            "ax.models.torch.botorch_moo_defaults.optimize_acqf_list",
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
                objective_thresholds=obj_t,
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
        Xs3, Ys3, Yvars3, _, _, _, _ = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        n = 3
        objective_weights = torch.tensor([1.0, 1.0, 0.0], **tkwargs)
        obj_t = torch.tensor([1.0, 1.0, 1.0], **tkwargs)
        model = MultiObjectiveBotorchModel()

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)
        acqfv_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)

        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                Xs=Xs1 + Xs2 + Xs3,
                Ys=Ys1 + Ys2 + Ys3,
                Yvars=Yvars1 + Yvars2 + Yvars3,
                search_space_digest=SearchSpaceDigest(
                    feature_names=fns,
                    bounds=bounds,
                    task_features=tfs,
                ),
                metric_names=mns,
            )
            _mock_fit_model.assert_called_once()

        # test wrong number of objective thresholds
        with self.assertRaises(AxError):
            model.gen(
                n,
                bounds,
                objective_weights,
                objective_thresholds=torch.tensor([1.0, 1.0]),
            )
        # test that objective thresholds and weights are properly subsetted
        obj_t = torch.tensor([1.0, 1.0, 1.0])
        with mock.patch.object(
            model,
            "acqf_constructor",
            wraps=botorch_moo.get_EHVI,
        ) as mock_get_ehvi, mock.patch(
            "ax.models.torch.botorch_defaults.optimize_acqf",
            return_value=(X_dummy, acqfv_dummy),
        ):
            model.gen(
                n,
                bounds,
                objective_weights,
                model_gen_options={"optimizer_kwargs": _get_optimizer_kwargs()},
                objective_thresholds=obj_t,
            )
            mock_get_ehvi.assert_called_once()
            _, ckwargs = mock_get_ehvi.call_args
            self.assertEqual(ckwargs["model"].num_outputs, 2)
            self.assertTrue(
                torch.equal(ckwargs["objective_weights"], objective_weights[:-1])
            )
            self.assertTrue(torch.equal(ckwargs["objective_thresholds"], obj_t[:-1]))
            self.assertIsNone(ckwargs["outcome_constraints"])
            # the second datapoint is out of bounds
            self.assertTrue(torch.equal(ckwargs["X_observed"], Xs1[0][:1]))
            self.assertIsNone(ckwargs["X_pending"])
        # test that outcome constraints are passed properly
        oc = (
            torch.tensor([[0.0, 0.0, 1.0]], **tkwargs),
            torch.tensor([[10.0]], **tkwargs),
        )
        with mock.patch.object(
            model,
            "acqf_constructor",
            wraps=botorch_moo.get_EHVI,
        ) as mock_get_ehvi, mock.patch(
            "ax.models.torch.botorch_defaults.optimize_acqf",
            return_value=(X_dummy, acqfv_dummy),
        ):
            model.gen(
                n,
                bounds,
                objective_weights,
                outcome_constraints=oc,
                model_gen_options={"optimizer_kwargs": _get_optimizer_kwargs()},
                objective_thresholds=obj_t,
            )
            mock_get_ehvi.assert_called_once()
            _, ckwargs = mock_get_ehvi.call_args
            self.assertEqual(ckwargs["model"].num_outputs, 3)
            self.assertTrue(
                torch.equal(ckwargs["objective_weights"], objective_weights)
            )
            self.assertTrue(torch.equal(ckwargs["objective_thresholds"], obj_t))
            self.assertTrue(torch.equal(ckwargs["outcome_constraints"][0], oc[0]))
            self.assertTrue(torch.equal(ckwargs["outcome_constraints"][1], oc[1]))
            # the second datapoint is out of bounds
            self.assertTrue(torch.equal(ckwargs["X_observed"], Xs1[0][:1]))
            self.assertIsNone(ckwargs["X_pending"])
