#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack
from typing import Dict
from unittest import mock

import ax.models.torch.botorch_moo_defaults as botorch_moo_defaults
import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxError
from ax.models.torch.botorch_defaults import get_NEI
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.botorch_moo_defaults import (
    get_NEHVI,
    get_EHVI,
    infer_objective_thresholds,
)
from ax.models.torch.utils import HYPERSPHERE
from ax.utils.common.testutils import TestCase
from botorch.acquisition.multi_objective import monte_carlo as moo_monte_carlo
from botorch.models import ModelListGP
from botorch.models.transforms.input import Warp
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.testing import MockModel, MockPosterior

FIT_MODEL_MO_PATH = "ax.models.torch.botorch_defaults.fit_gpytorch_model"
SAMPLE_SIMPLEX_UTIL_PATH = "ax.models.torch.utils.sample_simplex"
SAMPLE_HYPERSPHERE_UTIL_PATH = "ax.models.torch.utils.sample_hypersphere"
CHEBYSHEV_SCALARIZATION_PATH = (
    "ax.models.torch.botorch_defaults.get_chebyshev_scalarization"
)
NEHVI_ACQF_PATH = (
    "botorch.acquisition.utils.moo_monte_carlo.qNoisyExpectedHypervolumeImprovement"
)
EHVI_ACQF_PATH = (
    "botorch.acquisition.utils.moo_monte_carlo.qExpectedHypervolumeImprovement"
)
NEHVI_PARTITIONING_PATH = (
    "botorch.acquisition.multi_objective.monte_carlo.FastNondominatedPartitioning"
)
EHVI_PARTITIONING_PATH = "botorch.acquisition.utils.FastNondominatedPartitioning"


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
    def test_BotorchMOOModel_double(self):
        self.test_BotorchMOOModel_with_random_scalarization(dtype=torch.double)

    def test_BotorchMOOModel_cuda(self):
        if torch.cuda.is_available():
            for dtype in (torch.float, torch.double):
                self.test_BotorchMOOModel_with_random_scalarization(
                    dtype=dtype, cuda=True
                )
                # test qEHVI
                self.test_BotorchMOOModel_with_qehvi(
                    dtype=dtype, cuda=True, use_qnehvi=False
                )
                self.test_BotorchMOOModel_with_qehvi_and_outcome_constraints(
                    dtype=dtype, cuda=True, use_qnehvi=False
                )
                # test qNEHVI
                self.test_BotorchMOOModel_with_qehvi(
                    dtype=dtype, cuda=True, use_qnehvi=True
                )
                self.test_BotorchMOOModel_with_qehvi_and_outcome_constraints(
                    dtype=dtype, cuda=True, use_qnehvi=True
                )

    def test_BotorchMOOModel_with_qnehvi(self):
        for dtype in (torch.float, torch.double):
            self.test_BotorchMOOModel_with_qehvi(dtype=dtype, use_qnehvi=True)
            self.test_BotorchMOOModel_with_qehvi_and_outcome_constraints(
                dtype=dtype, use_qnehvi=True
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

    def test_BotorchMOOModel_with_qehvi(
        self, dtype=torch.float, cuda=False, use_qnehvi=False
    ):
        if use_qnehvi:
            acqf_constructor = get_NEHVI
            partitioning_path = NEHVI_PARTITIONING_PATH
        else:
            acqf_constructor = get_EHVI
            partitioning_path = EHVI_PARTITIONING_PATH
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
        model = MultiObjectiveBotorchModel(acqf_constructor=acqf_constructor)

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
        with ExitStack() as es:
            _mock_acqf = es.enter_context(
                mock.patch(
                    NEHVI_ACQF_PATH,
                    wraps=moo_monte_carlo.qNoisyExpectedHypervolumeImprovement,
                )
            )
            if use_qnehvi:
                _mock_acqf = es.enter_context(
                    mock.patch(
                        NEHVI_ACQF_PATH,
                        wraps=moo_monte_carlo.qNoisyExpectedHypervolumeImprovement,
                    )
                )
            else:
                _mock_acqf = es.enter_context(
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
                    partitioning_path,
                    wraps=moo_monte_carlo.FastNondominatedPartitioning,
                )
            )
            _, _, gen_metadata, _ = model.gen(
                n,
                bounds,
                objective_weights,
                objective_thresholds=obj_t,
                model_gen_options={"optimizer_kwargs": _get_optimizer_kwargs()},
            )
            # the NEHVI acquisition function should be created only once.
            self.assertEqual(1, _mock_acqf.call_count)
            # check partitioning strategy
            # NEHVI should call FastNondominatedPartitioning 1 time
            # since a batched partitioning is used for 2 objectives
            _mock_partitioning.assert_called_once()
            self.assertTrue(
                torch.equal(gen_metadata["objective_thresholds"], obj_t.cpu())
            )
            _mock_fit_model = es.enter_context(mock.patch(FIT_MODEL_MO_PATH))
            # 3 objective
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
            model.gen(
                n,
                bounds,
                torch.tensor([1.0, 1.0, 1.0], **tkwargs),
                model_gen_options={"optimizer_kwargs": _get_optimizer_kwargs()},
                objective_thresholds=torch.tensor([1.0, 1.0, 1.0], **tkwargs),
            )
            # check partitioning strategy
            # NEHVI should call FastNondominatedPartitioning 129 times because
            # we have called gen twice: The first time, a batch partitioning is used
            # so there is one call to _mock_partitioning. The second time gen() is
            # called with three objectives so 128 calls are made to _mock_partitioning
            # because a BoxDecompositionList is used. qEHVI will only make 2 calls.
            self.assertEqual(
                len(_mock_partitioning.mock_calls), 129 if use_qnehvi else 2
            )

            # test inferred objective thresholds in gen()
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
            _mock_model_infer_objective_thresholds = es.enter_context(
                mock.patch(
                    "ax.models.torch.botorch_moo.infer_objective_thresholds",
                    wraps=infer_objective_thresholds,
                )
            )
            _mock_infer_reference_point = es.enter_context(
                mock.patch(
                    "ax.models.torch.botorch_moo_defaults.infer_reference_point",
                    wraps=infer_reference_point,
                )
            )
            # after subsetting, the model will only have two outputs
            _mock_num_outputs = es.enter_context(
                mock.patch(
                    "botorch.utils.testing.MockModel.num_outputs",
                    new_callable=mock.PropertyMock,
                )
            )
            _mock_num_outputs.return_value = 3
            preds = torch.tensor(
                [
                    [11.0, 2.0],
                    [9.0, 3.0],
                ],
                **tkwargs,
            )
            model.model = MockModel(
                MockPosterior(
                    mean=preds,
                    samples=preds,
                ),
            )
            subset_mock_model = MockModel(
                MockPosterior(
                    mean=preds,
                    samples=preds,
                ),
            )
            es.enter_context(
                mock.patch.object(
                    model.model,
                    "subset_output",
                    return_value=subset_mock_model,
                )
            )
            outcome_constraints = (
                torch.tensor([[1.0, 0.0, 0.0]], **tkwargs),
                torch.tensor([[10.0]], **tkwargs),
            )
            _, _, gen_metadata, _ = model.gen(
                n,
                bounds,
                objective_weights=torch.tensor([-1.0, -1.0, 0.0], **tkwargs),
                outcome_constraints=outcome_constraints,
                model_gen_options={"optimizer_kwargs": _get_optimizer_kwargs()},
            )
            # the NEHVI acquisition function should be created only once.
            self.assertEqual(_mock_acqf.call_count, 3)
            ckwargs = _mock_model_infer_objective_thresholds.call_args[1]
            X_observed = ckwargs["X_observed"]
            sorted_idcs = X_observed[:, 0].argsort()
            expected_X_observed = torch.tensor(
                [[1.0, 2.0, 3.0], [0.9, 1.9, 2.9]], **tkwargs
            )
            sorted_idcs2 = expected_X_observed[:, 0].argsort()
            self.assertTrue(
                torch.equal(
                    X_observed[sorted_idcs],
                    expected_X_observed[sorted_idcs2],
                )
            )
            self.assertTrue(
                torch.equal(
                    ckwargs["objective_weights"],
                    torch.tensor([-1.0, -1.0, 0.0], **tkwargs),
                )
            )
            oc = ckwargs["outcome_constraints"]
            self.assertTrue(torch.equal(oc[0], outcome_constraints[0]))
            self.assertTrue(torch.equal(oc[1], outcome_constraints[1]))
            self.assertIs(ckwargs["model"], subset_mock_model)
            self.assertTrue(
                torch.equal(
                    ckwargs["subset_idcs"],
                    torch.tensor([0, 1], device=tkwargs["device"]),
                )
            )
            _mock_infer_reference_point.assert_called_once()
            ckwargs = _mock_infer_reference_point.call_args[1]
            self.assertEqual(ckwargs["scale"], 0.1)
            self.assertTrue(
                torch.equal(
                    ckwargs["pareto_Y"], torch.tensor([[-9.0, -3.0]], **tkwargs)
                )
            )
            self.assertIn("objective_thresholds", gen_metadata)
            obj_t = gen_metadata["objective_thresholds"]
            self.assertTrue(
                torch.equal(obj_t[:2], torch.tensor([9.9, 3.3], dtype=tkwargs["dtype"]))
            )
            self.assertTrue(np.isnan(obj_t[2]))
            # test providing model with extra tracking metrics and objective thresholds
            provided_obj_t = torch.tensor([10.0, 4.0, float("nan")], **tkwargs)
            _, _, gen_metadata, _ = model.gen(
                n,
                bounds,
                objective_weights=torch.tensor([-1.0, -1.0, 0.0], **tkwargs),
                outcome_constraints=outcome_constraints,
                model_gen_options={"optimizer_kwargs": _get_optimizer_kwargs()},
                objective_thresholds=provided_obj_t,
            )
            self.assertIn("objective_thresholds", gen_metadata)
            obj_t = gen_metadata["objective_thresholds"]
            self.assertTrue(torch.equal(obj_t[:2], provided_obj_t[:2].cpu()))
            self.assertTrue(np.isnan(obj_t[2]))

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

    def test_BotorchMOOModel_with_qehvi_and_outcome_constraints(
        self, dtype=torch.float, cuda=False, use_qnehvi=False
    ):
        acqf_constructor = get_NEHVI if use_qnehvi else get_EHVI
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
        model = MultiObjectiveBotorchModel(acqf_constructor=acqf_constructor)

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
                objective_thresholds=torch.tensor([1.0, 1.0], **tkwargs),
            )
        # test that objective thresholds and weights are properly subsetted
        obj_t = torch.tensor([1.0, 1.0, 1.0], **tkwargs)
        with mock.patch.object(
            model,
            "acqf_constructor",
            wraps=botorch_moo_defaults.get_NEHVI,
        ) as mock_get_nehvi, mock.patch(
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
            mock_get_nehvi.assert_called_once()
            _, ckwargs = mock_get_nehvi.call_args
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
            wraps=botorch_moo_defaults.get_NEHVI,
        ) as mock_get_nehvi, mock.patch(
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
            mock_get_nehvi.assert_called_once()
            _, ckwargs = mock_get_nehvi.call_args
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
