#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from contextlib import ExitStack
from typing import Any, Dict
from unittest import mock

import ax.models.torch.botorch_moo_defaults as botorch_moo_defaults
import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxError
from ax.models.torch.botorch_defaults import get_NEI
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.botorch_moo_defaults import (
    get_EHVI,
    get_NEHVI,
    infer_objective_thresholds,
)
from ax.models.torch.utils import HYPERSPHERE
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import fast_botorch_optimize
from botorch.acquisition.multi_objective import monte_carlo as moo_monte_carlo
from botorch.models import ModelListGP
from botorch.models.transforms.input import Warp
from botorch.optim.optimize import optimize_acqf_list
from botorch.sampling.normal import IIDNormalSampler
from botorch.utils.datasets import FixedNoiseDataset
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.testing import MockModel, MockPosterior


FIT_MODEL_MO_PATH = "ax.models.torch.botorch_defaults.fit_gpytorch_mll"
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


# pyre-fixme[3]: Return type must be annotated.
def _get_torch_test_data(
    dtype: torch.dtype = torch.float,
    cuda: bool = False,
    constant_noise: bool = True,
    # pyre-fixme[2]: Parameter must be annotated.
    task_features=None,
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
    metric_names = ["y"]
    return Xs, Ys, Yvars, bounds, task_features, feature_names, metric_names


class BotorchMOOModelTest(TestCase):
    def test_BotorchMOOModel_double(self) -> None:
        self.test_BotorchMOOModel_with_random_scalarization(dtype=torch.double)

    def test_BotorchMOOModel_cuda(self) -> None:
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

    def test_BotorchMOOModel_with_qnehvi(self) -> None:
        for dtype in (torch.float, torch.double):
            self.test_BotorchMOOModel_with_qehvi(dtype=dtype, use_qnehvi=True)
            self.test_BotorchMOOModel_with_qehvi_and_outcome_constraints(
                dtype=dtype, use_qnehvi=True
            )

    @fast_botorch_optimize
    def test_BotorchMOOModel_with_random_scalarization(
        self,
        dtype: torch.dtype = torch.float,
        cuda: bool = False,
    ) -> None:
        tkwargs: Dict[str, Any] = {
            "device": torch.device("cuda") if cuda else torch.device("cpu"),
            "dtype": dtype,
        }
        Xs1, Ys1, Yvars1, bounds, tfs, fns, mns = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        training_data = [
            FixedNoiseDataset(X=Xs1[0], Y=Ys1[0], Yvar=Yvars1[0]),
            FixedNoiseDataset(X=Xs2[0], Y=Ys2[0], Yvar=Yvars2[0]),
        ]

        n = 3
        objective_weights = torch.tensor([1.0, 1.0], **tkwargs)
        obj_t = torch.tensor([1.0, 1.0], **tkwargs)

        search_space_digest = SearchSpaceDigest(
            feature_names=fns,
            bounds=bounds,
            task_features=tfs,
        )
        # pyre-fixme[6]: For 1st param expected `(Model, Tensor, Optional[Tuple[Tenso...
        model = MultiObjectiveBotorchModel(acqf_constructor=get_NEI)
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                datasets=training_data,
                metric_names=["y1", "y2"],
                search_space_digest=search_space_digest,
            )
            _mock_fit_model.assert_called_once()

        torch_opt_config = TorchOptConfig(
            objective_weights=objective_weights,
            objective_thresholds=obj_t,
            model_gen_options={
                "acquisition_function_kwargs": {"random_scalarization": True},
                "optimizer_kwargs": _get_optimizer_kwargs(),
                "subset_model": False,
            },
            is_moo=True,
        )
        with self.assertRaisesRegex(NotImplementedError, "Best observed"):
            model.best_point(
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )
        with mock.patch(
            SAMPLE_SIMPLEX_UTIL_PATH,
            autospec=True,
            return_value=torch.tensor([0.7, 0.3], **tkwargs),
        ) as _mock_sample_simplex:
            model.gen(
                n,
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )
            # Sample_simplex should be called once for generated candidate.
            self.assertEqual(n, _mock_sample_simplex.call_count)

        torch_opt_config.model_gen_options["acquisition_function_kwargs"] = {
            "random_scalarization": True,
            "random_scalarization_distribution": HYPERSPHERE,
        }
        with mock.patch(
            SAMPLE_HYPERSPHERE_UTIL_PATH,
            autospec=True,
            return_value=torch.tensor([0.6, 0.8], **tkwargs),
        ) as _mock_sample_hypersphere:
            model.gen(
                n,
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )
            # Sample_simplex should be called once per generated candidate.
            self.assertEqual(n, _mock_sample_hypersphere.call_count)

        # test input warping
        self.assertFalse(model.use_input_warping)
        model = MultiObjectiveBotorchModel(
            # pyre-fixme[6]: For 1st param expected `(Model, Tensor, Optional[Tuple[T...
            acqf_constructor=get_NEI,
            use_input_warping=True,
        )
        model.fit(
            datasets=training_data,
            metric_names=["y1", "y2"],
            search_space_digest=search_space_digest,
        )
        self.assertTrue(model.use_input_warping)
        self.assertIsInstance(model.model, ModelListGP)
        # pyre-fixme[16]: Optional type has no attribute `models`.
        for m in model.model.models:
            self.assertTrue(hasattr(m, "input_transform"))
            self.assertIsInstance(m.input_transform, Warp)
        self.assertFalse(hasattr(model.model, "input_transform"))

        # test loocv pseudo likelihood
        self.assertFalse(model.use_loocv_pseudo_likelihood)
        model = MultiObjectiveBotorchModel(
            # pyre-fixme[6]: For 1st param expected `(Model, Tensor, Optional[Tuple[T...
            acqf_constructor=get_NEI,
            use_loocv_pseudo_likelihood=True,
        )
        model.fit(
            datasets=training_data,
            metric_names=["y1", "y2"],
            search_space_digest=search_space_digest,
        )
        self.assertTrue(model.use_loocv_pseudo_likelihood)

    @fast_botorch_optimize
    def test_BotorchMOOModel_with_chebyshev_scalarization(
        self,
        dtype: torch.dtype = torch.float,
        cuda: bool = False,
    ) -> None:
        tkwargs: Dict[str, Any] = {
            "device": torch.device("cuda") if cuda else torch.device("cpu"),
            "dtype": dtype,
        }
        Xs1, Ys1, Yvars1, bounds, tfs, fns, mns = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        training_data = [
            FixedNoiseDataset(X=Xs1[0], Y=Ys1[0], Yvar=Yvars1[0]),
            FixedNoiseDataset(X=Xs2[0], Y=Ys2[0], Yvar=Yvars2[0]),
        ]

        n = 3
        objective_weights = torch.tensor([1.0, 1.0], **tkwargs)
        obj_t = torch.tensor([1.0, 1.0], **tkwargs)

        search_space_digest = SearchSpaceDigest(
            feature_names=fns,
            bounds=bounds,
            task_features=tfs,
        )
        # pyre-fixme[6]: For 1st param expected `(Model, Tensor, Optional[Tuple[Tenso...
        model = MultiObjectiveBotorchModel(acqf_constructor=get_NEI)
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                datasets=training_data,
                metric_names=["y1", "y2"],
                search_space_digest=search_space_digest,
            )
            _mock_fit_model.assert_called_once()

        torch_opt_config = TorchOptConfig(
            objective_weights=objective_weights,
            objective_thresholds=obj_t,
            model_gen_options={
                "acquisition_function_kwargs": {"chebyshev_scalarization": True},
                "optimizer_kwargs": _get_optimizer_kwargs(),
            },
        )
        with mock.patch(
            CHEBYSHEV_SCALARIZATION_PATH, wraps=get_chebyshev_scalarization
        ) as _mock_chebyshev_scalarization, mock.patch(
            "ax.models.torch.botorch_moo_defaults.optimize_acqf_list",
            wraps=optimize_acqf_list,
        ) as mock_optimize:
            model.gen(
                n,
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )
        # get_chebyshev_scalarization should be called once for generated candidate.
        self.assertEqual(n, _mock_chebyshev_scalarization.call_count)
        self.assertEqual(
            mock_optimize.call_args.kwargs["options"]["init_batch_limit"], 32
        )
        self.assertEqual(mock_optimize.call_args.kwargs["options"]["batch_limit"], 1)

    def test_BotorchMOOModel_with_qehvi(
        self,
        dtype: torch.dtype = torch.float,
        cuda: bool = False,
        use_qnehvi: bool = False,
    ) -> None:
        if use_qnehvi:
            acqf_constructor = get_NEHVI
            partitioning_path = NEHVI_PARTITIONING_PATH
        else:
            acqf_constructor = get_EHVI
            partitioning_path = EHVI_PARTITIONING_PATH
        tkwargs: Dict[str, Any] = {
            "device": torch.device("cuda") if cuda else torch.device("cpu"),
            "dtype": dtype,
        }
        Xs1, Ys1, Yvars1, bounds, tfs, fns, mns = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        training_data = [
            FixedNoiseDataset(X=Xs1[0], Y=Ys1[0], Yvar=Yvars1[0]),
            FixedNoiseDataset(X=Xs2[0], Y=Ys2[0], Yvar=Yvars2[0]),
        ]

        n = 3
        objective_weights = torch.tensor([1.0, 1.0], **tkwargs)
        obj_t = torch.tensor([1.0, 1.0], **tkwargs)
        # pyre-fixme[6]: For 1st param expected `(Model, Tensor, Optional[Tuple[Tenso...
        model = MultiObjectiveBotorchModel(acqf_constructor=acqf_constructor)

        X_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)
        acqfv_dummy = torch.tensor([[[1.0, 2.0, 3.0]]], **tkwargs)

        search_space_digest = SearchSpaceDigest(
            feature_names=fns,
            bounds=bounds,
            task_features=tfs,
        )
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                datasets=training_data,
                metric_names=["y1", "y2"],
                search_space_digest=search_space_digest,
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
            torch_opt_config = TorchOptConfig(
                objective_weights=objective_weights,
                objective_thresholds=obj_t,
                model_gen_options={"optimizer_kwargs": _get_optimizer_kwargs()},
            )
            gen_results = model.gen(
                n,
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )
            # the NEHVI acquisition function should be created only once.
            self.assertEqual(1, _mock_acqf.call_count)
            # check partitioning strategy
            # NEHVI should call FastNondominatedPartitioning 1 time
            # since a batched partitioning is used for 2 objectives
            _mock_partitioning.assert_called_once()
            self.assertTrue(
                torch.equal(
                    gen_results.gen_metadata["objective_thresholds"], obj_t.cpu()
                )
            )
            _mock_fit_model = es.enter_context(mock.patch(FIT_MODEL_MO_PATH))
            # 3 objective
            training_data_m3 = training_data + [training_data[-1]]

            model.fit(
                datasets=training_data_m3,
                metric_names=["y1", "y2", "y3"],
                search_space_digest=search_space_digest,
            )
            torch_opt_config = TorchOptConfig(
                objective_weights=torch.tensor([1.0, 1.0, 1.0], **tkwargs),
                objective_thresholds=torch.tensor([1.0, 1.0, 1.0], **tkwargs),
                model_gen_options={"optimizer_kwargs": _get_optimizer_kwargs()},
            )
            model.gen(
                n,
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
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
            training_data_multiple = [
                FixedNoiseDataset(X=Xs1[0], Y=Ys1[0], Yvar=Yvars1[0]),
                FixedNoiseDataset(X=Xs1[0], Y=Ys2[0], Yvar=Yvars2[0]),
            ]
            model.fit(
                datasets=training_data_multiple,
                metric_names=["y1", "y2", "dummy_metric"],
                search_space_digest=search_space_digest,
            )
            es.enter_context(
                mock.patch(
                    "ax.models.torch.botorch_moo_defaults._check_posterior_type",
                    wraps=lambda y: y,
                )
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
            es.enter_context(
                mock.patch(
                    "botorch.acquisition.utils.get_sampler",
                    return_value=IIDNormalSampler(sample_shape=torch.Size([2])),
                )
            )
            outcome_constraints = (
                torch.tensor([[1.0, 0.0, 0.0]], **tkwargs),
                torch.tensor([[10.0]], **tkwargs),
            )
            torch_opt_config = TorchOptConfig(
                objective_weights=torch.tensor([-1.0, -1.0, 0.0], **tkwargs),
                outcome_constraints=outcome_constraints,
                model_gen_options={
                    "optimizer_kwargs": _get_optimizer_kwargs(),
                    # do not used cached root decomposition since
                    # MockPosterior does not have an mvn attribute
                    "acquisition_function_kwargs": {
                        "cache_root": False,
                        "prune_baseline": False,
                    },
                },
            )
            gen_results = model.gen(
                n,
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
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
            self.assertIn("objective_thresholds", gen_results.gen_metadata)
            obj_t = gen_results.gen_metadata["objective_thresholds"]
            self.assertTrue(
                torch.equal(obj_t[:2], torch.tensor([9.9, 3.3], dtype=tkwargs["dtype"]))
            )
            self.assertTrue(np.isnan(obj_t[2]))
            # test providing model with extra tracking metrics and objective thresholds
            provided_obj_t = torch.tensor([10.0, 4.0, float("nan")], **tkwargs)
            gen_results = model.gen(
                n,
                search_space_digest=search_space_digest,
                torch_opt_config=dataclasses.replace(
                    torch_opt_config,
                    objective_thresholds=provided_obj_t,
                ),
            )
            self.assertIn("objective_thresholds", gen_results.gen_metadata)
            obj_t = gen_results.gen_metadata["objective_thresholds"]
            self.assertTrue(torch.equal(obj_t[:2], provided_obj_t[:2].cpu()))
            self.assertTrue(np.isnan(obj_t[2]))

    @fast_botorch_optimize
    def test_BotorchMOOModel_with_random_scalarization_and_outcome_constraints(
        self,
        dtype: torch.dtype = torch.float,
        cuda: bool = False,
    ) -> None:
        tkwargs: Dict[str, Any] = {
            "device": torch.device("cuda") if cuda else torch.device("cpu"),
            "dtype": dtype,
        }
        Xs1, Ys1, Yvars1, bounds, tfs, fns, mns = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        training_data = [
            FixedNoiseDataset(X=Xs1[0], Y=Ys1[0], Yvar=Yvars1[0]),
            FixedNoiseDataset(X=Xs2[0], Y=Ys2[0], Yvar=Yvars2[0]),
        ]

        n = 2
        objective_weights = torch.tensor([1.0, 1.0], **tkwargs)
        obj_t = torch.tensor([1.0, 1.0], **tkwargs)
        # pyre-fixme[6]: For 1st param expected `(Model, Tensor, Optional[Tuple[Tenso...
        model = MultiObjectiveBotorchModel(acqf_constructor=get_NEI)

        search_space_digest = SearchSpaceDigest(
            feature_names=fns,
            bounds=bounds,
            task_features=tfs,
        )
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                datasets=training_data,
                metric_names=mns,
                search_space_digest=search_space_digest,
            )
            _mock_fit_model.assert_called_once()

        with mock.patch(
            SAMPLE_SIMPLEX_UTIL_PATH,
            autospec=True,
            return_value=torch.tensor([0.7, 0.3], **tkwargs),
        ) as _mock_sample_simplex:
            model.gen(
                n,
                search_space_digest=search_space_digest,
                torch_opt_config=TorchOptConfig(
                    objective_weights=objective_weights,
                    outcome_constraints=(
                        torch.tensor([[1.0, 1.0]], **tkwargs),
                        torch.tensor([[10.0]], **tkwargs),
                    ),
                    model_gen_options={
                        "acquisition_function_kwargs": {"random_scalarization": True},
                        "optimizer_kwargs": _get_optimizer_kwargs(),
                    },
                    objective_thresholds=obj_t,
                ),
            )
            self.assertEqual(n, _mock_sample_simplex.call_count)

    @fast_botorch_optimize
    def test_BotorchMOOModel_with_chebyshev_scalarization_and_outcome_constraints(
        self,
        dtype: torch.dtype = torch.float,
        cuda: bool = False,
    ) -> None:
        tkwargs: Dict[str, Any] = {
            "device": torch.device("cuda") if cuda else torch.device("cpu"),
            "dtype": torch.float,
        }
        Xs1, Ys1, Yvars1, bounds, tfs, fns, mns = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        Xs2, Ys2, Yvars2, _, _, _, _ = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        training_data = [
            FixedNoiseDataset(X=Xs1[0], Y=Ys1[0], Yvar=Yvars1[0]),
            FixedNoiseDataset(X=Xs2[0], Y=Ys2[0], Yvar=Yvars2[0]),
        ]

        n = 2
        objective_weights = torch.tensor([1.0, 1.0], **tkwargs)
        obj_t = torch.tensor([1.0, 1.0], **tkwargs)
        # pyre-fixme[6]: For 1st param expected `(Model, Tensor, Optional[Tuple[Tenso...
        model = MultiObjectiveBotorchModel(acqf_constructor=get_NEI)

        search_space_digest = SearchSpaceDigest(
            feature_names=fns,
            bounds=bounds,
            task_features=tfs,
        )
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                datasets=training_data,
                metric_names=mns,
                search_space_digest=search_space_digest,
            )
            _mock_fit_model.assert_called_once()

        torch_opt_config = TorchOptConfig(
            objective_weights=objective_weights,
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
        with mock.patch(
            CHEBYSHEV_SCALARIZATION_PATH, wraps=get_chebyshev_scalarization
        ) as _mock_chebyshev_scalarization:
            model.gen(
                n,
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )
            # get_chebyshev_scalarization should be called once for generated candidate.
            self.assertEqual(n, _mock_chebyshev_scalarization.call_count)

    @fast_botorch_optimize
    def test_BotorchMOOModel_with_qehvi_and_outcome_constraints(
        self,
        dtype: torch.dtype = torch.float,
        cuda: bool = False,
        use_qnehvi: bool = False,
    ) -> None:
        acqf_constructor = get_NEHVI if use_qnehvi else get_EHVI
        tkwargs: Dict[str, Any] = {
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
        training_data = [
            FixedNoiseDataset(X=Xs1[0], Y=Ys1[0], Yvar=Yvars1[0]),
            FixedNoiseDataset(X=Xs2[0], Y=Ys2[0], Yvar=Yvars2[0]),
            FixedNoiseDataset(X=Xs3[0], Y=Ys3[0], Yvar=Yvars3[0]),
        ]

        n = 3
        objective_weights = torch.tensor([1.0, 1.0, 0.0], **tkwargs)
        obj_t = torch.tensor([1.0, 1.0, 1.0], **tkwargs)
        # pyre-fixme[6]: For 1st param expected `(Model, Tensor, Optional[Tuple[Tenso...
        model = MultiObjectiveBotorchModel(acqf_constructor=acqf_constructor)

        search_space_digest = SearchSpaceDigest(
            feature_names=fns,
            bounds=bounds,
            task_features=tfs,
        )
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                datasets=training_data,
                metric_names=mns,
                search_space_digest=search_space_digest,
            )
            _mock_fit_model.assert_called_once()

        # test wrong number of objective thresholds
        torch_opt_config = TorchOptConfig(
            objective_weights=objective_weights,
            objective_thresholds=torch.tensor([1.0, 1.0], **tkwargs),
        )
        with self.assertRaises(AxError):
            model.gen(
                n,
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )
        # test that objective thresholds and weights are properly subsetted
        obj_t = torch.tensor([1.0, 1.0, 1.0], **tkwargs)
        torch_opt_config = dataclasses.replace(
            torch_opt_config,
            model_gen_options={"optimizer_kwargs": _get_optimizer_kwargs()},
            objective_thresholds=obj_t,
        )
        with mock.patch.object(
            model,
            "acqf_constructor",
            wraps=botorch_moo_defaults.get_NEHVI,
        ) as mock_get_nehvi:
            model.gen(
                n,
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
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
        torch_opt_config = dataclasses.replace(
            torch_opt_config,
            outcome_constraints=oc,
        )
        with mock.patch.object(
            model,
            "acqf_constructor",
            wraps=botorch_moo_defaults.get_NEHVI,
        ) as mock_get_nehvi:
            model.gen(
                n,
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
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
