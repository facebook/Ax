#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Any, Dict
from unittest import mock

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_defaults import get_NEI
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.deterministic_metric import (
    get_and_fit_model_list_det,
    L1_norm_func,
)
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import fast_botorch_optimize
from botorch.models import ModelList
from botorch.utils.datasets import FixedNoiseDataset
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization


FIT_MODEL_MO_PATH = "ax.models.torch.deterministic_metric.fit_gpytorch_mll"
CHEBYSHEV_SCALARIZATION_PATH = (
    "ax.models.torch.botorch_defaults.get_chebyshev_scalarization"
)


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
    metric_names = ["y", "r", "d"]
    return Xs, Ys, Yvars, bounds, task_features, feature_names, metric_names


class BotorchDeterministicMetricMOOModelTest(TestCase):
    def test_l1_norm_func(self, cuda: bool = False) -> None:
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            init_point = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
            # testing a batch of two points
            sample_point = torch.tensor(
                [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], device=device, dtype=dtype
            )
            real_values = torch.norm(
                (sample_point - init_point), p=1, dim=-1, keepdim=True
            )
            computed_values = L1_norm_func(X=sample_point, init_point=init_point)
            self.assertTrue(torch.equal(real_values, computed_values))

    def test_l1_norm_func_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_l1_norm_func(cuda=True)

    @fast_botorch_optimize
    def test_deterministic_metric_BotorchMOOModel_with_cheby_scalarization(
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
        Xs3, Ys3, Yvars3, _, _, _, _ = _get_torch_test_data(
            dtype=dtype, cuda=cuda, constant_noise=True
        )
        n = 3
        objective_weights = torch.tensor([1.0, 1.0, 1.0], **tkwargs)
        obj_t = torch.tensor([1.0, 1.0, 1.0], **tkwargs)

        L1_norm_penalty = functools.partial(
            L1_norm_func,
            init_point=torch.zeros(3),
        )

        # test when det_metric_names is not any of the metric names
        det_metric_names = ["wrong_name"]
        det_metric_funcs = {"wrong_name": L1_norm_penalty}
        model = MultiObjectiveBotorchModel(
            # pyre-fixme[6]: For 1st param expected `(List[Tensor], List[Tensor], Lis...
            model_constructor=get_and_fit_model_list_det,
            # pyre-fixme[6]: For 2nd param expected `(Model, Tensor, Optional[Tuple[T...
            acqf_constructor=get_NEI,
            det_metric_names=det_metric_names,
            det_metric_funcs=det_metric_funcs,
        )
        datasets = [
            FixedNoiseDataset(X=X, Y=Y, Yvar=Y)
            for X, Y, Yvar in zip(
                Xs1 + Xs2 + Xs3, Ys1 + Ys2 + Ys3, Yvars1 + Yvars2 + Yvars3
            )
        ]
        with self.assertRaises(ValueError):
            model.fit(
                datasets=datasets,
                metric_names=mns,
                search_space_digest=SearchSpaceDigest(
                    feature_names=fns,
                    bounds=bounds,
                    task_features=tfs,
                ),
            )
        # test when det_metric_names matches
        det_metric_names = ["d"]
        det_metric_funcs = {"d": L1_norm_penalty}
        model = MultiObjectiveBotorchModel(
            # pyre-fixme[6]: For 1st param expected `(List[Tensor], List[Tensor], Lis...
            model_constructor=get_and_fit_model_list_det,
            # pyre-fixme[6]: For 2nd param expected `(Model, Tensor, Optional[Tuple[T...
            acqf_constructor=get_NEI,
            det_metric_names=det_metric_names,
            det_metric_funcs=det_metric_funcs,
        )
        # test that task_features are not supported
        with self.assertRaises(NotImplementedError):
            model.fit(
                datasets=datasets,
                metric_names=mns,
                search_space_digest=SearchSpaceDigest(
                    feature_names=fns,
                    bounds=bounds,
                    task_features=[0],
                ),
            )
        # test that fidelity_features are not supported
        with self.assertRaises(NotImplementedError):
            model.fit(
                datasets=datasets,
                metric_names=mns,
                search_space_digest=SearchSpaceDigest(
                    feature_names=fns,
                    bounds=bounds,
                    fidelity_features=[0],
                ),
            )
        # test fitting
        search_space_digest = SearchSpaceDigest(
            feature_names=fns,
            bounds=bounds,
            task_features=tfs,
        )
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                datasets=datasets,
                metric_names=mns,
                search_space_digest=search_space_digest,
            )
            # expect only fitting 2 GPs out of 3 objectives
            self.assertEqual(_mock_fit_model.call_count, 2)

        # test passing state_dict
        # pyre-fixme[16]: Optional type has no attribute `state_dict`
        state_dict = model.model.state_dict()
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.model_constructor(  # pyre-ignore: [28]
                Xs=model.Xs,
                Ys=model.Ys,
                Yvars=model.Yvars,
                task_features=model.task_features,
                state_dict=state_dict,
                fidelity_features=model.fidelity_features,
                metric_names=model.metric_names,
                refit_model=False,
                use_input_warping=model.use_input_warping,
                use_loocv_pseudo_likelihood=model.use_loocv_pseudo_likelihood,
                **model._kwargs,
            )
            # load state dict without fitting
            self.assertEqual(_mock_fit_model.call_count, 0)

        # test that use_loocv_pseudo_likelihood = True
        model = MultiObjectiveBotorchModel(
            # pyre-fixme[6]: For 1st param expected `(List[Tensor], List[Tensor], Lis...
            model_constructor=get_and_fit_model_list_det,
            # pyre-fixme[6]: For 2nd param expected `(Model, Tensor, Optional[Tuple[T...
            acqf_constructor=get_NEI,
            det_metric_names=det_metric_names,
            det_metric_funcs=det_metric_funcs,
            use_loocv_pseudo_likelihood=True,
        )
        self.assertTrue(model.use_loocv_pseudo_likelihood)
        with mock.patch(FIT_MODEL_MO_PATH) as _mock_fit_model:
            model.fit(
                datasets=datasets,
                metric_names=mns,
                search_space_digest=search_space_digest,
            )
            self.assertIsInstance(model.model, ModelList)
            # pyre-ignore
            self.assertEqual(len(model.model.models), 3)

        with mock.patch(
            CHEBYSHEV_SCALARIZATION_PATH, wraps=get_chebyshev_scalarization
        ) as _mock_chebyshev_scalarization:
            model.gen(
                n,
                search_space_digest=search_space_digest,
                torch_opt_config=TorchOptConfig(
                    objective_weights=objective_weights,
                    objective_thresholds=obj_t,
                    model_gen_options={
                        "acquisition_function_kwargs": {
                            "chebyshev_scalarization": True,
                        },
                    },
                ),
            )
            # get_chebyshev_scalarization should be called once for generated candidate.
            self.assertEqual(n, _mock_chebyshev_scalarization.call_count)
