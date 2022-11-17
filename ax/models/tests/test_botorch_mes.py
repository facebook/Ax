#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from unittest import mock

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_mes import _instantiate_MES, MaxValueEntropySearch
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import fast_botorch_optimize
from botorch.acquisition.max_value_entropy_search import (
    qMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models.transforms.input import Warp
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.datasets import FixedNoiseDataset


class MaxValueEntropySearchTest(TestCase):
    def setUp(self) -> None:
        self.tkwargs = {"device": torch.device("cpu"), "dtype": torch.double}
        self.training_data = [
            FixedNoiseDataset(
                # pyre-fixme[6]: For 2nd param expected `Optional[dtype]` but got
                #  `Union[device, dtype]`.
                # pyre-fixme[6]: For 2nd param expected `Union[None, str, device]`
                #  but got `Union[device, dtype]`.
                # pyre-fixme[6]: For 2nd param expected `bool` but got
                #  `Union[device, dtype]`.
                X=torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], **self.tkwargs),
                Y=torch.tensor([[3.0], [4.0]], **self.tkwargs),
                Yvar=torch.tensor([[0.0], [2.0]], **self.tkwargs),
            )
        ]
        self.bounds = [(0.0, 1.0), (1.0, 4.0), (2.0, 5.0)]
        self.feature_names = ["x1", "x2", "x3"]
        self.metric_names = ["y"]
        self.acq_options = {"num_fantasies": 30, "candidate_size": 100}
        self.objective_weights = torch.tensor([1.0], **self.tkwargs)
        self.optimizer_options = {
            "num_restarts": 12,
            "raw_samples": 12,
            "maxiter": 5,
            "batch_limit": 1,
        }
        self.optimize_acqf = "ax.models.torch.botorch_mes.optimize_acqf"
        self.search_space_digest = SearchSpaceDigest(
            feature_names=self.feature_names,
            bounds=self.bounds,
        )

    @fast_botorch_optimize
    def test_MaxValueEntropySearch(self) -> None:
        model = MaxValueEntropySearch()
        model.fit(
            # pyre-fixme[6]: For 1st param expected `List[SupervisedDataset]` but
            #  got `List[FixedNoiseDataset]`.
            datasets=self.training_data,
            metric_names=self.metric_names,
            search_space_digest=self.search_space_digest,
        )

        # test model.gen()
        torch_opt_config = TorchOptConfig(
            objective_weights=self.objective_weights,
            model_gen_options={
                "acquisition_function_kwargs": self.acq_options,
                "optimizer_kwargs": self.optimizer_options,
            },
        )
        new_X_dummy = torch.rand(1, 1, 3, **self.tkwargs)
        with mock.patch(self.optimize_acqf) as mock_optimize_acqf:
            mock_optimize_acqf.side_effect = [(new_X_dummy, None)]
            gen_results = model.gen(
                n=1,
                search_space_digest=self.search_space_digest,
                torch_opt_config=torch_opt_config,
            )
            self.assertTrue(torch.equal(gen_results.points, new_X_dummy.cpu()))
            self.assertTrue(
                torch.equal(
                    gen_results.weights, torch.ones(1, dtype=self.tkwargs["dtype"])
                )
            )
            mock_optimize_acqf.assert_called_once()

        # Check best point selection within bounds (some numerical tolerance)
        xbest = model.best_point(
            search_space_digest=self.search_space_digest,
            torch_opt_config=torch_opt_config,
        )
        lb = torch.tensor([b[0] for b in self.bounds]) - 1e-5
        ub = torch.tensor([b[1] for b in self.bounds]) + 1e-5
        self.assertTrue(torch.all(xbest <= ub))
        self.assertTrue(torch.all(xbest >= lb))

        # test error message in case of constraints
        linear_constraints = (
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([[0.5], [1.0]]),
        )
        with self.assertRaises(UnsupportedError):
            model.gen(
                n=1,
                search_space_digest=self.search_space_digest,
                torch_opt_config=dataclasses.replace(
                    torch_opt_config, linear_constraints=linear_constraints
                ),
            )

        # test error message in case of >1 objective weights
        objective_weights = torch.tensor([1.0, 1.0], **self.tkwargs)
        with self.assertRaises(UnsupportedError):
            model.gen(
                n=1,
                search_space_digest=self.search_space_digest,
                torch_opt_config=dataclasses.replace(
                    torch_opt_config, objective_weights=objective_weights
                ),
            )

        # test error message in best_point()
        with self.assertRaises(UnsupportedError):
            model.best_point(
                search_space_digest=self.search_space_digest,
                torch_opt_config=dataclasses.replace(
                    torch_opt_config, linear_constraints=linear_constraints
                ),
            )

        with self.assertRaises(RuntimeError):
            model.best_point(
                search_space_digest=dataclasses.replace(
                    self.search_space_digest, target_fidelities={2: 1.0}
                ),
                torch_opt_config=torch_opt_config,
            )
        # test input warping
        self.assertFalse(model.use_input_warping)
        model = MaxValueEntropySearch(use_input_warping=True)
        model.fit(
            # pyre-fixme[6]: For 1st param expected `List[SupervisedDataset]` but
            #  got `List[FixedNoiseDataset]`.
            datasets=self.training_data,
            metric_names=self.metric_names,
            search_space_digest=self.search_space_digest,
        )
        self.assertTrue(model.use_input_warping)
        self.assertTrue(hasattr(model.model, "input_transform"))
        # pyre-fixme[16]: Optional type has no attribute `input_transform`.
        self.assertIsInstance(model.model.input_transform, Warp)

        # test loocv pseudo likelihood
        self.assertFalse(model.use_loocv_pseudo_likelihood)
        model = MaxValueEntropySearch(use_loocv_pseudo_likelihood=True)
        model.fit(
            # pyre-fixme[6]: For 1st param expected `List[SupervisedDataset]` but
            #  got `List[FixedNoiseDataset]`.
            datasets=self.training_data,
            metric_names=self.metric_names,
            search_space_digest=self.search_space_digest,
        )
        self.assertTrue(model.use_loocv_pseudo_likelihood)

    @fast_botorch_optimize
    def test_MaxValueEntropySearch_MultiFidelity(self) -> None:
        search_space_digest = dataclasses.replace(
            self.search_space_digest,
            fidelity_features=[-1],
        )
        model = MaxValueEntropySearch()
        model.fit(
            # pyre-fixme[6]: For 1st param expected `List[SupervisedDataset]` but
            #  got `List[FixedNoiseDataset]`.
            datasets=self.training_data,
            metric_names=self.metric_names,
            search_space_digest=search_space_digest,
        )

        # Check best point selection within bounds (some numerical tolerance)
        torch_opt_config = TorchOptConfig(
            objective_weights=self.objective_weights,
        )
        xbest = model.best_point(
            search_space_digest=dataclasses.replace(
                search_space_digest,
                target_fidelities={2: 5.0},
            ),
            torch_opt_config=torch_opt_config,
        )
        lb = torch.tensor([b[0] for b in self.bounds]) - 1e-5
        ub = torch.tensor([b[1] for b in self.bounds]) + 1e-5
        self.assertTrue(torch.all(xbest <= ub))
        self.assertTrue(torch.all(xbest >= lb))

        # check error when no target fidelities are specified
        with self.assertRaises(RuntimeError):
            model.best_point(
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )

        # check error when target fidelity and fixed features have the same key
        with self.assertRaises(RuntimeError):
            model.best_point(
                search_space_digest=dataclasses.replace(
                    search_space_digest,
                    target_fidelities={2: 1.0},
                ),
                torch_opt_config=dataclasses.replace(
                    torch_opt_config,
                    fixed_features={2: 1.0},
                ),
            )

        # check generation

        n = 1
        new_X_dummy = torch.rand(1, n, 3, **self.tkwargs)
        with mock.patch(
            self.optimize_acqf, side_effect=[(new_X_dummy, None)]
        ) as mock_optimize_acqf:
            gen_results = model.gen(
                n=n,
                search_space_digest=dataclasses.replace(
                    search_space_digest,
                    target_fidelities={2: 1.0},
                ),
                torch_opt_config=dataclasses.replace(
                    torch_opt_config,
                    model_gen_options={
                        "acquisition_function_kwargs": self.acq_options,
                        "optimizer_kwargs": self.optimizer_options,
                    },
                ),
            )
            self.assertTrue(torch.equal(gen_results.points, new_X_dummy.cpu()))
            self.assertTrue(
                torch.equal(
                    gen_results.weights, torch.ones(n, dtype=self.tkwargs["dtype"])
                )
            )
            mock_optimize_acqf.assert_called()

        # test input warping
        self.assertFalse(model.use_input_warping)
        model = MaxValueEntropySearch(use_input_warping=True)
        model.fit(
            # pyre-fixme[6]: For 1st param expected `List[SupervisedDataset]` but
            #  got `List[FixedNoiseDataset]`.
            datasets=self.training_data,
            metric_names=self.metric_names,
            search_space_digest=SearchSpaceDigest(
                feature_names=self.feature_names,
                bounds=self.bounds,
                fidelity_features=[-1],
            ),
        )
        self.assertTrue(model.use_input_warping)
        self.assertTrue(hasattr(model.model, "input_transform"))
        # pyre-fixme[16]: Optional type has no attribute `input_transform`.
        self.assertIsInstance(model.model.input_transform, Warp)

        # test loocv pseudo likelihood
        self.assertFalse(model.use_loocv_pseudo_likelihood)
        model = MaxValueEntropySearch(use_loocv_pseudo_likelihood=True)
        model.fit(
            # pyre-fixme[6]: For 1st param expected `List[SupervisedDataset]` but
            #  got `List[FixedNoiseDataset]`.
            datasets=self.training_data,
            metric_names=self.metric_names,
            search_space_digest=search_space_digest,
        )
        self.assertTrue(model.use_loocv_pseudo_likelihood)

    @fast_botorch_optimize
    def test_instantiate_MES(self) -> None:

        model = MaxValueEntropySearch()
        model.fit(
            # pyre-fixme[6]: For 1st param expected `List[SupervisedDataset]` but
            #  got `List[FixedNoiseDataset]`.
            datasets=self.training_data,
            metric_names=self.metric_names,
            search_space_digest=SearchSpaceDigest(
                feature_names=self.feature_names,
                bounds=self.bounds,
            ),
        )

        # test acquisition setting
        X_dummy = torch.ones(1, 3, **self.tkwargs)
        candidate_set = torch.rand(10, 3, **self.tkwargs)
        # pyre-fixme[6]: For 1st param expected `Model` but got `Optional[Model]`.
        acq_function = _instantiate_MES(model=model.model, candidate_set=candidate_set)

        self.assertIsInstance(acq_function, qMaxValueEntropy)
        self.assertIsInstance(acq_function.sampler, SobolQMCNormalSampler)
        self.assertIsInstance(acq_function.fantasies_sampler, SobolQMCNormalSampler)
        self.assertEqual(acq_function.num_fantasies, 16)
        self.assertEqual(acq_function.num_mv_samples, 10)
        self.assertEqual(acq_function.use_gumbel, True)
        self.assertEqual(acq_function.maximize, True)

        acq_function = _instantiate_MES(
            # pyre-fixme[6]: For 1st param expected `Model` but got `Optional[Model]`.
            model=model.model,
            candidate_set=candidate_set,
            X_pending=X_dummy,
        )
        self.assertTrue(torch.equal(acq_function.X_pending, X_dummy))

        # multi-fidelity tests
        model = MaxValueEntropySearch()
        model.fit(
            # pyre-fixme[6]: For 1st param expected `List[SupervisedDataset]` but
            #  got `List[FixedNoiseDataset]`.
            datasets=self.training_data,
            metric_names=self.metric_names,
            search_space_digest=SearchSpaceDigest(
                feature_names=self.feature_names,
                bounds=self.bounds,
                fidelity_features=[-1],
            ),
        )

        candidate_set = torch.rand(10, 3, **self.tkwargs)
        acq_function = _instantiate_MES(
            # pyre-fixme[6]: For 1st param expected `Model` but got `Optional[Model]`.
            model=model.model,
            candidate_set=candidate_set,
            target_fidelities={2: 1.0},
        )
        self.assertIsInstance(acq_function, qMultiFidelityMaxValueEntropy)
        Xs = [self.training_data[0].X()]
        # pyre-fixme[29]: `Union[torch._tensor.Tensor,
        #  torch.nn.modules.module.Module]` is not a function.
        self.assertEqual(acq_function.expand(Xs), Xs)

        # test error that target fidelity and fidelity weight indices must match
        with self.assertRaises(RuntimeError):
            _instantiate_MES(
                # pyre-fixme[6]: For 1st param expected `Model` but got
                #  `Optional[Model]`.
                model=model.model,
                candidate_set=candidate_set,
                target_fidelities={1: 1.0},
                fidelity_weights={2: 1.0},
            )
