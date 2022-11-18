#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from unittest import mock

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_kg import _instantiate_KG, KnowledgeGradient
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import fast_botorch_optimize
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.acquisition.objective import (
    LinearMCObjective,
    ScalarizedPosteriorTransform,
)
from botorch.exceptions.errors import UnsupportedError
from botorch.models.transforms.input import Warp
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.datasets import FixedNoiseDataset


def dummy_func(X: torch.Tensor) -> torch.Tensor:
    return X


class KnowledgeGradientTest(TestCase):
    def setUp(self) -> None:
        self.tkwargs = {"device": torch.device("cpu"), "dtype": torch.double}
        self.dataset = FixedNoiseDataset(
            # pyre-fixme[6]: For 2nd param expected `Optional[dtype]` but got
            #  `Union[device, dtype]`.
            # pyre-fixme[6]: For 2nd param expected `Union[None, str, device]` but
            #  got `Union[device, dtype]`.
            # pyre-fixme[6]: For 2nd param expected `bool` but got `Union[device,
            #  dtype]`.
            X=torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], **self.tkwargs),
            Y=torch.tensor([[3.0], [4.0]], **self.tkwargs),
            Yvar=torch.tensor([[0.0], [2.0]], **self.tkwargs),
        )
        self.bounds = [(0.0, 1.0), (1.0, 4.0), (2.0, 5.0)]
        self.feature_names = ["x1", "x2", "x3"]
        self.metric_names = ["y"]
        self.acq_options = {"num_fantasies": 30, "mc_samples": 30}
        self.objective_weights = torch.tensor([1.0], **self.tkwargs)
        self.optimizer_options = {
            "num_restarts": 12,
            "raw_samples": 12,
            "maxiter": 5,
            "batch_limit": 1,
        }
        self.optimize_acqf = "ax.models.torch.botorch_kg.optimize_acqf"
        self.X_dummy = torch.ones(1, 3, **self.tkwargs)
        self.outcome_constraints = (torch.tensor([[1.0]]), torch.tensor([[0.5]]))
        self.objective_weights = torch.ones(1, **self.tkwargs)
        self.moo_objective_weights = torch.ones(2, **self.tkwargs)
        self.objective_thresholds = torch.tensor([0.5, 1.5])
        self.search_space_digest = SearchSpaceDigest(
            feature_names=self.feature_names,
            bounds=self.bounds,
        )

    @fast_botorch_optimize
    def test_KnowledgeGradient(self) -> None:
        model = KnowledgeGradient()
        model.fit(
            datasets=[self.dataset],
            metric_names=self.metric_names,
            search_space_digest=self.search_space_digest,
        )

        n = 2

        X_dummy = torch.rand(1, n, 4, **self.tkwargs)
        acq_dummy = torch.tensor(0.0, **self.tkwargs)

        torch_opt_config = TorchOptConfig(
            objective_weights=self.objective_weights,
            model_gen_options={
                "acquisition_function_kwargs": self.acq_options,
                "optimizer_kwargs": self.optimizer_options,
            },
        )

        with mock.patch(self.optimize_acqf) as mock_optimize_acqf:
            mock_optimize_acqf.side_effect = [(X_dummy, acq_dummy)]
            gen_results = model.gen(
                n=n,
                search_space_digest=self.search_space_digest,
                torch_opt_config=torch_opt_config,
            )
            self.assertTrue(torch.equal(gen_results.points, X_dummy.cpu()))
            self.assertTrue(
                torch.equal(
                    gen_results.weights, torch.ones(n, dtype=self.tkwargs["dtype"])
                )
            )

            # called once, the best point call is not caught by mock
            mock_optimize_acqf.assert_called_once()

        ini_dummy = torch.rand(10, 32, 3, **self.tkwargs)
        optimizer_options2 = {
            "num_restarts": 1,
            "raw_samples": 1,
            "maxiter": 5,
            "batch_limit": 1,
            "partial_restarts": 2,
        }
        torch_opt_config.model_gen_options["optimizer_kwargs"] = optimizer_options2
        with mock.patch(
            "ax.models.torch.botorch_kg.gen_one_shot_kg_initial_conditions",
            return_value=ini_dummy,
        ) as mock_warmstart_initialization:
            gen_results = model.gen(
                n=n,
                search_space_digest=self.search_space_digest,
                torch_opt_config=torch_opt_config,
            )
            mock_warmstart_initialization.assert_called_once()

        posterior_tf = ScalarizedPosteriorTransform(weights=self.objective_weights)
        # pyre-fixme[6]: For 1st param expected `Model` but got `Optional[Model]`.
        dummy_acq = PosteriorMean(model=model.model, posterior_transform=posterior_tf)
        with mock.patch(
            "ax.models.torch.utils.PosteriorMean", return_value=dummy_acq
        ) as mock_posterior_mean:
            gen_results = model.gen(
                n=n,
                search_space_digest=self.search_space_digest,
                torch_opt_config=torch_opt_config,
            )
            self.assertEqual(mock_posterior_mean.call_count, 2)

        # Check best point selection within bounds (some numerical tolerance)
        xbest = model.best_point(
            search_space_digest=self.search_space_digest,
            torch_opt_config=torch_opt_config,
        )
        lb = torch.tensor([b[0] for b in self.bounds]) - 1e-5
        ub = torch.tensor([b[1] for b in self.bounds]) + 1e-5
        self.assertTrue(torch.all(xbest <= ub))
        self.assertTrue(torch.all(xbest >= lb))

        # test error message
        torch_opt_config = dataclasses.replace(
            torch_opt_config,
            linear_constraints=(
                torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                torch.tensor([[0.5], [1.0]]),
            ),
        )
        with self.assertRaises(UnsupportedError):
            gen_results = model.gen(
                n=n,
                search_space_digest=self.search_space_digest,
                torch_opt_config=torch_opt_config,
            )

        # test input warping
        self.assertFalse(model.use_input_warping)
        model = KnowledgeGradient(use_input_warping=True)
        model.fit(
            datasets=[self.dataset],
            search_space_digest=self.search_space_digest,
            metric_names=self.metric_names,
        )
        self.assertTrue(model.use_input_warping)
        self.assertTrue(hasattr(model.model, "input_transform"))
        # pyre-fixme[16]: Optional type has no attribute `input_transform`.
        self.assertIsInstance(model.model.input_transform, Warp)

        # test loocv pseudo likelihood
        self.assertFalse(model.use_loocv_pseudo_likelihood)
        model = KnowledgeGradient(use_loocv_pseudo_likelihood=True)
        model.fit(
            datasets=[self.dataset],
            search_space_digest=self.search_space_digest,
            metric_names=self.metric_names,
        )
        self.assertTrue(model.use_loocv_pseudo_likelihood)

    @fast_botorch_optimize
    def test_KnowledgeGradient_multifidelity(self) -> None:
        search_space_digest = SearchSpaceDigest(
            feature_names=self.feature_names,
            bounds=self.bounds,
            fidelity_features=[2],
            target_fidelities={2: 5.0},
        )
        model = KnowledgeGradient()
        model.fit(
            datasets=[self.dataset],
            metric_names=["L2NormMetric"],
            search_space_digest=search_space_digest,
        )

        torch_opt_config = TorchOptConfig(
            objective_weights=self.objective_weights,
            model_gen_options={
                "acquisition_function_kwargs": self.acq_options,
                "optimizer_kwargs": self.optimizer_options,
            },
        )
        # Check best point selection within bounds (some numerical tolerance)
        xbest = model.best_point(
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
        )
        lb = torch.tensor([b[0] for b in self.bounds]) - 1e-5
        ub = torch.tensor([b[1] for b in self.bounds]) + 1e-5
        self.assertTrue(torch.all(xbest <= ub))
        self.assertTrue(torch.all(xbest >= lb))

        # check error when no target fidelities are specified
        with self.assertRaises(RuntimeError):
            model.best_point(
                search_space_digest=dataclasses.replace(
                    search_space_digest,
                    target_fidelities={},
                ),
                torch_opt_config=torch_opt_config,
            )

        # check generation
        n = 2
        X_dummy = torch.zeros(1, n, 3, **self.tkwargs)
        acq_dummy = torch.tensor(0.0, **self.tkwargs)
        dummy = (X_dummy, acq_dummy)
        with mock.patch(self.optimize_acqf, side_effect=[dummy]) as mock_optimize_acqf:
            gen_results = model.gen(
                n=n,
                search_space_digest=search_space_digest,
                torch_opt_config=torch_opt_config,
            )
            self.assertTrue(torch.equal(gen_results.points, X_dummy.cpu()))
            self.assertTrue(
                torch.equal(
                    gen_results.weights, torch.ones(n, dtype=self.tkwargs["dtype"])
                )
            )
            mock_optimize_acqf.assert_called()  # called twice, once for best_point

        # test error message
        with self.assertRaises(UnsupportedError):
            xbest = model.best_point(
                search_space_digest=search_space_digest,
                torch_opt_config=dataclasses.replace(
                    torch_opt_config,
                    linear_constraints=(
                        torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                        torch.tensor([[0.5], [1.0]]),
                    ),
                ),
            )

        # test input warping
        self.assertFalse(model.use_input_warping)
        model = KnowledgeGradient(use_input_warping=True)
        model.fit(
            datasets=[self.dataset],
            metric_names=["L2NormMetric"],
            search_space_digest=search_space_digest,
        )
        self.assertTrue(model.use_input_warping)
        self.assertTrue(hasattr(model.model, "input_transform"))
        # pyre-fixme[16]: Optional type has no attribute `input_transform`.
        self.assertIsInstance(model.model.input_transform, Warp)

        # test loocv pseudo likelihood
        self.assertFalse(model.use_loocv_pseudo_likelihood)
        model = KnowledgeGradient(use_loocv_pseudo_likelihood=True)
        model.fit(
            datasets=[self.dataset],
            metric_names=["L2NormMetric"],
            search_space_digest=search_space_digest,
        )
        self.assertTrue(model.use_loocv_pseudo_likelihood)

    @fast_botorch_optimize
    def test_KnowledgeGradient_helpers(self) -> None:
        model = KnowledgeGradient()
        model.fit(
            datasets=[self.dataset],
            metric_names=["L2NormMetric"],
            search_space_digest=SearchSpaceDigest(
                feature_names=self.feature_names,
                bounds=self.bounds,
            ),
        )

        # test _instantiate_KG
        posterior_tf = ScalarizedPosteriorTransform(weights=self.objective_weights)

        # test acquisition setting
        acq_function = _instantiate_KG(
            # pyre-fixme[6]: For 1st param expected `Model` but got `Optional[Model]`.
            model=model.model,
            posterior_transform=posterior_tf,
            n_fantasies=10,
            qmc=True,
        )
        self.assertIsInstance(acq_function.sampler, SobolQMCNormalSampler)
        self.assertIsInstance(
            acq_function.posterior_transform, ScalarizedPosteriorTransform
        )
        self.assertEqual(acq_function.num_fantasies, 10)

        acq_function = _instantiate_KG(
            # pyre-fixme[6]: For 1st param expected `Model` but got `Optional[Model]`.
            model=model.model,
            posterior_transform=posterior_tf,
            n_fantasies=10,
            qmc=False,
        )
        self.assertIsInstance(acq_function.sampler, IIDNormalSampler)

        acq_function = _instantiate_KG(
            # pyre-fixme[6]: For 1st param expected `Model` but got `Optional[Model]`.
            model=model.model,
            posterior_transform=posterior_tf,
            qmc=False,
        )
        self.assertIsNone(acq_function.inner_sampler)

        acq_function = _instantiate_KG(
            # pyre-fixme[6]: For 1st param expected `Model` but got `Optional[Model]`.
            model=model.model,
            posterior_transform=posterior_tf,
            qmc=True,
            X_pending=self.X_dummy,
        )
        self.assertIsNone(acq_function.inner_sampler)
        self.assertTrue(torch.equal(acq_function.X_pending, self.X_dummy))

        # test _get_best_point_acqf
        acq_function, non_fixed_idcs = model._get_best_point_acqf(
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            X_observed=self.X_dummy,
        )
        self.assertIsInstance(acq_function, qSimpleRegret)
        self.assertIsInstance(acq_function.sampler, SobolQMCNormalSampler)
        self.assertIsNone(non_fixed_idcs)

        acq_function, non_fixed_idcs = model._get_best_point_acqf(
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            X_observed=self.X_dummy,
            qmc=False,
        )
        self.assertIsInstance(acq_function.sampler, IIDNormalSampler)
        self.assertIsNone(non_fixed_idcs)

        with self.assertRaises(RuntimeError):
            model._get_best_point_acqf(
                objective_weights=self.objective_weights,
                outcome_constraints=self.outcome_constraints,
                X_observed=self.X_dummy,
                target_fidelities={1: 1.0},
            )

        # multi-fidelity tests

        model = KnowledgeGradient()
        model.fit(
            datasets=[self.dataset],
            metric_names=["L2NormMetric"],
            search_space_digest=SearchSpaceDigest(
                feature_names=self.feature_names,
                bounds=self.bounds,
                fidelity_features=[-1],
            ),
        )

        acq_function = _instantiate_KG(
            # pyre-fixme[6]: For 1st param expected `Model` but got `Optional[Model]`.
            model=model.model,
            posterior_transform=posterior_tf,
            target_fidelities={2: 1.0},
            # pyre-fixme[6]: For 4th param expected `Optional[Tensor]` but got `int`.
            current_value=0,
        )
        self.assertIsInstance(acq_function, qMultiFidelityKnowledgeGradient)

        acq_function = _instantiate_KG(
            # pyre-fixme[6]: For 1st param expected `Model` but got `Optional[Model]`.
            model=model.model,
            objective=LinearMCObjective(weights=self.objective_weights),
        )
        self.assertIsInstance(acq_function.inner_sampler, SobolQMCNormalSampler)

        # test error that target fidelity and fidelity weight indices must match
        with self.assertRaises(RuntimeError):
            _instantiate_KG(
                # pyre-fixme[6]: For 1st param expected `Model` but got
                #  `Optional[Model]`.
                model=model.model,
                posterior_transform=posterior_tf,
                target_fidelities={1: 1.0},
                fidelity_weights={2: 1.0},
                # pyre-fixme[6]: For 5th param expected `Optional[Tensor]` but got
                #  `int`.
                current_value=0,
            )

        # test _get_best_point_acqf
        acq_function, non_fixed_idcs = model._get_best_point_acqf(
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            X_observed=self.X_dummy,
            target_fidelities={2: 1.0},
        )
        self.assertIsInstance(acq_function, FixedFeatureAcquisitionFunction)
        # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no attribute
        #  `sampler`.
        self.assertIsInstance(acq_function.acq_func.sampler, SobolQMCNormalSampler)
        self.assertEqual(non_fixed_idcs, [0, 1])

        acq_function, non_fixed_idcs = model._get_best_point_acqf(
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            X_observed=self.X_dummy,
            target_fidelities={2: 1.0},
            qmc=False,
        )
        self.assertIsInstance(acq_function, FixedFeatureAcquisitionFunction)
        # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no attribute
        #  `sampler`.
        self.assertIsInstance(acq_function.acq_func.sampler, IIDNormalSampler)
        self.assertEqual(non_fixed_idcs, [0, 1])

        # test error that fixed features are provided
        with self.assertRaises(RuntimeError):
            model._get_best_point_acqf(
                objective_weights=self.objective_weights,
                outcome_constraints=self.outcome_constraints,
                X_observed=self.X_dummy,
                qmc=False,
            )

        # test error if fixed features are also fidelity features
        with self.assertRaises(RuntimeError):
            model._get_best_point_acqf(
                objective_weights=self.objective_weights,
                outcome_constraints=self.outcome_constraints,
                X_observed=self.X_dummy,
                fixed_features={2: 2.0},
                target_fidelities={2: 1.0},
                qmc=False,
            )

        # TODO: Test subsetting multi-output model
