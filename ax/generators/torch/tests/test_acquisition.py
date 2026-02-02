#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import dataclasses
import itertools
from contextlib import ExitStack
from copy import deepcopy
from typing import Any
from unittest import mock
from unittest.mock import Mock

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxError, SearchSpaceExhausted
from ax.generators.torch.botorch_modular.acquisition import (
    _expand_and_set_single_feature_to_target,
    Acquisition,
    logger,
)
from ax.generators.torch.botorch_modular.multi_acquisition import MultiAcquisition
from ax.generators.torch.botorch_modular.optimizer_argparse import optimizer_argparse
from ax.generators.torch.botorch_modular.optimizer_defaults import (
    BATCH_LIMIT,
    INIT_BATCH_LIMIT,
    MAX_OPT_AGG_SIZE,
)
from ax.generators.torch.botorch_modular.surrogate import Surrogate
from ax.generators.torch.utils import (
    _get_X_pending_and_observed,
    get_botorch_objective_and_transform,
    subset_model,
    SubsetModelData,
)
from ax.generators.torch_base import TorchOptConfig
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import (
    mock_botorch_optimize,
    mock_botorch_optimize_context_manager,
)
from ax.utils.testing.utils import generic_equals
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import (
    _register_acqf_input_constructor,
    ACQF_INPUT_CONSTRUCTOR_REGISTRY,
    get_acqf_input_constructor,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.logei import qLogProbabilityOfFeasibility
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.objective import LinearMCObjective
from botorch.exceptions.warnings import OptimizationWarning
from botorch.optim.optimize import (
    optimize_acqf,
    optimize_acqf_discrete,
    optimize_acqf_mixed,
)
from botorch.optim.optimize_mixed import optimize_acqf_mixed_alternating
from botorch.utils.constraints import get_outcome_constraint_transforms
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.testing import MockPosterior, skip_if_import_error
from torch import Tensor


ACQUISITION_PATH: str = Acquisition.__module__
CURRENT_PATH: str = __name__
SURROGATE_PATH: str = Surrogate.__module__


# Used to avoid going through BoTorch `Acquisition.__init__` which
# requires valid kwargs (correct sizes and lengths of tensors, etc).
class DummyAcquisitionFunction(AcquisitionFunction):
    X_pending: Tensor | None = None

    def __init__(self, eta: float = 1e-3, model: Any = None, **kwargs: Any) -> None:
        # pyre-ignore [6]
        AcquisitionFunction.__init__(self, model=None)
        self.eta = eta
        self.model = model

    def forward(self, X: Tensor) -> Tensor:
        # take the norm and sum over the q-batch dim
        if len(X.shape) > 2:
            res = torch.linalg.norm(X, dim=-1).sum(-1)
        else:
            res = torch.linalg.norm(X, dim=-1).squeeze(-1)
        # At least 1d is required for sequential optimize_acqf.
        return torch.atleast_1d(res)


class DummyOneShotAcquisitionFunction(DummyAcquisitionFunction, qKnowledgeGradient):
    def evaluate(self, X: Tensor, **kwargs: Any) -> Tensor:
        return X.sum(dim=-1)


class AcquisitionTest(TestCase):
    acquisition_class = Acquisition

    def setUp(self) -> None:
        super().setUp()
        qNEI_input_constructor = get_acqf_input_constructor(qNoisyExpectedImprovement)
        self.mock_input_constructor = mock.MagicMock(
            qNEI_input_constructor, side_effect=qNEI_input_constructor
        )
        # Adding wrapping here to be able to count calls and inspect arguments.
        _register_acqf_input_constructor(
            acqf_cls=DummyAcquisitionFunction,
            input_constructor=self.mock_input_constructor,
        )
        _register_acqf_input_constructor(
            acqf_cls=DummyOneShotAcquisitionFunction,
            input_constructor=self.mock_input_constructor,
        )
        self.tkwargs: dict[str, Any] = {"dtype": torch.double}
        self.surrogate = Surrogate()
        self.X = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], **self.tkwargs)
        self.Y = torch.tensor([[3.0], [4.0]], **self.tkwargs)
        self.Yvar = torch.tensor([[0.0], [2.0]], **self.tkwargs)
        self.fidelity_features = [2]
        self.feature_names = ["a", "b", "c"]
        self.metric_signatures = ["metric"]
        self.training_data = [
            SupervisedDataset(
                X=self.X,
                Y=self.Y,
                feature_names=self.feature_names,
                outcome_names=self.metric_signatures,
            )
        ]
        self.search_space_digest = SearchSpaceDigest(
            feature_names=self.feature_names,
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            target_values={2: 1.0},
        )
        with mock_botorch_optimize_context_manager():
            self.surrogate.fit(
                datasets=self.training_data,
                search_space_digest=SearchSpaceDigest(
                    feature_names=self.search_space_digest.feature_names,
                    bounds=self.search_space_digest.bounds,
                    target_values=self.search_space_digest.target_values,
                ),
            )

        self.botorch_acqf_class = DummyAcquisitionFunction
        self.objective_weights = torch.tensor([1.0])
        self.objective_thresholds = None
        self.pending_observations = [torch.tensor([[1.0, 3.0, 4.0]], **self.tkwargs)]
        self.outcome_constraints = (
            torch.tensor([[1.0]], **self.tkwargs),
            torch.tensor([[0.5]], **self.tkwargs),
        )
        self.constraints = get_outcome_constraint_transforms(
            outcome_constraints=self.outcome_constraints
        )
        self.linear_constraints = None
        self.fixed_features = {1: 2.0}
        self.botorch_acqf_options = {"cache_root": False, "prune_baseline": False}
        self.options = {}
        self.inequality_constraints = [
            (
                torch.tensor([0, 1], dtype=torch.int),
                torch.tensor([-1.0, 1.0], **self.tkwargs),
                1,
            )
        ]
        self.rounding_func = lambda x: x
        self.optimizer_options = {Keys.NUM_RESTARTS: 20, Keys.RAW_SAMPLES: 1024}
        self.torch_opt_config = TorchOptConfig(
            objective_weights=self.objective_weights,
            objective_thresholds=self.objective_thresholds,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
        )
        self.botorch_acqf_classes_with_options = None

    def tearDown(self) -> None:
        # Avoid polluting the registry for other tests.
        ACQF_INPUT_CONSTRUCTOR_REGISTRY.pop(DummyAcquisitionFunction)

    def get_acquisition_function(
        self,
        fixed_features: dict[int, float] | None = None,
        one_shot: bool = False,
        target_point: Tensor | None = None,
    ) -> Acquisition:
        return self.acquisition_class(
            botorch_acqf_class=(
                DummyOneShotAcquisitionFunction if one_shot else self.botorch_acqf_class
            ),
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            torch_opt_config=dataclasses.replace(
                self.torch_opt_config,
                fixed_features=fixed_features or {},
                pruning_target_point=target_point,
            ),
            options=self.options,
            botorch_acqf_options=self.botorch_acqf_options,
            botorch_acqf_classes_with_options=self.botorch_acqf_classes_with_options,
        )

    def test_init_raises(self) -> None:
        with self.assertRaisesRegex(
            AxError,
            "One of botorch_acqf_class or botorch_acqf_classes"
            "_with_options is required.",
        ):
            Acquisition(
                surrogate=self.surrogate,
                search_space_digest=self.search_space_digest,
                torch_opt_config=self.torch_opt_config,
                botorch_acqf_class=None,
                botorch_acqf_options={},
            )

    @mock.patch(
        f"{ACQUISITION_PATH}._get_X_pending_and_observed",
        wraps=_get_X_pending_and_observed,
    )
    @mock.patch(f"{ACQUISITION_PATH}.subset_model", wraps=subset_model)
    def test_init(
        self,
        mock_subset_model: Mock,
        mock_get_X: Mock,
    ) -> None:
        acquisition = self.acquisition_class(
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            torch_opt_config=self.torch_opt_config,
            botorch_acqf_class=self.botorch_acqf_class,
            options=self.options,
            botorch_acqf_options=self.botorch_acqf_options,
            botorch_acqf_classes_with_options=self.botorch_acqf_classes_with_options,
        )

        # Check `_get_X_pending_and_observed` kwargs
        mock_get_X.assert_called_once()
        _, ckwargs = mock_get_X.call_args
        for X, dataset in zip(ckwargs["Xs"], self.training_data):
            self.assertTrue(torch.equal(X, dataset.X))
        for attr in (
            "pending_observations",
            "objective_weights",
            "outcome_constraints",
            "linear_constraints",
            "fixed_features",
        ):
            self.assertTrue(generic_equals(ckwargs[attr], getattr(self, attr)))
        self.assertIs(ckwargs["bounds"], self.search_space_digest.bounds)

        # Call `subset_model` only when needed
        mock_subset_model.assert_called_with(
            model=acquisition.surrogate.model,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            objective_thresholds=self.objective_thresholds,
        )

    # Mock so that we can check that arguments are passed correctly.
    @mock.patch(f"{ACQUISITION_PATH}._get_X_pending_and_observed")
    @mock.patch(
        f"{ACQUISITION_PATH}.subset_model",
        # pyre-fixme[6]: For 1st param expected `Model` but got `None`.
        # pyre-fixme[6]: For 5th param expected `Tensor` but got `None`.
        return_value=SubsetModelData(None, torch.ones(1), None, None, None),
    )
    @mock.patch(
        f"{ACQUISITION_PATH}.get_botorch_objective_and_transform",
        wraps=get_botorch_objective_and_transform,
    )
    def test_init_with_subset_model_false(
        self,
        mock_get_objective_and_transform: Mock,
        mock_subset_model: Mock,
        mock_get_X: Mock,
    ) -> None:
        botorch_objective = LinearMCObjective(weights=torch.tensor([1.0]))
        mock_get_objective_and_transform.return_value = (botorch_objective, None)
        mock_get_X.return_value = (self.pending_observations[0], self.X[:1])
        self.options[Keys.SUBSET_MODEL] = False
        with mock.patch(
            f"{ACQUISITION_PATH}.get_outcome_constraint_transforms",
            return_value=self.constraints,
        ) as mock_get_outcome_constraint_transforms:
            acquisition = Acquisition(
                surrogate=self.surrogate,
                search_space_digest=self.search_space_digest,
                torch_opt_config=self.torch_opt_config,
                botorch_acqf_class=self.botorch_acqf_class,
                options=self.options,
                botorch_acqf_options=self.botorch_acqf_options,
            )
        mock_subset_model.assert_not_called()
        # Check `get_botorch_objective_and_transform` kwargs
        mock_get_objective_and_transform.assert_called_once()
        _, ckwargs = mock_get_objective_and_transform.call_args
        self.assertIs(ckwargs["model"], acquisition.surrogate.model)
        self.assertIs(ckwargs["objective_weights"], self.objective_weights)
        self.assertIs(ckwargs["outcome_constraints"], self.outcome_constraints)
        self.assertTrue(torch.equal(ckwargs["X_observed"], self.X[:1]))
        # Check final `acqf` creation
        self.mock_input_constructor.assert_called_once()
        _, ckwargs = self.mock_input_constructor.call_args
        self.assertIs(ckwargs["model"], acquisition.surrogate.model)
        self.assertIs(ckwargs["objective"], botorch_objective)
        self.assertTrue(torch.equal(ckwargs["X_pending"], self.pending_observations[0]))
        for k, v in self.botorch_acqf_options.items():
            self.assertEqual(ckwargs[k], v)
        self.assertIs(
            ckwargs["constraints"],
            self.constraints,
        )
        mock_get_outcome_constraint_transforms.assert_called_once_with(
            outcome_constraints=self.outcome_constraints
        )

    @mock_botorch_optimize
    def test_optimize(self) -> None:
        for prune_irrelevant_parameters in (False, True):
            if prune_irrelevant_parameters:
                self.options = {
                    "prune_irrelevant_parameters": True,
                }
            else:
                self.options = {}
            acquisition = self.get_acquisition_function(
                fixed_features=self.fixed_features,
                target_point=torch.zeros(3, dtype=torch.double),
            )
            n = 5
            with (
                mock.patch(
                    f"{ACQUISITION_PATH}.optimizer_argparse", wraps=optimizer_argparse
                ) as mock_optimizer_argparse,
                mock.patch(
                    f"{ACQUISITION_PATH}.optimize_acqf", wraps=optimize_acqf
                ) as mock_optimize_acqf,
                mock.patch.object(
                    acquisition,
                    "_prune_irrelevant_parameters",
                    wraps=acquisition._prune_irrelevant_parameters,
                ) as mock_prune_irrelevant_parameters,
            ):
                acquisition.optimize(
                    n=n,
                    search_space_digest=self.search_space_digest,
                    inequality_constraints=self.inequality_constraints,
                    fixed_features=self.fixed_features,
                    rounding_func=self.rounding_func,
                    optimizer_options=self.optimizer_options,
                )
            mock_optimizer_argparse.assert_called_once_with(
                acquisition.acqf,
                optimizer_options=self.optimizer_options,
                optimizer="optimize_acqf",
            )
            mock_optimize_acqf.assert_called_with(
                acq_function=acquisition.acqf,
                sequential=True,
                bounds=mock.ANY,
                q=n,
                options={
                    "init_batch_limit": INIT_BATCH_LIMIT,
                    "batch_limit": BATCH_LIMIT,
                    "max_optimization_problem_aggregation_size": MAX_OPT_AGG_SIZE,
                },
                inequality_constraints=self.inequality_constraints,
                fixed_features=self.fixed_features,
                post_processing_func=self.rounding_func,
                acq_function_sequence=None,
                **self.optimizer_options,
            )
            if prune_irrelevant_parameters:
                mock_prune_irrelevant_parameters.assert_called_once()
                call_kwargs = mock_prune_irrelevant_parameters.call_args_list[0][1]
                for kw_name in (
                    "candidates",
                    "search_space_digest",
                    "inequality_constraints",
                    "fixed_features",
                ):
                    self.assertIsNotNone(call_kwargs[kw_name])
                self.assertIsNotNone(acquisition.num_pruned_dims)
            else:
                mock_prune_irrelevant_parameters.assert_not_called()
                self.assertIsNone(acquisition.num_pruned_dims)
            # can't use assert_called_with on bounds due to ambiguous bool comparison
            expected_bounds = torch.tensor(
                self.search_space_digest.bounds,
                dtype=acquisition.dtype,
                device=acquisition.device,
            ).transpose(0, 1)
            self.assertTrue(
                torch.equal(mock_optimize_acqf.call_args[1]["bounds"], expected_bounds)
            )

    def test_optimize_discrete(self) -> None:
        ssd1 = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(1, 2), (2, 3), (3, 4)],
            categorical_features=[0, 1, 2],
            discrete_choices={0: [1, 2], 1: [2, 3], 2: [3, 4]},
        )
        # check fixed_feature index validation
        with self.assertRaisesRegex(ValueError, "Invalid fixed_feature index"):
            acquisition = self.get_acquisition_function()
            acquisition.optimize(
                n=3,
                search_space_digest=ssd1,
                fixed_features={3: 2.0},
                rounding_func=self.rounding_func,
            )
        # check that SearchSpaceExhausted is raised correctly
        acquisition = self.get_acquisition_function()
        all_possible_choices = list(itertools.product(*ssd1.discrete_choices.values()))
        acquisition.X_observed = torch.tensor(all_possible_choices, **self.tkwargs)
        with self.assertRaisesRegex(
            SearchSpaceExhausted,
            "No more feasible choices in a fully discrete search space.",
        ):
            acquisition.optimize(
                n=1,
                search_space_digest=ssd1,
                rounding_func=self.rounding_func,
            )
        acquisition = self.get_acquisition_function()
        with self.assertWarnsRegex(
            OptimizationWarning,
            "only.*possible choices remain.",
        ):
            acquisition.optimize(
                n=8,
                search_space_digest=ssd1,
                rounding_func=self.rounding_func,
            )

        acquisition = self.get_acquisition_function()
        n = 2

        # Also check that it runs when optimizer options are provided, whether
        # `raw_samples` or `num_restarts` is present or not.
        for optimizer_options in [None, {"raw_samples": 8}, {"num_restarts": 8}]:
            with self.subTest(optimizer_options=optimizer_options):
                acquisition.optimize(
                    n=n,
                    search_space_digest=ssd1,
                    rounding_func=self.rounding_func,
                    optimizer_options=optimizer_options,
                )

        optimizer_options = {"batch_initial_conditions": None}
        with (
            self.subTest(optimizer_options=None),
            self.assertRaisesRegex(ValueError, "Argument "),
        ):
            acquisition.optimize(
                n=n,
                search_space_digest=ssd1,
                rounding_func=self.rounding_func,
                optimizer_options=optimizer_options,
            )

        # check this works without any fixed_feature specified
        # 2 candidates have acqf value 8, but [1, 3, 4] is pending and thus should
        # not be selected. [2, 3, 4] is the best point, but has already been picked
        with (
            mock.patch(
                f"{ACQUISITION_PATH}.optimizer_argparse", wraps=optimizer_argparse
            ) as mock_optimizer_argparse,
            mock.patch(
                f"{ACQUISITION_PATH}.optimize_acqf_discrete",
                wraps=optimize_acqf_discrete,
            ) as mock_optimize_acqf_discrete,
        ):
            X_selected, _, weights = acquisition.optimize(
                n=n,
                search_space_digest=ssd1,
                rounding_func=self.rounding_func,
            )
        mock_optimizer_argparse.assert_called_once_with(
            acquisition.acqf,
            optimizer_options=None,
            optimizer="optimize_acqf_discrete",
        )

        mock_optimize_acqf_discrete.assert_called_once_with(
            acq_function=acquisition.acqf,
            q=n,
            choices=mock.ANY,
            max_batch_size=2048,
            X_avoid=mock.ANY,
            inequality_constraints=None,
        )

        expected_choices = torch.tensor(all_possible_choices)
        expected_avoid = torch.cat([self.X, self.pending_observations[0]], dim=-2)

        kwargs = mock_optimize_acqf_discrete.call_args.kwargs
        self.assertTrue(torch.equal(expected_choices, kwargs["choices"]))
        self.assertTrue(torch.equal(expected_avoid, kwargs["X_avoid"]))

        expected = torch.tensor([[2, 2, 4], [2, 3, 3]]).to(self.X)
        self.assertTrue(X_selected.shape == (2, 3))
        self.assertTrue(
            all((x.unsqueeze(0) == expected).all(dim=-1).any() for x in X_selected)
        )
        self.assertTrue(torch.equal(weights, torch.ones(2)))
        # check with fixed feature
        # Since parameter 1 is fixed to 2, the best 3 candidates are
        # [4, 2, 4], [3, 2, 4], [4, 2, 3]
        ssd2 = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(0, 4) for _ in range(3)],
            categorical_features=[0, 1, 2],
            discrete_choices={k: [0, 1, 2, 3, 4] for k in range(3)},
        )
        with (
            mock.patch(
                f"{ACQUISITION_PATH}.optimizer_argparse", wraps=optimizer_argparse
            ) as mock_optimizer_argparse,
            mock.patch(
                f"{ACQUISITION_PATH}.optimize_acqf_discrete",
                wraps=optimize_acqf_discrete,
            ) as mock_optimize_acqf_discrete,
            mock.patch(
                "botorch.models.gp_regression.SingleTaskGP.batch_shape",
                torch.Size([16]),
            ),
        ):
            X_selected, _, weights = acquisition.optimize(
                n=3,
                search_space_digest=ssd2,
                fixed_features=self.fixed_features,
                rounding_func=self.rounding_func,
            )
        mock_optimizer_argparse.assert_called_once_with(
            acquisition.acqf,
            optimizer_options=None,
            optimizer="optimize_acqf_discrete",
        )
        mock_optimize_acqf_discrete.assert_called_once_with(
            acq_function=acquisition.acqf,
            q=3,
            choices=mock.ANY,
            max_batch_size=128,  # 2048 // 16 (mocked batch_shape).
            X_avoid=mock.ANY,
            inequality_constraints=None,
        )

        expected = torch.tensor([[4, 2, 4], [3, 2, 4], [4, 2, 3]]).to(self.X)
        self.assertTrue(X_selected.shape == (3, 3))
        self.assertTrue(
            all((x.unsqueeze(0) == expected).all(dim=-1).any() for x in X_selected)
        )
        self.assertTrue(torch.equal(weights, torch.ones(3)))
        # check with a constraint that -1 * x[0]  -1 * x[1] >= 0 which should make
        # [0, 0, 4] the best candidate.
        X_selected, _, weights = acquisition.optimize(
            n=1,
            search_space_digest=ssd2,
            rounding_func=self.rounding_func,
            inequality_constraints=[
                (torch.tensor([0, 1], dtype=torch.int64), -torch.ones(2), 0)
            ],
        )
        expected = torch.tensor([[0, 0, 4]]).to(self.X)
        self.assertTrue(torch.equal(expected, X_selected))
        self.assertTrue(torch.equal(weights, torch.tensor([1.0], dtype=self.X.dtype)))
        # Same thing but use two constraints instead
        X_selected, _, weights = acquisition.optimize(
            n=1,
            search_space_digest=ssd2,
            rounding_func=self.rounding_func,
            inequality_constraints=[
                (torch.tensor([0], dtype=torch.int64), -torch.ones(1), 0),
                (torch.tensor([1], dtype=torch.int64), -torch.ones(1), 0),
            ],
        )
        expected = torch.tensor([[0, 0, 4]]).to(self.X)
        self.assertTrue(torch.equal(expected, X_selected))
        self.assertTrue(torch.equal(weights, torch.tensor([1.0])))
        # With no X_observed or X_pending.
        acquisition = self.get_acquisition_function()
        acquisition.X_observed, acquisition.X_pending = None, None
        X_selected, _, weights = acquisition.optimize(
            n=2,
            search_space_digest=ssd1,
            rounding_func=self.rounding_func,
        )
        self.assertTrue(torch.equal(weights, torch.ones(2)))
        expected = torch.tensor([[1, 3, 4], [2, 3, 4]]).to(self.X)
        self.assertTrue(X_selected.shape == (2, 3))
        self.assertTrue(
            all((x.unsqueeze(0) == expected).all(dim=-1).any() for x in X_selected)
        )

    # mock `optimize_acqf_discrete_local_search` because it isn't handled by
    # `mock_botorch_optimize`
    @mock.patch(
        f"{ACQUISITION_PATH}.optimize_acqf_discrete_local_search",
        return_value=(Mock(), Mock()),
    )
    def test_optimize_acqf_discrete_local_search(
        self,
        mock_optimize_acqf_discrete_local_search: Mock,
    ) -> None:
        ssd = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(0, 1) for _ in range(3)],
            categorical_features=[0, 1, 2],
            discrete_choices={  # 30 * 60 * 90 > 100,000
                k: np.linspace(0, 1, 30 * (k + 1)).tolist() for k in range(3)
            },
        )
        acquisition = self.get_acquisition_function()
        with mock.patch(
            f"{ACQUISITION_PATH}.optimizer_argparse", wraps=optimizer_argparse
        ) as mock_optimizer_argparse:
            acquisition.optimize(
                n=3,
                search_space_digest=ssd,
                inequality_constraints=self.inequality_constraints,
                fixed_features=None,
                rounding_func=self.rounding_func,
                optimizer_options=self.optimizer_options,
            )
        mock_optimizer_argparse.assert_called_once_with(
            acquisition.acqf,
            optimizer_options=self.optimizer_options,
            optimizer="optimize_acqf_discrete_local_search",
        )
        mock_optimize_acqf_discrete_local_search.assert_called_once()
        args, kwargs = mock_optimize_acqf_discrete_local_search.call_args
        self.assertEqual(len(args), 0)
        self.assertSetEqual(
            {
                "acq_function",
                "discrete_choices",
                "q",
                "num_restarts",
                "raw_samples",
                "inequality_constraints",
                "X_avoid",
            },
            set(kwargs.keys()),
        )
        self.assertEqual(kwargs["acq_function"], acquisition.acqf)
        self.assertEqual(kwargs["q"], 3)
        self.assertEqual(kwargs["inequality_constraints"], self.inequality_constraints)
        self.assertEqual(kwargs["num_restarts"], self.optimizer_options["num_restarts"])
        self.assertEqual(kwargs["raw_samples"], self.optimizer_options["raw_samples"])
        self.assertTrue(
            all(
                torch.allclose(torch.linspace(0, 1, 30 * (k + 1), **self.tkwargs), c)
                for k, c in enumerate(kwargs["discrete_choices"])
            )
        )
        X_avoid_true = torch.cat((self.X, self.pending_observations[0]), dim=0)
        self.assertEqual(kwargs["X_avoid"].shape, X_avoid_true.shape)
        self.assertTrue(  # The order of the rows may not match
            all((X_avoid_true == x).all(dim=-1).any().item() for x in kwargs["X_avoid"])
        )

    @mock_botorch_optimize
    def test_optimize_acqf_discrete_too_many_choices(self) -> None:
        # Check that mixed optimizer is used when there are too many choices.
        # Otherwise, it should use local search.
        ssd_ordinal_integer = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(0, 100 * (i + 1)) for i in range(3)],
            ordinal_features=[0, 1, 2],
            discrete_choices={i: list(range(100 * (i + 1) + 1)) for i in range(3)},
        )
        ssd_categorical_integer = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(0, 100 * (i + 1)) for i in range(3)],
            categorical_features=[0, 1, 2],
            discrete_choices={i: list(range(100 * (i + 1) + 1)) for i in range(3)},
        )
        ssd_ordinal_noninteger_small = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(0, 99) for i in range(3)],
            ordinal_features=[0, 1, 2],
            discrete_choices={
                i: np.arange(0, 100, dtype=np.float64).tolist() for i in range(3)
            },
        )
        ssd_ordinal_noninteger_large = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(0, 100) for i in range(3)],
            ordinal_features=[0, 1, 2],
            discrete_choices={
                i: np.arange(0, 100 + 1, dtype=np.float64).tolist() for i in range(3)
            },
        )
        acquisition = self.get_acquisition_function()
        for ssd, expected_optimizer in [
            (ssd_ordinal_integer, "optimize_acqf_mixed_alternating"),
            (ssd_categorical_integer, "optimize_acqf_mixed_alternating"),
            (ssd_ordinal_noninteger_small, "optimize_acqf_discrete_local_search"),
            (ssd_ordinal_noninteger_large, "optimize_acqf_mixed_alternating"),
        ]:
            # Mock optimize_acqf_discrete_local_search because it isn't handled
            # by `mock_botorch_optimize`
            with (
                mock.patch(
                    f"{ACQUISITION_PATH}.optimizer_argparse", wraps=optimizer_argparse
                ) as mock_optimizer_argparse,
                mock.patch(
                    f"{ACQUISITION_PATH}.optimize_acqf_discrete_local_search",
                    return_value=(Mock(), Mock()),
                ),
            ):
                acquisition.optimize(
                    n=3,
                    search_space_digest=ssd,
                    inequality_constraints=self.inequality_constraints,
                    fixed_features=None,
                    rounding_func=self.rounding_func,
                    optimizer_options=self.optimizer_options,
                )
            mock_optimizer_argparse.assert_called_once_with(
                acquisition.acqf,
                optimizer_options=self.optimizer_options,
                optimizer=expected_optimizer,
            )

    @mock_botorch_optimize
    def test_optimize_mixed(self) -> None:
        ssd = SearchSpaceDigest(
            feature_names=["a", "b"],
            bounds=[(0, 1), (0, 2)],
            categorical_features=[1],
            discrete_choices={1: [0, 1, 2]},
        )
        acquisition = self.get_acquisition_function()
        with mock.patch(
            f"{ACQUISITION_PATH}.optimize_acqf_mixed", wraps=optimize_acqf_mixed
        ) as mock_optimize_acqf_mixed:
            acquisition.optimize(
                n=3,
                search_space_digest=ssd,
                inequality_constraints=self.inequality_constraints,
                fixed_features=None,
                rounding_func=self.rounding_func,
                optimizer_options=self.optimizer_options,
            )
        mock_optimize_acqf_mixed.assert_called_with(
            acq_function=acquisition.acqf,
            bounds=mock.ANY,
            q=3,
            options={"init_batch_limit": INIT_BATCH_LIMIT, "batch_limit": BATCH_LIMIT},
            fixed_features_list=[{1: 0}, {1: 1}, {1: 2}],
            inequality_constraints=self.inequality_constraints,
            post_processing_func=self.rounding_func,
            **self.optimizer_options,
        )
        # can't use assert_called_with on bounds due to ambiguous bool comparison
        expected_bounds = torch.tensor(ssd.bounds, **self.tkwargs).transpose(0, 1)
        self.assertTrue(
            torch.equal(
                mock_optimize_acqf_mixed.call_args[1]["bounds"], expected_bounds
            )
        )

    @mock_botorch_optimize
    def test_optimize_acqf_mixed_alternating(self) -> None:
        b_upper_bound = 15
        ssd = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(0, 1), (0, b_upper_bound), (0, 5)],
            ordinal_features=[1],
            discrete_choices={1: list(range(16))},
        )
        acquisition = self.get_acquisition_function()

        # Check with ordinal discrete features.
        with mock.patch(
            f"{ACQUISITION_PATH}.optimize_acqf_mixed_alternating",
            wraps=optimize_acqf_mixed_alternating,
        ) as mock_alternating:
            acquisition.optimize(
                n=3,
                search_space_digest=ssd,
                inequality_constraints=self.inequality_constraints,
                fixed_features={0: 0.5},
                rounding_func=self.rounding_func,
                optimizer_options={
                    "options": {"maxiter_alternating": 2},
                    "num_restarts": 2,
                    "raw_samples": 4,
                },
            )
        mock_alternating.assert_called_with(
            acq_function=acquisition.acqf,
            bounds=mock.ANY,
            discrete_dims={1: list(range(16))},
            cat_dims={},
            q=3,
            options={
                "init_batch_limit": INIT_BATCH_LIMIT,
                "batch_limit": BATCH_LIMIT,
                "maxiter_alternating": 2,
            },
            inequality_constraints=self.inequality_constraints,
            fixed_features={0: 0.5},
            post_processing_func=self.rounding_func,
            num_restarts=2,
            raw_samples=4,
        )

        # Check with cateogrial features but no non-integer features.
        ssd_categorical = dataclasses.replace(
            ssd, ordinal_features=[], categorical_features=[1]
        )
        optimizer_options = {
            "options": {"maxiter_alternating": 2},
            "num_restarts": 2,
            "raw_samples": 4,
        }
        with mock.patch(
            f"{ACQUISITION_PATH}.optimize_acqf_mixed_alternating",
            wraps=optimize_acqf_mixed_alternating,
        ) as mock_alternating:
            candidates, acqf_values, arm_weights = acquisition.optimize(
                n=3,
                search_space_digest=ssd_categorical,
                inequality_constraints=self.inequality_constraints,
                fixed_features={0: 0.5},
                rounding_func=self.rounding_func,
                optimizer_options=optimizer_options,
            )
        mock_alternating.assert_called_with(
            acq_function=acquisition.acqf,
            bounds=mock.ANY,
            discrete_dims={},
            cat_dims={1: list(range(b_upper_bound + 1))},
            q=3,
            options={
                "init_batch_limit": INIT_BATCH_LIMIT,
                "batch_limit": BATCH_LIMIT,
                "maxiter_alternating": 2,
            },
            inequality_constraints=self.inequality_constraints,
            fixed_features={0: 0.5},
            post_processing_func=self.rounding_func,
            num_restarts=2,
            raw_samples=4,
        )
        # Check fixed feature
        self.assertTrue((candidates[:, 0] == 0.5).all())
        # Check that one of the params that should be an int is an int
        cat_cand = candidates[1, 1].item()
        self.assertEqual(cat_cand, int(cat_cand))
        self.assertTrue((acqf_values >= 0).all())
        self.assertTrue((arm_weights == 1).all())

        # Check that it is used even if there are non-integer discrete dimensions.
        ssd_nonint = dataclasses.replace(
            ssd,
            bounds=[(0, 10), (0, 10), (0, 10)],
            ordinal_features=[0, 1],
            discrete_choices={
                0: np.arange(10 + 1, dtype=np.float64).tolist(),
                1: np.arange(10 + 1, dtype=np.float64).tolist(),
            },
        )
        with mock.patch(
            f"{ACQUISITION_PATH}.optimize_acqf_mixed_alternating",
            wraps=optimize_acqf_mixed_alternating,
        ) as mock_alternating:
            acquisition.optimize(n=3, search_space_digest=ssd_nonint)
        mock_alternating.assert_called()

        # Check if the `fixed_features` argument works for discrete features.
        ub = 10
        ssd_many_combinations = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            bounds=[(0, 1), (0, ub), (0, ub)],
            ordinal_features=[1, 2],
            discrete_choices={1: list(range(ub + 1)), 2: list(range(ub + 1))},
        )
        dict_args = {
            "n": 1,
            "search_space_digest": ssd_many_combinations,
            "fixed_features": {1: 0},
            "rounding_func": self.rounding_func,
            "optimizer_options": self.optimizer_options,
        }
        with mock.patch(
            f"{ACQUISITION_PATH}.optimize_acqf_mixed_alternating",
            wraps=optimize_acqf_mixed_alternating,
        ) as mock_alternating:
            acquisition.optimize(**dict_args)
        mock_alternating.assert_called()

        # Now that we have made sure alternating minimization is called, call the
        # optimizer for real.
        candidates, _, _ = acquisition.optimize(**dict_args)
        self.assertTrue((candidates[:, 1] == 0).all())

    @mock.patch(
        f"{DummyOneShotAcquisitionFunction.__module__}."
        "DummyOneShotAcquisitionFunction.evaluate",
        return_value=None,
    )
    @mock.patch(
        f"{DummyAcquisitionFunction.__module__}.DummyAcquisitionFunction.__call__",
        return_value=None,
    )
    def test_evaluate(self, mock_call: Mock, mock_evaluate: Mock) -> None:
        # Default acqf.
        acquisition = self.get_acquisition_function()
        acquisition.evaluate(X=self.X)
        mock_call.assert_called_with(X=self.X)
        # One-shot acqf.
        acquisition = self.get_acquisition_function(one_shot=True)
        acquisition.evaluate(X=self.X)
        mock_evaluate.assert_called_with(X=self.X)

    @mock_botorch_optimize
    @mock.patch(  # pyre-ignore
        "ax.generators.torch.botorch_moo_utils._check_posterior_type",
        wraps=lambda y: y,
    )
    @mock.patch(f"{ACQUISITION_PATH}._get_X_pending_and_observed")
    def test_init_moo(
        self,
        mock_get_X: Mock,
        _,
        with_no_X_observed: bool = False,
        with_outcome_constraints: bool = True,
        with_objective_thresholds: bool = True,
    ) -> None:
        acqf_class = (
            DummyAcquisitionFunction
            if with_no_X_observed
            else qNoisyExpectedHypervolumeImprovement
        )
        moo_training_data = [
            SupervisedDataset(
                X=self.X,
                Y=self.Y.repeat(1, 3),
                feature_names=self.feature_names,
                outcome_names=["m1", "m2", "m3"],
            )
        ]
        moo_objective_weights = torch.tensor([-1.0, -1.0, 0.0], **self.tkwargs)
        moo_objective_thresholds = (
            torch.tensor([0.5, 1.5, float("nan")], **self.tkwargs)
            if with_objective_thresholds
            else None
        )
        self.surrogate.fit(
            datasets=moo_training_data,
            search_space_digest=self.search_space_digest,
        )
        if with_no_X_observed:
            mock_get_X.return_value = (self.pending_observations[0], None)
        else:
            mock_get_X.return_value = (self.pending_observations[0], self.X[:1])
        outcome_constraints = (
            (
                torch.tensor([[1.0, 0.0, 0.0]], **self.tkwargs),
                torch.tensor([[10.0]], **self.tkwargs),
            )
            if with_outcome_constraints
            else None
        )

        torch_opt_config = dataclasses.replace(
            self.torch_opt_config,
            objective_weights=moo_objective_weights,
            outcome_constraints=outcome_constraints,
            objective_thresholds=moo_objective_thresholds,
            is_moo=True,
        )
        acquisition = Acquisition(
            surrogate=self.surrogate,
            botorch_acqf_class=acqf_class,
            search_space_digest=self.search_space_digest,
            torch_opt_config=torch_opt_config,
            options=self.options,
            botorch_acqf_options=self.botorch_acqf_options,
        )
        if moo_objective_thresholds is not None:
            self.assertTrue(
                torch.equal(
                    moo_objective_thresholds[:2],
                    # pyre-fixme[16]: Optional type has no attribute `__getitem__`.
                    acquisition.objective_thresholds[:2],
                )
            )
        self.assertTrue(np.isnan(acquisition.objective_thresholds[2].item()))
        # test inferred objective_thresholds
        with ExitStack() as es:
            preds = torch.tensor(
                [
                    [11.0, 2.0],
                    [9.0, 3.0],
                ],
                **self.tkwargs,
            )
            es.enter_context(
                mock.patch.object(
                    self.surrogate.model,
                    "posterior",
                    return_value=MockPosterior(
                        mean=preds,
                        samples=preds,
                    ),
                )
            )
            acquisition = Acquisition(
                surrogate=self.surrogate,
                search_space_digest=self.search_space_digest,
                botorch_acqf_class=acqf_class,
                torch_opt_config=dataclasses.replace(
                    torch_opt_config,
                    objective_thresholds=None,
                ),
                options=self.options,
                botorch_acqf_options=self.botorch_acqf_options,
            )
            if with_no_X_observed:
                self.assertIsNone(acquisition.objective_thresholds)
            else:
                self.assertTrue(
                    torch.equal(
                        acquisition.objective_thresholds[:2],
                        torch.tensor([9.9, 3.3], **self.tkwargs),
                    )
                )
                self.assertTrue(np.isnan(acquisition.objective_thresholds[2].item()))
            # With partial thresholds.
            acquisition = Acquisition(
                surrogate=self.surrogate,
                search_space_digest=self.search_space_digest,
                botorch_acqf_class=acqf_class,
                torch_opt_config=dataclasses.replace(
                    torch_opt_config,
                    objective_thresholds=torch.tensor(
                        [float("nan"), 5.5, float("nan")], **self.tkwargs
                    ),
                ),
                options=self.options,
                botorch_acqf_options=self.botorch_acqf_options,
            )
            if with_no_X_observed:
                # Thresholds are not updated.
                self.assertEqual(acquisition.objective_thresholds[1].item(), 5.5)
                self.assertTrue(np.isnan(acquisition.objective_thresholds[0].item()))
                self.assertTrue(np.isnan(acquisition.objective_thresholds[2].item()))
            else:
                self.assertTrue(
                    torch.equal(
                        acquisition.objective_thresholds[:2],
                        torch.tensor([9.9, 5.5], **self.tkwargs),
                    )
                )
                self.assertTrue(np.isnan(acquisition.objective_thresholds[2].item()))

    def test_init_no_X_observed(self) -> None:
        self.test_init_moo(with_no_X_observed=True, with_outcome_constraints=False)

    def test_init_inferred_thresholds_with_constraints(self) -> None:
        self.test_init_moo(
            with_outcome_constraints=True, with_objective_thresholds=False
        )

    @mock_botorch_optimize
    def test_init_p_feasible(self) -> None:
        # Acquisition initialization should succeed when there are no feasible
        # points and we're using an acqf that doesn't need thresholds.
        moo_training_data = [
            SupervisedDataset(
                X=self.X,
                Y=self.Y.repeat(1, 3),
                feature_names=self.feature_names,
                outcome_names=["m1", "m2", "m3"],
            )
        ]
        self.surrogate.fit(
            datasets=moo_training_data,
            search_space_digest=self.search_space_digest,
        )
        torch_opt_config = TorchOptConfig(
            objective_weights=torch.tensor([1.0, 1.0, 0.0], **self.tkwargs),
            outcome_constraints=(
                torch.tensor([[0.0, 0.0, 1.0]], **self.tkwargs),
                torch.tensor([[0.0]], **self.tkwargs),
            ),
            is_moo=True,
        )
        with self.assertLogs(logger=logger, level="WARNING") as logs:
            acquisition = Acquisition(
                surrogate=self.surrogate,
                search_space_digest=self.search_space_digest,
                botorch_acqf_class=qLogProbabilityOfFeasibility,
                torch_opt_config=torch_opt_config,
            )
        self.assertTrue(
            any("Failed to infer objective thresholds." in str(log) for log in logs)
        )
        self.assertIsInstance(acquisition.acqf, qLogProbabilityOfFeasibility)
        self.assertIsNone(acquisition._full_objective_thresholds)

    def test_expand_and_set_single_feature_to_target(self) -> None:
        # Test helper function
        X = torch.tensor([[1.0, 2.0, 3.0]])  # 1 x 3
        indices = torch.tensor([0, 2])  # indices to modify
        targets = torch.tensor([10.0, 30.0])  # target values

        result = _expand_and_set_single_feature_to_target(X, indices, targets)

        # Should return a 2 x 1 x 3 tensor
        self.assertEqual(result.shape, (2, 1, 3))
        # First row should have X[0] = 10.0
        self.assertEqual(result[0, 0, 0].item(), 10.0)
        self.assertEqual(result[0, 0, 1].item(), 2.0)  # unchanged
        self.assertEqual(result[0, 0, 2].item(), 3.0)  # unchanged
        # Second row should have X[2] = 30.0
        self.assertEqual(result[1, 0, 0].item(), 1.0)  # unchanged
        self.assertEqual(result[1, 0, 1].item(), 2.0)  # unchanged
        self.assertEqual(result[1, 0, 2].item(), 30.0)  # changed

    def test_prune_irrelevant_parameters_no_target_point(self) -> None:
        # Test that ValueError is raised when no target_point is provided
        acq = Acquisition(
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            torch_opt_config=self.torch_opt_config,
            botorch_acqf_class=DummyAcquisitionFunction,
            options={},  # No target_point
        )

        candidates = torch.tensor([[0.9, 0.1]])

        with self.assertRaisesRegex(
            AssertionError,
            "Must specify pruning_target_point to prune irrelevant parameters",
        ):
            acq._prune_irrelevant_parameters(
                candidates=candidates, search_space_digest=self.search_space_digest
            )

    def test_prune_irrelevant_parameters_with_log_acquisition(self) -> None:
        # Test pruning with log-transformed acquisition function

        acq = Acquisition(
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            torch_opt_config=dataclasses.replace(
                self.torch_opt_config, pruning_target_point=torch.tensor([0.5, 0.5])
            ),
            botorch_acqf_class=DummyAcquisitionFunction,
        )

        # Create mock acquisition function with log transformation
        mock_acqf = Mock()
        mock_acqf._log = True
        # Log values that when exp() give predictable pruning behavior
        evaluation_values = [
            torch.tensor([-30.0]),  # baseline value
            torch.tensor([0.0]),  # dense value
            torch.tensor([-0.1, -0.69]),
        ]
        acq.evaluate = Mock(side_effect=evaluation_values)
        acq.acqf = mock_acqf
        candidates = torch.tensor([[0.9, 0.1]])
        pruned_candidates, pruned_values = acq._prune_irrelevant_parameters(
            candidates=candidates, search_space_digest=self.search_space_digest
        )
        self.assertTrue(torch.equal(pruned_candidates, torch.tensor([[0.5, 0.1]])))
        self.assertTrue(torch.equal(pruned_values, torch.tensor([-0.1])))

    def test_prune_irrelevant_parameters_zero_acquisition_value(self) -> None:
        # Test handling of zero or negative acquisition values
        torch.manual_seed(0)

        acq = Acquisition(
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            torch_opt_config=dataclasses.replace(
                self.torch_opt_config, pruning_target_point=torch.tensor([0.5, 0.5])
            ),
            botorch_acqf_class=DummyAcquisitionFunction,
        )

        mock_acqf = Mock()
        mock_acqf._log = False
        mock_evaluate = Mock(return_value=torch.tensor([0.0]))  # Zero acquisition value
        acq.evaluate = mock_evaluate
        acq.acqf = mock_acqf

        candidates = torch.tensor([[0.9, 0.1]])

        pruned_candidates, pruned_values = acq._prune_irrelevant_parameters(
            candidates=candidates, search_space_digest=self.search_space_digest
        )

        # Should handle zero acquisition value gracefully
        self.assertEqual(pruned_candidates.shape, (1, 2))
        self.assertEqual(pruned_values.shape, (1,))
        # Original candidate should be unchanged when acquisition value is zero
        torch.testing.assert_close(pruned_candidates, candidates)

    def test_prune_irrelevant_parameters_single_dimension(self) -> None:
        # Test that pruning stops when only one dimension would remain
        torch.manual_seed(0)

        # Create search space digest for 1D problem
        search_space_digest_1d = SearchSpaceDigest(
            feature_names=["x1"], bounds=[(0.0, 1.0)]
        )

        acq = Acquisition(
            surrogate=self.surrogate,
            search_space_digest=search_space_digest_1d,
            torch_opt_config=dataclasses.replace(
                self.torch_opt_config, pruning_target_point=torch.tensor([0.5])
            ),
            botorch_acqf_class=DummyAcquisitionFunction,
        )

        mock_acqf = Mock()
        mock_acqf._log = False
        acq.acqf = mock_acqf
        acq.evaluate = Mock(side_effect=[torch.tensor([0.0]), torch.tensor([1.0])])

        candidates = torch.tensor([[0.9]])  # 1D candidate

        pruned_candidates, _ = acq._prune_irrelevant_parameters(
            candidates=candidates, search_space_digest=self.search_space_digest
        )

        # Should not prune the only dimension
        torch.testing.assert_close(pruned_candidates, candidates)

    def test_prune_irrelevant_parameters_with_fixed_features(self) -> None:
        # Test pruning with fixed features that should be excluded from pruning
        # Create search space with fixed features
        search_space_digest = SearchSpaceDigest(
            feature_names=["x1", "x2", "x3"],
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        )

        acq = Acquisition(
            surrogate=self.surrogate,
            search_space_digest=search_space_digest,
            torch_opt_config=dataclasses.replace(
                self.torch_opt_config,
                pruning_target_point=torch.tensor([0.5, 0.5, 0.5]),
            ),
            botorch_acqf_class=DummyAcquisitionFunction,
        )

        mock_acqf = Mock()
        mock_acqf._log = False
        # all pruned points will return the same AF value as
        # the dense point, so we should have pruned the first dimension
        # if it weren't fixed
        mock_evaluate = Mock(
            side_effect=[
                torch.tensor([1.0]),  # baseline value
                torch.tensor([1.0]),  # dense value
                torch.tensor([1.0, 1.0]),  # pruned values
            ]
        )
        acq.evaluate = mock_evaluate
        acq.acqf = mock_acqf
        acq._instantiate_acquisition = Mock()

        candidates = torch.tensor([[0.9, 0.1, 0.8]])
        fixed_features = {0: 0.9}  # Fix feature 0 to 0.9

        pruned_candidates, pruned_values = acq._prune_irrelevant_parameters(
            candidates=candidates,
            search_space_digest=search_space_digest,
            fixed_features=fixed_features,
        )
        self.assertTrue(torch.equal(pruned_candidates, torch.tensor([[0.9, 0.5, 0.8]])))
        self.assertTrue(torch.equal(pruned_values, torch.tensor([1.0])))

    def test_prune_irrelevant_parameters_with_custom_threshold(self) -> None:
        search_space_digest = SearchSpaceDigest(
            feature_names=["x1", "x2", "x3"],
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        )

        acq = Acquisition(
            surrogate=self.surrogate,
            search_space_digest=search_space_digest,
            torch_opt_config=dataclasses.replace(
                self.torch_opt_config,
                pruning_target_point=torch.tensor([0.5, 0.5, 0.5]),
            ),
            botorch_acqf_class=DummyAcquisitionFunction,
            options={"irrelevance_pruning_rtol": 1.0},
        )
        # with a rtol of 1, we should prune the first two dimensions
        mock_acqf = Mock()
        mock_acqf._log = False
        mock_evaluate = Mock(
            side_effect=[
                # baseline value is zero, since X_observed is empty
                torch.tensor([1.0]),  # dense value
                torch.tensor([0.3, 0.2, 0.1]),  # pruned values
                torch.tensor([0.2, 0.1]),  # pruned values
            ]
        )
        acq.evaluate = mock_evaluate
        acq.acqf = mock_acqf
        acq._instantiate_acquisition = Mock()

        candidates = torch.tensor([[0.9, 0.1, 0.8]])

        pruned_candidates, pruned_values = acq._prune_irrelevant_parameters(
            candidates=candidates, search_space_digest=search_space_digest
        )
        self.assertTrue(torch.equal(pruned_candidates, torch.tensor([[0.5, 0.5, 0.8]])))
        self.assertTrue(torch.equal(pruned_values, torch.tensor([0.2])))

    def test_prune_irrelevant_parameters_with_inequality_constraints(self) -> None:
        # Test pruning with inequality constraints that filter out infeasible candidates
        search_space_digest = SearchSpaceDigest(
            feature_names=["x1", "x2"],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
        )
        acq = Acquisition(
            surrogate=self.surrogate,
            search_space_digest=search_space_digest,
            torch_opt_config=dataclasses.replace(
                self.torch_opt_config, pruning_target_point=torch.tensor([0.2, 0.2])
            ),
            botorch_acqf_class=DummyAcquisitionFunction,
        )
        mock_acqf = Mock()
        mock_acqf._log = False
        acq.acqf = mock_acqf
        mock_evaluate = Mock(
            side_effect=[
                torch.tensor([1.0]),  # original dense value
                torch.tensor([0.91, 0.9]),  # pruned value (after constraint filtering)
                torch.tensor([0.9]),
            ]
        )
        acq.evaluate = mock_evaluate
        candidates = torch.tensor([[0.8, 0.8]])
        # Constraint: x1 + x2 >= 1.0
        inequality_constraints = [(torch.tensor([0, 1]), torch.tensor([1.0, 1.0]), 1.0)]
        pruned_candidates, pruned_values = acq._prune_irrelevant_parameters(
            candidates=candidates,
            search_space_digest=search_space_digest,
            inequality_constraints=inequality_constraints,
        )
        self.assertTrue(torch.equal(pruned_candidates, torch.tensor([[0.2, 0.8]])))
        self.assertTrue(torch.equal(pruned_values, torch.tensor([0.91])))

    def test_prune_irrelevant_parameters_already_at_target(self) -> None:
        # Test that features already at target point are excluded from pruning

        search_space_digest = SearchSpaceDigest(
            feature_names=["x1", "x2", "x3"],
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        )

        acq = Acquisition(
            surrogate=self.surrogate,
            search_space_digest=search_space_digest,
            torch_opt_config=dataclasses.replace(
                self.torch_opt_config,
                pruning_target_point=torch.tensor([0.5, 0.5, 0.5]),
            ),
            botorch_acqf_class=DummyAcquisitionFunction,
        )

        mock_acqf = Mock()
        mock_acqf._log = False

        acq.acqf = mock_acqf
        mock_evaluate = Mock(
            side_effect=[
                torch.tensor([1.0]),  # original value
                torch.tensor([0.88, 0.96]),
                torch.tensor([0.88]),
            ]
        )
        acq.evaluate = mock_evaluate

        # Candidate where feature 1 is already at target point
        # only dimension 2 should be pruned
        candidates = torch.tensor([[0.9, 0.5, 0.8]])

        pruned_candidates, pruned_values = acq._prune_irrelevant_parameters(
            candidates=candidates, search_space_digest=search_space_digest
        )

        self.assertTrue(torch.equal(pruned_candidates, torch.tensor([[0.9, 0.5, 0.5]])))
        self.assertTrue(torch.equal(pruned_values, torch.tensor([0.96])))

    def test_prune_irrelevant_parameters_specific_pruning_behavior(self) -> None:
        # Test specific pruning behavior with predictable acquisition function responses
        acq = Acquisition(
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            torch_opt_config=dataclasses.replace(
                self.torch_opt_config,
                pruning_target_point=torch.tensor([0.2, 0.8], dtype=torch.double),
            ),
            botorch_acqf_class=DummyAcquisitionFunction,
        )
        mock_acqf = Mock()
        mock_acqf._log = False
        acq.acqf = mock_acqf
        # Test case where first dimension should be pruned but second shouldn't
        original_candidate = torch.tensor([[0.9, 0.1]], dtype=torch.double)
        # Mock acquisition function responses:
        # 1. Original candidate value: 1.0
        # 2. Pruning dimension 0 to target: 0.95 (5% reduction - below 10% threshold)
        # 3. Pruning dimension 1 to target: 0.85 (15% reduction - above 10% threshold)
        acq.evaluate = Mock(
            side_effect=[
                torch.tensor([0.0], dtype=torch.double),  # baseline acquisition value
                torch.tensor(
                    [1.0], dtype=torch.double
                ),  # original dense acquisition value
                torch.tensor(
                    [0.95, 0.85], dtype=torch.double
                ),  # pruning dim 0: 5% reduction (should prune)
            ]
        )
        pruned_candidates, pruned_values = acq._prune_irrelevant_parameters(
            candidates=original_candidate, search_space_digest=self.search_space_digest
        )
        self.assertTrue(
            torch.equal(
                pruned_candidates, torch.tensor([[0.2, 0.1]], dtype=torch.double)
            )
        )
        self.assertTrue(
            torch.equal(pruned_values, torch.tensor([0.95], dtype=torch.double))
        )
        self.assertEqual(acq.num_pruned_dims, [1])

    def test_prune_irrelevant_parameters_no_pruning_above_threshold(self) -> None:
        # Test that no pruning occurs when all reductions are above threshold
        acq = Acquisition(
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            torch_opt_config=dataclasses.replace(
                self.torch_opt_config, pruning_target_point=torch.tensor([0.2, 0.8])
            ),
            botorch_acqf_class=DummyAcquisitionFunction,
        )

        mock_acqf = Mock()
        mock_acqf._log = False
        acq.acqf = mock_acqf

        original_candidate = torch.tensor([[0.9, 0.1]])

        # All pruning attempts result in reductions above threshold
        mock_evaluate = Mock(
            side_effect=[
                torch.tensor([0.0]),  # baseline acquisition value
                torch.tensor([1.0]),  # original dense acquisition value
                torch.tensor([0.7, 0.3]),
            ]
        )
        acq.evaluate = mock_evaluate

        pruned_candidates, pruned_values = acq._prune_irrelevant_parameters(
            candidates=original_candidate, search_space_digest=self.search_space_digest
        )

        # Both dimensions should remain unchanged
        self.assertTrue(torch.equal(pruned_candidates, original_candidate))
        self.assertTrue(torch.equal(pruned_values, torch.tensor([1.0])))

    def test_prune_irrelevant_parameters_multi_candidate_exact_values(self) -> None:
        # Test exact pruned values for multiple candidates
        acq = Acquisition(
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            torch_opt_config=dataclasses.replace(
                self.torch_opt_config, pruning_target_point=torch.tensor([0.2, 0.8])
            ),
            botorch_acqf_class=DummyAcquisitionFunction,
        )
        mock_acqf = Mock()
        mock_acqf._log = False
        acq.acqf = mock_acqf
        original_candidates = torch.tensor([[0.9, 0.1], [0.3, 0.7]])
        # Mock responses for both candidates:
        # Candidate 1: both dims should be pruned
        # Candidate 2: only dim 1 should be pruned
        mock_evaluate = Mock(
            side_effect=[
                # Candidate 1 evaluations
                torch.tensor([0.0]),  # baseline value
                torch.tensor([1.0]),  # original dense value
                torch.tensor([0.95, 0.98]),  # prune dim 1
                torch.tensor([-30.0]),  # compute incremental baseline
                torch.tensor([0.8]),  # original dense value
                torch.tensor([0.75, 0.6]),
            ]
        )
        acq.evaluate = mock_evaluate
        pruned_candidates, pruned_values = acq._prune_irrelevant_parameters(
            candidates=original_candidates, search_space_digest=self.search_space_digest
        )
        expected_candidates = torch.tensor(
            [
                [0.9, 0.8],  # Candidate 1: dim 1 pruned to target
                [0.2, 0.7],  # Candidate 2: only dim 0 pruned to target
            ]
        )
        self.assertTrue(torch.equal(pruned_candidates, expected_candidates))
        self.assertTrue(torch.equal(pruned_values, torch.tensor([0.98, 0.75])))
        self.assertEqual(acq.num_pruned_dims, [1, 1])

    def test_prune_irrelevant_parameters_with_constraints_exact_values(self) -> None:
        # Test exact pruned values when constraints filter out some candidates

        acq = Acquisition(
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            torch_opt_config=dataclasses.replace(
                self.torch_opt_config, pruning_target_point=torch.tensor([0.1, 0.1])
            ),
            botorch_acqf_class=DummyAcquisitionFunction,
        )
        mock_acqf = Mock()
        mock_acqf._log = False
        acq.acqf = mock_acqf
        acq._instantiate_acquisition = Mock()

        original_candidate = torch.tensor([[0.9, 1.0]])
        # pruning does not reduce AF value, but pruning the dim 1 violates
        # the constraint
        mock_evaluate = Mock(
            side_effect=[
                torch.tensor([0.0]),  # baseline af val
                torch.tensor([1.0]),  # dense af val
                # pruned af val for single pruned_candidate, since the other
                # pruned candidate is filtered out
                torch.tensor([1.0]),
            ]
        )
        acq.evaluate = mock_evaluate

        pruned_candidates, pruned_values = acq._prune_irrelevant_parameters(
            candidates=original_candidate,
            search_space_digest=self.search_space_digest,
            inequality_constraints=[
                (
                    torch.tensor([[0, 1]], dtype=torch.long),
                    torch.tensor([[0.0, 1.0]]),
                    1.0,
                )
            ],
        )

        # Only dimension 0 should be pruned
        expected_candidate = torch.tensor([[0.1, 1.0]])
        self.assertTrue(torch.equal(pruned_candidates, expected_candidate))
        self.assertTrue(torch.equal(pruned_values, torch.tensor([1.0])))

    def test_prune_irrelevant_parameters_with_task_and_fidelity_features(self) -> None:
        # Test pruning with both task and fidelity features that should be excluded
        # from pruning
        acq = Acquisition(
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            torch_opt_config=dataclasses.replace(
                self.torch_opt_config,
                pruning_target_point=torch.tensor([0.2, 0.0, 0.8, 0.2]),
            ),
            botorch_acqf_class=DummyAcquisitionFunction,
        )

        mock_acqf = Mock()
        mock_acqf._log = False
        acq.acqf = mock_acqf
        acq._instantiate_acquisition = Mock()

        original_candidate = torch.tensor([[0.9, 0.5, 0.1, 0.3]])

        # Only dimensions 2
        # (dimensions 0 and 1 are task/fidelity features)
        # dimension 3 is skipped since we don't prune all dimensions.
        mock_evaluate = Mock(
            side_effect=[
                torch.tensor([0.0]),  # baseline af val
                torch.tensor([1.0]),  # original dense acquisition value
                torch.tensor([0.92]),  # pruning dim 2
            ]
        )
        acq.evaluate = mock_evaluate

        pruned_candidates, pruned_values = acq._prune_irrelevant_parameters(
            candidates=original_candidate,
            search_space_digest=SearchSpaceDigest(
                feature_names=self.feature_names,
                bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
                task_features=[0],
                fidelity_features=[1],
            ),
        )
        expected_candidate = torch.tensor([[0.9, 0.5, 0.8, 0.3]])
        self.assertTrue(torch.equal(pruned_candidates, expected_candidate))
        self.assertTrue(torch.equal(pruned_values, torch.tensor([0.92])))

    def test_prune_irrelevant_parameters_hss(self) -> None:
        # Test with HSS. HSS shouldn't change the behavior
        # of pruning
        acq = Acquisition(
            surrogate=self.surrogate,
            search_space_digest=self.search_space_digest,
            torch_opt_config=dataclasses.replace(
                self.torch_opt_config,
                pruning_target_point=torch.tensor([0.2, 0.8], dtype=torch.double),
            ),
            botorch_acqf_class=DummyAcquisitionFunction,
        )
        mock_acqf = Mock()
        mock_acqf._log = False
        acq.acqf = mock_acqf
        # Test case where first dimension should be pruned but second shouldn't
        original_candidate = torch.tensor([[0.9, 0.1]], dtype=torch.double)
        # Mock acquisition function responses:
        # 1. Original candidate value: 1.0
        # 2. Pruning dimension 0 to target: 0.95 (5% reduction - below 10% threshold)
        # 3. Pruning dimension 1 to target: 0.85 (15% reduction - above 10% threshold)
        acq.evaluate = Mock(
            side_effect=[
                torch.tensor([0.0], dtype=torch.double),  # baseline acquisition value
                torch.tensor(
                    [1.0], dtype=torch.double
                ),  # original dense acquisition value
                torch.tensor(
                    [0.95, 0.85], dtype=torch.double
                ),  # pruning dim 0: 5% reduction (should prune)
            ]
        )
        pruned_candidates, pruned_values = acq._prune_irrelevant_parameters(
            candidates=original_candidate,
            search_space_digest=SearchSpaceDigest(
                feature_names=self.feature_names,
                bounds=[(0.0, 1.0), (0.0, 10.0)],
                ordinal_features=[0],
                hierarchical_dependencies={0: {0: [1]}},
            ),
        )
        self.assertTrue(
            torch.equal(
                pruned_candidates, torch.tensor([[0.2, 0.1]], dtype=torch.double)
            )
        )
        self.assertTrue(
            torch.equal(pruned_values, torch.tensor([0.95], dtype=torch.double))
        )

    @mock_botorch_optimize
    def test_no_pruning_with_qLogProbabilityOfFeasibility(self) -> None:
        # Test that pruning is NOT called when using qLogProbabilityOfFeasibility,
        # even when prune_irrelevant_parameters option is enabled
        self.options = {"prune_irrelevant_parameters": True}
        self.botorch_acqf_class = qLogProbabilityOfFeasibility  # pyre-ignore [8]
        self.botorch_acqf_options = {}
        acquisition = self.get_acquisition_function(
            fixed_features=self.fixed_features,
        )
        n = 1
        with mock.patch.object(
            acquisition,
            "_prune_irrelevant_parameters",
            wraps=acquisition._prune_irrelevant_parameters,
        ) as mock_prune_irrelevant_parameters:
            acquisition.optimize(
                n=n,
                search_space_digest=self.search_space_digest,
                inequality_constraints=self.inequality_constraints,
                fixed_features=self.fixed_features,
                rounding_func=self.rounding_func,
                optimizer_options=self.optimizer_options,
            )
            mock_prune_irrelevant_parameters.assert_not_called()
            self.assertIsNone(acquisition.num_pruned_dims)


class MultiAcquisitionTest(AcquisitionTest):
    acquisition_class = MultiAcquisition

    def setUp(self) -> None:
        super().setUp()
        self.botorch_acqf_classes_with_options = [
            (DummyAcquisitionFunction, {}),
            (DummyAcquisitionFunction, {"eta": 3.0}),
        ]

    def test_optimize_discrete(self) -> None:
        pass

    def test_optimize_acqf_discrete_local_search(self) -> None:
        pass

    def test_optimize_acqf_discrete_too_many_choices(self) -> None:
        pass

    def test_optimize_mixed(self) -> None:
        pass

    def test_optimize_acqf_mixed_alternating(self) -> None:
        pass

    # Mock so that we can check that arguments are passed correctly.
    @mock.patch(f"{ACQUISITION_PATH}._get_X_pending_and_observed")
    @mock.patch(
        f"{ACQUISITION_PATH}.subset_model",
        # pyre-fixme[6]: For 1st param expected `Model` but got `None`.
        # pyre-fixme[6]: For 5th param expected `Tensor` but got `None`.
        return_value=SubsetModelData(None, torch.ones(1), None, None, None),
    )
    @mock.patch(
        f"{ACQUISITION_PATH}.get_botorch_objective_and_transform",
        wraps=get_botorch_objective_and_transform,
    )
    def test_init_with_subset_model_false(
        self,
        mock_get_objective_and_transform: Mock,
        mock_subset_model: Mock,
        mock_get_X: Mock,
    ) -> None:
        botorch_objective = LinearMCObjective(weights=torch.tensor([1.0]))
        mock_get_objective_and_transform.return_value = (botorch_objective, None)
        mock_get_X.return_value = (self.pending_observations[0], self.X[:1])
        self.options[Keys.SUBSET_MODEL] = False
        with mock.patch(
            f"{ACQUISITION_PATH}.get_outcome_constraint_transforms",
            return_value=self.constraints,
        ) as mock_get_outcome_constraint_transforms:
            acquisition = MultiAcquisition(
                surrogate=self.surrogate,
                search_space_digest=self.search_space_digest,
                torch_opt_config=self.torch_opt_config,
                botorch_acqf_class=self.botorch_acqf_class,
                options=self.options,
                botorch_acqf_options=self.botorch_acqf_options,
                botorch_acqf_classes_with_options=(
                    self.botorch_acqf_classes_with_options
                ),
            )
        mock_subset_model.assert_not_called()
        # Check `get_botorch_objective_and_transform` kwargs
        self.assertEqual(mock_get_objective_and_transform.call_count, 2)
        _, ckwargs = mock_get_objective_and_transform.call_args
        self.assertIs(ckwargs["model"], acquisition.surrogate.model)
        self.assertIs(ckwargs["objective_weights"], self.objective_weights)
        self.assertIs(ckwargs["outcome_constraints"], self.outcome_constraints)
        self.assertTrue(torch.equal(ckwargs["X_observed"], self.X[:1]))
        # Check final `acqf` creation
        self.assertEqual(self.mock_input_constructor.call_count, 2)
        for call, (_, botorch_acqf_options) in zip(
            self.mock_input_constructor.call_args_list,
            self.botorch_acqf_classes_with_options,
        ):
            ckwargs = call.kwargs
            self.assertIs(ckwargs["model"], acquisition.surrogate.model)
            self.assertIs(ckwargs["objective"], botorch_objective)
            self.assertTrue(
                torch.equal(ckwargs["X_pending"], self.pending_observations[0])
            )
            for k, v in botorch_acqf_options.items():
                self.assertEqual(ckwargs[k], v)
            self.assertIs(
                ckwargs["constraints"],
                self.constraints,
            )
        self.assertEqual(mock_get_outcome_constraint_transforms.call_count, 2)
        for call in mock_get_outcome_constraint_transforms.call_args_list:
            self.assertEqual(
                call.kwargs["outcome_constraints"], self.outcome_constraints
            )

    @skip_if_import_error
    def test_optimize(self) -> None:
        from botorch.utils.multi_objective.optimize import optimize_with_nsgaii

        acquisition = self.get_acquisition_function(fixed_features=self.fixed_features)
        n = 5
        optimizer_options = {"max_gen": 3, "population_size": 10}
        with (
            mock.patch(
                f"{ACQUISITION_PATH}.optimizer_argparse", wraps=optimizer_argparse
            ) as mock_optimizer_argparse,
            mock.patch(
                f"{ACQUISITION_PATH}.optimize_with_nsgaii", wraps=optimize_with_nsgaii
            ) as mock_optimize_with_nsgaii,
        ):
            acquisition.optimize(
                n=n,
                search_space_digest=self.search_space_digest,
                inequality_constraints=self.inequality_constraints,
                fixed_features=self.fixed_features,
                rounding_func=self.rounding_func,
                optimizer_options=optimizer_options,
            )
        mock_optimizer_argparse.assert_called_once_with(
            acquisition.acqf,
            optimizer_options=optimizer_options,
            optimizer="optimize_with_nsgaii",
        )
        mock_optimize_with_nsgaii.assert_called_with(
            acq_function=acquisition.acqf,
            bounds=mock.ANY,
            q=n,
            num_objectives=2,
            fixed_features=self.fixed_features,
            **optimizer_options,
        )
        # can't use assert_called_with on bounds due to ambiguous bool comparison
        expected_bounds = torch.tensor(
            self.search_space_digest.bounds,
            dtype=acquisition.dtype,
            device=acquisition.device,
        ).transpose(0, 1)
        self.assertTrue(
            torch.equal(
                mock_optimize_with_nsgaii.call_args[1]["bounds"], expected_bounds
            )
        )

    def test_evaluate(self) -> None:
        acquisition = self.get_acquisition_function()
        with mock.patch.object(acquisition.acqf, "forward") as mock_forward:
            acquisition.evaluate(X=self.X)
            mock_forward.assert_called_once_with(X=self.X)

    def test_ensemble_batch_instantiate_acq(self) -> None:
        surrogate = deepcopy(self.surrogate)
        model_mocks = [mock.MagicMock() for _ in range(3)]
        surrogate._model = model_mocks[0]
        models_for_gen_mock = mock.MagicMock()
        models_for_gen_mock.return_value = (
            ["a", "b"],
            [model_mocks[1], model_mocks[2]],
        )
        surrogate.models_for_gen = models_for_gen_mock
        acq = MultiAcquisition(
            surrogate=surrogate,
            search_space_digest=self.search_space_digest,
            torch_opt_config=self.torch_opt_config,
            botorch_acqf_class=DummyAcquisitionFunction,
            n=1,
        )
        self.assertEqual(acq.acqf.model, model_mocks[0])
        self.assertEqual(acq.acq_function_sequence, None)
