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
from itertools import chain
from typing import Any, Dict, Optional
from unittest import mock
from unittest.mock import Mock

import numpy as np
import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxWarning, SearchSpaceExhausted
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch.utils import (
    _get_X_pending_and_observed,
    subset_model,
    SubsetModelData,
)
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import fast_botorch_optimize
from ax.utils.testing.utils import generic_equals
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.input_constructors import (
    _register_acqf_input_constructor,
    ACQF_INPUT_CONSTRUCTOR_REGISTRY,
    get_acqf_input_constructor,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.objective import LinearMCObjective
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.constraints import get_outcome_constraint_transforms
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.testing import MockPosterior
from torch import Tensor


ACQUISITION_PATH: str = Acquisition.__module__
CURRENT_PATH: str = __name__
SURROGATE_PATH: str = Surrogate.__module__


# Used to avoid going through BoTorch `Acquisition.__init__` which
# requires valid kwargs (correct sizes and lengths of tensors, etc).
class DummyAcquisitionFunction(AcquisitionFunction):
    # pyre-fixme[4]: Attribute must be annotated.
    X_pending = None

    def __init__(self, **kwargs: Any) -> None:
        # pyre-fixme[6]: For 1st param expected `Model` but got `None`.
        AcquisitionFunction.__init__(self, model=None)

    def __call__(self, X: Tensor, **kwargs: Any) -> Tensor:
        return X.sum(dim=-1)

    # pyre-fixme[14]: `set_X_pending` overrides method defined in
    #  `AcquisitionFunction` inconsistently.
    def set_X_pending(self, X: Tensor, **kwargs: Any) -> None:
        self.X_pending = X

    # pyre-fixme[15]: `forward` overrides method defined in `AcquisitionFunction`
    #  inconsistently.
    def forward(self, X: torch.Tensor) -> None:
        pass


class DummyOneShotAcquisitionFunction(DummyAcquisitionFunction, qKnowledgeGradient):
    def evaluate(self, X: Tensor, **kwargs: Any) -> Tensor:
        return X.sum(dim=-1)


class AcquisitionTest(TestCase):
    @fast_botorch_optimize
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
        tkwargs: Dict[str, Any] = {"dtype": torch.double}
        self.botorch_model_class = SingleTaskGP
        self.surrogate = Surrogate(botorch_model_class=self.botorch_model_class)
        self.X = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], **tkwargs)
        self.Y = torch.tensor([[3.0], [4.0]], **tkwargs)
        self.Yvar = torch.tensor([[0.0], [2.0]], **tkwargs)
        self.fidelity_features = [2]
        self.feature_names = ["a", "b", "c"]
        self.metric_names = ["metric"]
        self.training_data = [
            SupervisedDataset(
                X=self.X,
                Y=self.Y,
                feature_names=self.feature_names,
                outcome_names=self.metric_names,
            )
        ]
        self.search_space_digest = SearchSpaceDigest(
            feature_names=self.feature_names,
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            target_values={2: 1.0},
        )
        self.surrogate.fit(
            datasets=self.training_data,
            search_space_digest=SearchSpaceDigest(
                feature_names=self.search_space_digest.feature_names[:1],
                bounds=self.search_space_digest.bounds,
                target_values=self.search_space_digest.target_values,
            ),
        )

        self.botorch_acqf_class = DummyAcquisitionFunction
        self.objective_weights = torch.tensor([1.0])
        self.objective_thresholds = None
        self.pending_observations = [torch.tensor([[1.0, 3.0, 4.0]], **tkwargs)]
        self.outcome_constraints = (
            torch.tensor([[1.0]], **tkwargs),
            torch.tensor([[0.5]], **tkwargs),
        )
        self.constraints = get_outcome_constraint_transforms(
            outcome_constraints=self.outcome_constraints
        )
        self.linear_constraints = None
        self.fixed_features = {1: 2.0}
        self.options = {"cache_root": False, "prune_baseline": False}
        self.inequality_constraints = [
            (torch.tensor([0, 1], **tkwargs), torch.tensor([-1.0, 1.0], **tkwargs), 1)
        ]
        self.rounding_func = lambda x: x
        self.optimizer_options = {Keys.NUM_RESTARTS: 20, Keys.RAW_SAMPLES: 1024}
        self.tkwargs = tkwargs
        self.torch_opt_config = TorchOptConfig(
            objective_weights=self.objective_weights,
            objective_thresholds=self.objective_thresholds,
            pending_observations=self.pending_observations,
            outcome_constraints=self.outcome_constraints,
            linear_constraints=self.linear_constraints,
            fixed_features=self.fixed_features,
        )

    def get_acquisition_function(
        self, fixed_features: Optional[Dict[int, float]] = None, one_shot: bool = False
    ) -> Acquisition:
        return Acquisition(
            botorch_acqf_class=(
                DummyOneShotAcquisitionFunction if one_shot else self.botorch_acqf_class
            ),
            surrogates={"surrogate": self.surrogate},
            search_space_digest=self.search_space_digest,
            torch_opt_config=dataclasses.replace(
                self.torch_opt_config, fixed_features=fixed_features or {}
            ),
            options=self.options,
        )

    def tearDown(self) -> None:
        # Avoid polluting the registry for other tests.
        ACQF_INPUT_CONSTRUCTOR_REGISTRY.pop(DummyAcquisitionFunction)

    def test_init_raises_when_missing_acqf_cls(self) -> None:
        with self.assertRaisesRegex(TypeError, ".* missing .* 'botorch_acqf_class'"):
            # pyre-ignore[20]: Argument `botorch_acqf_class` expected.
            Acquisition(
                surrogates={"surrogate": self.surrogate},
                search_space_digest=self.search_space_digest,
                torch_opt_config=self.torch_opt_config,
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
        acquisition = Acquisition(
            surrogates={"surrogate": self.surrogate},
            search_space_digest=self.search_space_digest,
            torch_opt_config=self.torch_opt_config,
            botorch_acqf_class=self.botorch_acqf_class,
            options=self.options,
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
            model=acquisition.surrogates["surrogate"].model,
            objective_weights=self.objective_weights,
            outcome_constraints=self.outcome_constraints,
            objective_thresholds=self.objective_thresholds,
        )

    @mock.patch(f"{ACQUISITION_PATH}._get_X_pending_and_observed")
    @mock.patch(
        f"{ACQUISITION_PATH}.subset_model",
        # pyre-fixme[6]: For 1st param expected `Model` but got `None`.
        # pyre-fixme[6]: For 5th param expected `Tensor` but got `None`.
        return_value=SubsetModelData(None, torch.ones(1), None, None, None),
    )
    @mock.patch(
        f"{ACQUISITION_PATH}.get_botorch_objective_and_transform",
    )
    @mock.patch(
        f"{CURRENT_PATH}.Acquisition.compute_model_dependencies",
        return_value={"eta": 0.1},
    )
    @mock.patch(
        f"{DummyAcquisitionFunction.__module__}.DummyAcquisitionFunction.__init__",
        return_value=None,
    )
    def test_init_with_subset_model_false(
        self,
        mock_botorch_acqf_class: Mock,
        mock_compute_model_deps: Mock,
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
                surrogates={"surrogate": self.surrogate},
                search_space_digest=self.search_space_digest,
                torch_opt_config=self.torch_opt_config,
                botorch_acqf_class=self.botorch_acqf_class,
                options=self.options,
            )
            mock_subset_model.assert_not_called()
            # Check `get_botorch_objective_and_transform` kwargs
            mock_get_objective_and_transform.assert_called_once()
            _, ckwargs = mock_get_objective_and_transform.call_args
            self.assertIs(ckwargs["model"], acquisition.surrogates["surrogate"].model)
            self.assertIs(ckwargs["objective_weights"], self.objective_weights)
            self.assertIs(ckwargs["outcome_constraints"], self.outcome_constraints)
            self.assertTrue(torch.equal(ckwargs["X_observed"], self.X[:1]))
            # Check final `acqf` creation
            model_deps = {"eta": 0.1}
            self.mock_input_constructor.assert_called_once()
            mock_botorch_acqf_class.assert_called_once()
            _, ckwargs = self.mock_input_constructor.call_args
            self.assertIs(ckwargs["model"], acquisition.surrogates["surrogate"].model)
            self.assertIs(ckwargs["objective"], botorch_objective)
            self.assertTrue(
                torch.equal(ckwargs["X_pending"], self.pending_observations[0])
            )
            for k, v in chain(self.options.items(), model_deps.items()):
                self.assertEqual(ckwargs[k], v)
            self.assertIs(
                ckwargs["constraints"],
                self.constraints,
            )
            mock_get_outcome_constraint_transforms.assert_called_once_with(
                outcome_constraints=self.outcome_constraints
            )

    @mock.patch(f"{ACQUISITION_PATH}.optimize_acqf", return_value=(Mock(), Mock()))
    def test_optimize(self, mock_optimize_acqf: Mock) -> None:
        acquisition = self.get_acquisition_function(fixed_features=self.fixed_features)
        acquisition.optimize(
            n=3,
            search_space_digest=self.search_space_digest,
            inequality_constraints=self.inequality_constraints,
            fixed_features=self.fixed_features,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
        )
        mock_optimize_acqf.assert_called_with(
            acq_function=acquisition.acqf,
            sequential=True,
            bounds=mock.ANY,
            q=3,
            options={"init_batch_limit": 32, "batch_limit": 5},
            inequality_constraints=self.inequality_constraints,
            fixed_features=self.fixed_features,
            post_processing_func=self.rounding_func,
            **self.optimizer_options,
        )
        # can't use assert_called_with on bounds due to ambiguous bool comparison
        expected_bounds = torch.tensor(
            self.search_space_digest.bounds,
            dtype=acquisition.dtype,
            device=acquisition.device,
        ).transpose(0, 1)
        self.assertTrue(
            torch.equal(mock_optimize_acqf.call_args[1]["bounds"], expected_bounds)
        )
        # Test with custom post processing function.
        # First, without the rounding func.
        post_processing_func = Mock(side_effect=lambda x: x**2)
        optimizer_options = self.optimizer_options.copy()
        optimizer_options["post_processing_func"] = post_processing_func
        # Check that `force_use_optimize_acqf` does not lead to errors.
        optimizer_options["force_use_optimize_acqf"] = True
        mock_optimize_acqf.reset_mock()
        acquisition.optimize(
            n=3,
            search_space_digest=self.search_space_digest,
            inequality_constraints=self.inequality_constraints,
            fixed_features=self.fixed_features,
            rounding_func=None,
            optimizer_options=optimizer_options,
        )
        mock_optimize_acqf.assert_called_with(
            acq_function=acquisition.acqf,
            sequential=True,
            bounds=mock.ANY,
            q=3,
            options={"init_batch_limit": 32, "batch_limit": 5},
            inequality_constraints=self.inequality_constraints,
            fixed_features=self.fixed_features,
            post_processing_func=post_processing_func,
            **self.optimizer_options,
        )
        # Now using both rounding func and post processing func.
        mock_optimize_acqf.reset_mock()
        rounding_func = Mock(side_effect=lambda x: x // 4)
        acquisition.optimize(
            n=3,
            search_space_digest=self.search_space_digest,
            inequality_constraints=self.inequality_constraints,
            fixed_features=self.fixed_features,
            rounding_func=rounding_func,
            optimizer_options=optimizer_options,
        )
        actual_func = mock_optimize_acqf.call_args[-1]["post_processing_func"]
        # Call it with a known input to check that the functions are called in
        # the correct order.
        self.assertEqual(actual_func(3.0), 2)  # (3 ** 2) // 4 = 2
        post_processing_func.assert_called_once_with(3.0)
        rounding_func.assert_called_once_with(9.0)

    def test_optimize_discrete(self) -> None:
        ssd1 = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            # pyre-fixme[6]: For 2nd param expected `List[Tuple[Union[float, int],
            #  Union[float, int]]]` but got `List[Tuple[int, int, int]]`.
            bounds=[(1, 2, 3), (2, 3, 4)],
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
            AxWarning,
            "only.*possible choices remain.",
        ):
            acquisition.optimize(
                n=8,
                search_space_digest=ssd1,
                rounding_func=self.rounding_func,
            )

        # check this works without any fixed_feature specified
        # 2 candidates have acqf value 8, but [1, 3, 4] is pending and thus should
        # not be selected. [2, 3, 4] is the best point, but has already been picked
        acquisition = self.get_acquisition_function()
        X_selected, _, weights = acquisition.optimize(
            n=2,
            search_space_digest=ssd1,
            rounding_func=self.rounding_func,
        )
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
            # pyre-fixme[6]: For 2nd param expected `List[Tuple[Union[float, int],
            #  Union[float, int]]]` but got `List[Tuple[int, int, int]]`.
            bounds=[(0, 0, 0), (4, 4, 4)],
            categorical_features=[0, 1, 2],
            # pyre-fixme[6]: For 4th param expected `Dict[int, List[Union[float,
            #  int]]]` but got `Dict[int, List[int]]`.
            discrete_choices={k: [0, 1, 2, 3, 4] for k in range(3)},
        )
        X_selected, _, weights = acquisition.optimize(
            n=3,
            search_space_digest=ssd2,
            fixed_features=self.fixed_features,
            rounding_func=self.rounding_func,
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

    @mock.patch(
        f"{ACQUISITION_PATH}.optimize_acqf_discrete_local_search",
        return_value=(Mock(), Mock()),
    )
    def test_optimize_acqf_discrete_local_search(
        self,
        mock_optimize_acqf_discrete_local_search: Mock,
    ) -> None:
        tkwargs = {"dtype": self.X.dtype, "device": self.X.device}
        ssd = SearchSpaceDigest(
            feature_names=["a", "b", "c"],
            # pyre-fixme[6]: For 2nd param expected `List[Tuple[Union[float, int],
            #  Union[float, int]]]` but got `List[Tuple[int, int, int]]`.
            bounds=[(0, 0, 0), (1, 1, 1)],
            categorical_features=[0, 1, 2],
            discrete_choices={  # 30 * 60 * 90 > 100,000
                k: np.linspace(0, 1, 30 * (k + 1)).tolist() for k in range(3)
            },
        )
        acquisition = self.get_acquisition_function()
        acquisition.optimize(
            n=3,
            search_space_digest=ssd,
            inequality_constraints=self.inequality_constraints,
            fixed_features=None,
            rounding_func=self.rounding_func,
            optimizer_options=self.optimizer_options,
        )
        mock_optimize_acqf_discrete_local_search.assert_called_once()
        args, kwargs = mock_optimize_acqf_discrete_local_search.call_args
        self.assertEqual(kwargs["acq_function"], acquisition.acqf)
        self.assertEqual(kwargs["q"], 3)
        self.assertEqual(kwargs["inequality_constraints"], self.inequality_constraints)
        self.assertEqual(kwargs["num_restarts"], self.optimizer_options["num_restarts"])
        self.assertEqual(kwargs["raw_samples"], self.optimizer_options["raw_samples"])
        self.assertTrue(
            all(
                torch.allclose(torch.linspace(0, 1, 30 * (k + 1), **tkwargs), c)
                for k, c in enumerate(kwargs["discrete_choices"])
            )
        )
        X_avoid_true = torch.cat((self.X, self.pending_observations[0]), dim=0)
        self.assertEqual(kwargs["X_avoid"].shape, X_avoid_true.shape)
        self.assertTrue(  # The order of the rows may not match
            all((X_avoid_true == x).all(dim=-1).any().item() for x in kwargs["X_avoid"])
        )

    @mock.patch(
        f"{ACQUISITION_PATH}.optimize_acqf_mixed", return_value=(Mock(), Mock())
    )
    def test_optimize_mixed(self, mock_optimize_acqf_mixed: Mock) -> None:
        tkwargs = {"dtype": self.X.dtype, "device": self.X.device}
        ssd = SearchSpaceDigest(
            feature_names=["a", "b"],
            bounds=[(0, 1), (0, 2)],
            categorical_features=[1],
            discrete_choices={1: [0, 1, 2]},
        )
        acquisition = self.get_acquisition_function()
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
            sequential=True,
            bounds=mock.ANY,
            q=3,
            options={"init_batch_limit": 32, "batch_limit": 5},
            fixed_features_list=[{1: 0}, {1: 1}, {1: 2}],
            inequality_constraints=self.inequality_constraints,
            post_processing_func=self.rounding_func,
            **self.optimizer_options,
        )
        # can't use assert_called_with on bounds due to ambiguous bool comparison
        expected_bounds = torch.tensor(ssd.bounds, **tkwargs).transpose(0, 1)
        self.assertTrue(
            torch.equal(
                mock_optimize_acqf_mixed.call_args[1]["bounds"], expected_bounds
            )
        )
        # Check that we don't use mixed optimizer if force_use_optimize_acqf is True.
        mock_optimize_acqf_mixed.reset_mock()
        optimizer_options = self.optimizer_options.copy()
        optimizer_options["force_use_optimize_acqf"] = True
        with mock.patch(
            f"{ACQUISITION_PATH}.optimize_acqf", return_value=(Mock(), Mock())
        ) as mock_optimize_acqf:
            acquisition.optimize(
                n=3,
                search_space_digest=ssd,
                inequality_constraints=self.inequality_constraints,
                fixed_features=None,
                rounding_func=self.rounding_func,
                optimizer_options=optimizer_options,
            )
        self.assertEqual(mock_optimize_acqf_mixed.call_count, 0)
        mock_optimize_acqf.assert_called_once()
        mock_optimize_acqf.assert_called_with(
            acq_function=acquisition.acqf,
            sequential=True,
            bounds=mock.ANY,
            q=3,
            options={"init_batch_limit": 32, "batch_limit": 5},
            inequality_constraints=self.inequality_constraints,
            fixed_features=None,
            post_processing_func=self.rounding_func,
            **self.optimizer_options,
        )

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

    @fast_botorch_optimize
    @mock.patch(  # pyre-ignore
        "ax.models.torch.botorch_moo_defaults._check_posterior_type",
        wraps=lambda y: y,
    )
    @mock.patch(f"{ACQUISITION_PATH}._get_X_pending_and_observed")
    def test_init_moo(
        self,
        mock_get_X: Mock,
        _,
        with_no_X_observed: bool = False,
        with_outcome_constraints: bool = True,
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
        moo_objective_thresholds = torch.tensor(
            [0.5, 1.5, float("nan")], **self.tkwargs
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
        )
        acquisition = Acquisition(
            surrogates={"surrogate": self.surrogate},
            botorch_acqf_class=acqf_class,
            search_space_digest=self.search_space_digest,
            torch_opt_config=torch_opt_config,
            options=self.options,
        )
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
                surrogates={"surrogate": self.surrogate},
                search_space_digest=self.search_space_digest,
                botorch_acqf_class=acqf_class,
                torch_opt_config=dataclasses.replace(
                    torch_opt_config,
                    objective_thresholds=None,
                ),
                options=self.options,
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
                surrogates={"surrogate": self.surrogate},
                search_space_digest=self.search_space_digest,
                botorch_acqf_class=acqf_class,
                torch_opt_config=dataclasses.replace(
                    torch_opt_config,
                    objective_thresholds=torch.tensor(
                        [float("nan"), 5.5, float("nan")], **self.tkwargs
                    ),
                ),
                options=self.options,
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
