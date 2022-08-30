# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Tuple
from unittest.mock import MagicMock, patch

import torch
from ax.models.torch.utils import (
    _get_X_pending_and_observed,
    get_botorch_objective_and_transform,
)
from ax.utils.common.testutils import TestCase
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    ScalarizedPosteriorTransform,
)
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.models.model import Model


class TorchUtilsTest(TestCase):
    # pyre-fixme[3]: Return type must be annotated.
    def setUp(self):
        self.device = torch.device("cpu")
        self.dtype = torch.double
        self.mock_botorch_model = MagicMock(Model)
        tkwargs = {"dtype": self.dtype, "device": self.device}
        self.X_dummy = torch.ones(1, 3, **tkwargs)
        self.outcome_constraints = (
            torch.tensor([[1.0]], **tkwargs),
            torch.tensor([[0.5]], **tkwargs),
        )
        self.objective_weights = torch.ones(1, **tkwargs)
        self.moo_objective_weights = torch.ones(2, **tkwargs)
        self.objective_thresholds = torch.tensor([0.5, 1.5], **tkwargs)

    # pyre-fixme[3]: Return type must be annotated.
    def test_get_X_pending_and_observed(self):
        def _to_obs_set(X: torch.Tensor) -> Set[Tuple[float]]:
            return {tuple(float(x_i) for x_i in x) for x in X}

        # Apply filter normally
        Xs = [torch.tensor([[0.0, 0.0], [0.0, 1.0]])]
        objective_weights = torch.tensor([1.0])
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        fixed_features = {1: 1.0}
        _, X_observed = _get_X_pending_and_observed(
            Xs=Xs,
            objective_weights=self.objective_weights,
            bounds=bounds,
            fixed_features=fixed_features,
        )
        expected = Xs[0][1:]
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
        self.assertEqual(_to_obs_set(expected), _to_obs_set(X_observed))

        # Filter too strict; return unfiltered X_observed
        fixed_features = {0: 1.0}
        _, X_observed = _get_X_pending_and_observed(
            Xs=Xs,
            objective_weights=objective_weights,
            bounds=bounds,
            fixed_features=fixed_features,
        )
        expected = Xs[0]
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
        self.assertEqual(_to_obs_set(expected), _to_obs_set(X_observed))

    @patch(
        f"{get_botorch_objective_and_transform.__module__}.get_infeasible_cost",
        return_value=1.0,
    )
    # pyre-fixme[3]: Return type must be annotated.
    def test_get_botorch_objective(self, _):
        # If there are outcome constraints, use a ConstrainedMCObjective
        obj, tf = get_botorch_objective_and_transform(
            model=self.mock_botorch_model,
            outcome_constraints=self.outcome_constraints,
            objective_weights=self.objective_weights,
            X_observed=self.X_dummy,
        )
        self.assertIsInstance(obj, ConstrainedMCObjective)
        self.assertIsNone(tf)

        # By default, `ScalarizedPosteriorTransform` should be picked in absence of
        # outcome constraints.
        obj, tf = get_botorch_objective_and_transform(
            model=self.mock_botorch_model,
            objective_weights=self.objective_weights,
            X_observed=self.X_dummy,
        )
        self.assertIsNone(obj)
        self.assertIsInstance(tf, ScalarizedPosteriorTransform)

        # Test MOO case.
        with self.assertRaises(BotorchTensorDimensionError):
            get_botorch_objective_and_transform(
                model=self.mock_botorch_model,
                objective_weights=self.objective_weights,  # Only has 1 objective.
                X_observed=self.X_dummy,
                objective_thresholds=self.objective_thresholds,
            )

        obj, tf = get_botorch_objective_and_transform(
            model=self.mock_botorch_model,
            objective_weights=self.moo_objective_weights,  # Has 2 objectives.
            X_observed=self.X_dummy,
            objective_thresholds=self.objective_thresholds,
        )
        self.assertIsInstance(obj, WeightedMCMultiOutputObjective)
        self.assertIsNone(tf)
