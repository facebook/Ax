# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import MagicMock, patch

import torch
from ax.generators.torch.utils import (
    _get_X_pending_and_observed,
    get_botorch_objective_and_transform,
)
from ax.utils.common.testutils import TestCase
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    GenericMCObjective,
    ScalarizedPosteriorTransform,
)
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.models.model import Model
from pyre_extensions import none_throws


class TorchUtilsTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.device = torch.device("cpu")
        self.dtype = torch.double
        self.mock_botorch_model = MagicMock(Model)
        tkwargs = {"dtype": self.dtype, "device": self.device}
        self.X_dummy = torch.ones(1, 3, **tkwargs)
        self.outcome_constraints = (
            torch.tensor([[1.0]], **tkwargs),
            torch.tensor([[0.5]], **tkwargs),
        )
        self.objective_weights = torch.ones(1, 1, **tkwargs)
        self.moo_objective_weights = torch.tensor([[1.0], [1.0]], **tkwargs)
        self.objective_thresholds = torch.tensor([0.5, 1.5], **tkwargs)

    def test_get_X_pending_and_observed(self) -> None:
        def _to_obs_set(X: torch.Tensor) -> set[tuple[float]]:
            return {tuple(float(x_i) for x_i in x) for x in X}

        # Apply filter normally
        Xs = [torch.tensor([[0.0, 0.0], [0.0, 1.0]])]
        objective_weights = torch.tensor([[1.0]])
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        fixed_features = {1: 1.0}
        _, X_observed = _get_X_pending_and_observed(
            Xs=Xs,
            objective_weights=self.objective_weights,
            bounds=bounds,
            fixed_features=fixed_features,
        )
        expected = Xs[0][1:]
        self.assertEqual(_to_obs_set(expected), _to_obs_set(none_throws(X_observed)))

        # Filter too strict; return unfiltered X_observed
        fixed_features = {0: 1.0}
        _, X_observed = _get_X_pending_and_observed(
            Xs=Xs,
            objective_weights=objective_weights,
            bounds=bounds,
            fixed_features=fixed_features,
        )
        expected = Xs[0]
        self.assertEqual(_to_obs_set(expected), _to_obs_set(none_throws(X_observed)))

        # Out of design observations are filtered out
        Xs = [torch.tensor([[2.0, 3.0], [3.0, 4.0]])]
        _, X_observed = _get_X_pending_and_observed(
            Xs=Xs,
            objective_weights=objective_weights,
            bounds=bounds,
            fixed_features=fixed_features,
            fit_out_of_design=False,
        )
        self.assertIsNone(X_observed)

        # Keep out of design observations
        _, X_observed = _get_X_pending_and_observed(
            Xs=Xs,
            objective_weights=objective_weights,
            bounds=bounds,
            fixed_features=fixed_features,
            fit_out_of_design=True,
        )
        expected = Xs[0]
        self.assertEqual(_to_obs_set(expected), _to_obs_set(none_throws(X_observed)))

    @patch(
        f"{get_botorch_objective_and_transform.__module__}.get_infeasible_cost",
        return_value=1.0,
    )
    def test_get_botorch_objective(self, _) -> None:
        # For KG with outcome constraints, use a ConstrainedMCObjective
        obj, tf = get_botorch_objective_and_transform(
            botorch_acqf_class=qKnowledgeGradient,
            model=self.mock_botorch_model,
            outcome_constraints=self.outcome_constraints,
            objective_weights=self.objective_weights,
            X_observed=self.X_dummy,
        )
        self.assertIsInstance(obj, ConstrainedMCObjective)
        self.assertIsNone(tf)

        # SampleReducingMCAcquisitionFunctions with outcome constraints, handle
        # the constraints separately
        obj, tf = get_botorch_objective_and_transform(
            botorch_acqf_class=qLogNoisyExpectedImprovement,
            model=self.mock_botorch_model,
            outcome_constraints=self.outcome_constraints,
            objective_weights=self.objective_weights,
            X_observed=self.X_dummy,
        )
        self.assertIsInstance(obj, GenericMCObjective)
        self.assertIsNone(tf)
        # test no X_observed (e.g., if there are no feasible points) with qLogNEI
        obj, tf = get_botorch_objective_and_transform(
            botorch_acqf_class=qLogNoisyExpectedImprovement,
            model=self.mock_botorch_model,
            outcome_constraints=self.outcome_constraints,
            objective_weights=self.objective_weights,
            X_observed=None,
        )
        self.assertIsInstance(obj, GenericMCObjective)
        self.assertIsNone(tf)

        # By default, `ScalarizedPosteriorTransform` should be picked in absence of
        # outcome constraints.
        obj, tf = get_botorch_objective_and_transform(
            botorch_acqf_class=qNoisyExpectedImprovement,
            model=self.mock_botorch_model,
            objective_weights=self.objective_weights,
            X_observed=self.X_dummy,
        )
        self.assertIsNone(obj)
        self.assertIsInstance(tf, ScalarizedPosteriorTransform)

        # Test MOO case.
        with self.assertRaises(BotorchTensorDimensionError):
            get_botorch_objective_and_transform(
                botorch_acqf_class=qNoisyExpectedHypervolumeImprovement,
                model=self.mock_botorch_model,
                objective_weights=self.objective_weights,  # Only has 1 objective.
                X_observed=self.X_dummy,
            )

        obj, tf = get_botorch_objective_and_transform(
            botorch_acqf_class=qNoisyExpectedHypervolumeImprovement,
            model=self.mock_botorch_model,
            objective_weights=self.moo_objective_weights,  # Has 2 objectives.
            X_observed=self.X_dummy,
        )
        self.assertIsInstance(obj, WeightedMCMultiOutputObjective)
        self.assertIsNone(tf)
