# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Tuple
from unittest.mock import MagicMock, Mock, patch

import torch
from ax.exceptions.core import UnsupportedError
from ax.models.torch.utils import (
    _get_X_pending_and_observed,
    get_botorch_objective_and_transform,
)
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast, not_none
from botorch.acquisition.multi_objective.multi_output_risk_measures import (
    MARS,
    MultiOutputExpectation,
)
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    ScalarizedPosteriorTransform,
)
from botorch.acquisition.risk_measures import Expectation
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.models.model import Model


class TorchUtilsTest(TestCase):
    def setUp(self) -> None:
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

    def test_get_X_pending_and_observed(self) -> None:
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
        self.assertEqual(_to_obs_set(expected), _to_obs_set(not_none(X_observed)))

        # Filter too strict; return unfiltered X_observed
        fixed_features = {0: 1.0}
        _, X_observed = _get_X_pending_and_observed(
            Xs=Xs,
            objective_weights=objective_weights,
            bounds=bounds,
            fixed_features=fixed_features,
        )
        expected = Xs[0]
        self.assertEqual(_to_obs_set(expected), _to_obs_set(not_none(X_observed)))

    @patch(
        f"{get_botorch_objective_and_transform.__module__}.get_infeasible_cost",
        return_value=1.0,
    )
    def test_get_botorch_objective(self, _) -> None:
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

    @patch(f"{MARS.__module__}.MARS.set_baseline_Y")
    def test_get_botorch_objective_w_risk_measures(
        self, mock_set_baseline_Y: Mock
    ) -> None:
        # Test outcome constraint error.
        with self.assertRaisesRegex(NotImplementedError, "Outcome constraints"):
            get_botorch_objective_and_transform(
                model=self.mock_botorch_model,
                objective_weights=self.objective_weights,
                outcome_constraints=self.outcome_constraints,
                risk_measure=Expectation(n_w=5),
            )
        # Test user supplied preprocessing function error.
        with self.assertRaisesRegex(UnsupportedError, "User supplied"):
            get_botorch_objective_and_transform(
                model=self.mock_botorch_model,
                objective_weights=self.objective_weights,
                risk_measure=Expectation(
                    n_w=5,
                    preprocessing_function=WeightedMCMultiOutputObjective(
                        weights=torch.tensor([1.0, 1.0])
                    ),
                ),
            )
        # Test MARS errors.
        with self.assertRaisesRegex(UnsupportedError, "X_observed"):
            get_botorch_objective_and_transform(
                model=self.mock_botorch_model,
                objective_weights=self.moo_objective_weights,
                risk_measure=MARS(alpha=0.8, n_w=5, chebyshev_weights=[]),
            )
        # Test single objective case.
        risk_measure, _ = get_botorch_objective_and_transform(
            model=self.mock_botorch_model,
            objective_weights=torch.tensor([-1.0]),
            risk_measure=Expectation(n_w=2),
        )
        self.assertTrue(
            torch.allclose(
                not_none(risk_measure)(torch.tensor([[1.0], [2.0]])),
                torch.tensor([-1.5]),
            )
        )
        # Test scalarized objective with single objective risk measure.
        risk_measure, _ = get_botorch_objective_and_transform(
            model=self.mock_botorch_model,
            objective_weights=torch.tensor([-1.0, 1.0, 0.0]),
            risk_measure=Expectation(n_w=2),
        )
        Y = torch.tensor([[1.0, -1.0, 3.0], [2.0, -2.0, 3.0]])
        self.assertTrue(
            torch.allclose(
                not_none(risk_measure)(Y),
                torch.tensor([-3.0]),
            )
        )
        # Test MO risk measure.
        risk_measure, _ = get_botorch_objective_and_transform(
            model=self.mock_botorch_model,
            objective_weights=torch.tensor([-1.0, 2.0, 0.0]),
            risk_measure=MultiOutputExpectation(n_w=2),
        )
        self.assertTrue(
            torch.allclose(
                not_none(risk_measure)(Y),
                torch.tensor([-1.5, -3.0]),
            )
        )
        # Test MARS.
        risk_measure, _ = get_botorch_objective_and_transform(
            model=self.mock_botorch_model,
            objective_weights=torch.tensor([-1.0, 2.0, 0.0]),
            risk_measure=MARS(
                alpha=0.8,
                n_w=2,
                chebyshev_weights=[],
                baseline_Y=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            ),
            X_observed=torch.ones(2, 2),  # dummy
        )
        mock_set_baseline_Y.assert_called_once()
        risk_measure = checked_cast(MARS, risk_measure)
        self.assertIsInstance(
            risk_measure.preprocessing_function, WeightedMCMultiOutputObjective
        )
        ch_weights = risk_measure.chebyshev_weights
        self.assertEqual(ch_weights.shape, torch.Size([2]))
        self.assertTrue(torch.allclose(ch_weights.sum(), torch.tensor(1.0)))
        # Overwrite ch weights to simplify the test.
        risk_measure.chebyshev_weights = [0.0, 1.0]
        self.assertTrue(
            torch.allclose(
                not_none(risk_measure)(Y),
                torch.tensor([-4.0]),
            )
        )
