# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

from ax.analysis.plotly.surface.utils import get_fixed_values_for_slice_or_contour
from ax.core.arm import Arm
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.common.testutils import TestCase


class TestGetFixedValuesForSliceOrContour(TestCase):
    def _create_client(self, with_moo: bool = False) -> AxClient:
        client = AxClient(random_seed=42)
        objectives = (
            {
                "obj1": ObjectiveProperties(minimize=True),
                "obj2": ObjectiveProperties(minimize=False),
            }
            if with_moo
            else {"obj": ObjectiveProperties(minimize=True)}
        )
        client.create_experiment(
            is_test=True,
            name="test",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-1.0, 1.0]},
                {"name": "y", "type": "range", "bounds": [-1.0, 1.0]},
            ],
            objectives=objectives,
        )
        return client

    def test_returns_status_quo_when_available(self) -> None:
        client = self._create_client()
        client.experiment.status_quo = Arm(
            parameters={"x": 0.5, "y": -0.5}, name="status_quo_arm"
        )

        fixed_values, description = get_fixed_values_for_slice_or_contour(
            experiment=client.experiment, generation_strategy=None
        )

        self.assertEqual(fixed_values, {"x": 0.5, "y": -0.5})
        self.assertEqual(description, "their status_quo value (Arm status_quo_arm)")

    def test_returns_best_trial_when_no_status_quo(self) -> None:
        client = self._create_client()

        with patch(
            "ax.analysis.plotly.surface.utils._get_best_trial_info",
            return_value=({"x": 0.1, "y": 0.2}, 0, "0_0"),
        ):
            fixed_values, description = get_fixed_values_for_slice_or_contour(
                experiment=client.experiment,
                generation_strategy=client.generation_strategy,
            )

        self.assertEqual(fixed_values, {"x": 0.1, "y": 0.2})
        self.assertEqual(description, "their best trial value (Arm 0_0)")

    def test_returns_center_when_no_best_trial(self) -> None:
        client = self._create_client()

        fixed_values, description = get_fixed_values_for_slice_or_contour(
            experiment=client.experiment, generation_strategy=None
        )

        self.assertEqual(fixed_values, {"x": 0.0, "y": 0.0})
        self.assertEqual(description, "the center of the search space")

    def test_ignores_status_quo_outside_search_space(self) -> None:
        client = self._create_client()
        client.experiment.status_quo = Arm(
            parameters={"x": 5.0, "y": 5.0}, name="status_quo_arm"
        )

        with patch(
            "ax.analysis.plotly.surface.utils._get_best_trial_info",
            return_value=({"x": 0.3, "y": -0.4}, 1, "1_0"),
        ):
            fixed_values, description = get_fixed_values_for_slice_or_contour(
                experiment=client.experiment,
                generation_strategy=client.generation_strategy,
            )

        self.assertEqual(fixed_values, {"x": 0.3, "y": -0.4})
        self.assertEqual(description, "their best trial value (Arm 1_0)")

    def test_returns_center_when_no_optimization_config(self) -> None:
        """Test center of search space is returned when optimization_config is None."""
        client = self._create_client()
        # Remove optimization config to trigger early return in _get_best_trial_info
        client.experiment._optimization_config = None

        fixed_values, description = get_fixed_values_for_slice_or_contour(
            experiment=client.experiment,
            generation_strategy=client.generation_strategy,
        )

        self.assertEqual(fixed_values, {"x": 0.0, "y": 0.0})
        self.assertEqual(description, "the center of the search space")

    def test_returns_center_when_best_point_returns_none(self) -> None:
        """Test center of search space is returned when best point returns None."""
        client = self._create_client()

        with patch(
            "ax.analysis.plotly.surface.utils."
            "get_best_parameters_from_model_predictions_with_trial_index",
            return_value=None,
        ):
            fixed_values, description = get_fixed_values_for_slice_or_contour(
                experiment=client.experiment,
                generation_strategy=client.generation_strategy,
            )

        self.assertEqual(fixed_values, {"x": 0.0, "y": 0.0})
        self.assertEqual(description, "the center of the search space")
