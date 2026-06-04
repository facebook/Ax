#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings

from ax.adapter.registry import Generators
from ax.core.types import TParameterization
from ax.exceptions.core import UnsupportedError
from ax.generation_strategy.generation_strategy import (
    GenerationNode,
    GenerationStrategy,
)
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.metrics.branin import branin
from ax.service.managed_loop import optimize
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize


def _branin_evaluation_function(
    parameterization: TParameterization,
) -> dict[str, tuple[float, float]]:
    x1, x2 = float(parameterization["x1"]), float(parameterization["x2"])
    return {
        "branin": (float(branin(x1, x2)), 0.0),
        "constrained_metric": (float(-branin(x1, x2)), 0.0),
    }


class TestOptimize(TestCase):
    """Tests for the deprecated optimize() function."""

    def setUp(self) -> None:
        super().setUp()
        # optimize() emits a DeprecationWarning; suppress it for all tests
        # except test_optimize_returns_deprecation_warning, which overrides
        # the filter locally via assertWarnsRegex.
        ctx = warnings.catch_warnings()
        ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        warnings.simplefilter("ignore", DeprecationWarning)

    def test_optimize_returns_deprecation_warning(self) -> None:
        with self.assertWarnsRegex(
            DeprecationWarning, expected_regex="optimize is deprecated"
        ):
            optimize(
                parameters=[
                    {"name": "x1", "type": "range", "bounds": [-10.0, 10.0]},
                    {"name": "x2", "type": "range", "bounds": [-10.0, 10.0]},
                ],
                evaluation_function=lambda p: (p["x1"] + 2 * p["x2"] - 7) ** 2
                + (2 * p["x1"] + p["x2"] - 5) ** 2,
                minimize=True,
                total_trials=5,
            )

    def test_optimize_single_metric(self) -> None:
        """Tests optimization with a single scalar return value."""
        best, vals, exp, model = optimize(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-10.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [-10.0, 10.0]},
            ],
            evaluation_function=lambda p: (p["x1"] + 2 * p["x2"] - 7) ** 2
            + (2 * p["x1"] + p["x2"] - 5) ** 2,
            minimize=True,
            total_trials=5,
        )
        self.assertIn("x1", best)
        self.assertIn("x2", best)
        self.assertIsNotNone(vals)
        self.assertIsNone(model)

    def test_optimize_tuple_return(self) -> None:
        """Tests optimization when eval function returns (value, SEM)."""
        best, vals, exp, model = optimize(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-10.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [-10.0, 10.0]},
            ],
            evaluation_function=lambda p: (
                (p["x1"] + 2 * p["x2"] - 7) ** 2 + (2 * p["x1"] + p["x2"] - 5) ** 2,
                0.0,
            ),
            minimize=True,
            total_trials=5,
        )
        self.assertIn("x1", best)
        self.assertIn("x2", best)

    def test_optimize_tuple_none_sem(self) -> None:
        """Tests optimization when eval function returns (value, None)."""
        best, vals, exp, model = optimize(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-10.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [-10.0, 10.0]},
            ],
            evaluation_function=lambda p: (
                (p["x1"] + 2 * p["x2"] - 7) ** 2 + (2 * p["x1"] + p["x2"] - 5) ** 2,
                None,
            ),
            minimize=True,
            total_trials=5,
        )
        self.assertIn("x1", best)
        self.assertIn("x2", best)

    @mock_botorch_optimize
    def test_optimize_dict_return(self) -> None:
        """Tests optimization when eval function returns a dict."""
        best, vals, exp, model = optimize(
            parameters=[
                {
                    "name": "x1",
                    "type": "range",
                    "bounds": [-5.0, 10.0],
                },
                {"name": "x2", "type": "range", "bounds": [0.0, 10.0]},
            ],
            experiment_name="test",
            objective_name="branin",
            minimize=True,
            evaluation_function=_branin_evaluation_function,
            outcome_constraints=["constrained_metric <= 10"],
            total_trials=6,
        )
        self.assertIn("x1", best)
        self.assertIn("x2", best)

    def test_optimize_with_parameter_constraints(self) -> None:
        """Tests optimization with parameter constraints."""
        best, vals, exp, model = optimize(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-10.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [-10.0, 10.0]},
            ],
            evaluation_function=lambda p: (p["x1"] + 2 * p["x2"] - 7) ** 2
            + (2 * p["x1"] + p["x2"] - 5) ** 2,
            minimize=True,
            parameter_constraints=["x1 + x2 <= 5"],
            total_trials=5,
        )
        self.assertIn("x1", best)
        self.assertIn("x2", best)

    def test_optimize_choice_parameters(self) -> None:
        """Tests optimization with choice parameters."""
        best, vals, exp, model = optimize(
            parameters=[
                {
                    "name": "x1",
                    "type": "choice",
                    "values": [1, 2, 3, 4],
                    "value_type": "int",
                },
                {
                    "name": "x2",
                    "type": "choice",
                    "values": [1, 2, 3, 4],
                    "value_type": "int",
                },
            ],
            evaluation_function=lambda p: (p["x1"] - 3) ** 2 + (p["x2"] - 2) ** 2,
            minimize=True,
            total_trials=5,
        )
        self.assertIn("x1", best)
        self.assertIn("x2", best)

    def test_optimize_search_space_exhausted(self) -> None:
        """Tests that optimization handles search space exhaustion gracefully."""
        best, vals, exp, model = optimize(
            parameters=[
                {
                    "name": "x1",
                    "type": "choice",
                    "values": [1, 2],
                    "value_type": "int",
                },
                {
                    "name": "x2",
                    "type": "choice",
                    "values": [1, 2],
                    "value_type": "int",
                },
            ],
            evaluation_function=lambda p: (p["x1"] - 1) ** 2 + (p["x2"] - 1) ** 2,
            minimize=True,
            total_trials=10,
        )
        # Should have stopped at 4 trials (2x2 search space)
        self.assertLessEqual(len(exp.trials), 4)
        self.assertIn("x1", best)
        self.assertIn("x2", best)

    def test_optimize_int_range_parameter(self) -> None:
        """Tests optimization with integer range parameters (used by callers)."""
        best, vals, exp, model = optimize(
            parameters=[
                {
                    "name": "k",
                    "type": "range",
                    "bounds": [2, 10],
                    "value_type": "int",
                },
            ],
            evaluation_function=lambda p: (p["k"] - 5) ** 2,
            minimize=True,
            total_trials=5,
        )
        self.assertIn("k", best)

    def test_optimize_log_scale(self) -> None:
        """Tests optimization with log-scaled parameters."""
        best, vals, exp, model = optimize(
            parameters=[
                {
                    "name": "lr",
                    "type": "range",
                    "bounds": [0.001, 1.0],
                    "value_type": "float",
                    "log_scale": True,
                },
            ],
            evaluation_function=lambda p: (p["lr"] - 0.01) ** 2,
            minimize=True,
            total_trials=5,
        )
        self.assertIn("lr", best)

    def test_optimize_rejects_batch_trials(self) -> None:
        """Tests that arms_per_trial > 1 raises an error."""
        with self.assertRaisesRegex(UnsupportedError, "arms_per_trial=1"):
            optimize(
                parameters=[
                    {"name": "x1", "type": "range", "bounds": [-10.0, 10.0]},
                ],
                evaluation_function=lambda p: p["x1"] ** 2,
                total_trials=5,
                arms_per_trial=3,
            )

    def test_optimize_with_custom_generation_strategy(self) -> None:
        """Tests that a custom generation strategy is passed to the Client."""
        gs = GenerationStrategy(
            nodes=[
                GenerationNode(
                    name="Sobol",
                    generator_specs=[GeneratorSpec(generator_enum=Generators.SOBOL)],
                )
            ],
        )
        best, vals, exp, model = optimize(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-10.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [-10.0, 10.0]},
            ],
            evaluation_function=lambda p: (p["x1"] + 2 * p["x2"] - 7) ** 2
            + (2 * p["x1"] + p["x2"] - 5) ** 2,
            minimize=True,
            total_trials=5,
            generation_strategy=gs,
        )
        self.assertIn("x1", best)
        self.assertIn("x2", best)
