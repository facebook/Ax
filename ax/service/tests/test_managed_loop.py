#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import Mock, patch

import numpy as np
from ax.exceptions.core import UserInputError
from ax.metrics.branin import branin
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import MODEL_KEY_TO_MODEL_SETUP, Models
from ax.service.managed_loop import OptimizationLoop, optimize
from ax.utils.common.testutils import TestCase


def _branin_evaluation_function(parameterization, weight=None):
    if any(param_name not in parameterization.keys() for param_name in ["x1", "x2"]):
        raise ValueError("Parametrization does not contain x1 or x2")
    x1, x2 = parameterization["x1"], parameterization["x2"]
    return {
        "branin": (branin(x1, x2), 0.0),
        "constrained_metric": (-branin(x1, x2), 0.0),
    }


def _branin_evaluation_function_v2(parameterization, weight=None):
    if any(param_name not in parameterization.keys() for param_name in ["x1", "x2"]):
        raise ValueError("Parametrization does not contain x1 or x2")
    x1, x2 = parameterization["x1"], parameterization["x2"]
    return (branin(x1, x2), 0.0)


def _branin_evaluation_function_with_unknown_sem(parameterization, weight=None):
    if any(param_name not in parameterization.keys() for param_name in ["x1", "x2"]):
        raise ValueError("Parametrization does not contain x1 or x2")
    x1, x2 = parameterization["x1"], parameterization["x2"]
    return (branin(x1, x2), None)


class TestManagedLoop(TestCase):
    """Check functionality of optimization loop."""

    def setUp(self):
        # To avoid tests timing out due to GP fit / gen times.
        patch.dict(
            f"{Models.__module__}.MODEL_KEY_TO_MODEL_SETUP",
            {"GPEI": MODEL_KEY_TO_MODEL_SETUP["Sobol"]},
        ).start()

    def test_branin(self) -> None:
        """Basic async synthetic function managed loop case."""
        loop = OptimizationLoop.with_evaluation_function(
            parameters=[
                {
                    "name": "x1",
                    "type": "range",
                    "bounds": [-5.0, 10.0],
                    "value_type": "float",
                    "log_scale": False,
                },
                {"name": "x2", "type": "range", "bounds": [0.0, 10.0]},
            ],
            experiment_name="test",
            objective_name="branin",
            minimize=True,
            evaluation_function=_branin_evaluation_function,
            parameter_constraints=["x1 + x2 <= 20"],
            outcome_constraints=["constrained_metric <= 10"],
            total_trials=6,
        )
        bp, _ = loop.full_run().get_best_point()
        self.assertIn("x1", bp)
        self.assertIn("x2", bp)
        with self.assertRaisesRegex(ValueError, "Optimization is complete"):
            loop.run_trial()

    def test_branin_without_objective_name(self) -> None:
        loop = OptimizationLoop.with_evaluation_function(
            parameters=[
                {
                    "name": "x1",
                    "type": "range",
                    "bounds": [-5.0, 10.0],
                    "value_type": "float",
                    "log_scale": False,
                },
                {"name": "x2", "type": "range", "bounds": [0.0, 10.0]},
            ],
            minimize=True,
            evaluation_function=_branin_evaluation_function_v2,
            parameter_constraints=["x1 + x2 <= 20"],
            total_trials=6,
        )
        bp, _ = loop.full_run().get_best_point()
        self.assertIn("x1", bp)
        self.assertIn("x2", bp)

    def test_branin_with_unknown_sem(self) -> None:
        loop = OptimizationLoop.with_evaluation_function(
            parameters=[
                {
                    "name": "x1",
                    "type": "range",
                    "bounds": [-5.0, 10.0],
                    "value_type": "float",
                    "log_scale": False,
                },
                {"name": "x2", "type": "range", "bounds": [0.0, 10.0]},
            ],
            minimize=True,
            evaluation_function=_branin_evaluation_function_with_unknown_sem,
            parameter_constraints=["x1 + x2 <= 20"],
            total_trials=6,
        )
        bp, _ = loop.full_run().get_best_point()
        self.assertIn("x1", bp)
        self.assertIn("x2", bp)

    def test_branin_batch(self) -> None:
        """Basic async synthetic function managed loop case."""

        batch_branin = Mock(side_effect=_branin_evaluation_function)

        loop = OptimizationLoop.with_evaluation_function(
            parameters=[
                {
                    "name": "x1",
                    "type": "range",
                    "bounds": [-5.0, 10.0],
                    "value_type": "float",
                    "log_scale": False,
                },
                {"name": "x2", "type": "range", "bounds": [0.0, 10.0]},
            ],
            experiment_name="test",
            objective_name="branin",
            minimize=True,
            evaluation_function=batch_branin,
            parameter_constraints=["x1 + x2 <= 20"],
            outcome_constraints=["constrained_metric <= 10"],
            total_trials=5,
            arms_per_trial=3,
        )
        bp, vals = loop.full_run().get_best_point()
        branin_calls = batch_branin.call_args_list
        self.assertTrue(
            all(len(args) == 2 for args, _ in branin_calls),
            branin_calls,
        )
        self.assertTrue(
            all(type(args[1]) is np.float64 for args, _ in branin_calls),
            branin_calls,
        )
        self.assertIn("x1", bp)
        self.assertIn("x2", bp)
        assert vals is not None
        self.assertIn("branin", vals[0])
        self.assertIn("branin", vals[1])
        self.assertIn("branin", vals[1]["branin"])
        # Check that all total_trials * arms_per_trial * 2 metrics evaluations
        # are present in the dataframe.
        self.assertEqual(len(loop.experiment.fetch_data().df.index), 30)

    def test_optimize(self) -> None:
        """Tests optimization as a single call."""
        best, vals, exp, model = optimize(
            parameters=[  # pyre-fixme[6]
                {"name": "x1", "type": "range", "bounds": [-10.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [-10.0, 10.0]},
            ],
            # Booth function.
            evaluation_function=lambda p: (p["x1"] + 2 * p["x2"] - 7) ** 2
            + (2 * p["x1"] + p["x2"] - 5) ** 2,
            minimize=True,
            total_trials=5,
        )
        self.assertIn("x1", best)
        self.assertIn("x2", best)
        assert vals is not None
        self.assertIn("objective", vals[0])
        self.assertIn("objective", vals[1])
        self.assertIn("objective", vals[1]["objective"])

    @patch(
        "ax.service.managed_loop.get_best_from_model_predictions",
        autospec=True,
        return_value=({"x1": 2.0, "x2": 3.0}, ({"a": 9.0}, {"a": {"a": 3.0}})),
    )
    def test_optimize_with_predictions(self, _) -> None:
        """Tests optimization as a single call."""
        best, vals, exp, model = optimize(
            parameters=[  # pyre-fixme[6]
                {"name": "x1", "type": "range", "bounds": [-10.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [-10.0, 10.0]},
            ],
            # Booth function.
            evaluation_function=lambda p: (p["x1"] + 2 * p["x2"] - 7) ** 2
            + (2 * p["x1"] + p["x2"] - 5) ** 2,
            minimize=True,
            total_trials=6,
            objective_name="a",
        )
        self.assertIn("x1", best)
        self.assertIn("x2", best)
        assert vals is not None
        self.assertIn("a", vals[0])
        self.assertIn("a", vals[1])
        self.assertIn("a", vals[1]["a"])

    def test_optimize_unknown_sem(self) -> None:
        """Tests optimization as a single call."""
        best, vals, exp, model = optimize(
            parameters=[  # pyre-fixme[6]
                {"name": "x1", "type": "range", "bounds": [-10.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [-10.0, 10.0]},
            ],
            # Booth function.
            evaluation_function=lambda p: (
                (p["x1"] + 2 * p["x2"] - 7) ** 2 + (2 * p["x1"] + p["x2"] - 5) ** 2,
                None,
            ),
            minimize=True,
            total_trials=6,
        )
        self.assertIn("x1", best)
        self.assertIn("x2", best)
        self.assertIsNotNone(vals)
        self.assertIn("objective", vals[0])
        self.assertIn("objective", vals[1])
        self.assertIn("objective", vals[1]["objective"])

    def test_optimize_propagates_random_seed(self) -> None:
        """Tests optimization as a single call."""
        _, _, _, model = optimize(
            parameters=[  # pyre-fixme[6]
                {"name": "x1", "type": "range", "bounds": [-10.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [-10.0, 10.0]},
            ],
            # Booth function.
            evaluation_function=lambda p: (p["x1"] + 2 * p["x2"] - 7) ** 2
            + (2 * p["x1"] + p["x2"] - 5) ** 2,
            minimize=True,
            total_trials=5,
            random_seed=12345,
        )
        self.assertEqual(12345, model.model.seed)

    def test_optimize_search_space_exhausted(self) -> None:
        """Tests optimization as a single call."""
        best, vals, exp, model = optimize(
            parameters=[  # pyre-fixme[6]
                {"name": "x1", "type": "choice", "values": [1, 2]},
                {"name": "x2", "type": "choice", "values": [1, 2]},
            ],
            # Booth function.
            evaluation_function=lambda p: (
                (p["x1"] + 2 * p["x2"] - 7) ** 2 + (2 * p["x1"] + p["x2"] - 5) ** 2,
                None,
            ),
            minimize=True,
            total_trials=6,
        )
        self.assertEqual(len(exp.trials), 4)
        self.assertIn("x1", best)
        self.assertIn("x2", best)
        self.assertIsNotNone(vals)
        self.assertIn("objective", vals[0])
        self.assertIn("objective", vals[1])
        self.assertIn("objective", vals[1]["objective"])

    def test_custom_gs(self) -> None:
        """Managed loop with custom generation strategy"""
        strategy0 = GenerationStrategy(
            name="Sobol", steps=[GenerationStep(model=Models.SOBOL, num_trials=-1)]
        )
        loop = OptimizationLoop.with_evaluation_function(
            parameters=[
                {
                    "name": "x1",
                    "type": "range",
                    "bounds": [-5.0, 10.0],
                    "value_type": "float",
                    "log_scale": False,
                },
                {"name": "x2", "type": "range", "bounds": [0.0, 10.0]},
            ],
            experiment_name="test",
            objective_name="branin",
            minimize=True,
            evaluation_function=_branin_evaluation_function,
            total_trials=6,
            generation_strategy=strategy0,
        )
        bp, _ = loop.full_run().get_best_point()
        self.assertIn("x1", bp)
        self.assertIn("x2", bp)

    def test_optimize_graceful_exit_on_exception(self) -> None:
        """Tests optimization as a single call, with exception during
        candidate generation.
        """
        best, vals, exp, model = optimize(
            parameters=[  # pyre-fixme[6]
                {"name": "x1", "type": "range", "bounds": [-10.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [-10.0, 10.0]},
            ],
            # Booth function.
            evaluation_function=lambda p: (
                (p["x1"] + 2 * p["x2"] - 7) ** 2 + (2 * p["x1"] + p["x2"] - 5) ** 2,
                None,
            ),
            minimize=True,
            total_trials=6,
            generation_strategy=GenerationStrategy(
                name="Sobol", steps=[GenerationStep(model=Models.SOBOL, num_trials=3)]
            ),
        )
        self.assertEqual(len(exp.trials), 3)  # Check that we stopped at 3 trials.
        # All the regular return values should still be present.
        self.assertIn("x1", best)
        self.assertIn("x2", best)
        self.assertIsNotNone(vals)
        self.assertIn("objective", vals[0])
        self.assertIn("objective", vals[1])
        self.assertIn("objective", vals[1]["objective"])

    @patch(
        "ax.core.experiment.Experiment.new_trial",
        side_effect=RuntimeError("cholesky_cpu error - bad matrix"),
    )
    def test_annotate_exception(self, _):
        strategy0 = GenerationStrategy(
            name="Sobol", steps=[GenerationStep(model=Models.SOBOL, num_trials=-1)]
        )
        loop = OptimizationLoop.with_evaluation_function(
            parameters=[
                {
                    "name": "x1",
                    "type": "range",
                    "bounds": [-5.0, 10.0],
                    "value_type": "float",
                    "log_scale": False,
                },
                {"name": "x2", "type": "range", "bounds": [0.0, 10.0]},
            ],
            experiment_name="test",
            objective_name="branin",
            minimize=True,
            evaluation_function=_branin_evaluation_function,
            total_trials=6,
            generation_strategy=strategy0,
        )
        with self.assertRaisesRegex(
            expected_exception=RuntimeError,
            expected_regex="Cholesky errors typically occur",
        ):
            loop.run_trial()

    def test_invalid_arms_per_trial(self) -> None:
        with self.assertRaisesRegex(
            UserInputError, "Invalid number of arms per trial: 0"
        ):
            loop = OptimizationLoop.with_evaluation_function(
                parameters=[  # pyre-fixme[6]
                    {"name": "x1", "type": "range", "bounds": [-10.0, 10.0]},
                    {"name": "x2", "type": "range", "bounds": [-10.0, 10.0]},
                ],
                experiment_name="test",
                objective_name="foo",
                evaluation_function=lambda p: 0.0,
                minimize=True,
                total_trials=5,
                arms_per_trial=0,
            )
            loop.run_trial()

    def test_eval_function_with_wrong_parameter_count_generates_error(self):
        with self.assertRaises(UserInputError):
            loop = OptimizationLoop.with_evaluation_function(
                parameters=[  # pyre-fixme[6]
                    {"name": "x1", "type": "range", "bounds": [-10.0, 10.0]},
                    {"name": "x2", "type": "range", "bounds": [-10.0, 10.0]},
                ],
                experiment_name="test",
                objective_name="foo",
                evaluation_function=lambda: 1.0,
                minimize=True,
                total_trials=5,
            )
            loop.run_trial()
