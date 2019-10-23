#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from enum import Enum
from unittest.mock import patch

from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.metrics.branin import branin
from ax.modelbridge.factory import get_sobol
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


# Patch the Models enum to replace GPEI with Sobol.
def get_experiment_data_sobol(experiment, data):
    return get_sobol(experiment.search_space)


class FakeModels(Enum):
    SOBOL = get_sobol
    GPEI = get_experiment_data_sobol


class TestManagedLoop(TestCase):
    """Check functionality of optimization loop."""

    @patch(
        "ax.modelbridge.torch.TorchModelBridge.gen",
        return_value=GeneratorRun(
            arms=[Arm(parameters={"x1": -2.73, "x2": 1.33})],
            best_arm_predictions=(
                Arm(name="1_0", parameters={"x1": 4.34, "x2": 2.60}),
                (
                    {"branin": 7.76, "constrained_metric": -7.76},
                    {
                        "branin": {"branin": 0.1, "constrained_metric": 0.0},
                        "constrained_metric": {
                            "branin": 0.0,
                            "constrained_metric": 0.1,
                        },
                    },
                ),
            ),
        ),
    )
    def test_branin(self, _) -> None:
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

    @patch("ax.modelbridge.dispatch_utils.Models", FakeModels)
    def test_branin_batch(self) -> None:
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
            total_trials=5,
            arms_per_trial=3,
        )
        bp, vals = loop.full_run().get_best_point()
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
