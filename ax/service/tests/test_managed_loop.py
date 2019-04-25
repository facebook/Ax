#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from enum import Enum
from unittest.mock import patch

from ax.metrics.branin import branin
from ax.modelbridge.factory import get_sobol
from ax.service.managed_loop import OptimizationLoop
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


class TestManagedLoop(TestCase):
    """Check functionality of optimization loop."""

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
            total_trials=5,
        )
        bp = loop.full_run().get_best_point()
        self.assertIn("x1", bp)
        self.assertIn("x2", bp)

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
            total_trials=5,
        )
        bp = loop.full_run().get_best_point()
        self.assertIn("x1", bp)
        self.assertIn("x2", bp)

    def test_branin_batch(self) -> None:
        """Basic async synthetic function managed loop case."""
        # Patch the Models enum to replace GPEI with Sobol.
        def get_experiment_data_sobol(experiment, data):
            return get_sobol(experiment.search_space)

        class FakeModels(Enum):
            SOBOL = get_sobol
            GPEI = get_experiment_data_sobol

        patch("ax.service.utils.dispatch.Models", FakeModels).start()
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
        bp = loop.full_run().get_best_point()
        self.assertIn("x1", bp)
        self.assertIn("x2", bp)
        # Check that all total_trials * arms_per_trial * 2 metrics evaluations
        # are present in the dataframe.
        self.assertEqual(len(loop.experiment.fetch_data().df.index), 30)
