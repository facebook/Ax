#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from ax.metrics.branin import branin
from ax.service.managed_loop import OptimizationLoop, OptimizationPlan, ScheduleConfig
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

    def testBranin(self) -> None:
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
            optimization_plan=OptimizationPlan(total_iterations=5),
            schedule_config=ScheduleConfig(wait_time=0, run_async=False),
        )
        bp = loop.full_run().get_best_point()
        self.assertIn("x1", bp)
        self.assertIn("x2", bp)

    def testBraninWithoutObjectiveName(self) -> None:
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
            optimization_plan=OptimizationPlan(total_iterations=5),
            schedule_config=ScheduleConfig(wait_time=0, run_async=False),
        )
        bp = loop.full_run().get_best_point()
        self.assertIn("x1", bp)
        self.assertIn("x2", bp)
