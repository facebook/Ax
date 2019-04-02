#!/usr/bin/env python3

from ax.core.metric import Metric
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.types import ComparisonOp
from ax.metrics.branin import branin
from ax.modelbridge.factory import get_GPEI, get_sobol
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.service.ax_client import AxClient
from ax.utils.common.testutils import TestCase


class TestServiceAPI(TestCase):
    """Tests service-like API functionality."""

    def test_default_generation_strategy(self):
        """Test that Sobol+GPEI is used if no GenerationStrategy is provided."""
        ax = AxClient()
        ax.create_experiment(
            name="test_branin",
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
            objective_name="branin",
            minimize=True,
        )
        self.assertEqual(ax.generation_strategy._model_factories, [get_sobol, get_GPEI])
        for _ in range(6):
            parameterization, trial_index = ax.get_next_trial()
            x1, x2 = parameterization.get("x1"), parameterization.get("x2")
            ax.complete_trial(trial_index, raw_data={"branin": (branin(x1, x2), 0.0)})

    def test_create_experiment(self):
        """Test basic experiment creation."""
        ax = AxClient(GenerationStrategy([get_sobol], [30]))
        ax.create_experiment(
            name="test_experiment",
            parameters=[
                {
                    "name": "x1",
                    "type": "range",
                    "bounds": [0.001, 0.1],
                    "value_type": "float",
                    "log_scale": True,
                },
                {
                    "name": "x2",
                    "type": "choice",
                    "values": [1, 2, 3],
                    "value_type": "int",
                    "is_ordered": True,
                },
                {"name": "x3", "type": "fixed", "value": 2, "value_type": "int"},
                {
                    "name": "x4",
                    "type": "range",
                    "bounds": [1.0, 3.0],
                    "value_type": "int",
                },
                {
                    "name": "x5",
                    "type": "choice",
                    "values": ["one", "two", "three"],
                    "value_type": "str",
                },
            ],
            objective_name="test_objective",
            minimize=True,
            outcome_constraints=["some_metric >= 3", "some_metric <= 4.0"],
            parameter_constraints=["x3 >= x4", "x3 + x4 >= 2"],
        )
        self.assertEqual(
            ax._experiment.search_space.parameters["x1"],
            RangeParameter(
                name="x1",
                parameter_type=ParameterType.FLOAT,
                lower=0.001,
                upper=0.1,
                log_scale=True,
            ),
        )
        self.assertEqual(
            ax._experiment.search_space.parameters["x2"],
            ChoiceParameter(
                name="x2",
                parameter_type=ParameterType.INT,
                values=[1, 2, 3],
                is_ordered=True,
            ),
        )
        self.assertEqual(
            ax._experiment.search_space.parameters["x3"],
            FixedParameter(name="x3", parameter_type=ParameterType.INT, value=2),
        )
        self.assertEqual(
            ax._experiment.search_space.parameters["x4"],
            RangeParameter(
                name="x4", parameter_type=ParameterType.INT, lower=1.0, upper=3.0
            ),
        )
        self.assertEqual(
            ax._experiment.search_space.parameters["x5"],
            ChoiceParameter(
                name="x5",
                parameter_type=ParameterType.STRING,
                values=["one", "two", "three"],
            ),
        )
        self.assertEqual(
            ax._experiment.optimization_config.outcome_constraints[0],
            OutcomeConstraint(
                metric=Metric(name="some_metric"),
                op=ComparisonOp.GEQ,
                bound=3.0,
                relative=False,
            ),
        )
        self.assertEqual(
            ax._experiment.optimization_config.outcome_constraints[1],
            OutcomeConstraint(
                metric=Metric(name="some_metric"),
                op=ComparisonOp.LEQ,
                bound=4.0,
                relative=False,
            ),
        )
        self.assertTrue(ax._experiment.optimization_config.objective.minimize)

    def test_constraint_same_as_objective(self):
        """Check that we do not allow constraints on the objective metric."""
        ax = AxClient(GenerationStrategy([get_sobol], [30]))
        with self.assertRaises(ValueError):
            ax.create_experiment(
                name="test_experiment",
                parameters=[
                    {"name": "x3", "type": "fixed", "value": 2, "value_type": "int"}
                ],
                objective_name="test_objective",
                outcome_constraints=["test_objective >= 3"],
            )
