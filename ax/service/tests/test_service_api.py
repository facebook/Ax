#!/usr/bin/env python3

from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.service.service_api import AELoopHandler
from ax.utils.common.testutils import TestCase


class TestServiceAPI(TestCase):
    """Tests service-like API functionality."""

    def test_create_expreriment(self):
        ax = AELoopHandler(GenerationStrategy([get_sobol], [30]))
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
        )
        self.assertEqual(
            ax.experiment.search_space.parameters["x1"],
            RangeParameter(
                name="x1",
                parameter_type=ParameterType.FLOAT,
                lower=0.001,
                upper=0.1,
                log_scale=True,
            ),
        )
        self.assertEqual(
            ax.experiment.search_space.parameters["x2"],
            ChoiceParameter(
                name="x2",
                parameter_type=ParameterType.INT,
                values=[1, 2, 3],
                is_ordered=True,
            ),
        )
        self.assertEqual(
            ax.experiment.search_space.parameters["x3"],
            FixedParameter(name="x3", parameter_type=ParameterType.INT, value=2),
        )
        self.assertEqual(
            ax.experiment.search_space.parameters["x4"],
            RangeParameter(
                name="x4", parameter_type=ParameterType.INT, lower=1.0, upper=3.0
            ),
        )
        self.assertEqual(
            ax.experiment.search_space.parameters["x5"],
            ChoiceParameter(
                name="x5",
                parameter_type=ParameterType.STRING,
                values=["one", "two", "three"],
            ),
        )
