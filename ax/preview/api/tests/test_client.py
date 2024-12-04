# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.core.parameter import (
    ChoiceParameter,
    ParameterType as CoreParameterType,
    RangeParameter,
)
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.preview.api.client import Client
from ax.preview.api.configs import (
    ChoiceParameterConfig,
    ExperimentConfig,
    ParameterType,
    RangeParameterConfig,
)
from ax.utils.common.testutils import TestCase
from pyre_extensions import none_throws


class TestClient(TestCase):
    def test_configure_experiment(self) -> None:
        client = Client()

        float_parameter = RangeParameterConfig(
            name="float_param",
            parameter_type=ParameterType.FLOAT,
            bounds=(0, 1),
        )
        int_parameter = RangeParameterConfig(
            name="int_param",
            parameter_type=ParameterType.INT,
            bounds=(0, 1),
        )
        choice_parameter = ChoiceParameterConfig(
            name="choice_param",
            parameter_type=ParameterType.STRING,
            values=["a", "b", "c"],
        )

        experiment_config = ExperimentConfig(
            name="test_experiment",
            parameters=[float_parameter, int_parameter, choice_parameter],
            parameter_constraints=["int_param <= float_param"],
            description="test description",
            owner="miles",
        )

        client.configure_experiment(experiment_config=experiment_config)
        self.assertEqual(
            client._experiment,
            Experiment(
                search_space=SearchSpace(
                    parameters=[
                        RangeParameter(
                            name="float_param",
                            parameter_type=CoreParameterType.FLOAT,
                            lower=0,
                            upper=1,
                        ),
                        RangeParameter(
                            name="int_param",
                            parameter_type=CoreParameterType.INT,
                            lower=0,
                            upper=1,
                        ),
                        ChoiceParameter(
                            name="choice_param",
                            parameter_type=CoreParameterType.STRING,
                            values=["a", "b", "c"],
                            is_ordered=False,
                            sort_values=False,
                        ),
                    ],
                    parameter_constraints=[
                        ParameterConstraint(
                            constraint_dict={"int_param": 1, "float_param": -1}, bound=0
                        )
                    ],
                ),
                name="test_experiment",
                description="test description",
                properties={"owners": ["miles"]},
            ),
        )

        with self.assertRaisesRegex(UnsupportedError, "Experiment already configured"):
            client.configure_experiment(experiment_config=experiment_config)

    def test_configure_optimization(self) -> None:
        client = Client()

        float_parameter = RangeParameterConfig(
            name="float_param",
            parameter_type=ParameterType.FLOAT,
            bounds=(0, 1),
        )
        int_parameter = RangeParameterConfig(
            name="int_param",
            parameter_type=ParameterType.INT,
            bounds=(0, 1),
        )
        choice_parameter = ChoiceParameterConfig(
            name="choice_param",
            parameter_type=ParameterType.STRING,
            values=["a", "b", "c"],
        )

        experiment_config = ExperimentConfig(
            name="test_experiment",
            parameters=[float_parameter, int_parameter, choice_parameter],
            parameter_constraints=["int_param <= float_param"],
            description="test description",
            owner="miles",
        )

        client.configure_experiment(experiment_config=experiment_config)

        client.configure_optimization(
            objective="-ne",
            outcome_constraints=["qps >= 0"],
        )

        self.assertEqual(
            none_throws(client._experiment).optimization_config,
            OptimizationConfig(
                objective=Objective(metric=Metric(name="ne"), minimize=True),
                outcome_constraints=[
                    OutcomeConstraint(
                        metric=Metric(name="qps"),
                        op=ComparisonOp.GEQ,
                        bound=0.0,
                        relative=False,
                    )
                ],
            ),
        )

        empty_client = Client()

        with self.assertRaisesRegex(AssertionError, "Experiment not set"):
            empty_client.configure_optimization(
                objective="ne",
                outcome_constraints=["qps >= 0"],
            )
