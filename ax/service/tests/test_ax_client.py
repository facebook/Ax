#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
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
from ax.modelbridge.factory import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.service.ax_client import AxClient
from ax.utils.common.testutils import TestCase


class TestServiceAPI(TestCase):
    """Tests service-like API functionality."""

    def test_default_generation_strategy(self) -> None:
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
        self.assertEqual(
            [s.model for s in ax.generation_strategy._steps],
            [Models.SOBOL, Models.GPEI],
        )
        for _ in range(6):
            parameterization, trial_index = ax.get_next_trial()
            x1, x2 = parameterization.get("x1"), parameterization.get("x2")
            ax.complete_trial(trial_index, raw_data={"branin": (branin(x1, x2), 0.0)})

    def test_create_experiment(self) -> None:
        """Test basic experiment creation."""
        ax = AxClient(
            GenerationStrategy(steps=[GenerationStep(model=Models.SOBOL, num_arms=30)])
        )
        with self.assertRaisesRegex(ValueError, "Experiment not set on Ax client"):
            ax.experiment
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
        assert ax._experiment is not None
        self.assertEqual(ax._experiment, ax.experiment)
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
        ax = AxClient(
            GenerationStrategy(steps=[GenerationStep(model=Models.SOBOL, num_arms=30)])
        )
        with self.assertRaises(ValueError):
            ax.create_experiment(
                name="test_experiment",
                parameters=[
                    {"name": "x3", "type": "fixed", "value": 2, "value_type": "int"}
                ],
                objective_name="test_objective",
                outcome_constraints=["test_objective >= 3"],
            )

    def test_raw_data_format(self):
        ax = AxClient()
        ax.create_experiment(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        for _ in range(6):
            parameterization, trial_index = ax.get_next_trial()
            x1, x2 = parameterization.get("x1"), parameterization.get("x2")
            ax.complete_trial(trial_index, raw_data=(branin(x1, x2), 0.0))
        with self.assertRaisesRegex(ValueError, "Raw data has an invalid type"):
            ax.complete_trial(
                trial_index, raw_data=[(branin(x1, x2), 0.0), (branin(x1, x2), 0.0)]
            )

    def test_keep_generating_without_data(self):
        # Check that normally numebr of arms to generate is enforced.
        ax = AxClient()
        ax.create_experiment(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        for _ in range(5):
            parameterization, trial_index = ax.get_next_trial()
        with self.assertRaisesRegex(ValueError, "All trials for current model"):
            ax.get_next_trial()
        # Check thatwith enforce_sequential_optimization off, we can keep
        # generating.
        ax = AxClient(enforce_sequential_optimization=False)
        ax.create_experiment(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        for _ in range(10):
            parameterization, trial_index = ax.get_next_trial()

    def test_trial_completion(self):
        ax = AxClient()
        ax.create_experiment(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        params, idx = ax.get_next_trial()
        ax.complete_trial(trial_index=idx, raw_data={"objective": (0, 0.0)})
        self.assertEqual(ax.get_best_parameters()[0], params)
        params2, idx2 = ax.get_next_trial()
        ax.complete_trial(trial_index=idx2, raw_data=(-1, 0.0))
        self.assertEqual(ax.get_best_parameters()[0], params2)
        params3, idx3 = ax.get_next_trial()
        ax.complete_trial(trial_index=idx3, raw_data=-2, metadata={"dummy": "test"})
        self.assertEqual(ax.get_best_parameters()[0], params3)
        self.assertEqual(ax.experiment.trials.get(2).run_metadata.get("dummy"), "test")
        self.assertEqual(
            ax.get_best_parameters()[1],
            ({"objective": -2.0}, {"objective": {"objective": 0.0}}),
        )

    def test_fail_on_batch(self):
        ax = AxClient()
        ax.create_experiment(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        batch_trial = ax.experiment.new_batch_trial(
            generator_run=GeneratorRun(
                arms=[
                    Arm(parameters={"x1": 0, "x2": 1}),
                    Arm(parameters={"x1": 0, "x2": 1}),
                ]
            )
        )
        with self.assertRaises(NotImplementedError):
            ax.complete_trial(batch_trial.index, 0)

    def test_log_failure(self):
        ax = AxClient()
        ax.create_experiment(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        _, idx = ax.get_next_trial()
        ax.log_trial_failure(idx, metadata={"dummy": "test"})
        self.assertTrue(ax.experiment.trials.get(idx).status.is_failed)
        self.assertEqual(
            ax.experiment.trials.get(idx).run_metadata.get("dummy"), "test"
        )

    def test_attach_trial(self):
        ax = AxClient()
        ax.create_experiment(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        params, idx = ax.attach_trial(parameters={"x1": 0, "x2": 1})
        ax.complete_trial(trial_index=idx, raw_data=5)
        self.assertEqual(ax.get_best_parameters()[0], params)
