#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
import time
from math import ceil
from typing import List, Tuple
from unittest.mock import patch

import numpy as np
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
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
from ax.exceptions.core import DataRequiredError, UnsupportedPlotError
from ax.exceptions.core import UnsupportedError
from ax.metrics.branin import branin
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import MODEL_KEY_TO_MODEL_SETUP, Models
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.storage.sqa_store.db import init_test_engine_and_session_factory
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.storage.sqa_store.structs import DBSettings
from ax.utils.common.testutils import TestCase
from ax.utils.common.timeutils import current_timestamp_in_millis
from ax.utils.common.typeutils import checked_cast, not_none
from ax.utils.testing.modeling_stubs import get_observation1, get_observation1trans


def run_trials_using_recommended_parallelism(
    ax_client: AxClient,
    recommended_parallelism: List[Tuple[int, int]],
    total_trials: int,
) -> int:
    remaining_trials = total_trials
    for num_trials, parallelism_setting in recommended_parallelism:
        if num_trials == -1:
            num_trials = remaining_trials
        for _ in range(ceil(num_trials / parallelism_setting)):
            in_flight_trials = []
            if parallelism_setting > remaining_trials:
                parallelism_setting = remaining_trials
            for _ in range(parallelism_setting):
                params, idx = ax_client.get_next_trial()
                in_flight_trials.append((params, idx))
                remaining_trials -= 1
            for _ in range(parallelism_setting):
                params, idx = in_flight_trials.pop()
                ax_client.complete_trial(idx, branin(params["x"], params["y"]))
    # If all went well and no errors were raised, remaining_trials should be 0.
    return remaining_trials


class TestAxClient(TestCase):
    """Tests service-like API functionality."""

    def setUp(self):
        # To avoid tests timing out due to GP fit / gen times.
        patch.dict(
            f"{Models.__module__}.MODEL_KEY_TO_MODEL_SETUP",
            {"GPEI": MODEL_KEY_TO_MODEL_SETUP["Sobol"]},
        ).start()

    def test_interruption(self) -> None:
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test",
            parameters=[  # pyre-fixme[6]: expected union that should include
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            objective_name="branin",
            minimize=True,
        )
        for i in range(6):
            parameterization, trial_index = ax_client.get_next_trial()
            self.assertFalse(  # There should be non-complete trials.
                all(t.status.is_terminal for t in ax_client.experiment.trials.values())
            )
            x, y = parameterization.get("x"), parameterization.get("y")
            ax_client.complete_trial(
                trial_index,
                raw_data=checked_cast(
                    float, branin(checked_cast(float, x), checked_cast(float, y))
                ),
            )
            old_client = ax_client
            serialized = ax_client.to_json_snapshot()
            ax_client = AxClient.from_json_snapshot(serialized)
            self.assertEqual(len(ax_client.experiment.trials.keys()), i + 1)
            self.assertIsNot(ax_client, old_client)
            self.assertTrue(  # There should be no non-complete trials.
                all(t.status.is_terminal for t in ax_client.experiment.trials.values())
            )

    @patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=([get_observation1()]),
    )
    @patch(
        "ax.modelbridge.random.RandomModelBridge.get_training_data",
        autospec=True,
        return_value=([get_observation1()]),
    )
    @patch(
        "ax.modelbridge.random.RandomModelBridge._predict",
        autospec=True,
        return_value=[get_observation1trans().data],
    )
    @patch(
        "ax.modelbridge.random.RandomModelBridge.feature_importances",
        autospec=True,
        return_value={"x": 0.9, "y": 1.1},
    )
    def test_default_generation_strategy_continuous(self, _a, _b, _c, _d) -> None:
        """Test that Sobol+GPEI is used if no GenerationStrategy is provided."""
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[  # pyre-fixme[6]: expected union that should include
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            objective_name="a",
            minimize=True,
        )
        self.assertEqual(
            [s.model for s in not_none(ax_client.generation_strategy)._steps],
            [Models.SOBOL, Models.GPEI],
        )
        with self.assertRaisesRegex(ValueError, ".* no trials"):
            ax_client.get_optimization_trace(objective_optimum=branin.fmin)
        for i in range(6):
            parameterization, trial_index = ax_client.get_next_trial()
            x, y = parameterization.get("x"), parameterization.get("y")
            ax_client.complete_trial(
                trial_index,
                raw_data={
                    "a": (
                        checked_cast(
                            float,
                            branin(checked_cast(float, x), checked_cast(float, y)),
                        ),
                        0.0,
                    )
                },
                sample_size=i,
            )
        self.assertEqual(ax_client.generation_strategy.model._model_key, "GPEI")
        ax_client.get_optimization_trace(objective_optimum=branin.fmin)
        ax_client.get_contour_plot()
        ax_client.get_feature_importances()
        trials_df = ax_client.get_trials_data_frame()
        self.assertIn("x", trials_df)
        self.assertIn("y", trials_df)
        self.assertIn("a", trials_df)
        self.assertEqual(len(trials_df), 6)

    @patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=([get_observation1()]),
    )
    @patch(
        "ax.modelbridge.random.RandomModelBridge.get_training_data",
        autospec=True,
        return_value=([get_observation1()]),
    )
    @patch(
        "ax.modelbridge.random.RandomModelBridge._predict",
        autospec=True,
        return_value=[get_observation1trans().data],
    )
    @patch(
        "ax.modelbridge.random.RandomModelBridge.feature_importances",
        autospec=True,
        return_value={"x": 0.9, "y": 1.1},
    )
    def test_default_generation_strategy_continuous_for_moo(
        self, _a, _b, _c, _d
    ) -> None:
        """Test that Sobol+MOO is used if no GenerationStrategy is provided."""
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[  # pyre-fixme[6]: expected union that should include
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            objectives={
                "a": ObjectiveProperties(minimize=True, threshold=1.0),
                "b": ObjectiveProperties(minimize=True, threshold=1.0),
            },
        )
        self.assertEqual(
            [s.model for s in not_none(ax_client.generation_strategy)._steps],
            [Models.SOBOL, Models.MOO],
        )
        with self.assertRaisesRegex(ValueError, ".* no trials"):
            ax_client.get_optimization_trace(objective_optimum=branin.fmin)
        for i in range(6):
            parameterization, trial_index = ax_client.get_next_trial()
            x, y = parameterization.get("x"), parameterization.get("y")
            ax_client.complete_trial(
                trial_index,
                raw_data={
                    "a": (
                        checked_cast(
                            float,
                            branin(checked_cast(float, x), checked_cast(float, y)),
                        ),
                        0.0,
                    ),
                    "b": (
                        checked_cast(
                            float,
                            branin(checked_cast(float, x), checked_cast(float, y)),
                        ),
                        0.0,
                    ),
                },
                sample_size=i,
            )
        self.assertEqual(ax_client.generation_strategy.model._model_key, "MOO")
        ax_client.get_contour_plot(metric_name="a")
        ax_client.get_contour_plot(metric_name="b")
        ax_client.get_feature_importances()
        trials_df = ax_client.get_trials_data_frame()
        self.assertIn("x", trials_df)
        self.assertIn("y", trials_df)
        self.assertIn("a", trials_df)
        self.assertIn("b", trials_df)
        self.assertEqual(len(trials_df), 6)

        with self.subTest("it raises UnsupportedError for get_optimization_trace"):
            with self.assertRaises(UnsupportedError):
                ax_client.get_optimization_trace(objective_optimum=branin.fmin)

        with self.subTest(
            "it raises UnsupportedError for get_contour_plot without metric"
        ):
            with self.assertRaises(UnsupportedError):
                ax_client.get_contour_plot()

    def test_create_experiment(self) -> None:
        """Test basic experiment creation."""
        ax_client = AxClient(
            GenerationStrategy(
                steps=[GenerationStep(model=Models.SOBOL, num_trials=30)]
            )
        )
        with self.assertRaisesRegex(ValueError, "Experiment not set on Ax client"):
            ax_client.experiment
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [0.001, 0.1],
                    "value_type": "float",
                    "log_scale": True,
                    "digits": 6,
                },
                {
                    "name": "y",
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
                {
                    "name": "x6",
                    "type": "range",
                    "bounds": [1.0, 3.0],
                    "value_type": "int",
                },
            ],
            objective_name="test_objective",
            minimize=True,
            outcome_constraints=["some_metric >= 3", "some_metric <= 4.0"],
            parameter_constraints=["x4 <= x6"],
            tracking_metric_names=["test_tracking_metric"],
            is_test=True,
        )
        assert ax_client._experiment is not None
        self.assertEqual(ax_client._experiment, ax_client.experiment)
        self.assertEqual(
            ax_client._experiment.search_space.parameters["x"],
            RangeParameter(
                name="x",
                parameter_type=ParameterType.FLOAT,
                lower=0.001,
                upper=0.1,
                log_scale=True,
                digits=6,
            ),
        )
        self.assertEqual(
            ax_client._experiment.search_space.parameters["y"],
            ChoiceParameter(
                name="y",
                parameter_type=ParameterType.INT,
                values=[1, 2, 3],
                is_ordered=True,
            ),
        )
        self.assertEqual(
            ax_client._experiment.search_space.parameters["x3"],
            FixedParameter(name="x3", parameter_type=ParameterType.INT, value=2),
        )
        self.assertEqual(
            ax_client._experiment.search_space.parameters["x4"],
            RangeParameter(
                name="x4", parameter_type=ParameterType.INT, lower=1.0, upper=3.0
            ),
        )
        self.assertEqual(
            ax_client._experiment.search_space.parameters["x5"],
            ChoiceParameter(
                name="x5",
                parameter_type=ParameterType.STRING,
                values=["one", "two", "three"],
            ),
        )
        self.assertEqual(
            ax_client._experiment.optimization_config.outcome_constraints[0],
            OutcomeConstraint(
                metric=Metric(name="some_metric"),
                op=ComparisonOp.GEQ,
                bound=3.0,
                relative=False,
            ),
        )
        self.assertEqual(
            ax_client._experiment.optimization_config.outcome_constraints[1],
            OutcomeConstraint(
                metric=Metric(name="some_metric"),
                op=ComparisonOp.LEQ,
                bound=4.0,
                relative=False,
            ),
        )
        self.assertTrue(ax_client._experiment.optimization_config.objective.minimize)
        self.assertDictEqual(
            ax_client._experiment._tracking_metrics,
            {"test_tracking_metric": Metric(name="test_tracking_metric")},
        )
        self.assertTrue(ax_client._experiment.immutable_search_space_and_opt_config)
        self.assertTrue(ax_client.experiment.is_test)

        with self.subTest("objective_name"):
            self.assertEqual(ax_client.objective_name, "test_objective")

        with self.subTest("objective_names"):
            self.assertEqual(ax_client.objective_names, ["test_objective"])

    def test_create_single_objective_experiment_with_objectives_dict(self) -> None:
        """Test basic experiment creation."""
        ax_client = AxClient(
            GenerationStrategy(
                steps=[GenerationStep(model=Models.SOBOL, num_trials=30)]
            )
        )
        with self.assertRaisesRegex(ValueError, "Experiment not set on Ax client"):
            ax_client.experiment
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [0.001, 0.1],
                    "value_type": "float",
                    "log_scale": True,
                    "digits": 6,
                },
                {
                    "name": "y",
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
                {
                    "name": "x6",
                    "type": "range",
                    "bounds": [1.0, 3.0],
                    "value_type": "int",
                },
            ],
            objectives={
                "test_objective": ObjectiveProperties(minimize=True, threshold=2.0),
            },
            outcome_constraints=["some_metric >= 3", "some_metric <= 4.0"],
            parameter_constraints=["x4 <= x6"],
            tracking_metric_names=["test_tracking_metric"],
            is_test=True,
        )
        assert ax_client._experiment is not None
        self.assertEqual(ax_client.objective_name, "test_objective")
        self.assertTrue(ax_client.objective.minimize)

        with self.subTest("objective_name"):
            self.assertEqual(ax_client.objective_name, "test_objective")

        with self.subTest("objective_names"):
            self.assertEqual(ax_client.objective_names, ["test_objective"])

    def test_it_does_not_accept_both_legacy_and_new_objective_params(self) -> None:
        """Test basic experiment creation."""
        ax_client = AxClient(
            GenerationStrategy(
                steps=[GenerationStep(model=Models.SOBOL, num_trials=30)]
            )
        )
        with self.assertRaisesRegex(ValueError, "Experiment not set on Ax client"):
            ax_client.experiment
        params = {
            "name": "test_experiment",
            "parameters": [
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [0.001, 0.1],
                    "value_type": "float",
                    "log_scale": True,
                    "digits": 6,
                },
                {
                    "name": "y",
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
                {
                    "name": "x6",
                    "type": "range",
                    "bounds": [1.0, 3.0],
                    "value_type": "int",
                },
            ],
            "objectives": {
                "test_objective": ObjectiveProperties(minimize=True, threshold=2.0),
            },
            "outcome_constraints": ["some_metric >= 3", "some_metric <= 4.0"],
            "parameter_constraints": ["x4 <= x6"],
            "tracking_metric_names": ["test_tracking_metric"],
            "is_test": True,
        }
        with self.subTest("objective_name"):
            with self.assertRaises(UnsupportedError):
                ax_client.create_experiment(objective_name="something", **params)
        with self.subTest("minimize"):
            with self.assertRaises(UnsupportedError):
                ax_client.create_experiment(minimize=False, **params)
        with self.subTest("both"):
            with self.assertRaises(UnsupportedError):
                ax_client.create_experiment(
                    objective_name="another thing", minimize=False, **params
                )

    def test_create_moo_experiment(self) -> None:
        """Test basic experiment creation."""
        ax_client = AxClient(
            GenerationStrategy(
                steps=[GenerationStep(model=Models.SOBOL, num_trials=30)]
            )
        )
        with self.assertRaisesRegex(ValueError, "Experiment not set on Ax client"):
            ax_client.experiment
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {
                    "name": "x",
                    "type": "range",
                    "bounds": [0.001, 0.1],
                    "value_type": "float",
                    "log_scale": True,
                    "digits": 6,
                },
                {
                    "name": "y",
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
                {
                    "name": "x6",
                    "type": "range",
                    "bounds": [1.0, 3.0],
                    "value_type": "int",
                },
            ],
            objectives={
                "test_objective_1": ObjectiveProperties(minimize=True, threshold=2.0),
                "test_objective_2": ObjectiveProperties(minimize=False, threshold=7.0),
            },
            outcome_constraints=["some_metric >= 3", "some_metric <= 4.0"],
            parameter_constraints=["x4 <= x6"],
            tracking_metric_names=["test_tracking_metric"],
            is_test=True,
        )
        assert ax_client._experiment is not None
        self.assertEqual(ax_client._experiment, ax_client.experiment)
        self.assertEqual(
            ax_client._experiment.search_space.parameters["x"],
            RangeParameter(
                name="x",
                parameter_type=ParameterType.FLOAT,
                lower=0.001,
                upper=0.1,
                log_scale=True,
                digits=6,
            ),
        )
        self.assertEqual(
            ax_client._experiment.search_space.parameters["y"],
            ChoiceParameter(
                name="y",
                parameter_type=ParameterType.INT,
                values=[1, 2, 3],
                is_ordered=True,
            ),
        )
        self.assertEqual(
            ax_client._experiment.search_space.parameters["x3"],
            FixedParameter(name="x3", parameter_type=ParameterType.INT, value=2),
        )
        self.assertEqual(
            ax_client._experiment.search_space.parameters["x4"],
            RangeParameter(
                name="x4", parameter_type=ParameterType.INT, lower=1.0, upper=3.0
            ),
        )
        self.assertEqual(
            ax_client._experiment.search_space.parameters["x5"],
            ChoiceParameter(
                name="x5",
                parameter_type=ParameterType.STRING,
                values=["one", "two", "three"],
            ),
        )
        optimization_config = ax_client._experiment.optimization_config
        self.assertEqual(
            [m.name for m in optimization_config.objective.metrics],
            ["test_objective_1", "test_objective_2"],
        )
        self.assertEqual(
            [o.minimize for o in optimization_config.objective.objectives],
            [True, False],
        )
        self.assertEqual(
            [m.lower_is_better for m in optimization_config.objective.metrics],
            [None, None],
        )
        self.assertEqual(
            [t.metric.name for t in optimization_config.objective_thresholds],
            ["test_objective_1", "test_objective_2"],
        )
        self.assertEqual(
            [t.bound for t in optimization_config.objective_thresholds],
            [2.0, 7.0],
        )
        self.assertEqual(
            [t.op for t in optimization_config.objective_thresholds],
            [ComparisonOp.LEQ, ComparisonOp.GEQ],
        )
        self.assertEqual(
            [t.relative for t in optimization_config.objective_thresholds],
            [False, False],
        )
        self.assertEqual(
            optimization_config.outcome_constraints[0],
            OutcomeConstraint(
                metric=Metric(name="some_metric"),
                op=ComparisonOp.GEQ,
                bound=3.0,
                relative=False,
            ),
        )
        self.assertEqual(
            optimization_config.outcome_constraints[1],
            OutcomeConstraint(
                metric=Metric(name="some_metric"),
                op=ComparisonOp.LEQ,
                bound=4.0,
                relative=False,
            ),
        )
        self.assertDictEqual(
            ax_client._experiment._tracking_metrics,
            {"test_tracking_metric": Metric(name="test_tracking_metric")},
        )
        self.assertTrue(ax_client._experiment.immutable_search_space_and_opt_config)
        self.assertTrue(ax_client.experiment.is_test)

        with self.subTest("objective_name name raises UnsupportedError"):
            with self.assertRaises(UnsupportedError):
                ax_client.objective_name

        with self.subTest("objective_names"):
            self.assertEqual(
                ax_client.objective_names, ["test_objective_1", "test_objective_2"]
            )

    def test_constraint_same_as_objective(self):
        """Check that we do not allow constraints on the objective metric."""
        ax_client = AxClient(
            GenerationStrategy(
                steps=[GenerationStep(model=Models.SOBOL, num_trials=30)]
            )
        )
        with self.assertRaises(ValueError):
            ax_client.create_experiment(
                name="test_experiment",
                parameters=[
                    {"name": "x3", "type": "fixed", "value": 2, "value_type": "int"}
                ],
                objective_name="test_objective",
                outcome_constraints=["test_objective >= 3"],
            )

    def test_raw_data_format(self):
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        for _ in range(6):
            parameterization, trial_index = ax_client.get_next_trial()
            x, y = parameterization.get("x"), parameterization.get("y")
            ax_client.complete_trial(trial_index, raw_data=(branin(x, y), 0.0))
        with self.assertRaisesRegex(
            ValueError, AxClient.TRIAL_RAW_DATA_FORMAT_ERROR_MESSAGE
        ):
            ax_client.update_trial_data(trial_index, raw_data="invalid_data")

    def test_raw_data_format_with_map_results(self):
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
            ],
            minimize=True,
            support_intermediate_data=True,
        )
        for _ in range(6):
            parameterization, trial_index = ax_client.get_next_trial()
            x, y = parameterization.get("x"), parameterization.get("y")
            ax_client.complete_trial(
                trial_index,
                raw_data=[
                    ({"y": y / 2.0}, {"objective": (branin(x, y / 2.0), 0.0)}),
                    ({"y": y}, {"objective": (branin(x, y), 0.0)}),
                ],
            )

    def test_keep_generating_without_data(self):
        # Check that normally numebr of arms to generate is enforced.
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        for _ in range(5):
            parameterization, trial_index = ax_client.get_next_trial()
        with self.assertRaisesRegex(DataRequiredError, "All trials for current model"):
            ax_client.get_next_trial()
        # Check thatwith enforce_sequential_optimization off, we can keep
        # generating.
        ax_client = AxClient(enforce_sequential_optimization=False)
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        self.assertFalse(
            ax_client.generation_strategy._steps[0].enforce_num_trials, False
        )
        self.assertFalse(ax_client.generation_strategy._steps[1].max_parallelism, None)
        for _ in range(10):
            parameterization, trial_index = ax_client.get_next_trial()

    def test_update_running_trial_with_intermediate_data(self):
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
            ],
            minimize=True,
            support_intermediate_data=True,
        )
        parameterization, trial_index = ax_client.get_next_trial()
        # Launch Trial and update it 3 times with additional data.
        for t in range(3):
            x, y = parameterization.get("x"), parameterization.get("y")
            if t < 2:
                ax_client.update_running_trial_with_intermediate_data(
                    0,
                    raw_data=[({"t": t}, {"objective": (branin(x, y) + t, 0.0)})],
                )
            if t == 2:
                ax_client.complete_trial(
                    0,
                    raw_data=[({"t": t}, {"objective": (branin(x, y) + t, 0.0)})],
                )
            current_data = ax_client.experiment.fetch_data().df
            self.assertEqual(len(current_data), 0 if t < 2 else 3)

        no_intermediate_data_ax_client = AxClient()
        no_intermediate_data_ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 1.0]},
            ],
            minimize=True,
            support_intermediate_data=False,
        )
        parameterization, trial_index = no_intermediate_data_ax_client.get_next_trial()
        with self.assertRaises(ValueError):
            no_intermediate_data_ax_client.update_running_trial_with_intermediate_data(
                0,
                raw_data=[
                    ({"t": p_t}, {"objective": (branin(x, y) + t, 0.0)})
                    for p_t in range(t + 1)
                ],
            )

    def test_trial_completion(self):
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        params, idx = ax_client.get_next_trial()
        # Can't update before completing.
        with self.assertRaisesRegex(ValueError, ".* not yet"):
            ax_client.update_trial_data(
                trial_index=idx, raw_data={"objective": (0, 0.0)}
            )
        ax_client.complete_trial(trial_index=idx, raw_data={"objective": (0, 0.0)})
        # Cannot complete a trial twice, should use `update_trial_data`.
        with self.assertRaisesRegex(ValueError, ".* already been completed"):
            ax_client.complete_trial(trial_index=idx, raw_data={"objective": (0, 0.0)})
        # Cannot update trial data with observation for a metric it already has.
        with self.assertRaisesRegex(ValueError, ".* contained an observation"):
            ax_client.update_trial_data(
                trial_index=idx, raw_data={"objective": (0, 0.0)}
            )
        # Same as above, except objective name should be getting inferred.
        with self.assertRaisesRegex(ValueError, ".* contained an observation"):
            ax_client.update_trial_data(trial_index=idx, raw_data=1.0)
        ax_client.update_trial_data(trial_index=idx, raw_data={"m1": (1, 0.0)})
        metrics_in_data = ax_client.experiment.fetch_data().df["metric_name"].values
        self.assertNotIn("m1", metrics_in_data)
        self.assertIn("objective", metrics_in_data)
        self.assertEqual(ax_client.get_best_parameters()[0], params)
        params2, idy = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=idy, raw_data=(-1, 0.0))
        self.assertEqual(ax_client.get_best_parameters()[0], params2)
        params3, idx3 = ax_client.get_next_trial()
        ax_client.complete_trial(
            trial_index=idx3, raw_data=-2, metadata={"dummy": "test"}
        )
        self.assertEqual(ax_client.get_best_parameters()[0], params3)
        self.assertEqual(
            ax_client.experiment.trials.get(2).run_metadata.get("dummy"), "test"
        )
        best_trial_values = ax_client.get_best_parameters()[1]
        self.assertEqual(best_trial_values[0], {"objective": -2.0})
        self.assertTrue(math.isnan(best_trial_values[1]["objective"]["objective"]))

    def test_abandon_trial(self):
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )

        # An abandoned trial adds no data.
        params, idx = ax_client.get_next_trial()
        ax_client.abandon_trial(trial_index=idx)
        data = ax_client.experiment.fetch_data()
        self.assertEqual(len(data.df.index), 0)

        # Can't update a completed trial.
        params2, idx2 = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=idx2, raw_data={"objective": (0, 0.0)})
        with self.assertRaisesRegex(ValueError, ".* in a terminal state."):
            ax_client.abandon_trial(trial_index=idx2)

    def test_ttl_trial(self):
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )

        # A ttl trial that ends adds no data.
        params, idx = ax_client.get_next_trial(ttl_seconds=1)
        self.assertTrue(ax_client.experiment.trials.get(idx).status.is_running)
        time.sleep(1)  # Wait for TTL to elapse.
        self.assertTrue(ax_client.experiment.trials.get(idx).status.is_failed)
        # Also make sure we can no longer complete the trial as it is failed.
        with self.assertRaisesRegex(
            ValueError, ".* has been marked FAILED, so it no longer expects data."
        ):
            ax_client.complete_trial(trial_index=idx, raw_data={"objective": (0, 0.0)})

        params2, idy = ax_client.get_next_trial(ttl_seconds=1)
        ax_client.complete_trial(trial_index=idy, raw_data=(-1, 0.0))
        self.assertEqual(ax_client.get_best_parameters()[0], params2)

    def test_start_and_end_time_in_trial_completion(self):
        start_time = current_timestamp_in_millis()
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        params, idx = ax_client.get_next_trial()
        ax_client.complete_trial(
            trial_index=idx,
            raw_data=1.0,
            metadata={
                "start_time": start_time,
                "end_time": current_timestamp_in_millis(),
            },
        )
        dat = ax_client.experiment.fetch_data().df
        self.assertGreater(dat["end_time"][0], dat["start_time"][0])

    def test_fail_on_batch(self):
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        batch_trial = ax_client.experiment.new_batch_trial(
            generator_run=GeneratorRun(
                arms=[
                    Arm(parameters={"x": 0, "y": 1}),
                    Arm(parameters={"x": 0, "y": 1}),
                ]
            )
        )
        with self.assertRaises(NotImplementedError):
            ax_client.complete_trial(batch_trial.index, 0)

    def test_log_failure(self):
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        _, idx = ax_client.get_next_trial()
        ax_client.log_trial_failure(idx, metadata={"dummy": "test"})
        self.assertTrue(ax_client.experiment.trials.get(idx).status.is_failed)
        self.assertEqual(
            ax_client.experiment.trials.get(idx).run_metadata.get("dummy"), "test"
        )
        with self.assertRaisesRegex(ValueError, ".* no longer expects"):
            ax_client.complete_trial(idx, {})

    def test_attach_trial_and_get_trial_parameters(self):
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        params, idx = ax_client.attach_trial(parameters={"x": 0.0, "y": 1.0})
        ax_client.complete_trial(trial_index=idx, raw_data=5)
        self.assertEqual(ax_client.get_best_parameters()[0], params)
        self.assertEqual(
            ax_client.get_trial_parameters(trial_index=idx), {"x": 0, "y": 1}
        )
        with self.assertRaises(ValueError):
            ax_client.get_trial_parameters(
                trial_index=10
            )  # No trial #10 in experiment.
        with self.assertRaisesRegex(ValueError, ".* is of type"):
            ax_client.attach_trial({"x": 1, "y": 2})

    def test_attach_trial_ttl_seconds(self):
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        params, idx = ax_client.attach_trial(
            parameters={"x": 0.0, "y": 1.0}, ttl_seconds=1
        )
        self.assertTrue(ax_client.experiment.trials.get(idx).status.is_running)
        time.sleep(1)  # Wait for TTL to elapse.
        self.assertTrue(ax_client.experiment.trials.get(idx).status.is_failed)
        # Also make sure we can no longer complete the trial as it is failed.
        with self.assertRaisesRegex(
            ValueError, ".* has been marked FAILED, so it no longer expects data."
        ):
            ax_client.complete_trial(trial_index=idx, raw_data=5)

        params2, idx2 = ax_client.attach_trial(
            parameters={"x": 0.0, "y": 1.0}, ttl_seconds=1
        )
        ax_client.complete_trial(trial_index=idx2, raw_data=5)
        self.assertEqual(ax_client.get_best_parameters()[0], params2)
        self.assertEqual(
            ax_client.get_trial_parameters(trial_index=idx2), {"x": 0, "y": 1}
        )

    def test_attach_trial_numpy(self):
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        params, idx = ax_client.attach_trial(parameters={"x": 0.0, "y": 1.0})
        ax_client.complete_trial(trial_index=idx, raw_data=np.int32(5))
        self.assertEqual(ax_client.get_best_parameters()[0], params)

    def test_relative_oc_without_sq(self):
        """Must specify status quo to have relative outcome constraint."""
        ax_client = AxClient()
        with self.assertRaises(ValueError):
            ax_client.create_experiment(
                name="test_experiment",
                parameters=[
                    {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                    {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
                ],
                objective_name="test_objective",
                minimize=True,
                outcome_constraints=["some_metric <= 4.0%"],
            )

    def test_recommended_parallelism(self):
        ax_client = AxClient()
        with self.assertRaisesRegex(ValueError, "No generation strategy"):
            ax_client.get_max_parallelism()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        self.assertEqual(ax_client.get_max_parallelism(), [(5, 5), (-1, 3)])
        self.assertEqual(
            run_trials_using_recommended_parallelism(
                ax_client, ax_client.get_max_parallelism(), 20
            ),
            0,
        )
        # With incorrect parallelism setting, the 'need more data' error should
        # still be raised.
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        with self.assertRaisesRegex(DataRequiredError, "All trials for current model "):
            run_trials_using_recommended_parallelism(ax_client, [(6, 6), (-1, 3)], 20)

    @patch.dict(sys.modules, {"ax.storage.sqa_store.structs": None})
    @patch.dict(sys.modules, {"sqalchemy": None})
    @patch("ax.service.ax_client.DBSettings", None)
    def test_no_sqa(self):
        # Make sure we couldn't import sqa_store.structs (this could happen when
        # SQLAlchemy is not installed).
        with self.assertRaises(ModuleNotFoundError):
            import ax_client.storage.sqa_store.structs  # noqa F401
        # Make sure we can still import ax_client.
        __import__("ax.service.ax_client")
        AxClient()  # Make sure we still can instantiate client w/o db settings.
        # DBSettings should be defined in `ax_client` now, but incorrectly typed
        # `db_settings` argument should still make instantiation fail.
        with self.assertRaisesRegex(ValueError, "`db_settings` argument should "):
            AxClient(db_settings="badly_typed_db_settings")

    def test_plotting_validation(self):
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x3", "type": "fixed", "value": 2, "value_type": "int"}
            ]
        )
        with self.assertRaisesRegex(ValueError, ".* there are no trials"):
            ax_client.get_contour_plot()
        with self.assertRaisesRegex(ValueError, ".* there are no trials"):
            ax_client.get_feature_importances()
        ax_client.get_next_trial()
        with self.assertRaisesRegex(ValueError, ".* less than 2 parameters"):
            ax_client.get_contour_plot()
        ax_client = AxClient()
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ]
        )
        ax_client.get_next_trial()
        with self.assertRaisesRegex(ValueError, "If `param_x` is provided"):
            ax_client.get_contour_plot(param_x="y")
        with self.assertRaisesRegex(ValueError, "If `param_x` is provided"):
            ax_client.get_contour_plot(param_y="y")
        with self.assertRaisesRegex(ValueError, 'Parameter "x3"'):
            ax_client.get_contour_plot(param_x="x3", param_y="x3")
        with self.assertRaisesRegex(ValueError, 'Parameter "x4"'):
            ax_client.get_contour_plot(param_x="x", param_y="x4")
        with self.assertRaisesRegex(ValueError, 'Metric "nonexistent"'):
            ax_client.get_contour_plot(
                param_x="x", param_y="y", metric_name="nonexistent"
            )
        with self.assertRaisesRegex(UnsupportedPlotError, "Could not obtain contour"):
            ax_client.get_contour_plot(
                param_x="x", param_y="y", metric_name="objective"
            )
        with self.assertRaisesRegex(ValueError, "Could not obtain feature"):
            ax_client.get_feature_importances()

    def test_sqa_storage(self):
        init_test_engine_and_session_factory(force_init=True)
        config = SQAConfig()
        encoder = Encoder(config=config)
        decoder = Decoder(config=config)
        db_settings = DBSettings(encoder=encoder, decoder=decoder)
        ax_client = AxClient(db_settings=db_settings)
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        for _ in range(5):
            parameters, trial_index = ax_client.get_next_trial()
            ax_client.complete_trial(
                trial_index=trial_index, raw_data=branin(*parameters.values())
            )
        gs = ax_client.generation_strategy
        ax_client = AxClient(db_settings=db_settings)
        ax_client.load_experiment_from_database("test_experiment")
        # Trial #4 was completed after the last time the generation strategy
        # generated candidates, so pre-save generation strategy was not
        # "aware" of completion of trial #4. Post-restoration generation
        # strategy is aware of it, however, since it gets restored with most
        # up-to-date experiment data. Do adding trial #4 to the seen completed
        # trials of pre-storage GS to check their equality otherwise.
        gs._seen_trial_indices_by_status[TrialStatus.COMPLETED].add(4)
        self.assertEqual(gs, ax_client.generation_strategy)
        with self.assertRaises(ValueError):
            # Overwriting existing experiment.
            ax_client.create_experiment(
                name="test_experiment",
                parameters=[
                    {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                    {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
                ],
                minimize=True,
            )
        with self.assertRaises(ValueError):
            # Overwriting existing experiment with overwrite flag with present
            # DB settings. This should fail as we no longer allow overwriting
            # experiments stored in the DB.
            ax_client.create_experiment(
                name="test_experiment",
                parameters=[{"name": "x", "type": "range", "bounds": [-5.0, 10.0]}],
                overwrite_existing_experiment=True,
            )
        # Original experiment should still be in DB and not have been overwritten.
        self.assertEqual(len(ax_client.experiment.trials), 5)

    def test_overwrite(self):
        init_test_engine_and_session_factory(force_init=True)
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )

        # Log a trial
        parameters, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(
            trial_index=trial_index, raw_data=branin(*parameters.values())
        )

        with self.assertRaises(ValueError):
            # Overwriting existing experiment.
            ax_client.create_experiment(
                name="test_experiment",
                parameters=[
                    {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                    {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
                ],
                minimize=True,
            )
        # Overwriting existing experiment with overwrite flag.
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
            overwrite_existing_experiment=True,
        )
        # There should be no trials, as we just put in a fresh experiment.
        self.assertEqual(len(ax_client.experiment.trials), 0)

        # Log a trial
        parameters, trial_index = ax_client.get_next_trial()
        self.assertIn("x1", parameters.keys())
        self.assertIn("x2", parameters.keys())
        ax_client.complete_trial(
            trial_index=trial_index, raw_data=branin(*parameters.values())
        )

    def test_fixed_random_seed_reproducibility(self):
        ax_client = AxClient(random_seed=239)
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ]
        )
        for _ in range(5):
            params, idx = ax_client.get_next_trial()
            ax_client.complete_trial(idx, branin(params.get("x"), params.get("y")))
        trial_parameters_1 = [
            t.arm.parameters for t in ax_client.experiment.trials.values()
        ]
        ax_client = AxClient(random_seed=239)
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ]
        )
        for _ in range(5):
            params, idx = ax_client.get_next_trial()
            ax_client.complete_trial(idx, branin(params.get("x"), params.get("y")))
        trial_parameters_2 = [
            t.arm.parameters for t in ax_client.experiment.trials.values()
        ]
        self.assertEqual(trial_parameters_1, trial_parameters_2)

    def test_init_position_saved(self):
        ax_client = AxClient(random_seed=239)
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            name="sobol_init_position_test",
        )
        for _ in range(4):
            # For each generated trial, snapshot the client before generating it,
            # then recreate client, regenerate the trial and compare the trial
            # generated before and after snapshotting. If the state of Sobol is
            # recorded correctly, the newly generated trial will be the same as
            # the one generated before the snapshotting.
            serialized = ax_client.to_json_snapshot()
            params, idx = ax_client.get_next_trial()
            ax_client = AxClient.from_json_snapshot(serialized)
            with self.subTest(ax=ax_client, params=params, idx=idx):
                new_params, new_idx = ax_client.get_next_trial()
                self.assertEqual(params, new_params)
                self.assertEqual(idx, new_idx)
                self.assertEqual(
                    ax_client.experiment.trials[
                        idx
                    ]._generator_run._model_state_after_gen["init_position"],
                    idx + 1,
                )
            ax_client.complete_trial(idx, branin(params.get("x"), params.get("y")))

    def test_unnamed_experiment_snapshot(self):
        ax_client = AxClient(random_seed=239)
        ax_client.create_experiment(
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ]
        )
        serialized = ax_client.to_json_snapshot()
        ax_client = AxClient.from_json_snapshot(serialized)
        self.assertIsNone(ax_client.experiment._name)

    @patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=([get_observation1()]),
    )
    @patch(
        "ax.modelbridge.random.RandomModelBridge.get_training_data",
        autospec=True,
        return_value=([get_observation1()]),
    )
    @patch(
        "ax.modelbridge.random.RandomModelBridge._predict",
        autospec=True,
        return_value=[get_observation1trans().data],
    )
    def test_get_model_predictions(self, _predict, _tr_data, _obs_from_data):
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
            objective_name="a",
        )
        ax_client.get_next_trial()
        ax_client.experiment.trials[0].arm._name = "1_1"
        self.assertEqual(ax_client.get_model_predictions(), {0: {"a": (9.0, 1.0)}})

    def test_deprecated_save_load_method_errors(self):
        ax_client = AxClient()
        with self.assertRaises(NotImplementedError):
            ax_client.save()
        with self.assertRaises(NotImplementedError):
            ax_client.load()
        with self.assertRaises(NotImplementedError):
            ax_client.load_experiment("test_experiment")
        with self.assertRaises(NotImplementedError):
            ax_client.get_recommended_max_parallelism()

    def test_find_last_trial_with_parameterization(self):
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
            objective_name="a",
        )
        params, trial_idx = ax_client.get_next_trial()
        found_trial_idx = ax_client._find_last_trial_with_parameterization(
            parameterization=params
        )
        self.assertEqual(found_trial_idx, trial_idx)
        # Check that it's indeed the _last_ trial with params that is found.
        _, new_trial_idx = ax_client.attach_trial(parameters=params)
        found_trial_idx = ax_client._find_last_trial_with_parameterization(
            parameterization=params
        )
        self.assertEqual(found_trial_idx, new_trial_idx)
        with self.assertRaisesRegex(ValueError, "No .* matches"):
            found_trial_idx = ax_client._find_last_trial_with_parameterization(
                parameterization={k: v + 1.0 for k, v in params.items()}
            )

    def test_verify_parameterization(self):
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
            objective_name="a",
        )
        params, trial_idx = ax_client.get_next_trial()
        self.assertTrue(
            ax_client.verify_trial_parameterization(
                trial_index=trial_idx, parameterization=params
            )
        )
        # Make sure it still works if ordering in the parameterization is diff.
        self.assertTrue(
            ax_client.verify_trial_parameterization(
                trial_index=trial_idx,
                parameterization={k: params[k] for k in reversed(list(params.keys()))},
            )
        )
        self.assertFalse(
            ax_client.verify_trial_parameterization(
                trial_index=trial_idx,
                parameterization={k: v + 1.0 for k, v in params.items()},
            )
        )

    @patch(
        "ax.core.experiment.Experiment.new_trial",
        side_effect=RuntimeError("cholesky_cpu error - bad matrix"),
    )
    def test_annotate_exception(self, _):
        ax_client = AxClient()
        ax_client.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "y", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
            objective_name="a",
        )
        with self.assertRaisesRegex(
            expected_exception=RuntimeError,
            expected_regex="Cholesky errors typically occur",
        ):
            ax_client.get_next_trial()
