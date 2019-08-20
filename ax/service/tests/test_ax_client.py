#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import sys
from enum import Enum
from math import ceil
from typing import List, Tuple
from unittest.mock import patch

import numpy as np
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
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient
from ax.storage.sqa_store.db import init_test_engine_and_session_factory
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.storage.sqa_store.structs import DBSettings
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast, not_none


class FakeModels(Enum):
    SOBOL = get_sobol
    GPEI = get_sobol


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
                ax_client.complete_trial(idx, branin(params["x1"], params["x2"]))
    # If all went well and no errors were raised, remaining_trials should be 0.
    return remaining_trials


class TestServiceAPI(TestCase):
    """Tests service-like API functionality."""

    def test_interruption(self) -> None:
        ax = AxClient()
        ax.create_experiment(
            name="test",
            parameters=[  # pyre-fixme[6]: expected union that should include
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
            objective_name="branin",
            minimize=True,
        )
        for i in range(6):
            parameterization, trial_index = ax.get_next_trial()
            self.assertFalse(  # There should be non-complete trials.
                all(t.status.is_terminal for t in ax.experiment.trials.values())
            )
            x1, x2 = parameterization.get("x1"), parameterization.get("x2")
            ax.complete_trial(
                trial_index,
                raw_data=checked_cast(
                    float, branin(checked_cast(float, x1), checked_cast(float, x2))
                ),
            )
            old_client = ax
            serialized = ax.to_json_snapshot()
            ax = AxClient.from_json_snapshot(serialized)
            self.assertEqual(len(ax.experiment.trials.keys()), i + 1)
            self.assertIsNot(ax, old_client)
            self.assertTrue(  # There should be no non-complete trials.
                all(t.status.is_terminal for t in ax.experiment.trials.values())
            )

    def test_default_generation_strategy(self) -> None:
        """Test that Sobol+GPEI is used if no GenerationStrategy is provided."""
        ax = AxClient()
        ax.create_experiment(
            parameters=[  # pyre-fixme[6]: expected union that should include
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
            objective_name="branin",
            minimize=True,
        )
        self.assertEqual(
            [s.model for s in not_none(ax.generation_strategy)._steps],
            [Models.SOBOL, Models.GPEI],
        )
        with self.assertRaisesRegex(ValueError, ".* no trials."):
            ax.get_optimization_trace(objective_optimum=branin.fmin)
        for i in range(6):
            parameterization, trial_index = ax.get_next_trial()
            x1, x2 = parameterization.get("x1"), parameterization.get("x2")
            ax.complete_trial(
                trial_index,
                raw_data={
                    "branin": (
                        checked_cast(
                            float,
                            branin(checked_cast(float, x1), checked_cast(float, x2)),
                        ),
                        0.0,
                    )
                },
                sample_size=i,
            )
            if i < 5:
                with self.assertRaisesRegex(ValueError, "Could not obtain contour"):
                    ax.get_contour_plot(param_x="x1", param_y="x2")
        ax.get_optimization_trace(objective_optimum=branin.fmin)
        ax.get_contour_plot()
        # Test that Sobol is chosen when all parameters are choice.
        ax = AxClient()
        ax.create_experiment(
            parameters=[  # pyre-fixme[6]: expected union that should include
                {"name": "x1", "type": "choice", "values": [1, 2, 3]},
                {"name": "x2", "type": "choice", "values": [1, 2, 3]},
            ]
        )
        self.assertEqual(
            [s.model for s in not_none(ax.generation_strategy)._steps], [Models.SOBOL]
        )
        self.assertEqual(ax.get_recommended_max_parallelism(), [(-1, -1)])

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
            ax.complete_trial(trial_index, raw_data="invalid_data")

    def test_raw_data_format_with_fidelities(self):
        ax = AxClient()
        ax.create_experiment(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 1.0]},
            ],
            minimize=True,
        )
        for _ in range(6):
            parameterization, trial_index = ax.get_next_trial()
            x1, x2 = parameterization.get("x1"), parameterization.get("x2")
            ax.complete_trial(
                trial_index,
                raw_data=[
                    ({"x2": x2 / 2.0}, {"objective": (branin(x1, x2 / 2.0), 0.0)}),
                    ({"x2": x2}, {"objective": (branin(x1, x2), 0.0)}),
                ],
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
        best_trial_values = ax.get_best_parameters()[1]
        self.assertEqual(best_trial_values[0], {"objective": -2.0})
        self.assertTrue(math.isnan(best_trial_values[1]["objective"]["objective"]))

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

    def test_attach_trial_and_get_trial_parameters(self):
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
        self.assertEqual(ax.get_trial_parameters(trial_index=idx), {"x1": 0, "x2": 1})
        with self.assertRaises(ValueError):
            ax.get_trial_parameters(trial_index=10)  # No trial #10 in experiment.

    def test_attach_trial_numpy(self):
        ax = AxClient()
        ax.create_experiment(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        params, idx = ax.attach_trial(parameters={"x1": 0, "x2": 1})
        ax.complete_trial(trial_index=idx, raw_data=np.int32(5))
        self.assertEqual(ax.get_best_parameters()[0], params)

    def test_relative_oc_without_sq(self):
        """Must specify status quo to have relative outcome constraint."""
        ax = AxClient()
        with self.assertRaises(ValueError):
            ax.create_experiment(
                name="test_experiment",
                parameters=[
                    {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                    {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
                ],
                objective_name="test_objective",
                minimize=True,
                outcome_constraints=["some_metric <= 4.0%"],
            )

    @patch("ax.service.utils.dispatch.Models", FakeModels)
    def test_recommended_parallelism(self):
        ax = AxClient()
        with self.assertRaisesRegex(ValueError, "`get_recommended_max_parallelism`"):
            ax.get_recommended_max_parallelism()
        ax.create_experiment(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        self.assertEqual(ax.get_recommended_max_parallelism(), [(5, 5), (-1, 3)])
        self.assertEqual(
            run_trials_using_recommended_parallelism(
                ax, ax.get_recommended_max_parallelism(), 20
            ),
            0,
        )
        # With incorrect parallelism setting, the 'need more data' error should
        # still be raised.
        ax = AxClient()
        ax.create_experiment(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        with self.assertRaisesRegex(ValueError, "All trials for current model "):
            run_trials_using_recommended_parallelism(ax, [(6, 6), (-1, 3)], 20)

    @patch.dict(sys.modules, {"ax.storage.sqa_store.structs": None})
    def test_no_sqa(self):
        # Pretend we couldn't import sqa_store.structs (this could happen when
        # SQLAlchemy is not installed).
        patcher = patch("ax.service.ax_client.DBSettings", None)
        patcher.start()
        with self.assertRaises(ModuleNotFoundError):
            import ax.storage.sqa_store.structs  # noqa F401
        AxClient()  # Make sure we still can instantiate client w/o db settings.
        # Even with correctly typed DBSettings, `AxClient` instantiation should
        # fail here, because `DBSettings` are mocked to None in `ax_client`.
        db_settings = DBSettings()
        self.assertIsInstance(db_settings, DBSettings)
        with self.assertRaisesRegex(ValueError, "`db_settings` argument should "):
            AxClient(db_settings=db_settings)
        patcher.stop()
        # DBSettings should be defined in `ax_client` now, but incorrectly typed
        # `db_settings` argument should still make instantiation fail.
        with self.assertRaisesRegex(ValueError, "`db_settings` argument should "):
            AxClient(db_settings="badly_typed_db_settings")

    def test_plotting_validation(self):
        ax = AxClient()
        ax.create_experiment(
            parameters=[
                {"name": "x3", "type": "fixed", "value": 2, "value_type": "int"}
            ]
        )
        with self.assertRaisesRegex(ValueError, ".* there are no trials"):
            ax.get_contour_plot()
        ax.get_next_trial()
        with self.assertRaisesRegex(ValueError, ".* less than 2 parameters"):
            ax.get_contour_plot()
        ax = AxClient()
        ax.create_experiment(
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ]
        )
        ax.get_next_trial()
        with self.assertRaisesRegex(ValueError, "If `param_x` is provided"):
            ax.get_contour_plot(param_x="x2")
        with self.assertRaisesRegex(ValueError, "If `param_x` is provided"):
            ax.get_contour_plot(param_y="x2")
        with self.assertRaisesRegex(ValueError, 'Parameter "x3"'):
            ax.get_contour_plot(param_x="x3", param_y="x3")
        with self.assertRaisesRegex(ValueError, 'Parameter "x4"'):
            ax.get_contour_plot(param_x="x1", param_y="x4")
        with self.assertRaisesRegex(ValueError, 'Metric "nonexistent"'):
            ax.get_contour_plot(param_x="x1", param_y="x2", metric_name="nonexistent")
        with self.assertRaisesRegex(ValueError, "Could not obtain contour"):
            ax.get_contour_plot(param_x="x1", param_y="x2", metric_name="objective")

    def test_sqa_storage(self):
        init_test_engine_and_session_factory(force_init=True)
        config = SQAConfig()
        encoder = Encoder(config=config)
        decoder = Decoder(config=config)
        db_settings = DBSettings(encoder=encoder, decoder=decoder)
        ax = AxClient(db_settings=db_settings)
        ax.create_experiment(
            name="test_experiment",
            parameters=[
                {"name": "x1", "type": "range", "bounds": [-5.0, 10.0]},
                {"name": "x2", "type": "range", "bounds": [0.0, 15.0]},
            ],
            minimize=True,
        )
        for _ in range(5):
            parameters, trial_index = ax.get_next_trial()
            ax.complete_trial(
                trial_index=trial_index, raw_data=branin(*parameters.values())
            )
        gs = ax.generation_strategy
        ax = AxClient(db_settings=db_settings)
        ax.load_experiment("test_experiment")
        self.assertEqual(gs, ax.generation_strategy)
