# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
from collections.abc import Mapping
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
from ax.analysis.analysis_card import AnalysisCard
from ax.analysis.plotly.parallel_coordinates import ParallelCoordinatesPlot
from ax.api.client import Client
from ax.api.configs import (
    ChoiceParameterConfig,
    DerivedParameterConfig,
    RangeParameterConfig,
    StorageConfig,
)
from ax.api.protocols.metric import IMetric
from ax.api.protocols.runner import IRunner
from ax.api.types import TParameterization
from ax.core.evaluations_to_data import DataType
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.map_metric import MapMetric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.core.parameter import (
    ChoiceParameter,
    ParameterType as CoreParameterType,
    RangeParameter,
)
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.core.trial import Trial
from ax.core.trial_status import TrialStatus
from ax.early_stopping.strategies import PercentileEarlyStoppingStrategy
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.service.utils.with_db_settings_base import (
    _save_generation_strategy_to_db_if_possible,
)
from ax.storage.sqa_store.db import init_test_engine_and_session_factory
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_optimization_config,
    get_percentile_early_stopping_strategy,
)
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import get_generation_strategy
from pyre_extensions import assert_is_instance, none_throws, override


class TestClient(TestCase):
    def test_configure_experiment(self) -> None:
        client = Client()

        float_parameter = RangeParameterConfig(
            name="float_param",
            parameter_type="float",
            bounds=(0, 1),
        )
        int_parameter = RangeParameterConfig(
            name="int_param",
            parameter_type="int",
            bounds=(0, 1),
        )
        choice_parameter = ChoiceParameterConfig(
            name="choice_param",
            parameter_type="str",
            values=["a", "b", "c"],
        )

        client.configure_experiment(
            name="test_experiment",
            parameters=[float_parameter, int_parameter, choice_parameter],
            parameter_constraints=["int_param <= float_param"],
            description="test description",
            owner="miles",
        )
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
                default_data_type=DataType.MAP_DATA,
            ),
        )

        with self.assertRaisesRegex(UnsupportedError, "Experiment already configured"):
            client.configure_experiment(
                name="test_experiment",
                parameters=[float_parameter, int_parameter, choice_parameter],
                parameter_constraints=["int_param <= float_param"],
                description="test description",
                owner="miles",
            )

    def test_configure_optimization(self) -> None:
        client = Client()

        float_parameter = RangeParameterConfig(
            name="float_param",
            parameter_type="float",
            bounds=(0, 1),
        )
        int_parameter = RangeParameterConfig(
            name="int_param",
            parameter_type="int",
            bounds=(0, 1),
        )
        choice_parameter = ChoiceParameterConfig(
            name="choice_param",
            parameter_type="str",
            values=["a", "b", "c"],
        )

        client.configure_experiment(
            name="test_experiment",
            parameters=[float_parameter, int_parameter, choice_parameter],
            parameter_constraints=["int_param <= float_param"],
            description="test description",
            owner="miles",
        )

        client.configure_optimization(
            objective="-ne",
            outcome_constraints=["qps >= 0"],
        )

        self.assertEqual(
            client._experiment.optimization_config,
            OptimizationConfig(
                objective=Objective(
                    metric=MapMetric(name="ne", lower_is_better=True), minimize=True
                ),
                outcome_constraints=[
                    OutcomeConstraint(
                        metric=MapMetric(name="qps"),
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

        # Test that metrics not in the objective string are downgraded to tracking
        # metrics rather than being removed
        client = Client()
        client.configure_experiment(
            name="test_experiment",
            parameters=[float_parameter],
        )

        custom_metric1 = DummyMetric(name="metric1")
        custom_metric2 = DummyMetric(name="metric2")
        custom_metric3 = DummyMetric(name="metric3")
        client.configure_metrics(
            metrics=[custom_metric1, custom_metric2, custom_metric3]
        )

        # Verify all metrics are added as tracking metrics
        self.assertEqual(len(client._experiment.tracking_metrics), 3)
        self.assertIn(custom_metric1, client._experiment.tracking_metrics)
        self.assertIn(custom_metric2, client._experiment.tracking_metrics)
        self.assertIn(custom_metric3, client._experiment.tracking_metrics)

        # Configure optimization with only metric1 in the objective
        client.configure_optimization(
            objective="metric1",
        )

        # Verify metric2 and metric3 are still present as a tracking metric
        self.assertIn(custom_metric2, client._experiment.tracking_metrics)
        self.assertIn(custom_metric3, client._experiment.tracking_metrics)

        # Verify metric1 is now part of the objective and no longer a tracking metric
        optimization_config = client._experiment.optimization_config
        self.assertIsNotNone(optimization_config)
        objective = assert_is_instance(
            optimization_config.objective,
            Objective,
        )
        self.assertEqual(objective.metric.name, "metric1")

        # Verify no metrics were removed, just moved from tracking to objective
        all_metrics = [objective.metric] + list(client._experiment.tracking_metrics)
        self.assertEqual(len(all_metrics), 3)

    def test_configure_runner(self) -> None:
        client = Client()
        runner = DummyRunner()

        with self.assertRaisesRegex(AssertionError, "Experiment not set"):
            client.configure_runner(runner=runner)

        client.set_experiment(experiment=get_branin_experiment())
        client.configure_runner(runner=runner)

        self.assertEqual(client._experiment.runner, runner)

    def test_configure_metric(self) -> None:
        client = Client()
        custom_metric = DummyMetric(name="custom")

        with self.assertRaisesRegex(AssertionError, "Experiment not set"):
            client.configure_metrics(metrics=[custom_metric])

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(0, 1))
            ],
            name="foo",
        )

        # Test replacing a single objective
        client.configure_optimization(objective="custom")
        client.configure_metrics(metrics=[custom_metric])

        self.assertEqual(
            custom_metric,
            none_throws(client._experiment.optimization_config).objective.metric,
        )

        # Test replacing a multi-objective
        client.configure_optimization(objective="custom, foo")
        client.configure_metrics(metrics=[custom_metric])

        self.assertIn(
            custom_metric,
            assert_is_instance(
                none_throws(client._experiment.optimization_config).objective,
                MultiObjective,
            ).metrics,
        )
        # Test replacing a scalarized objective
        client.configure_optimization(objective="custom + foo")
        client.configure_metrics(metrics=[custom_metric])

        self.assertIn(
            custom_metric,
            assert_is_instance(
                none_throws(client._experiment.optimization_config).objective,
                ScalarizedObjective,
            ).metrics,
        )

        # Test replacing an outcome constraint
        client.configure_optimization(
            objective="foo", outcome_constraints=["custom >= 0"]
        )
        client.configure_metrics(metrics=[custom_metric])

        self.assertEqual(
            custom_metric,
            none_throws(client._experiment.optimization_config)
            .outcome_constraints[0]
            .metric,
        )

        # Test adding a tracking metric
        client = Client()  # Start a fresh Client
        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(0, 1))
            ],
            name="foo",
        )
        client.configure_metrics(metrics=[custom_metric])

        self.assertEqual(
            custom_metric,
            client._experiment.tracking_metrics[0],
        )

    def test_set_experiment(self) -> None:
        client = Client()
        experiment = get_branin_experiment()

        client.set_experiment(experiment=experiment)

        self.assertEqual(client._experiment, experiment)

    def test_set_optimization_config(self) -> None:
        client = Client()
        optimization_config = get_branin_optimization_config()

        with self.assertRaisesRegex(AssertionError, "Experiment not set"):
            client.set_optimization_config(optimization_config=optimization_config)

        client.set_experiment(experiment=get_branin_experiment())
        client.set_optimization_config(
            optimization_config=optimization_config,
        )

        self.assertEqual(client._experiment.optimization_config, optimization_config)

    def test_set_generation_strategy(self) -> None:
        client = Client()
        client.set_experiment(experiment=get_branin_experiment())

        generation_strategy = get_generation_strategy()

        client.set_generation_strategy(generation_strategy=generation_strategy)
        self.assertEqual(client._generation_strategy, generation_strategy)

    def test_set_early_stopping_strategy(self) -> None:
        client = Client()
        early_stopping_strategy = get_percentile_early_stopping_strategy()

        client.set_early_stopping_strategy(
            early_stopping_strategy=early_stopping_strategy
        )
        self.assertEqual(client._early_stopping_strategy, early_stopping_strategy)

    def test_get_next_trials(self) -> None:
        client = Client()

        with self.assertRaisesRegex(AssertionError, "Experiment not set"):
            client.get_next_trials(max_trials=1)

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
                RangeParameterConfig(name="x2", parameter_type="float", bounds=(-1, 1)),
            ],
            name="foo",
        )

        with self.assertRaisesRegex(UnsupportedError, "OptimizationConfig not set"):
            client.get_next_trials(max_trials=1)

        client.configure_optimization(objective="foo")
        client.configure_generation_strategy(
            # Set this to a large number so test runs fast
            initialization_budget=999,
        )

        # Test can generate one trial
        trials = client.get_next_trials(max_trials=1)
        self.assertEqual(len(trials), 1)
        self.assertEqual({*trials[0].keys()}, {"x1", "x2"})
        for parameter in ["x1", "x2"]:
            value = assert_is_instance(trials[0][parameter], float)
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)

        # Test can generate multiple trials
        trials = client.get_next_trials(max_trials=2)
        self.assertEqual(len(trials), 2)

        # Test respects fixed features
        with mock.patch(
            "ax.service.utils.with_db_settings_base"
            "._save_generation_strategy_to_db_if_possible"
        ) as mock_save:
            trials = client.get_next_trials(max_trials=1, fixed_parameters={"x1": 0.5})
        value = assert_is_instance(trials[3]["x1"], float)
        self.assertEqual(value, 0.5)

        # Check that GS is not saved to the DB.
        mock_save.assert_not_called()

    def test_get_next_trials_with_db(self) -> None:
        init_test_engine_and_session_factory(force_init=True)
        client = Client(storage_config=StorageConfig())

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
                RangeParameterConfig(name="x2", parameter_type="float", bounds=(-1, 1)),
            ],
            name="foo",
        )

        client.configure_optimization(objective="foo")
        client.configure_generation_strategy(
            initialization_budget=1, initialize_with_center=False
        )

        # Test can generate one trial
        trials = client.get_next_trials(max_trials=1)
        self.assertEqual(len(trials), 1)
        self.assertIn(0, trials)
        trial = client._experiment.trials[0]
        self.assertEqual(trial.generator_runs[0]._model_key, "Sobol")
        client.complete_trial(
            trial_index=trial.index, raw_data={"foo": (random.random(), 1.0)}
        )
        # Generate one more trial, so that GS transitions to BO.
        with mock.patch(
            "ax.service.utils.with_db_settings_base"
            "._save_generation_strategy_to_db_if_possible",
            wraps=_save_generation_strategy_to_db_if_possible,
        ) as mock_save:
            trials = client.get_next_trials(max_trials=1)
        # Check that GS was saved after generating the trials.
        self.assertEqual(client._generation_strategy.current_node_name, "MBM")
        mock_save.assert_called_once()

        # Check that loading the GS from the DB results in using BO.
        # This will only happen if the GS is saved in get_next_trials
        client2 = Client.load_from_database(
            client._experiment.name, storage_config=StorageConfig()
        )
        self.assertEqual(client2._generation_strategy.current_node_name, "MBM")
        trials = client2.get_next_trials(max_trials=1)
        self.assertEqual(len(trials), 1)
        self.assertIn(2, trials)
        trial = client2._experiment.trials[2]
        self.assertEqual(trial.generator_runs[0]._model_key, "BoTorch")

    def test_attach_data(self) -> None:
        client = Client()

        with self.assertRaisesRegex(AssertionError, "Experiment not set"):
            client.attach_data(trial_index=0, raw_data={"foo": 1.0})

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
            ],
            name="foo",
        )
        client.configure_optimization(objective="foo")

        # Vanilla case with no progression argument
        trial_index = [*client.get_next_trials(max_trials=1).keys()][0]
        client.attach_data(trial_index=trial_index, raw_data={"foo": 1.0})

        self.assertEqual(
            client._experiment.trials[trial_index].status,
            TrialStatus.RUNNING,
        )
        self.assertTrue(
            assert_is_instance(
                client._experiment.lookup_data(trial_indices=[trial_index]),
                MapData,
            ).map_df.equals(
                pd.DataFrame(
                    {
                        "trial_index": {0: 0},
                        "arm_name": {0: "0_0"},
                        "metric_name": {0: "foo"},
                        "metric_signature": {0: "foo"},
                        "mean": {0: 1.0},
                        "sem": {0: np.nan},
                        "step": {0: np.nan},
                    }
                )
            ),
        )

        # With progression argument
        client.attach_data(trial_index=0, raw_data={"foo": 2.0}, progression=10)

        self.assertEqual(
            client._experiment.trials[trial_index].status,
            TrialStatus.RUNNING,
        )
        self.assertTrue(
            assert_is_instance(
                client._experiment.lookup_data(trial_indices=[trial_index]),
                MapData,
            ).map_df.equals(
                pd.DataFrame(
                    {
                        "trial_index": {0: 0, 1: 0},
                        "arm_name": {0: "0_0", 1: "0_0"},
                        "metric_name": {0: "foo", 1: "foo"},
                        "metric_signature": {0: "foo", 1: "foo"},
                        "mean": {0: 1.0, 1: 2.0},
                        "sem": {0: np.nan, 1: np.nan},
                        "step": {0: np.nan, 1: 10.0},
                    }
                )
            ),
        )

        # With extra metrics
        # Try and attach data for a metric that doesn't exist
        with self.assertRaisesRegex(
            UserInputError,
            "Unable to find the metric signature for one or more metrics.",
        ):
            client.attach_data(
                trial_index=trial_index,
                raw_data={"foo": 1.0, "bar": 2.0},
            )

        client.configure_metrics(metrics=[DummyMetric(name="bar")])
        client.attach_data(
            trial_index=trial_index,
            raw_data={"foo": 1.0, "bar": 2.0},
        )
        self.assertEqual(
            client._experiment.trials[trial_index].status,
            TrialStatus.RUNNING,
        )
        self.assertTrue(
            assert_is_instance(
                client._experiment.lookup_data(trial_indices=[trial_index]),
                MapData,
            ).map_df.equals(
                pd.DataFrame(
                    {
                        "trial_index": {0: 0, 1: 0, 2: 0},
                        "arm_name": {0: "0_0", 1: "0_0", 2: "0_0"},
                        "metric_name": {0: "foo", 1: "foo", 2: "bar"},
                        "metric_signature": {0: "foo", 1: "foo", 2: "bar"},
                        "mean": {0: 2.0, 1: 1.0, 2: 2.0},
                        "sem": {0: np.nan, 1: np.nan, 2: np.nan},
                        "step": {0: 10.0, 1: np.nan, 2: np.nan},
                    }
                )
            ),
        )

    def test_complete_trial(self) -> None:
        client = Client()

        with self.assertRaisesRegex(AssertionError, "Experiment not set"):
            client.complete_trial(trial_index=0, raw_data={"foo": 1.0})

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
            ],
            name="foo",
        )
        client.configure_optimization(objective="foo", outcome_constraints=["bar >= 0"])

        # Vanilla case with no progression argument
        trial_index = [*client.get_next_trials(max_trials=1).keys()][0]
        client.complete_trial(
            trial_index=trial_index, raw_data={"foo": 1.0, "bar": 2.0}
        )

        self.assertEqual(
            client._experiment.trials[trial_index].status,
            TrialStatus.COMPLETED,
        )
        self.assertTrue(
            assert_is_instance(
                client._experiment.lookup_data(trial_indices=[trial_index]),
                MapData,
            ).map_df.equals(
                pd.DataFrame(
                    {
                        "trial_index": {0: 0, 1: 0},
                        "arm_name": {0: "0_0", 1: "0_0"},
                        "metric_name": {0: "foo", 1: "bar"},
                        "metric_signature": {0: "foo", 1: "bar"},
                        "mean": {0: 1.0, 1: 2.0},
                        "sem": {0: np.nan, 1: np.nan},
                        "step": {0: np.nan, 1: np.nan},
                    }
                )
            ),
        )

        # With progression argument
        trial_index = [*client.get_next_trials(max_trials=1).keys()][0]
        client.complete_trial(
            trial_index=trial_index, raw_data={"foo": 1.0, "bar": 2.0}, progression=10
        )

        self.assertEqual(
            client._experiment.trials[trial_index].status,
            TrialStatus.COMPLETED,
        )

        self.assertTrue(
            assert_is_instance(
                client._experiment.lookup_data(trial_indices=[trial_index]),
                MapData,
            ).map_df.equals(
                pd.DataFrame(
                    {
                        "trial_index": {0: 1, 1: 1},
                        "arm_name": {0: "1_0", 1: "1_0"},
                        "metric_name": {0: "foo", 1: "bar"},
                        "metric_signature": {0: "foo", 1: "bar"},
                        "mean": {0: 1.0, 1: 2.0},
                        "sem": {0: np.nan, 1: np.nan},
                        "step": {0: 10.0, 1: 10.0},
                    }
                )
            ),
        )

        # With missing metrics
        trial_index = [*client.get_next_trials(max_trials=1).keys()][0]
        client.complete_trial(trial_index=trial_index, raw_data={"foo": 1.0})

        self.assertEqual(
            client._experiment.trials[trial_index].status,
            TrialStatus.FAILED,
        )
        self.assertTrue(
            assert_is_instance(
                client._experiment.lookup_data(trial_indices=[trial_index]),
                MapData,
            ).map_df.equals(
                pd.DataFrame(
                    {
                        "trial_index": {0: 2},
                        "arm_name": {0: "2_0"},
                        "metric_name": {0: "foo"},
                        "metric_signature": {0: "foo"},
                        "mean": {0: 1.0},
                        "sem": {0: np.nan},
                        "step": {0: np.nan},
                    }
                )
            ),
        )

    def test_attach_trial(self) -> None:
        client = Client()

        with self.assertRaisesRegex(AssertionError, "Experiment not set"):
            client.attach_trial(parameters={"x1": 0.5})

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
            ],
            name="foo",
        )
        client.configure_optimization(objective="foo")

        trial_index = client.attach_trial(parameters={"x1": 0.5}, arm_name="bar")
        trial = assert_is_instance(client._experiment.trials[trial_index], Trial)
        self.assertEqual(none_throws(trial.arm).parameters, {"x1": 0.5})
        self.assertEqual(none_throws(trial.arm).name, "bar")
        self.assertEqual(trial.status, TrialStatus.RUNNING)

    def test_attach_baseline(self) -> None:
        client = Client()

        with self.assertRaisesRegex(AssertionError, "Experiment not set"):
            client.attach_baseline(parameters={"x1": 0.5})

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
            ],
            name="foo",
        )
        client.configure_optimization(objective="foo")

        trial_index = client.attach_baseline(parameters={"x1": 0.5})
        trial = assert_is_instance(client._experiment.trials[trial_index], Trial)
        self.assertEqual(none_throws(trial.arm).parameters, {"x1": 0.5})
        self.assertEqual(none_throws(trial.arm).name, "baseline")
        self.assertEqual(trial.status, TrialStatus.RUNNING)

        self.assertEqual(client._experiment.status_quo, trial.arm)

    def test_mark_trial_failed(self) -> None:
        client = Client()

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
            ],
            name="foo",
        )
        client.configure_optimization(objective="foo")

        trial_index = [*client.get_next_trials(max_trials=1).keys()][0]
        client.mark_trial_failed(
            trial_index=trial_index, failed_reason="testing the optional parameter"
        )
        self.assertEqual(
            client._experiment.trials[trial_index].status,
            TrialStatus.FAILED,
        )
        self.assertEqual(
            client._experiment.trials[trial_index]._failed_reason,
            "testing the optional parameter",
        )

    def test_mark_trial_abandoned(self) -> None:
        client = Client()

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
            ],
            name="foo",
        )
        client.configure_optimization(objective="foo")

        trial_index = [*client.get_next_trials(max_trials=1).keys()][0]
        client.mark_trial_abandoned(trial_index=trial_index)
        self.assertEqual(
            client._experiment.trials[trial_index].status,
            TrialStatus.ABANDONED,
        )

    def test_mark_trial_early_stopped(self) -> None:
        client = Client()

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
            ],
            name="foo",
        )
        client.configure_optimization(objective="foo")

        trial_index = [*client.get_next_trials(max_trials=1).keys()][0]

        with self.assertRaisesRegex(
            UnsupportedError, "Cannot mark trial early stopped"
        ):
            client.mark_trial_early_stopped(trial_index=trial_index)

        client.attach_data(
            trial_index=trial_index, raw_data={"foo": 0.0}, progression=1
        )
        client.mark_trial_early_stopped(trial_index=trial_index)
        self.assertEqual(
            client._experiment.trials[trial_index].status,
            TrialStatus.EARLY_STOPPED,
        )
        self.assertTrue(
            assert_is_instance(
                client._experiment.lookup_data(trial_indices=[trial_index]),
                MapData,
            ).map_df.equals(
                pd.DataFrame(
                    {
                        "trial_index": {0: 0},
                        "arm_name": {0: "0_0"},
                        "metric_name": {0: "foo"},
                        "metric_signature": {0: "foo"},
                        "mean": {0: 0.0},
                        "sem": {0: np.nan},
                        "step": {0: 1.0},
                    }
                )
            ),
        )

    def test_should_stop_trial_early(self) -> None:
        client = Client()

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
            ],
            name="foo",
        )
        client.configure_optimization(objective="foo")

        client.set_early_stopping_strategy(
            early_stopping_strategy=PercentileEarlyStoppingStrategy(
                metric_signatures=["foo"]
            )
        )

        client.get_next_trials(max_trials=1)
        self.assertFalse(client.should_stop_trial_early(trial_index=0))

    def test_run_trials(self) -> None:
        client = Client()

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
            ],
            name="foo",
        )
        client.configure_optimization(objective="foo")
        client.configure_metrics(metrics=[DummyMetric(name="foo")])
        client.configure_runner(runner=DummyRunner())

        client.run_trials(max_trials=4)

        self.assertEqual(len(client._experiment.trials), 4)
        self.assertEqual(
            [
                trial.index
                for trial in client._experiment.trials_by_status[TrialStatus.COMPLETED]
            ],
            [0, 1, 2, 3],
        )

        self.assertTrue(
            assert_is_instance(
                client._experiment.lookup_data(),
                MapData,
            ).map_df.equals(
                pd.DataFrame(
                    {
                        "trial_index": {0: 0, 1: 1, 2: 2, 3: 3},
                        "arm_name": {0: "0_0", 1: "1_0", 2: "2_0", 3: "3_0"},
                        "metric_name": {0: "foo", 1: "foo", 2: "foo", 3: "foo"},
                        "metric_signature": {0: "foo", 1: "foo", 2: "foo", 3: "foo"},
                        "mean": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                        "sem": {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan},
                        "step": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                    }
                )
            ),
        )

    def test_get_next_trials_then_run_trials(self) -> None:
        client = Client()

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
            ],
            name="foo",
        )
        client.configure_optimization(objective="foo")
        client.configure_metrics(metrics=[DummyMetric(name="foo")])
        client.configure_runner(runner=DummyRunner())

        # First use Client in ask-tell
        # Complete two trials
        for index, _parameters in client.get_next_trials(max_trials=2).items():
            client.complete_trial(trial_index=index, raw_data={"foo": 1.0})

        # Leave one trial RUNNING
        _ = client.get_next_trials(max_trials=1)

        self.assertEqual(
            len(client._experiment.trials_by_status[TrialStatus.COMPLETED]),
            2,
        )
        self.assertEqual(
            len(client._experiment.trials_by_status[TrialStatus.RUNNING]),
            1,
        )

        # Configure runners and Metrics Run another two trials
        client.configure_metrics(metrics=[DummyMetric(name="foo")])
        client.configure_runner(runner=DummyRunner())
        client.run_trials(max_trials=2)

        # All trials should be COMPLETED
        self.assertEqual(
            len(client._experiment.trials_by_status[TrialStatus.COMPLETED]),
            5,
        )

    def test_summarize(self) -> None:
        client = Client()

        client.configure_experiment(
            name="test_experiment",
            parameters=[
                RangeParameterConfig(
                    name="x1",
                    parameter_type="float",
                    bounds=(0, 1),
                ),
                RangeParameterConfig(
                    name="x2",
                    parameter_type="float",
                    bounds=(0, 1),
                ),
            ],
        )
        client.configure_optimization(objective="foo, bar")
        # No GS, no data.
        summary_df = client.summarize()
        self.assertTrue(summary_df.empty)

        # Add manual trial.
        index = client.attach_trial(
            parameters={"x1": 0.5, "x2": 0.5}, arm_name="manual"
        )
        client.attach_data(trial_index=index, raw_data={"foo": 0.0, "bar": 0.5})
        summary_df = client.summarize()
        expected_columns = {
            "trial_index",
            "arm_name",
            "trial_status",
            "foo",
            "bar",
            "x1",
            "x2",
        }
        self.assertEqual(set(summary_df.columns), expected_columns)

        # Get two trials and fail one, giving us a ragged structure
        client.get_next_trials(max_trials=2)
        client.complete_trial(trial_index=1, raw_data={"foo": 1.0, "bar": 2.0})
        client.mark_trial_failed(trial_index=2)

        summary_df = client.summarize()
        expected_columns.add("generation_node")
        self.assertEqual(set(summary_df.columns), expected_columns)

        trial_0_parameters = none_throws(
            assert_is_instance(client._experiment.trials[0], Trial).arm
        ).parameters
        trial_1_parameters = none_throws(
            assert_is_instance(client._experiment.trials[1], Trial).arm
        ).parameters
        trial_2_parameters = none_throws(
            assert_is_instance(client._experiment.trials[2], Trial).arm
        ).parameters
        expected = pd.DataFrame(
            {
                "trial_index": {0: 0, 1: 1, 2: 2},
                "arm_name": {0: "manual", 1: "1_0", 2: "2_0"},
                "trial_status": {0: "RUNNING", 1: "COMPLETED", 2: "FAILED"},
                "generation_node": {0: None, 1: "CenterOfSearchSpace", 2: "Sobol"},
                "foo": {0: 0.0, 1: 1.0, 2: np.nan},  # NaN because trial 2 failed
                "bar": {0: 0.5, 1: 2.0, 2: np.nan},
                "x1": {
                    0: trial_0_parameters["x1"],
                    1: trial_1_parameters["x1"],
                    2: trial_2_parameters["x1"],
                },
                "x2": {
                    0: trial_0_parameters["x2"],
                    1: trial_1_parameters["x2"],
                    2: trial_2_parameters["x2"],
                },
            }
        )
        pd.testing.assert_frame_equal(summary_df, expected)

        # Test with trial_indices parameter
        # Only include trials 0 and 1
        summary_df_filtered = client.summarize(trial_indices=[0, 1])
        expected_filtered = pd.DataFrame(
            {
                "trial_index": {0: 0, 1: 1},
                "arm_name": {0: "manual", 1: "1_0"},
                "trial_status": {0: "RUNNING", 1: "COMPLETED"},
                "generation_node": {0: None, 1: "CenterOfSearchSpace"},
                "foo": {0: 0.0, 1: 1.0},
                "bar": {0: 0.5, 1: 2.0},
                "x1": {
                    0: trial_0_parameters["x1"],
                    1: trial_1_parameters["x1"],
                },
                "x2": {
                    0: trial_0_parameters["x2"],
                    1: trial_1_parameters["x2"],
                },
            }
        )
        pd.testing.assert_frame_equal(summary_df_filtered, expected_filtered)

        # Test with only one trial index
        summary_df_single = client.summarize(trial_indices=[1])
        expected_single = pd.DataFrame(
            {
                "trial_index": {0: 1},
                "arm_name": {0: "1_0"},
                "trial_status": {0: "COMPLETED"},
                "generation_node": {0: "CenterOfSearchSpace"},
                "foo": {0: 1.0},
                "bar": {0: 2.0},
                "x1": {0: trial_1_parameters["x1"]},
                "x2": {0: trial_1_parameters["x2"]},
            }
        )
        pd.testing.assert_frame_equal(summary_df_single, expected_single)

        # Test with trial_status parameter
        summary_df_completed = client.summarize(trial_statuses=["completed"])
        expected_completed = pd.DataFrame(
            {
                "trial_index": {0: 1},
                "arm_name": {0: "1_0"},
                "trial_status": {0: "COMPLETED"},
                "generation_node": {0: "CenterOfSearchSpace"},
                "foo": {0: 1.0},
                "bar": {0: 2.0},
                "x1": {0: trial_1_parameters["x1"]},
                "x2": {0: trial_1_parameters["x2"]},
            }
        )
        pd.testing.assert_frame_equal(summary_df_completed, expected_completed)

        # Test with trial_status parameter for running trials
        summary_df_running = client.summarize(trial_statuses=["running"])
        expected_running = pd.DataFrame(
            {
                "trial_index": {0: 0},
                "arm_name": {0: "manual"},
                "trial_status": {0: "RUNNING"},
                "foo": {0: 0.0},
                "bar": {0: 0.5},
                "x1": {0: trial_0_parameters["x1"]},
                "x2": {0: trial_0_parameters["x2"]},
            }
        )

        assert summary_df_running.equals(expected_running)

        # Test with multiple trial_status values
        summary_df_multi_status = client.summarize(
            trial_statuses=["completed", "running"]
        )
        expected_multi_status = pd.DataFrame(
            {
                "trial_index": {0: 0, 1: 1},
                "arm_name": {0: "manual", 1: "1_0"},
                "trial_status": {0: "RUNNING", 1: "COMPLETED"},
                "generation_node": {0: None, 1: "CenterOfSearchSpace"},
                "foo": {0: 0.0, 1: 1.0},
                "bar": {0: 0.5, 1: 2.0},
                "x1": {
                    0: trial_0_parameters["x1"],
                    1: trial_1_parameters["x1"],
                },
                "x2": {
                    0: trial_0_parameters["x2"],
                    1: trial_1_parameters["x2"],
                },
            }
        )
        self.assertTrue(summary_df_multi_status.equals(expected_multi_status))

    def test_compute_analyses(self) -> None:
        client = Client()

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
            ],
            name="foo",
        )
        client.configure_optimization(objective="foo")
        client.configure_generation_strategy()

        with self.assertLogs(logger="ax.analysis", level="ERROR") as lg:
            analysis = ParallelCoordinatesPlot()
            cards = client.compute_analyses(analyses=[analysis])

            self.assertEqual(len(cards), 1)
            self.assertEqual(cards[0].name, "ParallelCoordinatesPlot")
            self.assertEqual(cards[0].title, "ParallelCoordinatesPlot Error")
            self.assertEqual(
                cards[0].subtitle,
                "ValueError encountered while computing ParallelCoordinatesPlot.",
            )
            self.assertIn("Traceback", assert_is_instance(cards[0], AnalysisCard).blob)
            self.assertTrue(
                any(("No data found for metric") in msg for msg in lg.output)
            )

        for trial_index, _ in client.get_next_trials(max_trials=1).items():
            client.complete_trial(trial_index=trial_index, raw_data={"foo": 1.0})

        cards = client.compute_analyses(analyses=[ParallelCoordinatesPlot()])

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].name, "ParallelCoordinatesPlot")

    @mock_botorch_optimize
    def test_get_best_parameterization(self) -> None:
        client = Client()

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
            ],
            name="foo",
        )

        with self.assertRaisesRegex(
            UnsupportedError, "No optimization config has been set"
        ):
            client.get_best_parameterization()

        client.configure_optimization(objective="foo")
        # Set initialization_budget=3 so we can reach a predictive GenerationNode
        # quickly
        client.configure_generation_strategy(initialization_budget=3)

        with self.assertRaisesRegex(UnsupportedError, "No trials have been run yet"):
            client.get_best_parameterization()

        for _ in range(3):
            for index, parameters in client.get_next_trials(max_trials=1).items():
                client.complete_trial(
                    trial_index=index,
                    raw_data={
                        "foo": assert_is_instance(parameters["x1"], float) ** 2,
                    },
                )

        parameters, prediction, index, name = client.get_best_parameterization()
        self.assertIn(
            name,
            [
                none_throws(assert_is_instance(trial, Trial).arm).name
                for trial in client._experiment.trials.values()
            ],
        )
        self.assertTrue(
            client._experiment.search_space.check_membership(
                parameterization=parameters
            )
        )
        self.assertEqual({*prediction.keys()}, {"foo"})

        # Run a non-Sobol trial
        for index, parameters in client.get_next_trials(max_trials=1).items():
            client.complete_trial(
                trial_index=index,
                raw_data={
                    "foo": assert_is_instance(parameters["x1"], float) ** 2,
                },
            )
        parameters, prediction, index, name = client.get_best_parameterization()
        self.assertIn(
            name,
            [
                none_throws(assert_is_instance(trial, Trial).arm).name
                for trial in client._experiment.trials.values()
            ],
        )
        self.assertTrue(
            client._experiment.search_space.check_membership(
                parameterization=parameters
            )
        )
        self.assertEqual({*prediction.keys()}, {"foo"})

        # Try calling after setting OptimizationConfig to MOO problem
        client.configure_optimization("foo, bar")
        with self.assertRaisesRegex(
            UnsupportedError, "Please call get_pareto_frontier"
        ):
            client.get_best_parameterization()

    @mock_botorch_optimize
    def test_get_pareto_frontier(self) -> None:
        client = Client()

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
            ],
            name="foo",
        )

        with self.assertRaisesRegex(
            UnsupportedError, "No optimization config has been set"
        ):
            client.get_pareto_frontier()

        client.configure_optimization(objective="foo, bar")
        # Set initialization_budget=3 so we can reach a predictive GenerationNode
        # quickly
        client.configure_generation_strategy(initialization_budget=3)

        with self.assertRaisesRegex(UnsupportedError, "No trials have been run yet"):
            client.get_pareto_frontier()

        for _ in range(3):
            for index, parameters in client.get_next_trials(max_trials=1).items():
                client.complete_trial(
                    trial_index=index,
                    raw_data={
                        "foo": assert_is_instance(parameters["x1"], float) ** 2,
                        "bar": 0.0,
                    },
                )

        frontier = client.get_pareto_frontier(False)
        for parameters, prediction, index, name in frontier:
            self.assertEqual(
                none_throws(
                    assert_is_instance(client._experiment.trials[index], Trial).arm
                ).name,
                name,
            )
            self.assertIn(
                name,
                [
                    none_throws(assert_is_instance(trial, Trial).arm).name
                    for trial in client._experiment.trials.values()
                ],
            )
            self.assertTrue(
                client._experiment.search_space.check_membership(
                    parameterization=parameters
                )
            )
            self.assertEqual({*prediction.keys()}, {"foo", "bar"})

        # Run a non-Sobol trial
        for index, parameters in client.get_next_trials(max_trials=1).items():
            client.complete_trial(
                trial_index=index,
                raw_data={
                    "foo": assert_is_instance(parameters["x1"], float) ** 2,
                    "bar": 0.0,
                },
            )
        frontier = client.get_pareto_frontier()
        for parameters, prediction, index, name in frontier:
            self.assertEqual(
                none_throws(
                    assert_is_instance(client._experiment.trials[index], Trial).arm
                ).name,
                name,
            )
            self.assertIn(
                name,
                [
                    none_throws(assert_is_instance(trial, Trial).arm).name
                    for trial in client._experiment.trials.values()
                ],
            )
            self.assertTrue(
                client._experiment.search_space.check_membership(
                    parameterization=parameters
                )
            )
            self.assertEqual({*prediction.keys()}, {"foo", "bar"})

        # Try calling after setting OptimizationConfig to single objective problem
        client.configure_optimization("foo")
        with self.assertRaisesRegex(
            UnsupportedError, "Please call get_best_parameterization"
        ):
            client.get_pareto_frontier()

    @mock_botorch_optimize
    def test_predict(self) -> None:
        client = Client()

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
            ],
            name="foo",
        )
        client.configure_optimization(objective="foo", outcome_constraints=["bar >= 0"])
        # Set initialization_budget=3 so we can reach a predictive GenerationNode
        # quickly
        client.configure_generation_strategy(initialization_budget=3)

        with self.assertRaisesRegex(ValueError, "but search space has parameters"):
            client.predict(points=[{"x0": 0}])

        with self.assertRaisesRegex(UnsupportedError, "not predictive"):
            client.predict(points=[{"x1": 0}])

        client.configure_metrics(metrics=[DummyMetric(name="baz")])
        for _ in range(4):
            for index, parameters in client.get_next_trials(max_trials=1).items():
                client.complete_trial(
                    trial_index=index,
                    raw_data={
                        "foo": assert_is_instance(parameters["x1"], float) ** 2,
                        "bar": 0.0,
                    },
                )

        # Check we've predicted something for foo and bar but not baz (which is a
        # tracking metric)
        (prediction,) = client.predict(points=[{"x1": 0.5}])
        self.assertEqual(set(prediction), {"foo", "bar"})
        # Check that we're returning SEM not variance.
        pred_mean = {"foo": [0.25], "bar": [0.0]}
        pred_cov = {
            "foo": {"foo": [4.0], "bar": [9.0]},
            "bar": {"foo": [9.0], "bar": [16.0]},
        }
        with mock.patch(
            "ax.adapter.torch.TorchAdapter.predict", return_value=(pred_mean, pred_cov)
        ) as mock_predict:
            (prediction,) = client.predict(points=[{"x1": 0.5}])
        mock_predict.assert_called_once()
        self.assertEqual(prediction, {"foo": (0.25, 2.0), "bar": (0.0, 4.0)})

    def test_json_storage(self) -> None:
        client = Client()

        # Experiment with relatively complicated search space
        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
                RangeParameterConfig(name="x2", parameter_type="int", bounds=(-1, 1)),
                ChoiceParameterConfig(
                    name="x3",
                    parameter_type="str",
                    values=["a", "b"],
                ),
                ChoiceParameterConfig(
                    name="x4",
                    parameter_type="int",
                    values=[1, 2, 3],
                    is_ordered=True,
                ),
                ChoiceParameterConfig(name="x5", parameter_type="int", values=[1]),
            ],
            name="foo",
        )

        # Relatively complicated optimization config
        client.configure_optimization(
            objective="foo + 2 * bar", outcome_constraints=["baz >= 0"]
        )

        # Specified generation strategy
        client.configure_generation_strategy(
            initialization_budget=2,
        )

        # Use the Client a bit
        _ = client.get_next_trials(max_trials=2)

        snapshot = client._to_json_snapshot()
        other_client = Client._from_json_snapshot(snapshot=snapshot)

        self.assertEqual(client._experiment, other_client._experiment)
        # Don't check for deep equality of GenerationStrategy since the other gs will
        # not have all its attributes initialized, but ensure they have the same repr
        self.assertEqual(
            str(client._generation_strategy), str(other_client._generation_strategy)
        )

    def test_sql_storage(self) -> None:
        init_test_engine_and_session_factory(force_init=True)
        client = Client(storage_config=StorageConfig())

        # Experiment with relatively complicated search space
        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
                RangeParameterConfig(name="x2", parameter_type="int", bounds=(-1, 1)),
                ChoiceParameterConfig(
                    name="x3",
                    parameter_type="str",
                    values=["a", "b"],
                ),
                ChoiceParameterConfig(
                    name="x4",
                    parameter_type="int",
                    values=[1, 2, 3],
                    is_ordered=True,
                ),
                ChoiceParameterConfig(name="x5", parameter_type="int", values=[1]),
            ],
            name="unique_test_experiment",
        )

        # Relatively complicated optimization config
        client.configure_optimization(
            objective="foo + 2 * bar", outcome_constraints=["baz >= 0"]
        )

        # Specified generation strategy
        client.configure_generation_strategy(
            initialization_budget=3,
        )

        other_client = Client.load_from_database(
            experiment_name="unique_test_experiment", storage_config=StorageConfig()
        )

        self.assertEqual(client._experiment, other_client._experiment)
        # Don't check for deep equality of GenerationStrategy since the other gs will
        # not have all its attributes initialized, but ensure they have the same repr
        self.assertEqual(
            str(client._generation_strategy), str(other_client._generation_strategy)
        )

    def test_overwrite_metric(self) -> None:
        client = Client()

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(0, 1))
            ],
            name="foo_exp",
        )

        # Start with single-objective + outcome constraint
        client.configure_optimization(objective="foo", outcome_constraints=["bar >= 0"])

        # test 1: Replace the single-objective metric
        foo_metric = DummyMetric(name="foo")
        client.configure_metrics(metrics=[foo_metric])
        objective = none_throws(client._experiment.optimization_config).objective
        self.assertIsInstance(objective, Objective)
        self.assertIs(objective.metric, foo_metric)

        # test 2: Replace the outcome-constraint metric
        bar_metric = DummyMetric(name="bar")
        client.configure_metrics(metrics=[bar_metric])
        oc = none_throws(client._experiment.optimization_config).outcome_constraints[0]
        self.assertIs(oc.metric, bar_metric)

        # test 3: Add a tracking metric, then replace it by name
        baz_metric_1 = DummyMetric(name="baz")
        client.configure_metrics(metrics=[baz_metric_1])
        self.assertIn("baz", client._experiment._tracking_metrics)
        self.assertIs(client._experiment._tracking_metrics["baz"], baz_metric_1)

        baz_metric_2 = DummyMetric(name="baz")
        client.configure_metrics(metrics=[baz_metric_2])
        self.assertIs(client._experiment._tracking_metrics["baz"], baz_metric_2)

        # test 4: Metric name not present anywhere, should be added as tracking + warn
        quux_metric = DummyMetric(name="quux")
        with self.assertLogs(logger="ax.api.client", level="WARNING") as logs:
            client.configure_metrics(metrics=[quux_metric])
        self.assertIn("quux", client._experiment._tracking_metrics)
        self.assertIs(client._experiment._tracking_metrics["quux"], quux_metric)
        self.assertTrue(
            any("added as tracking metric" in msg for msg in logs.output),
            "Expected a warning that the metric was added as a tracking metric.",
        )

        # test 5: Replace inside a MultiObjective
        client.configure_optimization(objective="foo, qux")
        qux_metric_moo = DummyMetric(name="qux")
        client.configure_metrics(metrics=[qux_metric_moo])
        moo = assert_is_instance(
            none_throws(client._experiment.optimization_config).objective,
            MultiObjective,
        )
        self.assertIn(qux_metric_moo, moo.metrics)

        # test 6: Replace inside a ScalarizedObjective
        client.configure_optimization(objective="foo + qux")
        qux_metric_scalar = DummyMetric(name="qux")
        client.configure_metrics(metrics=[qux_metric_scalar])
        scalar = assert_is_instance(
            none_throws(client._experiment.optimization_config).objective,
            ScalarizedObjective,
        )
        self.assertIn(qux_metric_scalar, scalar.metrics)

    def test_configure_generation_strategy_with_simplify(self) -> None:
        client = Client()

        client.configure_experiment(
            parameters=[
                RangeParameterConfig(name="x1", parameter_type="float", bounds=(-1, 1)),
            ],
            name="foo",
        )

        # Test with no generation strategy
        client.configure_optimization(objective="foo")

        # Test with generation strategy
        client.configure_generation_strategy()
        self.assertFalse(
            client._generation_strategy._nodes[2]
            .generator_specs[0]
            .model_kwargs["acquisition_options"]["prune_irrelevant_parameters"]
        )
        client.configure_generation_strategy(simplify_parameter_changes=True)
        self.assertTrue(
            client._generation_strategy._nodes[2]
            .generator_specs[0]
            .model_kwargs["acquisition_options"]["prune_irrelevant_parameters"]
        )

    def test_configure_experiment_with_derived_parameter(self) -> None:
        # Setup: Create parameters including a derived parameter

        client = Client()
        x1 = RangeParameterConfig(name="x1", parameter_type="float", bounds=(0.0, 1.0))
        x2 = RangeParameterConfig(name="x2", parameter_type="float", bounds=(0.0, 1.0))
        derived = DerivedParameterConfig(
            name="x3",
            expression_str="1.0 - x1 - x2",
            parameter_type="float",
        )

        # Execute: Configure experiment with derived parameter
        client.configure_experiment(
            name="test_derived_param",
            parameters=[x1, x2, derived],
        )

        # Assert: Verify derived parameter is correctly configured
        experiment = client._experiment
        self.assertEqual(len(experiment.search_space.parameters), 3)
        from ax.core.parameter import DerivedParameter

        self.assertIsInstance(
            experiment.search_space.parameters["x3"], DerivedParameter
        )

    def test_configure_experiment_with_multiple_derived_parameters(self) -> None:
        # Setup: Create multiple derived parameters

        client = Client()
        x1 = RangeParameterConfig(name="x1", parameter_type="float", bounds=(0.0, 1.0))
        x2 = RangeParameterConfig(name="x2", parameter_type="float", bounds=(0.0, 1.0))
        derived1 = DerivedParameterConfig(
            name="sum_x1_x2",
            expression_str="x1 + x2",
            parameter_type="float",
        )
        derived2 = DerivedParameterConfig(
            name="complement",
            expression_str="1.0 - x1 - x2",
            parameter_type="float",
        )

        # Execute: Configure with multiple derived parameters
        client.configure_experiment(
            name="test_multiple_derived",
            parameters=[x1, x2, derived1, derived2],
        )

        # Assert: Verify all parameters exist
        experiment = client._experiment
        self.assertEqual(len(experiment.search_space.parameters), 4)
        from ax.core.parameter import DerivedParameter

        self.assertIsInstance(
            experiment.search_space.parameters["sum_x1_x2"], DerivedParameter
        )
        self.assertIsInstance(
            experiment.search_space.parameters["complement"], DerivedParameter
        )

    def test_get_next_trials_with_derived_parameters(self) -> None:
        # Setup: Configure experiment with derived parameter

        client = Client()
        x1 = RangeParameterConfig(name="x1", parameter_type="float", bounds=(0.0, 1.0))
        x2 = RangeParameterConfig(name="x2", parameter_type="float", bounds=(0.0, 1.0))
        derived = DerivedParameterConfig(
            name="x3",
            expression_str="1.0 - x1 - x2",
            parameter_type="float",
        )

        client.configure_experiment(
            name="test_trials_derived",
            parameters=[x1, x2, derived],
        )
        client.configure_optimization(objective="objective")

        # Execute: Generate trials
        trials = client.get_next_trials(max_trials=3)

        # Assert: Verify trials include derived parameter with correct values
        self.assertEqual(len(trials), 3)
        for trial_params in trials.values():
            self.assertIn("x1", trial_params)
            self.assertIn("x2", trial_params)
            self.assertIn("x3", trial_params)
            # Verify derived parameter is correctly computed
            # pyre-fixme[58]: Arithmetic operations on TParameterValue
            expected_x3 = 1.0 - trial_params["x1"] - trial_params["x2"]
            # pyre-fixme[6]: Type mismatch on assertAlmostEqual
            self.assertAlmostEqual(trial_params["x3"], expected_x3, places=6)

    def test_complete_trial_with_derived_parameters(self) -> None:
        # Setup: Configure experiment with derived parameter and generate trial

        client = Client()
        x1 = RangeParameterConfig(name="x1", parameter_type="float", bounds=(0.0, 1.0))
        x2 = RangeParameterConfig(name="x2", parameter_type="float", bounds=(0.0, 1.0))
        derived = DerivedParameterConfig(
            name="x_sum",
            expression_str="x1 + x2",
            parameter_type="float",
        )

        client.configure_experiment(
            name="test_complete_trial",
            parameters=[x1, x2, derived],
        )
        client.configure_optimization(objective="objective")
        trials = client.get_next_trials(max_trials=1)
        trial_index = list(trials.keys())[0]

        # Execute: Complete trial with data
        status = client.complete_trial(
            trial_index=trial_index, raw_data={"objective": 0.5}
        )

        # Assert: Trial completed successfully
        self.assertTrue(status.is_completed)

    def test_get_best_parameterization_with_derived_parameters(self) -> None:
        # Setup: Run experiment with derived parameters

        client = Client()
        x1 = RangeParameterConfig(name="x1", parameter_type="float", bounds=(0.0, 1.0))
        x2 = RangeParameterConfig(name="x2", parameter_type="float", bounds=(0.0, 1.0))
        derived = DerivedParameterConfig(
            name="x_sum",
            expression_str="x1 + x2",
            parameter_type="float",
        )

        client.configure_experiment(
            name="test_best_param",
            parameters=[x1, x2, derived],
        )
        client.configure_optimization(objective="loss")

        # Generate and complete trials
        for _ in range(3):
            trials = client.get_next_trials(max_trials=1)
            trial_index = list(trials.keys())[0]
            client.complete_trial(
                trial_index=trial_index, raw_data={"loss": float(trial_index)}
            )

        # Execute: Get best parameterization
        best_params, best_values, best_trial, best_arm = (
            client.get_best_parameterization(use_model_predictions=False)
        )

        # Assert: Best parameterization includes derived parameter
        self.assertIn("x1", best_params)
        self.assertIn("x2", best_params)
        self.assertIn("x_sum", best_params)
        self.assertIn("loss", best_values)

    def test_summarize_with_derived_parameters(self) -> None:
        # Setup: Run experiment with derived parameters

        client = Client()
        param1 = RangeParameterConfig(
            name="param1", parameter_type="float", bounds=(0.0, 1.0)
        )
        param2 = RangeParameterConfig(
            name="param2", parameter_type="float", bounds=(0.0, 1.0)
        )
        derived = DerivedParameterConfig(
            name="param_sum",
            expression_str="param1 + param2",
            parameter_type="float",
        )

        client.configure_experiment(
            name="test_summarize",
            parameters=[param1, param2, derived],
        )
        client.configure_optimization(objective="score")

        # Complete some trials
        for i in range(3):
            trials = client.get_next_trials(max_trials=1)
            trial_index = list(trials.keys())[0]
            client.complete_trial(
                trial_index=trial_index, raw_data={"score": float(i) * 0.1}
            )

        # Execute: Get summary
        summary_df = client.summarize()

        # Assert: Summary includes derived parameter
        self.assertIn("param1", summary_df.columns)
        self.assertIn("param2", summary_df.columns)
        self.assertIn("param_sum", summary_df.columns)
        self.assertEqual(len(summary_df), 3)


class DummyRunner(IRunner):
    @override
    def run_trial(
        self, trial_index: int, parameterization: TParameterization
    ) -> dict[str, Any]:
        return {}

    @override
    def poll_trial(
        self, trial_index: int, trial_metadata: Mapping[str, Any]
    ) -> TrialStatus:
        return TrialStatus.COMPLETED

    @override
    def stop_trial(
        self, trial_index: int, trial_metadata: Mapping[str, Any]
    ) -> dict[str, Any]: ...


class DummyMetric(IMetric):
    def fetch(
        self,
        trial_index: int,
        trial_metadata: Mapping[str, Any],
    ) -> tuple[int, float | tuple[float, float]]:
        return 0, 0.0
