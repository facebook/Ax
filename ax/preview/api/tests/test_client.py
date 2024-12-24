# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Mapping

import numpy as np

import pandas as pd
from ax.analysis.plotly.parallel_coordinates import ParallelCoordinatesPlot
from ax.core.base_trial import TrialStatus

from ax.core.experiment import Experiment
from ax.core.formatting_utils import DataType
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
from ax.early_stopping.strategies import PercentileEarlyStoppingStrategy
from ax.exceptions.core import UnsupportedError
from ax.preview.api.client import Client
from ax.preview.api.configs import (
    ChoiceParameterConfig,
    ExperimentConfig,
    GenerationStrategyConfig,
    OrchestrationConfig,
    ParameterType,
    RangeParameterConfig,
    StorageConfig,
)
from ax.preview.api.protocols.metric import IMetric
from ax.preview.api.protocols.runner import IRunner
from ax.preview.api.types import TParameterization
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
                default_data_type=DataType.MAP_DATA,
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
            client._experiment.optimization_config,
            OptimizationConfig(
                objective=Objective(metric=MapMetric(name="ne"), minimize=True),
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
            experiment_config=ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(0, 1)
                    )
                ],
                name="foo",
            )
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

        # Test replacing a tracking metric
        client.configure_optimization(
            objective="foo",
        )
        client._experiment.add_tracking_metric(metric=MapMetric("custom"))
        client.configure_metrics(metrics=[custom_metric])

        self.assertEqual(
            custom_metric,
            client._experiment.tracking_metrics[0],
        )

        # Test adding a tracking metric
        client = Client()  # Start a fresh Client
        client.configure_experiment(
            experiment_config=ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(0, 1)
                    )
                ],
                name="foo",
            )
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
            client.get_next_trials()

        client.configure_experiment(
            ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                    RangeParameterConfig(
                        name="x2", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                ],
                name="foo",
            )
        )

        with self.assertRaisesRegex(UnsupportedError, "OptimizationConfig not set"):
            client.get_next_trials()

        client.configure_optimization(objective="foo")
        client.configure_generation_strategy(
            generation_strategy_config=GenerationStrategyConfig(
                # Set this to a large number so test runs fast
                initialization_budget=999,
            )
        )

        # Test can generate one trial
        trials = client.get_next_trials()
        self.assertEqual(len(trials), 1)
        self.assertEqual({*trials[0].keys()}, {"x1", "x2"})
        for parameter in ["x1", "x2"]:
            value = assert_is_instance(trials[0][parameter], float)
            self.assertGreaterEqual(value, -1.0)
            self.assertLessEqual(value, 1.0)

        # Test can generate multiple trials
        trials = client.get_next_trials(maximum_trials=2)
        self.assertEqual(len(trials), 2)

        # Test respects fixed features
        trials = client.get_next_trials(maximum_trials=1, fixed_parameters={"x1": 0.5})
        value = assert_is_instance(trials[3]["x1"], float)
        self.assertEqual(value, 0.5)

    def test_attach_data(self) -> None:
        client = Client()

        with self.assertRaisesRegex(AssertionError, "Experiment not set"):
            client.attach_data(trial_index=0, raw_data={"foo": 1.0})

        client.configure_experiment(
            ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                ],
                name="foo",
            )
        )
        client.configure_optimization(objective="foo")

        # Vanilla case with no progression argument
        trial_index = [*client.get_next_trials(maximum_trials=1).keys()][0]
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
                        "arm_name": {0: "0_0"},
                        "metric_name": {0: "foo"},
                        "mean": {0: 1.0},
                        "sem": {0: np.nan},
                        "trial_index": {0: 0},
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
                        "arm_name": {0: "0_0", 1: "0_0"},
                        "metric_name": {0: "foo", 1: "foo"},
                        "mean": {0: 1.0, 1: 2.0},
                        "sem": {0: np.nan, 1: np.nan},
                        "trial_index": {0: 0, 1: 0},
                        "step": {0: np.nan, 1: 10.0},
                    }
                )
            ),
        )

        # With extra metrics
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
                        "arm_name": {0: "0_0", 1: "0_0", 2: "0_0"},
                        "metric_name": {0: "foo", 1: "foo", 2: "bar"},
                        "mean": {0: 2.0, 1: 1.0, 2: 2.0},
                        "sem": {0: np.nan, 1: np.nan, 2: np.nan},
                        "trial_index": {0: 0, 1: 0, 2: 0},
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
            ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                ],
                name="foo",
            )
        )
        client.configure_optimization(objective="foo", outcome_constraints=["bar >= 0"])

        # Vanilla case with no progression argument
        trial_index = [*client.get_next_trials(maximum_trials=1).keys()][0]
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
                        "arm_name": {0: "0_0", 1: "0_0"},
                        "metric_name": {0: "foo", 1: "bar"},
                        "mean": {0: 1.0, 1: 2.0},
                        "sem": {0: np.nan, 1: np.nan},
                        "trial_index": {0: 0, 1: 0},
                        "step": {0: np.nan, 1: np.nan},
                    }
                )
            ),
        )

        # With progression argument
        trial_index = [*client.get_next_trials(maximum_trials=1).keys()][0]
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
                        "arm_name": {0: "1_0", 1: "1_0"},
                        "metric_name": {0: "foo", 1: "bar"},
                        "mean": {0: 1.0, 1: 2.0},
                        "sem": {0: np.nan, 1: np.nan},
                        "trial_index": {0: 1, 1: 1},
                        "step": {0: 10.0, 1: 10.0},
                    }
                )
            ),
        )

        # With missing metrics
        trial_index = [*client.get_next_trials(maximum_trials=1).keys()][0]
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
                        "arm_name": {0: "2_0"},
                        "metric_name": {0: "foo"},
                        "mean": {0: 1.0},
                        "sem": {0: np.nan},
                        "trial_index": {0: 2},
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
            ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                ],
                name="foo",
            )
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
            ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                ],
                name="foo",
            )
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
            ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                ],
                name="foo",
            )
        )
        client.configure_optimization(objective="foo")

        trial_index = [*client.get_next_trials(maximum_trials=1).keys()][0]
        client.mark_trial_failed(trial_index=trial_index)
        self.assertEqual(
            client._experiment.trials[trial_index].status,
            TrialStatus.FAILED,
        )

    def test_mark_trial_abandoned(self) -> None:
        client = Client()

        client.configure_experiment(
            ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                ],
                name="foo",
            )
        )
        client.configure_optimization(objective="foo")

        trial_index = [*client.get_next_trials(maximum_trials=1).keys()][0]
        client.mark_trial_abandoned(trial_index=trial_index)
        self.assertEqual(
            client._experiment.trials[trial_index].status,
            TrialStatus.ABANDONED,
        )

    def test_mark_trial_early_stopped(self) -> None:
        client = Client()

        client.configure_experiment(
            ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                ],
                name="foo",
            )
        )
        client.configure_optimization(objective="foo")

        trial_index = [*client.get_next_trials(maximum_trials=1).keys()][0]
        client.mark_trial_early_stopped(
            trial_index=trial_index, raw_data={"foo": 0.0}, progression=1
        )
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
                        "arm_name": {0: "0_0"},
                        "metric_name": {0: "foo"},
                        "mean": {0: 0.0},
                        "sem": {0: np.nan},
                        "trial_index": {0: 0},
                        "step": {0: 1.0},
                    }
                )
            ),
        )

    def test_should_stop_trial_early(self) -> None:
        client = Client()

        client.configure_experiment(
            ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                ],
                name="foo",
            )
        )
        client.configure_optimization(objective="foo")

        client.set_early_stopping_strategy(
            early_stopping_strategy=PercentileEarlyStoppingStrategy(
                metric_names=["foo"]
            )
        )

        client.get_next_trials(maximum_trials=1)
        self.assertFalse(client.should_stop_trial_early(trial_index=0))

    def test_run_trials(self) -> None:
        client = Client()

        client.configure_experiment(
            ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                ],
                name="foo",
            )
        )
        client.configure_optimization(objective="foo")
        client.configure_metrics(metrics=[DummyMetric(name="foo")])
        client.configure_runner(runner=DummyRunner())

        client.run_trials(maximum_trials=4, options=OrchestrationConfig())

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
                        "arm_name": {0: "0_0", 1: "1_0", 2: "2_0", 3: "3_0"},
                        "metric_name": {0: "foo", 1: "foo", 2: "foo", 3: "foo"},
                        "mean": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                        "sem": {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan},
                        "trial_index": {0: 0, 1: 1, 2: 2, 3: 3},
                        "progression": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
                    }
                )
            ),
        )

    def test_get_next_trials_then_run_trials(self) -> None:
        client = Client()

        client.configure_experiment(
            ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                ],
                name="foo",
            )
        )
        client.configure_optimization(objective="foo")
        client.configure_metrics(metrics=[DummyMetric(name="foo")])
        client.configure_runner(runner=DummyRunner())

        # First use Client in ask-tell
        # Complete two trials
        for index, _parameters in client.get_next_trials(maximum_trials=2).items():
            client.complete_trial(trial_index=index, raw_data={"foo": 1.0})

        # Leave one trial RUNNING
        _ = client.get_next_trials(maximum_trials=1)

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
        client.run_trials(maximum_trials=2, options=OrchestrationConfig())

        # All trials should be COMPLETED
        self.assertEqual(
            len(client._experiment.trials_by_status[TrialStatus.COMPLETED]),
            5,
        )

    def test_compute_analyses(self) -> None:
        client = Client()

        client.configure_experiment(
            ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                ],
                name="foo",
            )
        )
        client.configure_optimization(objective="foo")
        client.configure_generation_strategy(
            generation_strategy_config=GenerationStrategyConfig()
        )

        with self.assertLogs(logger="ax.analysis", level="ERROR") as lg:
            cards = client.compute_analyses(analyses=[ParallelCoordinatesPlot()])

            self.assertEqual(len(cards), 1)
            self.assertEqual(cards[0].name, "ParallelCoordinatesPlot")
            self.assertEqual(cards[0].title, "ParallelCoordinatesPlot Error")
            self.assertEqual(
                cards[0].subtitle,
                f"An error occurred while computing {ParallelCoordinatesPlot()}",
            )
            self.assertIn("Traceback", cards[0].blob)
            self.assertTrue(
                any(
                    (
                        "Failed to compute ParallelCoordinatesPlot: "
                        "No data found for metric "
                    )
                    in msg
                    for msg in lg.output
                )
            )

        for trial_index, _ in client.get_next_trials(maximum_trials=1).items():
            client.complete_trial(trial_index=trial_index, raw_data={"foo": 1.0})

        cards = client.compute_analyses(analyses=[ParallelCoordinatesPlot()])

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].name, "ParallelCoordinatesPlot")

    @mock_botorch_optimize
    def test_get_best_parameterization(self) -> None:
        client = Client()

        client.configure_experiment(
            experiment_config=ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                ],
                name="foo",
            )
        )
        client.configure_optimization(objective="foo")
        # Set initialization_budget=3 so we can reach a predictive GenerationNode
        # quickly
        client.configure_generation_strategy(
            generation_strategy_config=GenerationStrategyConfig(initialization_budget=3)
        )

        with self.assertRaisesRegex(UnsupportedError, "No trials have been run yet"):
            client.get_best_parameterization()

        for _ in range(3):
            for index, parameters in client.get_next_trials(maximum_trials=1).items():
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
        for index, parameters in client.get_next_trials(maximum_trials=1).items():
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

    @mock_botorch_optimize
    def test_get_pareto_frontier(self) -> None:
        client = Client()

        client.configure_experiment(
            experiment_config=ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                ],
                name="foo",
            )
        )
        client.configure_optimization(objective="foo, bar")
        # Set initialization_budget=3 so we can reach a predictive GenerationNode
        # quickly
        client.configure_generation_strategy(
            generation_strategy_config=GenerationStrategyConfig(initialization_budget=3)
        )

        with self.assertRaisesRegex(UnsupportedError, "No trials have been run yet"):
            client.get_pareto_frontier()

        for _ in range(3):
            for index, parameters in client.get_next_trials(maximum_trials=1).items():
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
        for index, parameters in client.get_next_trials(maximum_trials=1).items():
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

    @mock_botorch_optimize
    def test_predict(self) -> None:
        client = Client()

        client.configure_experiment(
            experiment_config=ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                ],
                name="foo",
            )
        )
        client.configure_optimization(objective="foo", outcome_constraints=["bar >= 0"])
        # Set initialization_budget=3 so we can reach a predictive GenerationNode
        # quickly
        client.configure_generation_strategy(
            generation_strategy_config=GenerationStrategyConfig(initialization_budget=3)
        )

        with self.assertRaisesRegex(ValueError, "but search space has parameters"):
            client.predict(points=[{"x0": 0}])

        with self.assertRaisesRegex(UnsupportedError, "not predictive"):
            client.predict(points=[{"x1": 0}])

        client.configure_metrics(metrics=[DummyMetric(name="baz")])
        for _ in range(4):
            for index, parameters in client.get_next_trials(maximum_trials=1).items():
                client.complete_trial(
                    trial_index=index,
                    raw_data={
                        "foo": assert_is_instance(parameters["x1"], float) ** 2,
                        "bar": 0.0,
                    },
                )

        # Check we've predicted something for foo and bar but not baz (which is a
        # tracking metric)
        point = client.predict(points=[{"x1": 0.5}])
        self.assertEqual({*point[0].keys()}, {"foo", "bar"})

    def test_json_storage(self) -> None:
        client = Client()

        # Experiment with relatively complicated search space
        client.configure_experiment(
            experiment_config=ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                    RangeParameterConfig(
                        name="x2", parameter_type=ParameterType.INT, bounds=(-1, 1)
                    ),
                    ChoiceParameterConfig(
                        name="x3",
                        parameter_type=ParameterType.STRING,
                        values=["a", "b"],
                    ),
                    ChoiceParameterConfig(
                        name="x4",
                        parameter_type=ParameterType.INT,
                        values=[1, 2, 3],
                        is_ordered=True,
                    ),
                    ChoiceParameterConfig(
                        name="x5", parameter_type=ParameterType.INT, values=[1]
                    ),
                ],
                name="foo",
            )
        )

        # Relatively complicated optimization config
        client.configure_optimization(
            objective="foo + 2 * bar", outcome_constraints=["baz >= 0"]
        )

        # Specified generation strategy
        client.configure_generation_strategy(
            generation_strategy_config=GenerationStrategyConfig(
                initialization_budget=2,
            )
        )

        # Use the Client a bit
        _ = client.get_next_trials(maximum_trials=2)

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
            experiment_config=ExperimentConfig(
                parameters=[
                    RangeParameterConfig(
                        name="x1", parameter_type=ParameterType.FLOAT, bounds=(-1, 1)
                    ),
                    RangeParameterConfig(
                        name="x2", parameter_type=ParameterType.INT, bounds=(-1, 1)
                    ),
                    ChoiceParameterConfig(
                        name="x3",
                        parameter_type=ParameterType.STRING,
                        values=["a", "b"],
                    ),
                    ChoiceParameterConfig(
                        name="x4",
                        parameter_type=ParameterType.INT,
                        values=[1, 2, 3],
                        is_ordered=True,
                    ),
                    ChoiceParameterConfig(
                        name="x5", parameter_type=ParameterType.INT, values=[1]
                    ),
                ],
                name="unique_test_experiment",
            )
        )

        # Relatively complicated optimization config
        client.configure_optimization(
            objective="foo + 2 * bar", outcome_constraints=["baz >= 0"]
        )

        # Specified generation strategy
        client.configure_generation_strategy(
            generation_strategy_config=GenerationStrategyConfig(
                initialization_budget=3,
            )
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
