# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import Mock

import numpy as np
import pandas as pd
from ax.benchmark.benchmark_metric import (
    _get_no_metadata_msg,
    BenchmarkMapMetric,
    BenchmarkMetric,
)
from ax.benchmark.benchmark_trial_metadata import BenchmarkTrialMetadata
from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.metric import MetricFetchE
from ax.core.trial import Trial
from ax.utils.common.result import Err
from ax.utils.common.testutils import TestCase
from ax.utils.testing.backend_simulator import BackendSimulator, BackendSimulatorOptions
from ax.utils.testing.core_stubs import get_experiment
from pyre_extensions import none_throws


def get_test_trial(
    map_data: bool = False, batch: bool = False, has_metadata: bool = True
) -> Trial | BatchTrial:
    experiment = get_experiment()

    arm1 = Arm(parameters={"w": 1.0, "x": 1, "y": "foo", "z": True}, name="0_0")
    arm2 = Arm(parameters={"w": 1.0, "x": 2, "y": "foo", "z": True}, name="0_1")

    if batch:
        trial = experiment.new_batch_trial()
        trial.add_arms_and_weights(arms=[arm1, arm2])
    else:
        trial = experiment.new_trial()
        trial.add_arm(arm=arm1)

    if not has_metadata:
        return trial

    dfs = {
        "test_metric1": pd.DataFrame(
            {
                "arm_name": ["0_0", "0_1"] if batch else ["0_0"],
                "metric_name": "test_metric1",
                "mean": [1.0, 2.5] if batch else [1.0],
                "sem": [0.1, 0.0] if batch else [0.1],
                "step": 0,
                "virtual runtime": 0,
                "trial_index": 0,
            }
        ),
        "test_metric2": pd.DataFrame(
            {
                "arm_name": ["0_0", "0_1"] if batch else ["0_0"],
                "metric_name": "test_metric2",
                "mean": [0.5, 1.5] if batch else [0.5],
                "sem": [0.1, 0.0] if batch else [0.1],
                "step": 0,
                "virtual runtime": 0,
                "trial_index": 0,
            }
        ),
    }

    if map_data:
        backend_simulator = BackendSimulator(
            options=BackendSimulatorOptions(
                max_concurrency=1, internal_clock=0, use_update_as_start_time=True
            ),
            verbose_logging=False,
        )
        n_steps = 3
        dfs = {
            k: pd.concat([df] * n_steps)
            .assign(step=np.repeat(np.arange(n_steps), 2 if batch else 1))
            .assign(**{"virtual runtime": lambda df: df["step"]})
            for k, df in dfs.items()
        }

        backend_simulator.run_trial(trial_index=trial.index, runtime=1)
        metadata = BenchmarkTrialMetadata(
            dfs=dfs,
            backend_simulator=backend_simulator,
        )
    else:
        metadata = BenchmarkTrialMetadata(dfs=dfs)

    trial.update_run_metadata({"benchmark_metadata": metadata})
    return trial


class BenchmarkMetricTest(TestCase):
    def setUp(self) -> None:
        self.outcome_names = ["test_metric1", "test_metric2"]
        self.metric1, self.metric2 = (
            BenchmarkMetric(name=name, lower_is_better=True)
            for name in self.outcome_names
        )
        self.map_metric1, self.map_metric2 = (
            BenchmarkMapMetric(name=name, lower_is_better=True)
            for name in self.outcome_names
        )

    def test_fetch_trial_data(self) -> None:
        with self.subTest("Error for multiple metrics in BenchmarkMetric"):
            trial = get_test_trial()
            map_trial = get_test_trial(map_data=True)
            trial.run_metadata["benchmark_metadata"] = map_trial.run_metadata[
                "benchmark_metadata"
            ]
            with self.assertRaisesRegex(
                ValueError, "Trial 0 has data from multiple time steps"
            ):
                self.metric1.fetch_trial_data(trial=trial)

        for map_data, metric in [(False, self.metric1), (True, self.map_metric1)]:
            with self.subTest(f"No-metadata error, map_data={map_data}"):
                trial = get_test_trial(has_metadata=False, map_data=map_data)
                result = metric.fetch_trial_data(trial=trial)
                self.assertIsInstance(result, Err)
                self.assertIsInstance(result.value, MetricFetchE)
                self.assertEqual(
                    result.value.message,
                    _get_no_metadata_msg(trial_index=trial.index),
                )

        trial = get_test_trial()
        with self.assertRaisesRegex(
            NotImplementedError,
            "Arguments {'foo'} are not supported in Benchmark",
        ):
            self.metric1.fetch_trial_data(trial, foo="bar")
        df1 = self.metric1.fetch_trial_data(trial=trial).value.df
        self.assertEqual(len(df1), 1)
        expected_results = {
            "arm_name": "0_0",
            "metric_name": self.outcome_names[0],
            "mean": 1.0,
            "sem": 0.1,
            "trial_index": 0,
        }
        self.assertDictEqual(df1.iloc[0].to_dict(), expected_results)
        df2 = self.metric2.fetch_trial_data(trial=trial).value.df
        self.assertEqual(len(df2), 1)
        expected_results = {
            "arm_name": "0_0",
            "metric_name": self.outcome_names[1],
            "mean": 0.5,
            "sem": 0.1,
            "trial_index": 0,
        }
        self.assertDictEqual(df2.iloc[0].to_dict(), expected_results)

    def test_fetch_trial_map_data(self) -> None:
        trial = get_test_trial(map_data=True)
        with self.assertRaisesRegex(
            NotImplementedError,
            "Arguments {'foo'} are not supported in Benchmark",
        ):
            self.map_metric1.fetch_trial_data(trial, foo="bar")

        map_df1 = self.map_metric1.fetch_trial_data(trial=trial).value.map_df
        self.assertEqual(len(map_df1), 1)
        expected_results = {
            "arm_name": "0_0",
            "metric_name": self.outcome_names[0],
            "mean": 1.0,
            "sem": 0.1,
            "trial_index": 0,
            "step": 0,
        }

        self.assertDictEqual(map_df1.iloc[0].to_dict(), expected_results)
        map_df2 = self.map_metric2.fetch_trial_data(trial=trial).value.map_df
        self.assertEqual(len(map_df2), 1)
        expected_results = {
            "arm_name": "0_0",
            "metric_name": self.outcome_names[1],
            "mean": 0.5,
            "sem": 0.1,
            "trial_index": 0,
            "step": 0,
        }
        self.assertDictEqual(map_df2.iloc[0].to_dict(), expected_results)

        backend_simulator = trial.run_metadata["benchmark_metadata"].backend_simulator
        self.assertEqual(backend_simulator.time, 0)
        sim_trial = none_throws(backend_simulator.get_sim_trial_by_index(trial.index))
        self.assertIn(sim_trial, backend_simulator._running)
        backend_simulator.update()
        self.assertEqual(backend_simulator.time, 1)
        self.assertIn(sim_trial, backend_simulator._completed)
        backend_simulator.update()
        self.assertIn(sim_trial, backend_simulator._completed)
        self.assertEqual(backend_simulator.time, 2)
        map_df1 = self.map_metric1.fetch_trial_data(trial=trial).value.map_df
        self.assertEqual(len(map_df1), 2)
        self.assertEqual(set(map_df1["step"].tolist()), {0, 1})

    def _test_fetch_trial_data_batch_trial(self, map_data: bool) -> None:
        if map_data:
            metric1, metric2 = self.map_metric1, self.map_metric2
        else:
            metric1, metric2 = self.metric1, self.metric2
        trial = get_test_trial(map_data=map_data, batch=True)
        df1 = metric1.fetch_trial_data(trial=trial).value.df
        self.assertEqual(len(df1), 2)
        expected = {
            "arm_name": ["0_0", "0_1"],
            "metric_name": ["test_metric1", "test_metric1"],
            "mean": [1.0, 2.5],
            "sem": [0.1, 0.0],
            "trial_index": [0, 0],
        }
        if map_data:
            for col in ["step", "virtual runtime"]:
                expected[col] = [0, 0]

        for col in df1.columns:
            self.assertEqual(df1[col].to_list(), expected[col], col)

        df2 = metric2.fetch_trial_data(trial=trial).value.df
        self.assertEqual(len(df2), 2)
        expected = {
            "arm_name": ["0_0", "0_1"],
            "metric_name": ["test_metric2", "test_metric2"],
            "mean": [0.5, 1.5],
            "sem": [0.1, 0.0],
            "trial_index": [0, 0],
        }
        if map_data:
            for col in ["step", "virtual runtime"]:
                expected[col] = [0, 0]
        for col in df2.columns:
            self.assertEqual(df2[col].to_list(), expected[col], col)

    def test_fetch_trial_data_batch_trial(self) -> None:
        self._test_fetch_trial_data_batch_trial(map_data=False)
        self._test_fetch_trial_data_batch_trial(map_data=True)

    def test_sim_trial_completes_in_future_raises(self) -> None:
        simulator = BackendSimulator()
        simulator.run_trial(trial_index=0, runtime=0)
        simulator.update()
        simulator.options.internal_clock = -1
        metadata = BenchmarkTrialMetadata(
            dfs={"test_metric": pd.DataFrame({"t": [3]})}, backend_simulator=simulator
        )
        trial = Mock(spec=Trial)
        trial.index = 0
        trial.run_metadata = {"benchmark_metadata": metadata}
        metric = BenchmarkMapMetric(name="test_metric", lower_is_better=True)
        with self.assertRaisesRegex(RuntimeError, "in the future"):
            metric.fetch_trial_data(trial=trial)

    def test_map_data_without_map_metric_raises(self) -> None:
        metadata = BenchmarkTrialMetadata(
            dfs={"test_metric": pd.DataFrame({"step": [0, 1]})},
        )
        trial = Mock(spec=Trial)
        trial.run_metadata = {"benchmark_metadata": metadata}
        metric = BenchmarkMetric(name="test_metric", lower_is_better=True)
        with self.assertRaisesRegex(ValueError, "data from multiple time steps"):
            metric.fetch_trial_data(trial=trial)

    def test_abandoned_arms_not_supported(self) -> None:
        trial = get_test_trial(batch=True)
        trial.mark_arm_abandoned("0_0")
        with self.assertRaisesRegex(
            NotImplementedError, "does not support abandoned arms"
        ):
            self.map_metric1.fetch_trial_data(trial=trial)
