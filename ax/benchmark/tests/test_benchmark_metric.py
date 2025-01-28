# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from itertools import product
from unittest.mock import Mock

import numpy as np
import pandas as pd
from ax.benchmark.benchmark_metric import (
    _get_no_metadata_msg,
    BenchmarkMapMetric,
    BenchmarkMapUnavailableWhileRunningMetric,
    BenchmarkMetric,
    BenchmarkMetricBase,
    BenchmarkTimeVaryingMetric,
)
from ax.benchmark.benchmark_trial_metadata import BenchmarkTrialMetadata
from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.map_metric import MapMetric
from ax.core.metric import MetricFetchE
from ax.core.trial import Trial
from ax.utils.common.result import Err
from ax.utils.common.testutils import TestCase
from ax.utils.testing.backend_simulator import BackendSimulator, BackendSimulatorOptions
from ax.utils.testing.core_stubs import get_experiment
from pyre_extensions import assert_is_instance, none_throws


def _get_one_step_df(
    batch: bool, metric_name: str, step: int, observe_noise_sd: bool
) -> pd.DataFrame:
    if observe_noise_sd:
        sem = [0.1, 0.0] if batch else [0.1]
    else:
        sem = [np.nan, np.nan] if batch else [np.nan]
    if metric_name == "test_metric1":
        return pd.DataFrame(
            {
                "arm_name": ["0_0", "0_1"] if batch else ["0_0"],
                "metric_name": metric_name,
                "mean": [1.0, 2.5] if batch else [1.0],
                "sem": sem,
                "trial_index": 0,
                "step": step,
                "virtual runtime": step,
            }
        )
    return pd.DataFrame(
        {
            "arm_name": ["0_0", "0_1"] if batch else ["0_0"],
            "metric_name": metric_name,
            "mean": [0.5, 1.5] if batch else [0.5],
            "sem": sem,
            "trial_index": 0,
            "step": step,
            "virtual runtime": step,
        }
    )


def get_test_trial(
    multiple_time_steps: bool = False,
    batch: bool = False,
    has_metadata: bool = True,
    has_simulator: bool = False,
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

    n_steps = 3 if multiple_time_steps else 1
    dfs = {
        name: pd.concat(
            (
                _get_one_step_df(
                    batch=batch, metric_name=name, step=i, observe_noise_sd=True
                )
                for i in range(n_steps)
            )
        )
        for name in ["test_metric1", "test_metric2"]
    }

    if has_simulator:
        backend_simulator = BackendSimulator(
            options=BackendSimulatorOptions(
                max_concurrency=1, internal_clock=0, use_update_as_start_time=True
            ),
            verbose_logging=False,
        )
        backend_simulator.run_trial(trial_index=trial.index, runtime=1)
    else:
        backend_simulator = None
    metadata = BenchmarkTrialMetadata(
        dfs=dfs,
        backend_simulator=backend_simulator,
    )

    trial.update_run_metadata({"benchmark_metadata": metadata})
    return trial


class TestBenchmarkMetric(TestCase):
    def setUp(self) -> None:
        self.outcome_names = ["test_metric1", "test_metric2"]
        self.metrics = {
            name: BenchmarkMetric(
                name=name, lower_is_better=True, observe_noise_sd=observe_noise_sd
            )
            for name, observe_noise_sd in zip(
                self.outcome_names, (False, True), strict=True
            )
        }
        self.tv_metrics = {
            name: BenchmarkTimeVaryingMetric(name=name, lower_is_better=True)
            for name in self.outcome_names
        }
        self.map_metrics = {
            name: BenchmarkMapMetric(name=name, lower_is_better=True)
            for name in self.outcome_names
        }
        self.map_unavailable_while_running_metrics = {
            name: BenchmarkMapUnavailableWhileRunningMetric(
                name=name, lower_is_better=True
            )
            for name in self.outcome_names
        }

    def test_available_while_running(self) -> None:
        self.assertFalse(self.metrics["test_metric1"].is_available_while_running())
        self.assertFalse(BenchmarkMetric.is_available_while_running())

        self.assertTrue(self.map_metrics["test_metric1"].is_available_while_running())
        self.assertTrue(BenchmarkMapMetric.is_available_while_running())

        self.assertTrue(self.tv_metrics["test_metric1"].is_available_while_running())
        self.assertTrue(BenchmarkTimeVaryingMetric.is_available_while_running())

        self.assertFalse(
            self.map_unavailable_while_running_metrics[
                "test_metric1"
            ].is_available_while_running()
        )
        self.assertFalse(
            BenchmarkMapUnavailableWhileRunningMetric.is_available_while_running()
        )

    def test_exceptions(self) -> None:
        for metric in [
            self.metrics["test_metric1"],
            self.map_metrics["test_metric1"],
            self.tv_metrics["test_metric1"],
            self.map_unavailable_while_running_metrics["test_metric1"],
        ]:
            with self.subTest(
                f"No-metadata error, metric class={metric.__class__.__name__}"
            ):
                trial = get_test_trial(has_metadata=False)
                result = metric.fetch_trial_data(trial=trial)
                self.assertIsInstance(result, Err)
                self.assertIsInstance(result.value, MetricFetchE)
                self.assertEqual(
                    result.value.message,
                    _get_no_metadata_msg(trial_index=trial.index),
                    msg=f"{metric.__class__.__name__}",
                )

            trial = get_test_trial()
            with self.subTest(
                f"Unsupported kwargs, metric class={metric.__class__.__name__}"
            ), self.assertRaisesRegex(
                NotImplementedError,
                "Arguments {'foo'} are not supported in Benchmark",
            ):
                metric.fetch_trial_data(trial, foo="bar")

        with self.subTest("Error for multiple metrics in BenchmarkMetric"):
            trial = get_test_trial()
            map_trial = get_test_trial(multiple_time_steps=True)
            trial.run_metadata["benchmark_metadata"] = map_trial.run_metadata[
                "benchmark_metadata"
            ]
            with self.assertRaisesRegex(
                ValueError, "Trial has data from multiple time steps"
            ):
                self.metrics["test_metric1"].fetch_trial_data(trial=trial)

    def _test_fetch_trial_data_one_time_step(
        self, batch: bool, metric: BenchmarkMetricBase
    ) -> None:
        trial = get_test_trial(batch=batch, has_simulator=True)
        df1 = assert_is_instance(metric.fetch_trial_data(trial=trial).value, Data).df
        self.assertEqual(len(df1), len(trial.arms))
        expected_results = _get_one_step_df(
            batch=batch,
            metric_name=metric.name,
            step=0,
            observe_noise_sd=metric.observe_noise_sd,
        ).drop(columns=["virtual runtime"])
        if not isinstance(metric, MapMetric):
            expected_results = expected_results.drop(columns=["step"])
        df1_dict = df1.to_dict()
        expected_dict = expected_results.to_dict()
        # Compare SEM separately since NaN equality fails.
        sem, expected_sem = df1_dict.pop("sem"), expected_dict.pop("sem")
        try:
            self.assertEqual(sem, expected_sem)
        except AssertionError:
            self.assertTrue(all(math.isnan(s) for s in sem.values()))
        self.assertDictEqual(df1_dict, expected_dict)

    def test_fetch_trial_data_one_time_step(self) -> None:
        for batch, metrics in product(
            [False, True],
            [
                self.metrics,
                self.map_metrics,
                self.tv_metrics,
                self.map_unavailable_while_running_metrics,
            ],
        ):
            metric = metrics["test_metric1"]
            with self.subTest(
                batch=batch,
                metric_cls=type(metric),
            ):
                self._test_fetch_trial_data_one_time_step(batch=batch, metric=metric)

    def _test_fetch_trial_multiple_time_steps_with_simulator(self, batch: bool) -> None:
        """
        Cases for fetching data with multiple time steps:
        - Metric is 'BenchmarkMetric' -> exception, tested below
        - Has simulator, metric is 'BenchmarkMapMetric' -> df grows with each step
        - No simulator, metric is 'BenchmarkMapMetric' -> all data present while
          running (but realistically it would never be RUNNING)
        - Has simulator, metric is 'BenchmarkTimeVaryingMetric' -> one step at
          a time, evolving as we take steps
        - No simulator, metric is 'BenchmarkTimeVaryingMetric' -> completes
            immediately and returns last step

        See table in benchmark_metric.py for more details.
        """
        metric_name = "test_metric1"

        for metric, has_simulator in product(
            [
                self.map_metrics[metric_name],
                self.tv_metrics[metric_name],
                self.map_unavailable_while_running_metrics[metric_name],
            ],
            [False, True],
        ):
            trial = get_test_trial(
                has_metadata=True,
                batch=batch,
                multiple_time_steps=True,
                has_simulator=has_simulator,
            )
            data = metric.fetch_trial_data(trial=trial).value
            df_or_map_df = data.map_df if isinstance(metric, MapMetric) else data.df
            returns_full_data = (not has_simulator) and isinstance(metric, MapMetric)
            self.assertEqual(
                len(df_or_map_df), len(trial.arms) * (3 if returns_full_data else 1)
            )
            drop_cols = ["virtual runtime"]
            if not isinstance(metric, MapMetric):
                drop_cols += ["step"]

            expected_df = _get_one_step_df(
                batch=batch,
                metric_name=metric_name,
                step=0,
                observe_noise_sd=metric.observe_noise_sd,
            ).drop(columns=drop_cols)
            if returns_full_data:
                self.assertEqual(
                    df_or_map_df[df_or_map_df["step"] == 0].to_dict(),
                    expected_df.to_dict(),
                )
            else:
                self.assertEqual(df_or_map_df.to_dict(), expected_df.to_dict())

            backend_simulator = trial.run_metadata[
                "benchmark_metadata"
            ].backend_simulator
            self.assertEqual(backend_simulator is None, not has_simulator)
            if backend_simulator is None:
                continue
            self.assertEqual(backend_simulator.time, 0)
            sim_trial = none_throws(
                backend_simulator.get_sim_trial_by_index(trial.index)
            )
            self.assertIn(sim_trial, backend_simulator._running)
            backend_simulator.update()
            self.assertEqual(backend_simulator.time, 1)
            self.assertIn(sim_trial, backend_simulator._completed)
            backend_simulator.update()
            self.assertIn(sim_trial, backend_simulator._completed)
            self.assertEqual(backend_simulator.time, 2)
            data = metric.fetch_trial_data(trial=trial).value
            if isinstance(metric, MapMetric):
                map_df = data.map_df
                self.assertEqual(len(map_df), 2 * len(trial.arms))
                self.assertEqual(set(map_df["step"].tolist()), {0, 1})
            df = data.df
            self.assertEqual(len(df), len(trial.arms))
            expected_df = _get_one_step_df(
                batch=batch,
                metric_name=metric_name,
                step=1,
                observe_noise_sd=metric.observe_noise_sd,
            ).drop(columns=drop_cols)
            self.assertEqual(df.reset_index(drop=True).to_dict(), expected_df.to_dict())

    def test_fetch_trial_multiple_time_steps_with_simulator(self) -> None:
        self._test_fetch_trial_multiple_time_steps_with_simulator(batch=False)
        self._test_fetch_trial_multiple_time_steps_with_simulator(batch=True)

    def test_sim_trial_completes_in_future_raises(self) -> None:
        simulator = BackendSimulator()
        simulator.run_trial(trial_index=0, runtime=0)
        simulator.update()
        simulator.options.internal_clock = -1
        metadata = BenchmarkTrialMetadata(
            dfs={"test_metric": pd.DataFrame({"t": [3], "step": 0})},
            backend_simulator=simulator,
        )
        trial = Mock(spec=Trial)
        trial.index = 0
        trial.run_metadata = {"benchmark_metadata": metadata}
        metric = BenchmarkMapMetric(name="test_metric", lower_is_better=True)
        with self.assertRaisesRegex(RuntimeError, "in the future"):
            metric.fetch_trial_data(trial=trial)

    def test_multiple_time_steps_with_BenchmarkMetric_raises(self) -> None:
        trial = get_test_trial(multiple_time_steps=True)
        metric = self.metrics["test_metric1"]
        with self.assertRaisesRegex(ValueError, "data from multiple time steps"):
            metric.fetch_trial_data(trial=trial)

    def test_abandoned_arms_not_supported(self) -> None:
        trial = get_test_trial(batch=True)
        trial.mark_arm_abandoned("0_0")
        with self.assertRaisesRegex(
            NotImplementedError, "does not support abandoned arms"
        ):
            self.map_metrics["test_metric1"].fetch_trial_data(trial=trial)
