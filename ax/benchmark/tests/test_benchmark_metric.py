# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.benchmark.benchmark_metric import BenchmarkMetric
from ax.benchmark.benchmark_trial_metadata import BenchmarkTrialMetadata
from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.trial import Trial
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment


def get_test_trial() -> Trial:
    experiment = get_experiment()
    trial = experiment.new_trial()
    arm = Arm(parameters={"w": 1.0, "x": 1, "y": "foo", "z": True}, name="0_0")
    outcome_names = ["test_metric1", "test_metric2"]

    trial.add_arm(arm)
    metadata = BenchmarkTrialMetadata(
        Ys={arm.name: [1.0, 0.5]},
        Ystds={arm.name: [0.1, 0.1]},
        outcome_names=outcome_names,
    )

    trial.update_run_metadata({"benchmark_metadata": metadata})
    return trial


def get_test_batch_trial() -> BatchTrial:
    experiment = get_experiment()
    trial = experiment.new_batch_trial()
    arm1 = Arm(parameters={"w": 1.0, "x": 1, "y": "foo", "z": True}, name="0_0")
    arm2 = Arm(parameters={"w": 1.0, "x": 2, "y": "foo", "z": True}, name="0_1")
    trial.add_arms_and_weights(arms=[arm1, arm2])

    Ys = {"0_0": [1.0, 0.5], "0_1": [2.5, 1.5]}
    Ystds = {"0_0": [0.1, 0.1], "0_1": [0.0, 0.0]}
    outcome_names = ["test_metric1", "test_metric2"]
    metadata = BenchmarkTrialMetadata(outcome_names=outcome_names, Ys=Ys, Ystds=Ystds)

    trial.update_run_metadata({"benchmark_metadata": metadata})
    return trial


class BenchmarkMetricTest(TestCase):
    def setUp(self) -> None:
        self.outcome_names = ["test_metric1", "test_metric2"]
        self.metric1, self.metric2 = (
            BenchmarkMetric(name=name, lower_is_better=True)
            for name in self.outcome_names
        )

    def test_fetch_trial_data(self) -> None:
        trial = get_test_trial()
        with self.assertRaisesRegex(
            NotImplementedError,
            "Arguments {'foo'} are not supported in BenchmarkMetric",
        ):
            self.metric1.fetch_trial_data(trial, foo="bar")
        df1 = self.metric1.fetch_trial_data(trial=trial).value.df  # pyre-ignore [16]
        self.assertEqual(len(df1), 1)
        expected_results = {
            "arm_name": "0_0",
            "metric_name": self.outcome_names[0],
            "mean": 1.0,
            "sem": 0.1,
            "trial_index": 0,
        }
        self.assertDictEqual(df1.iloc[0].to_dict(), expected_results)
        df2 = self.metric2.fetch_trial_data(trial=trial).value.df  # pyre-ignore [16]
        self.assertEqual(len(df2), 1)
        expected_results = {
            "arm_name": "0_0",
            "metric_name": self.outcome_names[1],
            "mean": 0.5,
            "sem": 0.1,
            "trial_index": 0,
        }
        self.assertDictEqual(df2.iloc[0].to_dict(), expected_results)

    def test_fetch_trial_data_batch_trial(self) -> None:
        metric1, metric2 = self.metric1, self.metric2
        trial = get_test_batch_trial()
        df1 = metric1.fetch_trial_data(trial=trial).value.df  # pyre-ignore [16]
        self.assertEqual(len(df1), 2)
        expected = {
            "arm_name": {0: "0_0", 1: "0_1"},
            "metric_name": {0: "test_metric1", 1: "test_metric1"},
            "mean": {0: 1.0, 1: 2.5},
            "sem": {0: 0.1, 1: 0.0},
            "trial_index": {0: 0, 1: 0},
        }
        self.assertDictEqual(df1.to_dict(), expected)
        df2 = metric2.fetch_trial_data(trial=trial).value.df  # pyre-ignore [16]
        self.assertEqual(len(df2), 2)
        expected = {
            "arm_name": {0: "0_0", 1: "0_1"},
            "metric_name": {0: "test_metric2", 1: "test_metric2"},
            "mean": {0: 0.5, 1: 1.5},
            "sem": {0: 0.1, 1: 0.0},
            "trial_index": {0: 0, 1: 0},
        }
        self.assertDictEqual(df2.to_dict(), expected)
