# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.benchmark.benchmark_metric import BenchmarkMetric
from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.trial import Trial
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment


def get_test_trial() -> Trial:
    experiment = get_experiment()
    trial = experiment.new_trial()
    arm = Arm(parameters={"w": 1.0, "x": 1, "y": "foo", "z": True}, name="0_0")
    trial.add_arm(arm)
    trial.update_run_metadata(
        {
            "Ys": {"0_0": [1.0, 0.5]},
            "Ys_true": {"0_0": [1.1, 0.4]},
            "Ystds": {"0_0": [0.1, 0.1]},
            "outcome_names": ["test_metric1", "test_metric2"],
        }
    )
    return trial


def get_test_batch_trial() -> BatchTrial:
    experiment = get_experiment()
    trial = experiment.new_batch_trial()
    arm1 = Arm(parameters={"w": 1.0, "x": 1, "y": "foo", "z": True}, name="0_0")
    arm2 = Arm(parameters={"w": 1.0, "x": 2, "y": "foo", "z": True}, name="0_1")
    trial.add_arms_and_weights(arms=[arm1, arm2])
    trial.update_run_metadata(
        {
            "Ys": {"0_0": [1.0, 0.5], "0_1": [2.5, 1.5]},
            "Ys_true": {"0_0": [1.1, 0.4], "0_1": [2.5, 1.5]},
            "Ystds": {"0_0": [0.1, 0.1], "0_1": [0.0, 0.0]},
            "outcome_names": ["test_metric1", "test_metric2"],
        }
    )
    return trial


class BenchmarkMetricTest(TestCase):

    def test_fetch_trial_data(self) -> None:
        metric1 = BenchmarkMetric(name="test_metric1", lower_is_better=True)
        metric2 = BenchmarkMetric(name="test_metric2", lower_is_better=True)
        trial = get_test_trial()
        with self.assertRaisesRegex(
            NotImplementedError,
            "Arguments {'foo'} are not supported in BenchmarkMetric",
        ):
            metric1.fetch_trial_data(trial, foo="bar")
        df1 = metric1.fetch_trial_data(trial=trial).value.df  # pyre-ignore [16]
        self.assertEqual(len(df1), 1)
        self.assertDictEqual(
            df1.iloc[0].to_dict(),
            {
                "arm_name": "0_0",
                "metric_name": "test_metric1",
                "mean": 1.0,
                "sem": 0.1,
                "trial_index": 0,
            },
        )
        df2 = metric2.fetch_trial_data(trial=trial).value.df  # pyre-ignore [16]
        self.assertEqual(len(df2), 1)
        self.assertDictEqual(
            df2.iloc[0].to_dict(),
            {
                "arm_name": "0_0",
                "metric_name": "test_metric2",
                "mean": 0.5,
                "sem": 0.1,
                "trial_index": 0,
            },
        )

    def test_fetch_trial_data_batch_trial(self) -> None:
        metric1 = BenchmarkMetric(name="test_metric1", lower_is_better=True)
        metric2 = BenchmarkMetric(name="test_metric2", lower_is_better=True)
        trial = get_test_batch_trial()
        df1 = metric1.fetch_trial_data(trial=trial).value.df  # pyre-ignore [16]
        self.assertEqual(len(df1), 2)
        self.assertDictEqual(
            df1.to_dict(),
            {
                "arm_name": {0: "0_0", 1: "0_1"},
                "metric_name": {0: "test_metric1", 1: "test_metric1"},
                "mean": {0: 1.0, 1: 2.5},
                "sem": {0: 0.1, 1: 0.0},
                "trial_index": {0: 0, 1: 0},
            },
        )
        df2 = metric2.fetch_trial_data(trial=trial).value.df  # pyre-ignore [16]
        self.assertEqual(len(df2), 2)
        self.assertDictEqual(
            df2.to_dict(),
            {
                "arm_name": {0: "0_0", 1: "0_1"},
                "metric_name": {0: "test_metric2", 1: "test_metric2"},
                "mean": {0: 0.5, 1: 1.5},
                "sem": {0: 0.1, 1: 0.0},
                "trial_index": {0: 0, 1: 0},
            },
        )
