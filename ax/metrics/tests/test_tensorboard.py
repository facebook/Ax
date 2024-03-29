#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union
from unittest import mock

import numpy as np
import pandas as pd
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.tensorboard import TensorboardCurveMetric, TensorboardMetric
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_search_space, get_trial
from pyre_extensions import assert_is_instance
from tensorboard.backend.event_processing import event_multiplexer


@dataclass
class _TensorProto:
    double_val: List[float]


@dataclass
class _TensorEvent:
    step: int
    tensor_proto: _TensorProto


def _get_fake_multiplexer(
    fake_data: Sequence[float],
) -> event_multiplexer.EventMultiplexer:
    mul = event_multiplexer.EventMultiplexer()

    # pyre-ignore[8] Return fake tags when content is requested
    mul.PluginRunToTagToContent = lambda plugin: {".": {"loss": ""}}

    # pyre-ignore[8] Return fake data when tensors requested
    mul.Tensors = lambda run, tag: [
        _TensorEvent(step=i, tensor_proto=_TensorProto(double_val=[dat]))
        for i, dat in enumerate(fake_data)
    ]

    return mul


class TensorboardMetricTest(TestCase):
    def test_fetch_trial_data(self) -> None:
        fake_data = [8.0, 9.0, 2.0, 1.0]
        fake_multiplexer = _get_fake_multiplexer(fake_data=fake_data)

        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=fake_multiplexer,
        ):
            metric = TensorboardMetric(
                name="loss", tag="loss", lower_is_better=True, smoothing=0
            )
            trial = get_trial()

            result = metric.fetch_trial_data(trial=trial)

            df = assert_is_instance(result.unwrap(), MapData).map_df

            expected_df = pd.DataFrame(
                [
                    {
                        "arm_name": "0_0",
                        "metric_name": "loss",
                        "mean": fake_data[i],
                        "sem": float("nan"),
                        "trial_index": 0,
                        "step": float(i),
                    }
                    for i in range(len(fake_data))
                ]
            )

            self.assertTrue(df.equals(expected_df))

    def test_smoothing(self) -> None:
        fake_data = [8.0, 4.0, 2.0, 1.0]
        smoothing = 0.5
        smooth_data = pd.Series(fake_data).ewm(alpha=smoothing).mean().tolist()

        fake_multiplexer = _get_fake_multiplexer(fake_data=fake_data)

        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=fake_multiplexer,
        ):
            metric = TensorboardMetric(
                name="loss", tag="loss", lower_is_better=True, smoothing=smoothing
            )
            trial = get_trial()

            result = metric.fetch_trial_data(trial=trial)

            df = assert_is_instance(result.unwrap(), MapData).map_df

            expected_df = pd.DataFrame(
                [
                    {
                        "arm_name": "0_0",
                        "metric_name": "loss",
                        "mean": smooth_data[i],
                        "sem": float("nan"),
                        "trial_index": 0,
                        "step": float(i),
                    }
                    for i in range(len(fake_data))
                ]
            )

            self.assertTrue(df.equals(expected_df))

    def test_cumulative_best(self) -> None:
        fake_data = [4.0, 8.0, 2.0, 1.0]
        cummin_data = pd.Series(fake_data).cummin().tolist()

        fake_multiplexer = _get_fake_multiplexer(fake_data=fake_data)

        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=fake_multiplexer,
        ):
            metric = TensorboardMetric(
                name="loss",
                tag="loss",
                lower_is_better=True,
                cumulative_best=True,
                smoothing=0,
            )
            trial = get_trial()

            result = metric.fetch_trial_data(trial=trial)

            df = assert_is_instance(result.unwrap(), MapData).map_df

            expected_df = pd.DataFrame(
                [
                    {
                        "arm_name": "0_0",
                        "metric_name": "loss",
                        "mean": cummin_data[i],
                        "sem": float("nan"),
                        "trial_index": 0,
                        "step": float(i),
                    }
                    for i in range(len(fake_data))
                ]
            )

            self.assertTrue(df.equals(expected_df))


class TensorboardCurveMetricTest(TestCase):
    def test_tensorboard_curve_metric(self) -> None:
        def mock_get_tb_from_posix(
            path: str, tags: Optional[List[str]] = None
        ) -> Dict[str, pd.Series]:
            data = np.array([10, 3, 5, 2, 7, 1])
            return {"test_curve": pd.Series((int(path) + 1) * data)}

        mock_path = "ax.metrics.tensorboard.get_tb_from_posix"

        class FakeTensorboardCurveMetric(TensorboardCurveMetric):
            @classmethod
            def get_ids_from_trials(
                cls, trials: Iterable[BaseTrial]
            ) -> Dict[int, Union[int, str]]:
                result = {}
                for trial in trials:
                    result[trial.index] = trial.index
                return result

        with mock.patch(mock_path, side_effect=mock_get_tb_from_posix):
            # test simple
            experiment = Experiment(
                name="dummy_experiment",
                search_space=get_branin_search_space(),
                optimization_config=OptimizationConfig(
                    objective=Objective(
                        metric=FakeTensorboardCurveMetric(
                            name="test_metric",
                            curve_name="test_curve",
                            lower_is_better=True,
                            cumulative_best=False,
                        ),
                        minimize=True,
                    )
                ),
                runner=SyntheticRunner(),
            )
            for param in range(0, 2):
                trial = experiment.new_trial()
                trial.add_arm(Arm(parameters={"x1": float(param), "x2": 0.0}))
                trial.run()

            self.assertTrue(
                np.allclose(
                    # pyre-fixme[16]: `Data` has no attribute `map_df`.
                    experiment.fetch_data().map_df["mean"].to_numpy(),
                    np.array(
                        [10.0, 3.0, 5.0, 2.0, 7.0, 1.0, 20.0, 6.0, 10.0, 4.0, 14.0, 2.0]
                    ),
                )
            )

            # test cumulative best
            experiment = Experiment(
                name="dummy_experiment",
                search_space=get_branin_search_space(),
                optimization_config=OptimizationConfig(
                    objective=Objective(
                        metric=FakeTensorboardCurveMetric(
                            name="test_metric",
                            curve_name="test_curve",
                            lower_is_better=True,
                            cumulative_best=True,
                        ),
                        minimize=True,
                    )
                ),
                runner=SyntheticRunner(),
            )
            for param in range(0, 2):
                trial = experiment.new_trial()
                trial.add_arm(Arm(parameters={"x1": float(param), "x2": 0.0}))
                trial.run()

            self.assertTrue(
                np.allclose(
                    experiment.fetch_data().map_df["mean"].to_numpy(),
                    np.array(
                        [10.0, 3.0, 3.0, 2.0, 2.0, 1.0, 20.0, 6.0, 6.0, 4.0, 4.0, 2.0]
                    ),
                )
            )

            # test cumulative best (lower is worse)
            experiment = Experiment(
                name="dummy_experiment",
                search_space=get_branin_search_space(),
                optimization_config=OptimizationConfig(
                    objective=Objective(
                        metric=FakeTensorboardCurveMetric(
                            name="test_metric",
                            curve_name="test_curve",
                            lower_is_better=False,
                            cumulative_best=True,
                        ),
                        minimize=False,
                    )
                ),
                runner=SyntheticRunner(),
            )
            for param in range(0, 2):
                trial = experiment.new_trial()
                trial.add_arm(Arm(parameters={"x1": float(param), "x2": 0.0}))
                trial.run()

            self.assertTrue(
                np.allclose(
                    experiment.fetch_data().map_df["mean"].to_numpy(),
                    np.array(
                        [
                            10.0,
                            10.0,
                            10.0,
                            10.0,
                            10.0,
                            10.0,
                            20.0,
                            20.0,
                            20.0,
                            20.0,
                            20.0,
                            20.0,
                        ]
                    ),
                ),
            )

            # test smoothing
            experiment = Experiment(
                name="dummy_experiment",
                search_space=get_branin_search_space(),
                optimization_config=OptimizationConfig(
                    objective=Objective(
                        metric=FakeTensorboardCurveMetric(
                            name="test_metric",
                            curve_name="test_curve",
                            lower_is_better=True,
                            cumulative_best=False,
                            smoothing_window=3,
                        ),
                        minimize=True,
                    )
                ),
                runner=SyntheticRunner(),
            )
            for param in range(0, 2):
                trial = experiment.new_trial()
                trial.add_arm(Arm(parameters={"x1": float(param), "x2": 0.0}))
                trial.run()
            self.assertTrue(
                np.allclose(
                    experiment.fetch_data().map_df["mean"].to_numpy(),
                    np.array(
                        [
                            6.00000000,
                            6.00000000,
                            6.00000000,
                            3.33333333,
                            4.66666667,
                            3.33333333,
                            12.0,
                            12.0,
                            12.0,
                            6.66666667,
                            9.33333333,
                            6.66666667,
                        ]
                    ),
                )
            )
