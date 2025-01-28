#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Sequence
from dataclasses import dataclass
from unittest import mock

import numpy as np

import pandas as pd
from ax.core.map_data import MapData
from ax.core.metric import MetricFetchE
from ax.metrics.tensorboard import TensorboardMetric
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_trial
from pyre_extensions import assert_is_instance
from tensorboard.backend.event_processing import event_multiplexer


@dataclass
class _TensorProto:
    double_val: list[float]


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

    def test_fetch_trial_data_with_bad_data(self) -> None:
        nan_data = [1, 2, np.nan, 4]
        nan_multiplexer = _get_fake_multiplexer(fake_data=nan_data)

        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=nan_multiplexer,
        ):
            metric = TensorboardMetric(
                name="loss",
                tag="loss",
            )

            trial = get_trial()

            result = metric.fetch_trial_data(trial=trial)

            err = assert_is_instance(result.unwrap_err(), MetricFetchE)
            self.assertEqual(
                err.message,
                "Failed to fetch data for loss",
            )
            self.assertEqual(
                str(err.exception),
                "Found NaNs or Infs in data",
            )

        inf_data = [1, 2, np.inf, 4]
        inf_multiplexer = _get_fake_multiplexer(fake_data=inf_data)

        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=inf_multiplexer,
        ):
            metric = TensorboardMetric(
                name="loss",
                tag="loss",
            )

            trial = get_trial()

            result = metric.fetch_trial_data(trial=trial)

            err = assert_is_instance(result.unwrap_err(), MetricFetchE)
            self.assertEqual(
                err.message,
                "Failed to fetch data for loss",
            )
            self.assertEqual(
                str(err.exception),
                "Found NaNs or Infs in data",
            )

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

    def test_percentile(self) -> None:
        fake_data = [8.0, 4.0, 2.0, 1.0]
        percentile = 0.5
        percentile_data = pd.Series(fake_data).expanding().quantile(percentile)
        fake_multiplexer = _get_fake_multiplexer(fake_data=fake_data)
        trial = get_trial()

        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=fake_multiplexer,
        ):
            metric = TensorboardMetric(
                name="loss", tag="loss", lower_is_better=True, percentile=percentile
            )
            result = metric.fetch_trial_data(trial=trial)
        df = assert_is_instance(result.unwrap(), MapData).map_df
        expected_df = pd.DataFrame(
            [
                {
                    "arm_name": "0_0",
                    "metric_name": "loss",
                    "mean": percentile_data[i],
                    "sem": float("nan"),
                    "trial_index": 0,
                    "step": float(i),
                }
                for i in range(len(fake_data))
            ]
        )

        self.assertTrue(df.equals(expected_df))
