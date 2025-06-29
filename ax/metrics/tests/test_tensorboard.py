#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from collections.abc import Sequence
from dataclasses import dataclass
from unittest import mock

import numpy as np

import pandas as pd
from ax.core.map_data import MapData
from ax.core.metric import MetricFetchE
from ax.metrics.tensorboard import logger, TensorboardMetric
from ax.utils.common.result import Ok
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


def tb_smooth(scalars: list[float], weight: float) -> list[float]:
    """
    Debiased EMA implementation according to Tensorboard Github:
    https://fburl.com/6v6q8scg

    This function is used to test that the smoothing functaionality
    of pandas.series.ewm is equivalent to the smoothing functionality
    of Tensorboard.
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed


class TensorboardMetricTest(TestCase):
    def test_fetch_trial_data(self) -> None:
        fake_data = [8.0, 9.0, 2.0, 1.0]
        fake_multiplexer = _get_fake_multiplexer(fake_data=fake_data)
        metric = TensorboardMetric(
            name="loss", tag="loss", lower_is_better=True, smoothing=0
        )
        trial = get_trial()

        # This is mocked to avoid an xdb call
        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=fake_multiplexer,
        ):
            result = metric.fetch_trial_data(trial=trial)

        df = assert_is_instance(result.unwrap(), MapData).map_df

        expected_df = pd.DataFrame(
            [
                {
                    "trial_index": 0,
                    "arm_name": "0_0",
                    "metric_name": "loss",
                    "mean": fake_data[i],
                    "sem": float("nan"),
                    "step": float(i),
                }
                for i in range(len(fake_data))
            ]
        )

        self.assertTrue(df.equals(expected_df))

    def test_fetch_trial_data_with_bad_data(self) -> None:
        nan_data = [1, 2, np.nan, 4]
        nan_multiplexer = _get_fake_multiplexer(fake_data=nan_data)
        metric = TensorboardMetric(name="loss", tag="loss")

        trial = get_trial()
        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=nan_multiplexer,
        ), mock.patch.object(logger, "warning") as mock_warning:
            result = metric.fetch_trial_data(trial=trial)
        mock_warning.assert_called_once_with(
            "1 / 4 data points are NaNs or Infs. Filtering out non-finite values."
        )

        assert_is_instance(result, Ok)
        map_data = assert_is_instance(result.unwrap(), MapData)
        nan_array = np.array(nan_data)
        nan_array = nan_array[np.isfinite(nan_array)]
        self.assertTrue(
            np.array_equal(
                map_data.map_df["mean"].to_numpy(), nan_array, equal_nan=False
            )
        )

        # testing some inf data
        inf_data = [1, 2, np.inf, 4]
        inf_multiplexer = _get_fake_multiplexer(fake_data=inf_data)

        metric = TensorboardMetric(name="loss", tag="loss")

        trial = get_trial()
        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=inf_multiplexer,
        ):
            with mock.patch.object(logger, "warning") as mock_warning:
                result = metric.fetch_trial_data(trial=trial)
            mock_warning.assert_called_once_with(
                "1 / 4 data points are NaNs or Infs. Filtering out non-finite values."
            )

        assert_is_instance(result, Ok)
        map_data = assert_is_instance(result.unwrap(), MapData)
        inf_array = np.array(inf_data)
        inf_array = inf_array[np.isfinite(inf_array)]
        self.assertTrue(np.array_equal(map_data.map_df["mean"].to_numpy(), inf_array))

        # testing all non-finite data
        for non_finite_val in [np.nan, np.inf]:
            nf_data = [non_finite_val for _ in range(4)]
            nf_multiplexer = _get_fake_multiplexer(fake_data=nf_data)
            metric = TensorboardMetric(name="loss", tag="loss")

            trial = get_trial()
            with mock.patch.object(
                TensorboardMetric,
                "_get_event_multiplexer_for_trial",
                return_value=nf_multiplexer,
            ):
                result = metric.fetch_trial_data(trial=trial)

            err = assert_is_instance(result.unwrap_err(), MetricFetchE)
            self.assertEqual(err.message, "Failed to fetch data for loss")
            self.assertEqual(str(err.exception), "All values are NaNs or Infs.")

    def test_smoothing(self) -> None:
        fake_data = [8.0, 4.0, 2.0, 1.0]
        smoothing = 0.5
        smooth_data = tb_smooth(fake_data, smoothing)

        fake_multiplexer = _get_fake_multiplexer(fake_data=fake_data)
        metric = TensorboardMetric(
            name="loss", tag="loss", lower_is_better=True, smoothing=smoothing
        )
        trial = get_trial()

        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=fake_multiplexer,
        ):
            result = metric.fetch_trial_data(trial=trial)

        df = assert_is_instance(result.unwrap(), MapData).map_df

        expected_df = pd.DataFrame(
            [
                {
                    "trial_index": 0,
                    "arm_name": "0_0",
                    "metric_name": "loss",
                    "mean": smooth_data[i],
                    "sem": float("nan"),
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
        metric = TensorboardMetric(
            name="loss",
            tag="loss",
            lower_is_better=True,
            cumulative_best=True,
            smoothing=0,
        )
        trial = get_trial()

        with mock.patch.object(
            TensorboardMetric,
            "_get_event_multiplexer_for_trial",
            return_value=fake_multiplexer,
        ):
            result = metric.fetch_trial_data(trial=trial)

        df = assert_is_instance(result.unwrap(), MapData).map_df

        expected_df = pd.DataFrame(
            [
                {
                    "trial_index": 0,
                    "arm_name": "0_0",
                    "metric_name": "loss",
                    "mean": cummin_data[i],
                    "sem": float("nan"),
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
                    "trial_index": 0,
                    "arm_name": "0_0",
                    "metric_name": "loss",
                    "mean": percentile_data[i],
                    "sem": float("nan"),
                    "step": float(i),
                }
                for i in range(len(fake_data))
            ]
        )

        self.assertTrue(df.equals(expected_df))
