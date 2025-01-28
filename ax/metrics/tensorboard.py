#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging

from logging import Logger
from typing import Any

import numpy as np

import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.map_data import MapData, MapKeyInfo
from ax.core.map_metric import MapMetric
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.core.trial import Trial
from ax.utils.common.logger import get_logger
from ax.utils.common.result import Err, Ok
from pyre_extensions import assert_is_instance

logger: Logger = get_logger(__name__)

# Default in Tensorboard UI (https://fburl.com/workplace/1sq11640)
SMOOTHING_DEFAULT = 0
RUN_METADATA_KEY = "tb_log_dir"

try:
    from tensorboard.backend.event_processing import (
        plugin_event_multiplexer as event_multiplexer,
    )

    logging.getLogger("tensorboard").setLevel(logging.CRITICAL)

    class TensorboardMetric(MapMetric):
        """A *new* `MapMetric` for getting Tensorboard metrics."""

        map_key_info: MapKeyInfo[float] = MapKeyInfo(key="step", default_value=0.0)

        def __init__(
            self,
            name: str,
            tag: str,
            lower_is_better: bool | None = True,
            smoothing: float = SMOOTHING_DEFAULT,
            cumulative_best: bool = False,
            percentile: float | None = None,
        ) -> None:
            """
            Args:
                name: The name of the metric.
                tag: The name of the learning curve in the Tensorboard Scalars tab.
                lower_is_better: If True, lower curve values are considered better.
                smoothing: If > 0, apply exponential weighted mean to the curve. This
                    is the same postprocessing as the "smoothing" slider in the
                    Tensorboard UI.
                cumulative_best: If True, for each trial, apply cumulative best to
                    the curve (i.e., if lower is better, then we return a curve
                    representing the cumulative min of the raw curve).
                percentile: If not None, return the (rolling) percentile value
                    of the curve.
                    e.g. if the original curve is [0, 6, 4, 2] and percentile=0.5, then
                    the returned curve is [0, 3, 4, 3]. Rolling percentile is applied
                    after any potential smoothing or cumulative_best processing.
            """
            super().__init__(name=name, lower_is_better=lower_is_better)

            self.smoothing = smoothing
            self.tag = tag
            self.cumulative_best = cumulative_best
            self.percentile = percentile

        @classmethod
        def is_available_while_running(cls) -> bool:
            return True

        def bulk_fetch_trial_data(
            self, trial: BaseTrial, metrics: list[Metric], **kwargs: Any
        ) -> dict[str, MetricFetchResult]:
            """Fetch multiple metrics data for one trial, using instance attributes
            of the metrics.

            Returns Dict of metric_name => Result
            Default behavior calls `fetch_trial_data` for each metric. Subclasses should
            override this to perform trial data computation for multiple metrics.
            """
            tb_metrics = [
                assert_is_instance(metric, TensorboardMetric) for metric in metrics
            ]

            trial = assert_is_instance(trial, Trial)
            if trial.arm is None:
                raise ValueError("Trial must have arm set.")

            arm_name = trial.arm.name

            try:
                mul = self._get_event_multiplexer_for_trial(trial=trial)
            except Exception as e:
                return {
                    metric.name: Err(
                        MetricFetchE(
                            message=f"Failed to get event multiplexer for {trial=}",
                            exception=e,
                        )
                    )
                    for metric in tb_metrics
                }

            scalar_dict = mul.PluginRunToTagToContent("scalars")
            if len(scalar_dict) == 0:
                return {
                    metric.name: Err(
                        MetricFetchE(
                            message=(
                                "Tensorboard multiplexer is empty. This can happen if "
                                "TB data is not populated at the time of fetch. Check "
                                "the corresponding logs to confirm that Tensorboard "
                                "data is available."
                            ),
                            exception=None,
                        )
                    )
                    for metric in tb_metrics
                }

            res = {}
            for metric in tb_metrics:
                try:
                    records = [
                        {
                            "trial_index": trial.index,
                            "arm_name": arm_name,
                            "metric_name": metric.name,
                            self.map_key_info.key: t.step,
                            "mean": (
                                t.tensor_proto.double_val[0]
                                if t.tensor_proto.double_val
                                else t.tensor_proto.float_val[0]
                            ),
                            "sem": float("nan"),
                        }
                        for run_name, tags in scalar_dict.items()
                        for tag in tags
                        if tag == metric.tag
                        for t in mul.Tensors(run_name, tag)
                    ]

                    # If records is empty something has gone wrong: either the tag is
                    # not present on the multiplexer or the content referenced is empty
                    if len(records) == 0:
                        if metric.tag not in [
                            j for sub in scalar_dict.values() for j in sub
                        ]:
                            raise KeyError(
                                f"Tag {metric.tag} not found on multiplexer {mul=}. "
                                "Did you specify this tag exactly as it appears in "
                                "the TensorBoard UI's Scalars tab?"
                            )
                        else:
                            raise ValueError(
                                f"Found tag {metric.tag}, but no data found for it. Is "
                                "the curve empty in the TensorBoard UI?"
                            )

                    df = (
                        pd.DataFrame(records)
                        # If a metric has multiple records for the same arm, metric, and
                        # step (sometimes caused by restarts, etc) take the mean
                        .groupby(["arm_name", "metric_name", self.map_key_info.key])
                        .mean()
                        .reset_index()
                    )

                    # If there are any NaNs or Infs in the data, raise an Exception
                    if np.any(~np.isfinite(df["mean"])):
                        raise ValueError("Found NaNs or Infs in data")

                    # Apply per-metric post-processing
                    # Apply cumulative "best" (min if lower_is_better)
                    if metric.cumulative_best:
                        if metric.lower_is_better:
                            df["mean"] = df["mean"].cummin()
                        else:
                            df["mean"] = df["mean"].cummax()

                    # Apply smoothing
                    if metric.smoothing > 0:
                        df["mean"] = df["mean"].ewm(alpha=metric.smoothing).mean()

                    # Apply rolling percentile
                    if metric.percentile is not None:
                        df["mean"] = df["mean"].expanding().quantile(metric.percentile)

                    # Accumulate successfully extracted timeseries
                    res[metric.name] = Ok(
                        MapData(
                            df=df,
                            map_key_infos=[self.map_key_info],
                        )
                    )

                except Exception as e:
                    res[metric.name] = Err(
                        MetricFetchE(
                            message=f"Failed to fetch data for {metric.name}",
                            exception=e,
                        )
                    )

            self._clear_multiplexer_if_possible(multiplexer=mul)

            return res

        def fetch_trial_data(
            self, trial: BaseTrial, **kwargs: Any
        ) -> MetricFetchResult:
            """Fetch data for one trial."""

            return self.bulk_fetch_trial_data(trial=trial, metrics=[self], **kwargs)[
                self.name
            ]

        def _get_event_multiplexer_for_trial(
            self, trial: BaseTrial
        ) -> event_multiplexer.EventMultiplexer:
            """Get an event multiplexer with the logs for a given trial."""

            mul = event_multiplexer.EventMultiplexer(max_reload_threads=20)
            mul.AddRunsFromDirectory(trial.run_metadata[RUN_METADATA_KEY], None)
            mul.Reload()

            return mul

        def _clear_multiplexer_if_possible(
            self, multiplexer: event_multiplexer.EventMultiplexer
        ) -> None:
            """
            Clear the multiplexer of all data. This is a no-op here, but for some
            Multiplexers which may implement a clearing method this method can be
            important for managing memory consumption.
            """
            pass

except ImportError:
    logger.warning(
        "tensorboard package not found. If you would like to use "
        "TensorboardMetric, please install tensorboard."
    )
    pass
