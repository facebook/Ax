#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging

from logging import Logger
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Set, Union

import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.map_data import MapData, MapKeyInfo
from ax.core.map_metric import MapMetric
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.core.trial import Trial
from ax.metrics.curve import AbstractCurveMetric
from ax.utils.common.logger import get_logger
from ax.utils.common.result import Err, Ok
from pyre_extensions import assert_is_instance

logger: Logger = get_logger(__name__)

SMOOTHING_DEFAULT = 0.6  # Default in Tensorboard UI
RUN_METADATA_KEY = "tb_log_dir"

try:
    from tensorboard.backend.event_processing import (
        plugin_event_multiplexer as event_multiplexer,
    )
    from tensorboard.compat.proto import types_pb2

    logging.getLogger("tensorboard").setLevel(logging.CRITICAL)

    class TensorboardMetric(MapMetric):
        """A *new* `MapMetric` for getting Tensorboard metrics."""

        map_key_info: MapKeyInfo[float] = MapKeyInfo(key="step", default_value=0.0)

        def __init__(
            self,
            name: str,
            tag: str,
            lower_is_better: bool = True,
            smoothing: float = SMOOTHING_DEFAULT,
            cumulative_best: bool = False,
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
            """
            super().__init__(name=name, lower_is_better=lower_is_better)

            self.smoothing = smoothing
            self.tag = tag
            self.cumulative_best = cumulative_best

        @classmethod
        def is_available_while_running(cls) -> bool:
            return True

        def bulk_fetch_trial_data(
            self, trial: BaseTrial, metrics: List[Metric], **kwargs: Any
        ) -> Dict[str, MetricFetchResult]:
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

            if len(mul.PluginRunToTagToContent("scalars")) == 0:
                return {
                    metric.name: Err(
                        MetricFetchE(
                            message=(
                                "No 'scalar' data found for trial in multiplexer "
                                f"{mul=}"
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
                        for run_name, tb_metrics in mul.PluginRunToTagToContent(
                            "scalars"
                        ).items()
                        for tag in tb_metrics
                        if tag == metric.tag
                        for t in mul.Tensors(run_name, tag)
                    ]

                    df = (
                        pd.DataFrame(records)
                        # If a metric has multiple records for the same arm, metric, and
                        # step (sometimes caused by restarts, etc) take the mean
                        .groupby(["arm_name", "metric_name", self.map_key_info.key])
                        .mean()
                        .reset_index()
                    )

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

    class TensorboardCurveMetric(AbstractCurveMetric):
        """A `CurveMetric` for getting Tensorboard curves."""

        map_key_info: MapKeyInfo[float] = MapKeyInfo(key="steps", default_value=0.0)

        def get_curves_from_ids(
            self,
            ids: Iterable[Union[int, str]],
            names: Optional[Set[str]] = None,
        ) -> Dict[Union[int, str], Dict[str, pd.Series]]:
            """Get curve data from tensorboard logs.

            NOTE: If the ids are not simple paths/posix locations, subclass this metric
            and replace this method with an appropriate one that retrieves the log
            results.

            Args:
                ids: A list of string paths to tensorboard log directories.
                names: The names of the tags for which to fetch the curves.
                    If omitted, all tags are returned.

            Returns:
                A nested dictionary mapping ids (first level) and metric names (second
                level) to pandas Series of data.
            """
            return {idx: get_tb_from_posix(path=str(idx), tags=names) for idx in ids}

    def get_tb_from_posix(
        path: str,
        tags: Optional[Set[str]] = None,
    ) -> Dict[str, pd.Series]:
        r"""Get Tensorboard data from a posix path.

        Args:
            path: The posix path for the directory that contains the tensorboard logs.
            tags: The names of the tags for which to fetch the curves. If omitted,
                all tags are returned.
        Returns:
            A dictionary mapping tag names to pandas Series of data.
        """
        logger.debug(f"Reading TB logs from {path}.")
        mul = event_multiplexer.EventMultiplexer(max_reload_threads=20)
        mul.AddRunsFromDirectory(path, None)
        mul.Reload()
        scalar_dict = mul.PluginRunToTagToContent("scalars")

        raw_result = [
            {"tag": tag, "event": mul.Tensors(run, tag)}
            for run, run_dict in scalar_dict.items()
            for tag in run_dict
            if tags is None or tag in tags
        ]
        tb_run_data = {}
        for item in raw_result:
            latest_start_time = _get_latest_start_time(item["event"])
            steps = [e.step for e in item["event"] if e.wall_time >= latest_start_time]
            vals = [
                _get_event_value(e)
                for e in item["event"]
                if e.wall_time >= latest_start_time
            ]
            key = item["tag"]
            series = pd.Series(index=steps, data=vals).dropna()
            if key in tb_run_data:
                tb_run_data[key] = pd.concat(objs=[tb_run_data[key], series])
            else:
                tb_run_data[key] = series
        for key, series in tb_run_data.items():
            if any(series.index.duplicated()):
                # take average of repeated observations of the same "step"
                series = series.groupby(series.index).mean()
                logger.debug(
                    f"Found duplicate steps for tag {key}. "
                    "Removing duplicates by averaging."
                )
                tb_run_data[key] = series
        return tb_run_data

    # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
    #  `typing.List` to avoid runtime subscripting errors.
    def _get_latest_start_time(events: List) -> float:
        """In each directory, there may be previous training runs due to restarting
        training jobs.

        Args:
            events: A list of TensorEvents.

        Returns:
            The start time of the latest training run.
        """
        events.sort(key=lambda e: e.wall_time)
        start_time = events[0].wall_time
        for i in range(1, len(events)):
            # detect points in time where restarts occurred
            if events[i].step < events[i - 1].step:
                start_time = events[i].wall_time
        return start_time

    def _get_event_value(e: NamedTuple) -> float:
        r"""Helper function to check the dtype and then get the value
        stored in a TensorEvent."""
        tensor = e.tensor_proto  # pyre-ignore[16]
        if tensor.dtype == types_pb2.DT_FLOAT:
            return tensor.float_val[0]
        elif tensor.dtype == types_pb2.DT_DOUBLE:
            return tensor.double_val[0]
        elif tensor.dtype == types_pb2.DT_INT32:
            return tensor.int_val[0]
        else:
            raise ValueError(f"Tensorboard dtype {tensor.dtype} not supported.")

except ImportError:
    logger.warning(
        "tensorboard package not found. If you would like to use "
        "TensorboardCurveMetric, please install tensorboard."
    )
    pass
