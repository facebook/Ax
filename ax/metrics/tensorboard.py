#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging

from logging import Logger
from typing import Dict, Iterable, List, NamedTuple, Union

import pandas as pd
from ax.core.map_data import MapKeyInfo
from ax.metrics.curve import AbstractCurveMetric
from ax.utils.common.logger import get_logger

logger: Logger = get_logger(__name__)

try:
    from tensorboard.backend.event_processing import (
        plugin_event_multiplexer as event_multiplexer,
    )
    from tensorboard.compat.proto import types_pb2

    logging.getLogger("tensorboard").setLevel(logging.CRITICAL)

    class TensorboardCurveMetric(AbstractCurveMetric):
        """A `CurveMetric` for getting Tensorboard curves."""

        # pyre-fixme[4]: Attribute must be annotated.
        MAP_KEY = MapKeyInfo(key="steps", default_value=0.0)

        @classmethod
        def get_curves_from_ids(
            cls, ids: Iterable[Union[int, str]]
        ) -> Dict[Union[int, str], Dict[str, pd.Series]]:
            """Get curve data from tensorboard logs.

            NOTE: If the ids are not simple paths/posix locations, subclass this metric
            and replace this method with an appropriate one that retrieves the log
            results.

            Args:
                ids: A list of string paths to tensorboard log directories.

            Returns:
                A dictionary mapping metric names to pandas Series of data.
            """
            return {idx: get_tb_from_posix(str(idx)) for idx in ids}

    def get_tb_from_posix(path: str) -> Dict[str, pd.Series]:
        r"""Get Tensorboard data from a posix path.

        Args:
            path: The posix path for the directory that contains the tensorboard logs.

        Returns:
            A dictionary mapping metric names to pandas Series of data.
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
            if any(series.index.duplicated()):  # pyre-ignore[16]
                # take average of repeated observations of the same "step"
                series = series.groupby(series.index).mean()  # pyre-ignore[16]
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
