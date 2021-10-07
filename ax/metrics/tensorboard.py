#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
from typing import Iterable, Dict, Optional, Union

import pandas as pd
from ax.metrics.curve import AbstractCurveMetric
from ax.utils.common.logger import get_logger


logger = get_logger(__name__)

RESULTS_KEY = "vis_metrics"


try:
    from tensorboard.backend.event_processing import (
        plugin_event_multiplexer as event_multiplexer,
    )

    class TensorboardCurveMetric(AbstractCurveMetric):
        """A `CurveMetric` for getting Tensorboard curves."""

        MAP_KEY = "steps"

        @classmethod
        def get_curves_from_ids(
            cls, ids: Iterable[Union[int, str]]
        ) -> Dict[Union[int, str], Dict[str, pd.Series]]:
            """Get curve data from tensorboard logs.

            Args:
                ids: A list of string paths to tensorboard log directories.

            NOTE: If the ids are not simple paths/posix locations, subclass this
                metric and replace this method with an appropriate one that
                retrieves the log results.
            """
            result = {}
            for id_ in ids:
                tb = get_tb_from_posix(str(id_))
                if tb is not None:
                    result[id_] = tb
            return result

    def get_tb_from_posix(path: str) -> Optional[Dict[str, pd.Series]]:
        r"""Get Tensorboard data from a posix path.

        Args:
            path: The posix path for the directory that contains the
                tensorboard logs.

        Returns:
            A dictionary mapping metric names to pandas Series of data.
            If the path does not exist, return None.
        """
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
            steps = [e.step for e in item["event"]]
            vals = list(
                itertools.chain.from_iterable(
                    [e.tensor_proto.float_val for e in item["event"]]
                )
            )
            key = item["tag"]
            tb_run_data[key] = pd.Series(index=steps, data=vals).dropna()
        return tb_run_data


except ImportError:
    logger.warning(
        "tensorboard package not found. If you would like to use "
        "TensorboardCurveMetric, please install tensorboard."
    )
    pass
