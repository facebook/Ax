# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import Any, Mapping

from ax.preview.api.protocols.utils import _APIMetric
from pyre_extensions import override


class IMetric(_APIMetric):
    """
    Metrics automate the process of fetching data from external systems. They are used
    in conjunction with Runners in the run_n_trials method to facilitate closed-loop
    experimentation.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    @override
    def fetch(
        self,
        trial_index: int,
        trial_metadata: Mapping[str, Any],
    ) -> tuple[int, float | tuple[float, float]]:
        """
        Given trial metadata (the mapping returned from IRunner.run), fetches
        readings for the metric.

        Readings are returned as a pair (progression, outcome), where progression is
        an integer representing the progression of the trial (e.g. number of epochs
        for a training job, timestamp for a time series, etc.), and outcome is either
        direct reading or a (mean, sem) pair for the metric.
        """
        ...
