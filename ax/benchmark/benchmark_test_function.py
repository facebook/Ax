# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ax.core.types import TParamValue
from torch import Tensor


@dataclass(kw_only=True)
class BenchmarkTestFunction(ABC):
    """
    The basic Ax class for generating deterministic data to benchmark against.

    (Noise - if desired - is added by the runner.)

    Args:
        outcome_names: Names of the outcomes.
        n_steps: Number of data points produced per metric and per evaluation. 1
            if data is not time-series. If data is time-series, this will
            eventually become the number of values on a `MapMetric` for
            evaluations that run to completion.
    """

    outcome_names: Sequence[str]
    n_steps: int = 1

    @abstractmethod
    def evaluate_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        """
        Evaluate noiselessly.

        Returns:
            A 2d tensor of shape (len(self.outcome_names), self.n_steps).
        """
        ...
