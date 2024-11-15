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
    """

    outcome_names: Sequence[str]

    @abstractmethod
    def evaluate_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        """
        Evaluate noiselessly.

        Returns:
            A 2d tensor of shape (len(outcome_names), n_intervals).
            ``n_intervals`` is only relevant when using time-series data
            (``MapData``). Otherwise, it is 1.
        """
        ...
