# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from ax.core.types import TParamValue
from torch import Tensor


@dataclass(kw_only=True)
class BenchmarkTestFunction(ABC):
    """
    The basic Ax class for generating deterministic data to benchmark against.

    (Noise - if desired - is added by the runner.)
    """

    outcome_names: list[str]

    @abstractmethod
    def evaluate_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        """
        Evaluate noiselessly.

        Returns:
            1d tensor of shape (len(outcome_names),).
        """
        ...
