# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from ax.core.types import TParamValue


@runtime_checkable
class TBenchmarkStepRuntimeFunction(Protocol):
    def __call__(self, params: Mapping[str, TParamValue]) -> float:
        """
        Return the runtime for each step.

        Each step within an arm will take the same amount of time.
        """
        ...
