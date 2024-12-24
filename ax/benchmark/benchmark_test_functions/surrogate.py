# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from ax.core.observation import ObservationFeatures
from ax.core.types import TParamValue
from ax.modelbridge.torch import TorchModelBridge
from ax.utils.common.base import Base
from ax.utils.common.equality import equality_typechecker
from pyre_extensions import none_throws
from torch import Tensor


@dataclass(kw_only=True)
class SurrogateTestFunction(BenchmarkTestFunction):
    """
    Data-generating function for surrogate benchmark problems.

    Args:
        name: The name of the runner.
        outcome_names: Names of outcomes to return in `evaluate_true`, if the
            surrogate produces more outcomes than are needed.
        _surrogate: Either `None`, or a `TorchModelBridge` surrogate to use
            for generating observations. If `None`, `get_surrogate`
            must not be None and will be used to generate the surrogate when it
            is needed.
        get_surrogate: Function that returns the surrogate, to allow for lazy
            construction. If `get_surrogate` is not provided, `surrogate` must
            be provided and vice versa.
    """

    name: str
    outcome_names: Sequence[str]
    _surrogate: TorchModelBridge | None = None
    get_surrogate: None | Callable[[], TorchModelBridge] = None

    def __post_init__(self) -> None:
        if self.get_surrogate is None and self._surrogate is None:
            raise ValueError(
                "If `get_surrogate` is None, `_surrogate` must not be None, and"
                " vice versa."
            )

    @property
    def surrogate(self) -> TorchModelBridge:
        if self._surrogate is None:
            self._surrogate = none_throws(self.get_surrogate)()
        return none_throws(self._surrogate)

    def evaluate_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        # We're ignoring the uncertainty predictions of the surrogate model here and
        # use the mean predictions as the outcomes (before potentially adding noise)
        means, _ = self.surrogate.predict(
            # `dict` makes a copy so that parameters are not mutated
            observation_features=[ObservationFeatures(parameters=dict(params))]
        )
        means = [means[name][0] for name in self.outcome_names]
        return torch.tensor(
            means,
            device=self.surrogate.device,
            dtype=self.surrogate.dtype,
        )

    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        if type(other) is not type(self):
            return False

        # Don't check surrogate, datasets, or callable
        return self.name == other.name
