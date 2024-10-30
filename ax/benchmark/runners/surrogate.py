# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from ax.benchmark.runners.botorch_test import ParamBasedTestProblem
from ax.core.observation import ObservationFeatures
from ax.core.types import TParamValue
from ax.modelbridge.torch import TorchModelBridge
from ax.utils.common.base import Base
from ax.utils.common.equality import equality_typechecker
from botorch.utils.datasets import SupervisedDataset
from pyre_extensions import none_throws
from torch import Tensor


@dataclass(kw_only=True)
class SurrogateTestFunction(ParamBasedTestProblem):
    """
    Data-generating function for surrogate benchmark problems.

    Args:
        name: The name of the runner.
        outcome_names: Names of outcomes to return in `evaluate_true`, if the
            surrogate produces more outcomes than are needed.
        _surrogate: Either `None`, or a `TorchModelBridge` surrogate to use
            for generating observations. If `None`, `get_surrogate_and_datasets`
            must not be None and will be used to generate the surrogate when it
            is needed.
        _datasets: Either `None`, or the `SupervisedDataset`s used to fit
            the surrogate model. If `None`, `get_surrogate_and_datasets` must
            not be None and will be used to generate the datasets when they are
            needed.
        get_surrogate_and_datasets: Function that returns the surrogate and
            datasets, to allow for lazy construction. If
            `get_surrogate_and_datasets` is not provided, `surrogate` and
            `datasets` must be provided, and vice versa.
    """

    name: str
    outcome_names: list[str]
    _surrogate: TorchModelBridge | None = None
    _datasets: list[SupervisedDataset] | None = None
    get_surrogate_and_datasets: (
        None | Callable[[], tuple[TorchModelBridge, list[SupervisedDataset]]]
    ) = None

    def __post_init__(self) -> None:
        if self.get_surrogate_and_datasets is None and (
            self._surrogate is None or self._datasets is None
        ):
            raise ValueError(
                "If `get_surrogate_and_datasets` is None, `_surrogate` "
                "and `_datasets` must not be None, and vice versa."
            )

    def set_surrogate_and_datasets(self) -> None:
        self._surrogate, self._datasets = none_throws(self.get_surrogate_and_datasets)()

    @property
    def surrogate(self) -> TorchModelBridge:
        if self._surrogate is None:
            self.set_surrogate_and_datasets()
        return none_throws(self._surrogate)

    @property
    def datasets(self) -> list[SupervisedDataset]:
        if self._datasets is None:
            self.set_surrogate_and_datasets()
        return none_throws(self._datasets)

    def evaluate_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        # We're ignoring the uncertainty predictions of the surrogate model here and
        # use the mean predictions as the outcomes (before potentially adding noise)
        means, _ = self.surrogate.predict(
            # pyre-fixme[6]: params is a Mapping, but ObservationFeatures expects a Dict
            observation_features=[ObservationFeatures(params)]
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
