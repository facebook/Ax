# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from ax.adapter.torch import TorchAdapter
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from ax.core.observation import ObservationFeatures
from ax.core.types import TParamValue
from ax.utils.common.base import Base
from ax.utils.common.equality import equality_typechecker
from botorch.models.deterministic import (
    DeterministicModel,
    MatheronPathModel,
    PosteriorMeanModel,
)
from pyre_extensions import none_throws
from torch import Tensor

RANDOM_SURROGATE_TYPES: list[type[DeterministicModel]] = [MatheronPathModel]


@dataclass(kw_only=True)
class SurrogateTestFunction(BenchmarkTestFunction):
    """
    Data-generating function for surrogate benchmark problems.

    Args:
        name: The name of the runner.
        outcome_names: Names of outcomes to return in `evaluate_true`, if the
            surrogate produces more outcomes than are needed.
        _surrogate: Either `None`, or a `TorchAdapter` surrogate to use
            for generating observations. If `None`, `get_surrogate`
            must not be None and will be used to generate the surrogate when it
            is needed.
        get_surrogate: Function that returns the surrogate, to allow for lazy
            construction. If `get_surrogate` is not provided, `surrogate` must
            be provided and vice versa.
        surrogate_model_type: The type of surrogate model to use. We either pass
            in a type of deterministic model, (e.g. PosteriorMeanModel) or
            a function that returns a deterministic model
            (e.g. get_matheron_path_model).
    """

    name: str
    outcome_names: Sequence[str]
    _surrogate: TorchAdapter | None = None
    get_surrogate: None | Callable[[], TorchAdapter] = None
    surrogate_model_type: type[DeterministicModel] = PosteriorMeanModel
    seed: int = 0

    def __post_init__(self) -> None:
        if self.get_surrogate is None and self._surrogate is None:
            raise ValueError(
                "If `get_surrogate` is None, `_surrogate` must not be None, and"
                " vice versa."
            )

    def specify_surrogate_type(self) -> None:
        """Substitute the surrogate model for a deterministic model used for
        benchmarking."""
        # pyre-ignore[16]: `ax.generators.torch_base.TorchGenerator` has no attribute
        # `surrogate`.
        surrogate_model = none_throws(self._surrogate).generator.surrogate

        if isinstance(surrogate_model.model, DeterministicModel):
            return  # Already wrapped

        base_model = surrogate_model.model

        # Check if surrogate_model_type accepts a 'seed' argument
        if self.surrogate_model_type in RANDOM_SURROGATE_TYPES:
            # pyre-ignore[45]: Cannot instantiate abstract class `DeterministicModel`
            # with abstract method `forward`.
            wrapped_model = self.surrogate_model_type(base_model, seed=self.seed)
        else:
            # pyre-ignore[45]: Cannot instantiate abstract class `DeterministicModel`
            # with abstract method `forward`.
            wrapped_model = self.surrogate_model_type(base_model)

        surrogate_model._model = wrapped_model

    @property
    def surrogate(self) -> TorchAdapter:
        if self._surrogate is None:
            self._surrogate = none_throws(self.get_surrogate)()
        if not isinstance(
            # pyre-ignore[16]: `ax.generators.torch_base.TorchGenerator` has no
            # attribute `surrogate`.
            self._surrogate.generator.surrogate.model,
            DeterministicModel,
        ):
            self.specify_surrogate_type()
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
            dtype=torch.double,
        )

    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        if type(other) is not type(self):
            return False

        # Don't check surrogate, datasets, or callable
        return self.name == other.name
