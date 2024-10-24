# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from ax.benchmark.runners.base import BenchmarkRunner
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.observation import ObservationFeatures
from ax.core.search_space import SearchSpaceDigest
from ax.modelbridge.torch import TorchModelBridge
from ax.utils.common.base import Base
from ax.utils.common.equality import equality_typechecker
from botorch.utils.datasets import SupervisedDataset
from pyre_extensions import assert_is_instance, none_throws
from torch import Tensor


@dataclass
class SurrogateRunner(BenchmarkRunner):
    """Runner for surrogate benchmark problems.

    Args:
        name: The name of the runner.
        outcome_names: The names of the outcomes of the Surrogate.
        _surrogate: Either `None`, or a `TorchModelBridge` surrogate to use
            for generating observations. If `None`, `get_surrogate_and_datasets`
            must not be None and will be used to generate the surrogate when it
            is needed.
        _datasets: Either `None`, or the `SupervisedDataset`s used to fit
            the surrogate model. If `None`, `get_surrogate_and_datasets` must
            not be None and will be used to generate the datasets when they are
            needed.
        noise_stds: Noise standard deviations to add to the surrogate output(s).
            If a single float is provided, noise with that standard deviation
            is added to all outputs. Alternatively, a dictionary mapping outcome
            names to noise standard deviations can be provided to specify different
            noise levels for different outputs.
        get_surrogate_and_datasets: Function that returns the surrogate and
            datasets, to allow for lazy construction. If
            `get_surrogate_and_datasets` is not provided, `surrogate` and
            `datasets` must be provided, and vice versa.
        search_space_digest: Used to get the target task and fidelity at
            which the oracle is evaluated.
    """

    name: str
    _surrogate: TorchModelBridge | None = None
    _datasets: list[SupervisedDataset] | None = None
    noise_stds: float | dict[str, float] = 0.0
    get_surrogate_and_datasets: (
        None | Callable[[], tuple[TorchModelBridge, list[SupervisedDataset]]]
    ) = None
    statuses: dict[int, TrialStatus] = field(default_factory=dict)

    def __post_init__(self, search_space_digest: SearchSpaceDigest | None) -> None:
        super().__post_init__(search_space_digest=search_space_digest)
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

    def get_noise_stds(self) -> dict[str, float]:
        noise_std = self.noise_stds
        if isinstance(noise_std, float):
            return {name: noise_std for name in self.outcome_names}
        return noise_std

    # pyre-fixme[14]: Inconsistent override
    def get_Y_true(self, params: Mapping[str, float | int]) -> Tensor:
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

    def run(self, trial: BaseTrial) -> dict[str, Any]:
        """Run the trial by evaluating its parameterization(s) on the surrogate model.

        Note: This also sets the status of the trial to COMPLETED.

        Args:
            trial: The trial to evaluate.

        Returns:
            A dictionary with the following keys:
                - outcome_names: The names of the metrics being evaluated.
                - Ys: A dict mapping arm names to lists of corresponding outcomes,
                    where the order of the outcomes is the same as in `outcome_names`.
                - Ystds: A dict mapping arm names to lists of corresponding outcome
                    noise standard deviations (possibly nan if the noise level is
                    unobserved), where the order of the outcomes is the same as in
                    `outcome_names`.
                - Ys_true: A dict mapping arm names to lists of corresponding ground
                    truth outcomes, where the order of the outcomes is the same as
                    in `outcome_names`.
        """
        self.statuses[trial.index] = TrialStatus.COMPLETED
        run_metadata = super().run(trial=trial)
        run_metadata["outcome_names"] = self.outcome_names
        return run_metadata

    @property
    def is_noiseless(self) -> bool:
        if self.noise_stds is None:
            return True
        if isinstance(self.noise_stds, float):
            return self.noise_stds == 0.0
        return all(
            std == 0.0 for std in assert_is_instance(self.noise_stds, dict).values()
        )

    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        if type(other) is not type(self):
            return False

        # Don't check surrogate, datasets, or callable
        return (
            (self.name == other.name)
            and (self.outcome_names == other.outcome_names)
            and (self.noise_stds == other.noise_stds)
            # pyre-fixme[16]: `SurrogateRunner` has no attribute `search_space_digest`.
            and (self.search_space_digest == other.search_space_digest)
        )
