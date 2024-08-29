# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from typing import Any, Callable, Optional, Union

import torch
from ax.benchmark.runners.base import BenchmarkRunner
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.observation import ObservationFeatures
from ax.core.search_space import SearchSpace, SearchSpaceDigest
from ax.modelbridge.torch import TorchModelBridge
from ax.utils.common.base import Base
from ax.utils.common.equality import equality_typechecker
from ax.utils.common.serialization import TClassDecoderRegistry, TDecoderRegistry
from botorch.utils.datasets import SupervisedDataset
from pyre_extensions import assert_is_instance, none_throws
from torch import Tensor


class SurrogateRunner(BenchmarkRunner):
    def __init__(
        self,
        *,
        name: str,
        search_space: SearchSpace,
        outcome_names: list[str],
        surrogate: Optional[TorchModelBridge] = None,
        datasets: Optional[list[SupervisedDataset]] = None,
        noise_stds: Union[float, dict[str, float]] = 0.0,
        get_surrogate_and_datasets: Optional[
            Callable[[], tuple[TorchModelBridge, list[SupervisedDataset]]]
        ] = None,
        search_space_digest: SearchSpaceDigest | None = None,
    ) -> None:
        """Runner for surrogate benchmark problems.

        Args:
            name: The name of the runner.
            surrogate: The modular BoTorch model `Surrogate` to use for
                generating observations.
            search_space: The search space of the problem (used for
                parameter transforms).
            datasets: The data sets used to fit the surrogate model.
            outcome_names: The names of the outcomes of the Surrogate.
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
        if get_surrogate_and_datasets is None and (
            surrogate is None or datasets is None
        ):
            raise ValueError(
                "If get_surrogate_and_datasets is not provided, surrogate and "
                "datasets must be provided, and vice versa."
            )
        super().__init__(search_space_digest=search_space_digest)
        self.get_surrogate_and_datasets = get_surrogate_and_datasets
        self.name = name
        self._surrogate = surrogate
        self.outcome_names = outcome_names
        self._datasets = datasets
        self.search_space = search_space
        self.noise_stds = noise_stds
        self.statuses: dict[int, TrialStatus] = {}

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

    def get_noise_stds(self) -> Union[None, float, dict[str, float]]:
        return self.noise_stds

    def get_Y_true(self, arm: Arm) -> Tensor:
        # We're ignoring the uncertainty predictions of the surrogate model here and
        # use the mean predictions as the outcomes (before potentially adding noise)
        means, _ = self.surrogate.predict(
            observation_features=[ObservationFeatures(arm.parameters)]
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

    @classmethod
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def serialize_init_args(cls, obj: Any) -> dict[str, Any]:
        """Serialize the properties needed to initialize the runner.
        Used for storage.

        WARNING: Because of issues with consistently saving and loading BoTorch and
        GPyTorch modules the SurrogateRunner cannot be serialized at this time.
        At load time the runner will be replaced with a SyntheticRunner.
        """
        warnings.warn(
            "Because of issues with consistently saving and loading BoTorch and "
            f"GPyTorch modules, {cls.__name__} cannot be serialized at this time. "
            "At load time the runner will be replaced with a SyntheticRunner.",
            stacklevel=3,
        )
        return {}

    @classmethod
    def deserialize_init_args(
        cls,
        args: dict[str, Any],
        decoder_registry: Optional[TDecoderRegistry] = None,
        class_decoder_registry: Optional[TClassDecoderRegistry] = None,
    ) -> dict[str, Any]:
        return {}

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

        # Checking the whole datasets' equality here would be too expensive to be
        # worth it; just check names instead
        return self.name == other.name
