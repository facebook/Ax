# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Union

import torch
from ax.benchmark.runners.base import BenchmarkRunner
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.observation import ObservationFeatures
from ax.core.search_space import SearchSpace
from ax.modelbridge.torch import TorchModelBridge
from ax.utils.common.serialization import TClassDecoderRegistry, TDecoderRegistry
from botorch.utils.datasets import SupervisedDataset
from torch import Tensor


class SurrogateRunner(BenchmarkRunner):
    def __init__(
        self,
        name: str,
        surrogate: TorchModelBridge,
        datasets: List[SupervisedDataset],
        search_space: SearchSpace,
        outcome_names: List[str],
        noise_stds: Union[float, Dict[str, float]] = 0.0,
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
        """
        self.name = name
        self.surrogate = surrogate
        self._outcome_names = outcome_names
        self.datasets = datasets
        self.search_space = search_space
        self.noise_stds = noise_stds
        self.statuses: Dict[int, TrialStatus] = {}

    @property
    def outcome_names(self) -> List[str]:
        return self._outcome_names

    def get_noise_stds(self) -> Union[None, float, Dict[str, float]]:
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

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
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

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
        return {TrialStatus.COMPLETED: {t.index for t in trials}}

    @classmethod
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def serialize_init_args(cls, obj: Any) -> Dict[str, Any]:
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
        args: Dict[str, Any],
        decoder_registry: Optional[TDecoderRegistry] = None,
        class_decoder_registry: Optional[TClassDecoderRegistry] = None,
    ) -> Dict[str, Any]:
        return {}
