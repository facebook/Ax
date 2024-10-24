# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, InitVar
from math import sqrt
from typing import Any

import torch
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.runner import Runner
from ax.core.search_space import SearchSpaceDigest
from ax.core.trial import Trial
from ax.core.types import TParamValue
from ax.exceptions.core import UnsupportedError
from ax.utils.common.serialization import TClassDecoderRegistry, TDecoderRegistry

from ax.utils.common.typeutils import checked_cast
from numpy import ndarray
from torch import Tensor


@dataclass(kw_only=True)
class BenchmarkRunner(Runner, ABC):
    """
    A Runner that produces both observed and ground-truth values.

    Observed values equal ground-truth values plus noise, with the noise added
    according to the standard deviations returned by `get_noise_stds()`.

    This runner does require that every benchmark has a ground truth, which
    won't necessarily be true for real-world problems. Such problems fall into
    two categories:
        - If they are deterministic, they can be used with this runner by
          viewing them as noiseless problems where the observed values are the
          ground truth. The observed values will be used for tracking the
          progress of optimization.
        - If they are not deterministc, they are not supported. It is not
          conceptually clear how to benchmark such problems, so we decided to
          not over-engineer for that before such a use case arrives.
    """

    outcome_names: list[str]
    # pyre-fixme[16]: Pyre doesn't understand InitVars
    search_space_digest: InitVar[SearchSpaceDigest | None] = None
    target_fidelity_and_task: Mapping[str, float | int] = field(init=False)

    def __post_init__(self, search_space_digest: SearchSpaceDigest | None) -> None:
        if search_space_digest is not None:
            self.target_fidelity_and_task: dict[str, float | int] = {
                search_space_digest.feature_names[i]: target
                for i, target in search_space_digest.target_values.items()
            }
        else:
            self.target_fidelity_and_task = {}

    def get_Y_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        """
        Return the ground truth values for a given arm.

        Synthetic noise is added as part of the Runner's `run()` method.
        """
        ...

    # pyre-fixme[24]: Generic type `ndarray` expects 2 type parameters.
    def evaluate_oracle(self, parameters: Mapping[str, TParamValue]) -> ndarray:
        """
        Evaluate oracle metric values at a parameterization. In the base class,
        oracle values are underlying noiseless function values evaluated at the
        target task and fidelity (if applicable).

        This method can be customized for more complex setups based on different
        notions of what the "oracle" value should be. For example, with a
        preference-learned objective, the values might be true metrics evaluated
        at the true utility function (which would be unobserved in reality).
        """
        params = {**parameters, **self.target_fidelity_and_task}
        return self.get_Y_true(params=params).numpy()

    @abstractmethod
    def get_noise_stds(self) -> dict[str, float]:
        """
        Return the standard errors for the synthetic noise to be applied to the
        observed values.
        """
        ...

    def run(self, trial: BaseTrial) -> dict[str, Any]:
        """Run the trial by evaluating its parameterization(s).

        Args:
            trial: The trial to evaluate.

        Returns:
            A dictionary with the following keys:
                - Ys: A dict mapping arm names to lists of corresponding outcomes,
                    where the order of the outcomes is the same as in `outcome_names`.
                - Ystds: A dict mapping arm names to lists of corresponding outcome
                    noise standard deviations (possibly nan if the noise level is
                    unobserved), where the order of the outcomes is the same as in
                    `outcome_names`.
                - "outcome_names": A list of metric names.
        """
        Ys, Ystds = {}, {}
        noise_stds = self.get_noise_stds()

        if noise_stds is not None:
            # extract arm weights to adjust noise levels accordingly
            if isinstance(trial, BatchTrial):
                # normalize arm weights (we assume that the noise level is defined)
                # w.r.t. to a single arm allocated all of the sample budget
                nlzd_arm_weights = {
                    arm: weight / sum(trial.arm_weights.values())
                    for arm, weight in trial.arm_weights.items()
                }
            else:
                nlzd_arm_weights = {checked_cast(Trial, trial).arm: 1.0}
            # generate a tensor of noise levels that we'll reuse below
            if isinstance(noise_stds, float):
                noise_stds_tsr = torch.full(
                    (len(self.outcome_names),),
                    noise_stds,
                    dtype=torch.double,
                )
            else:
                noise_stds_tsr = torch.tensor(
                    [noise_stds[metric_name] for metric_name in self.outcome_names],
                    dtype=torch.double,
                )

        for arm in trial.arms:
            # Case where we do have a ground truth
            Y_true = self.get_Y_true(arm.parameters)
            if noise_stds is None:
                # No noise, so just return the true outcome.
                Ystds[arm.name] = [0.0] * len(Y_true)
                Ys[arm.name] = Y_true.tolist()
            else:
                # We can scale the noise std by the inverse of the relative sample
                # budget allocation to each arm. This works b/c (i) we assume that
                # observations per unit sample budget are i.i.d. and (ii) the
                # normalized weights sum to one.
                # pyre-fixme[61]: `nlzd_arm_weights` is undefined, or not always
                #  defined.
                std = noise_stds_tsr.to(Y_true) / sqrt(nlzd_arm_weights[arm])
                Ystds[arm.name] = std.tolist()
                Ys[arm.name] = (Y_true + std * torch.randn_like(Y_true)).tolist()

        run_metadata = {
            "Ys": Ys,
            "Ystds": Ystds,
            "outcome_names": self.outcome_names,
        }
        return run_metadata

    # This will need to be udpated once asynchronous benchmarks are supported.
    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        return {TrialStatus.COMPLETED: {t.index for t in trials}}

    @classmethod
    # pyre-fixme [2]: Parameter `obj` must have a type other than `Any``
    def serialize_init_args(cls, obj: Any) -> dict[str, Any]:
        """
        It is tricky to use SerializationMixin with instances that have Ax
        objects as attributes, as BenchmarkRunners do. Therefore, serialization
        is not supported.
        """
        raise UnsupportedError(
            "serialize_init_args is not a supported method for BenchmarkRunners."
        )

    @classmethod
    def deserialize_init_args(
        cls,
        args: dict[str, Any],
        decoder_registry: TDecoderRegistry | None = None,
        class_decoder_registry: TClassDecoderRegistry | None = None,
    ) -> dict[str, Any]:
        """
        It is tricky to use SerializationMixin with instances that have Ax
        objects as attributes, as BenchmarkRunners do. Therefore, serialization
        is not supported.
        """
        raise UnsupportedError(
            "deserialize_init_args is not a supported method for BenchmarkRunners."
        )
