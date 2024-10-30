# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, InitVar
from math import sqrt
from typing import Any

import numpy.typing as npt

import torch
from ax.benchmark.runners.botorch_test import ParamBasedTestProblem
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.runner import Runner
from ax.core.search_space import SearchSpaceDigest
from ax.core.trial import Trial
from ax.core.types import TParamValue
from ax.exceptions.core import UnsupportedError
from ax.utils.common.serialization import TClassDecoderRegistry, TDecoderRegistry

from ax.utils.common.typeutils import checked_cast
from torch import Tensor


@dataclass(kw_only=True)
class BenchmarkRunner(Runner):
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

    Args:
        outcome_names: The names of the outcomes returned by the problem.
        test_problem: A ``ParamBasedTestProblem`` from which to generate
            deterministic data before adding noise.
        noise_std: The standard deviation of the noise added to the data. Can be
            a list or dict to be per-metric.
        search_space_digest: Used to extract target fidelity and task.
    """

    outcome_names: list[str]
    test_problem: ParamBasedTestProblem
    noise_std: float | list[float] | dict[str, float] = 0.0
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
        """Evaluates the test problem.

        Returns:
            An `m`-dim tensor of ground truth (noiseless) evaluations.
        """
        return torch.atleast_1d(self.test_problem.evaluate_true(params=params))

    def evaluate_oracle(self, parameters: Mapping[str, TParamValue]) -> npt.NDArray:
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

    def get_noise_stds(self) -> dict[str, float]:
        noise_std = self.noise_std
        if isinstance(noise_std, float):
            return {name: noise_std for name in self.outcome_names}
        elif isinstance(noise_std, dict):
            if not set(noise_std.keys()) == set(self.outcome_names):
                raise ValueError(
                    "Noise std must have keys equal to outcome names if given as "
                    "a dict."
                )
            return noise_std
        # list of floats
        return dict(zip(self.outcome_names, noise_std, strict=True))

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

        noiseless = all(v == 0 for v in noise_stds.values())

        if not noiseless:
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
            noise_stds_tsr = torch.tensor(
                [noise_stds[metric_name] for metric_name in self.outcome_names],
                dtype=torch.double,
            )

        for arm in trial.arms:
            # Case where we do have a ground truth
            Y_true = self.get_Y_true(arm.parameters)
            if noiseless:
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
