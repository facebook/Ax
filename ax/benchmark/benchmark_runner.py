# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import torch
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from ax.benchmark.benchmark_trial_metadata import BenchmarkTrialMetadata
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.runner import Runner
from ax.core.trial import Trial
from ax.core.types import TParamValue
from ax.exceptions.core import UnsupportedError
from ax.runners.simulated_backend import SimulatedBackendRunner
from ax.utils.common.serialization import TClassDecoderRegistry, TDecoderRegistry
from ax.utils.testing.backend_simulator import BackendSimulator, BackendSimulatorOptions
from pyre_extensions import assert_is_instance
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

    If ``trial_runtime_func`` and ``max_concurrency`` are both left as default,
    trials run serially and complete immediately. Otherwise, a
    ``SimulatedBackendRunner`` is constructed to track the status of trials.

    Args:
        test_function: A ``BenchmarkTestFunction`` from which to generate
            deterministic data before adding noise.
        noise_std: The standard deviation of the noise added to the data. Can be
            a list or dict to be per-metric.
        trial_runtime_func: A callable that takes a trial and returns its
            runtime, in simulated seconds. If `None`, each trial completes in
            one simulated second.
        max_concurrency: The maximum number of trials that can be running at a
            given time. Typically, this is ``max_pending_trials`` from the
            ``scheduler_options`` on the ``BenchmarkMethod``.
    """

    test_function: BenchmarkTestFunction
    noise_std: float | list[float] | dict[str, float] = 0.0
    trial_runtime_func: Callable[[BaseTrial], int] | None = None
    max_concurrency: int = 1
    simulated_backend_runner: SimulatedBackendRunner | None = field(init=False)

    def __post_init__(self) -> None:
        if self.max_concurrency > 1:
            simulator = BackendSimulator(
                options=BackendSimulatorOptions(
                    max_concurrency=self.max_concurrency,
                    # Always use virtual rather than real time for benchmarking
                    internal_clock=0,
                    use_update_as_start_time=False,
                ),
                verbose_logging=False,
            )
            self.simulated_backend_runner = SimulatedBackendRunner(
                simulator=simulator,
                sample_runtime_func=self.trial_runtime_func
                if self.trial_runtime_func is not None
                else lambda _: 1,
            )
        else:
            self.simulated_backend_runner = None

    @property
    def outcome_names(self) -> Sequence[str]:
        """The names of the outcomes."""
        return self.test_function.outcome_names

    def get_Y_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        """Evaluates the test problem.

        Returns:
            An `m`-dim tensor of ground truth (noiseless) evaluations.
        """
        return torch.atleast_1d(self.test_function.evaluate_true(params=params))

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

    def run(self, trial: BaseTrial) -> dict[str, BenchmarkTrialMetadata]:
        """Run the trial by evaluating its parameterization(s).

        Args:
            trial: The trial to evaluate.

        Returns:
            A dictionary {"benchmark_metadata": metadata}, where ``metadata`` is
            a ``BenchmarkTrialMetadata``.
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
                nlzd_arm_weights = {assert_is_instance(trial, Trial).arm: 1.0}
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

        metadata = BenchmarkTrialMetadata(
            Ys=Ys,
            Ystds=Ystds,
            outcome_names=self.outcome_names,
        )
        if self.simulated_backend_runner is not None:
            self.simulated_backend_runner.run(trial=trial)
        return {"benchmark_metadata": metadata}

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        if self.simulated_backend_runner is None:
            return {TrialStatus.COMPLETED: {t.index for t in trials}}
        return self.simulated_backend_runner.poll_trial_status(trials=trials)

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

    def stop(self, trial: BaseTrial, reason: str | None = None) -> dict[str, Any]:
        if self.simulated_backend_runner is None:
            raise UnsupportedError(
                "stop() is not supported for a `BenchmarkRunner` without a "
                "`simulated_backend_runner`, becauase trials complete "
                "immediately."
            )
        return self.simulated_backend_runner.stop(trial=trial, reason=reason)
