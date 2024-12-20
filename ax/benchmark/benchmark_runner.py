# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from ax.benchmark.benchmark_step_runtime_function import TBenchmarkStepRuntimeFunction

from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from ax.benchmark.benchmark_trial_metadata import BenchmarkTrialMetadata
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.runner import Runner
from ax.core.types import TParamValue
from ax.exceptions.core import UnsupportedError
from ax.runners.simulated_backend import SimulatedBackendRunner
from ax.utils.common.serialization import TClassDecoderRegistry, TDecoderRegistry
from ax.utils.testing.backend_simulator import BackendSimulator, BackendSimulatorOptions
from pyre_extensions import assert_is_instance


def _dict_of_arrays_to_df(
    Y_true_by_arm: Mapping[str, npt.NDArray],
    step_duration_by_arm: Mapping[str, float],
    outcome_names: Sequence[str],
) -> pd.DataFrame:
    """
    Return a DataFrame with columns
    ["metric_name", "arm_name", "Y_true", "step", and "virtual runtime"].

    When the trial produces MapData, the "step" column is 0, 1, 2, ...., and
    "virtual runtime" contains cumulative time for each element of the
    progression. When the trial does not produce MapData, the "step" column is
    just 0, and "virtual runtime" is the total runtime of the trial.

    Args:
        Y_true_by_arm: A mapping from arm name to a 2D arrays each with shape
            (len(outcome_names), n_steps).
        step_duration_by_arm: A mapping from arm name to a number representing
            the runtime of each step.
        outcome_names: The names of the outcomes; will be mapped to the first
            dimension of each array in ``Y_true_by_arm``.
    """
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "metric_name": outcome_name,
                    "arm_name": arm_name,
                    "Y_true": y_true[i, :],
                    "step": np.arange(y_true.shape[1], dtype=int),
                    "virtual runtime": np.arange(1, y_true.shape[1] + 1, dtype=int)
                    * step_duration_by_arm[arm_name],
                }
            )
            for i, outcome_name in enumerate(outcome_names)
            for arm_name, y_true in Y_true_by_arm.items()
        ],
        ignore_index=True,
    )
    return df


def _add_noise(
    df: pd.DataFrame,
    noise_stds: Mapping[str, float],
    arm_weights: Mapping[str, float] | None,
) -> pd.DataFrame:
    """
    For each ``Y_true`` in ``df``, with metric name ``metric_name`` and
    arm name ``arm_name``, add noise with standard deviation
    ``noise_stds[metric_name] / sqrt_nlzd_arm_weights[arm_name]``,
    where ``sqrt_nlzd_arm_weights = sqrt(arm_weights[arm_name] /
    sum(arm_weights.values())])``.

    Args:
        df: A DataFrame with columns including
            ["metric_name", "arm_name", "Y_true"].
        noise_stds: A mapping from metric name to what the standard
            deviation would be if one arm received the entire
            sample budget.
        arm_weights: Either ``None`` if there is only one ``Arm``, or a
            mapping from ``Arm`` name to the arm's allocation. Using arm
            weights will increase noise levels, since each ``Arm`` is
            assumed to receive a fraction of the total sample budget.

    Returns:
        The original ``df``, now with additional columns ["mean", "sem"].
    """
    noiseless = all(v == 0 for v in noise_stds.values())
    if not noiseless:
        noise_std_ser = df["metric_name"].map(noise_stds)
        if arm_weights is not None:
            nlzd_arm_weights_sqrt = {
                arm_name: sqrt(weight / sum(arm_weights.values()))
                for arm_name, weight in arm_weights.items()
            }
            arm_weights_ser = df["arm_name"].map(nlzd_arm_weights_sqrt)
            df["sem"] = noise_std_ser / arm_weights_ser

        else:
            df["sem"] = noise_std_ser

        df["mean"] = df["Y_true"] + np.random.normal(loc=0, scale=df["sem"])

    else:
        df["sem"] = 0.0
        df["mean"] = df["Y_true"]
    return df


def get_total_runtime(
    trial: BaseTrial,
    step_runtime_function: TBenchmarkStepRuntimeFunction | None,
    n_steps: int,
) -> float:
    """Get the total runtime of a trial."""
    # By default, each step takes 1 virtual second.
    if step_runtime_function is not None:
        max_step_runtime = max(
            (step_runtime_function(arm.parameters) for arm in trial.arms)
        )
    else:
        max_step_runtime = 1
    return n_steps * max_step_runtime


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

    If ``max_concurrency`` is left as default (1), trials run serially and
    complete immediately. Otherwise, a ``SimulatedBackendRunner`` is constructed
    to track the status of trials.

    Args:
        test_function: A ``BenchmarkTestFunction`` from which to generate
            deterministic data before adding noise.
        noise_std: The standard deviation of the noise added to the data. Can be
            a list or dict to be per-metric.
        step_runtime_function: A function that takes in parameters
            (in ``TParameterization`` format) and returns the runtime of a step.
        max_concurrency: The maximum number of trials that can be running at a
            given time. Typically, this is ``max_pending_trials`` from the
            ``scheduler_options`` on the ``BenchmarkMethod``.
    """

    test_function: BenchmarkTestFunction
    noise_std: float | Sequence[float] | Mapping[str, float] = 0.0
    step_runtime_function: TBenchmarkStepRuntimeFunction | None = None
    max_concurrency: int = 1
    simulated_backend_runner: SimulatedBackendRunner | None = field(init=False)

    def __post_init__(self) -> None:
        if self.max_concurrency > 1:
            simulator = BackendSimulator(
                options=BackendSimulatorOptions(
                    max_concurrency=self.max_concurrency,
                    # Always use virtual rather than real time for benchmarking
                    internal_clock=0,
                    use_update_as_start_time=True,
                ),
            )
            self.simulated_backend_runner = SimulatedBackendRunner(
                simulator=simulator,
                sample_runtime_func=lambda trial: get_total_runtime(
                    trial=trial,
                    step_runtime_function=self.step_runtime_function,
                    n_steps=self.test_function.n_steps,
                ),
            )
        else:
            self.simulated_backend_runner = None

    @property
    def outcome_names(self) -> Sequence[str]:
        """The names of the outcomes."""
        return self.test_function.outcome_names

    def get_Y_true(self, params: Mapping[str, TParamValue]) -> npt.NDArray:
        """Evaluates the test problem.

        Returns:
            An array of ground truth (noiseless) evaluations, with shape
            (len(outcome_names), n_intervals) if is_map is True, and
            (len(outcome_names), 1) otherwise.
        """
        result = np.atleast_1d(self.test_function.evaluate_true(params=params).numpy())
        if result.ndim == 1:
            return result[:, None]
        return result

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
        return dict(
            zip(self.outcome_names, assert_is_instance(noise_std, list), strict=True)
        )

    def run(self, trial: BaseTrial) -> dict[str, BenchmarkTrialMetadata]:
        """Run the trial by evaluating its parameterization(s).

        Args:
            trial: The trial to evaluate.

        Returns:
            A dictionary {"benchmark_metadata": metadata}, where ``metadata`` is
            a ``BenchmarkTrialMetadata``.
        """
        Y_true_by_arm = {
            arm.name: self.get_Y_true(arm.parameters) for arm in trial.arms
        }

        step_duration_by_arm = {
            arm.name: 1
            if self.step_runtime_function is None
            else self.step_runtime_function(arm.parameters)
            for arm in trial.arms
        }
        for arm_name, duration in step_duration_by_arm.items():
            if duration < 0:
                raise ValueError(
                    "Step duration must be non-negative for each arm. For arm "
                    f"{arm_name}, duration is {duration}."
                )

        df = _dict_of_arrays_to_df(
            Y_true_by_arm=Y_true_by_arm,
            step_duration_by_arm=step_duration_by_arm,
            outcome_names=self.outcome_names,
        )

        arm_weights = (
            {arm.name: w for arm, w in trial.arm_weights.items()}
            if isinstance(trial, BatchTrial)
            else None
        )

        df = _add_noise(
            df=df, noise_stds=self.get_noise_stds(), arm_weights=arm_weights
        )
        df["trial_index"] = trial.index
        df.drop(columns=["Y_true"], inplace=True)

        if self.simulated_backend_runner is not None:
            self.simulated_backend_runner.run(trial=trial)

        dfs = {
            outcome_name: df[df["metric_name"] == outcome_name]
            for outcome_name in self.outcome_names
        }

        metadata = BenchmarkTrialMetadata(
            dfs=dfs,
            backend_simulator=None
            if self.simulated_backend_runner is None
            else self.simulated_backend_runner.simulator,
        )
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
