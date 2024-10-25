# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice

import torch
from ax.benchmark.runners.base import BenchmarkRunner
from ax.core.types import TParamValue
from botorch.test_functions.synthetic import BaseTestProblem, ConstrainedBaseTestProblem
from botorch.utils.transforms import normalize, unnormalize
from torch import Tensor


@dataclass(kw_only=True)
class ParamBasedTestProblem(ABC):
    """
    The basic Ax class for generating deterministic data to benchmark against.

    (Noise - if desired - is added by the runner.)
    """

    @abstractmethod
    def evaluate_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        """
        Evaluate noiselessly.

        Returns:
            1d tensor of shape (num_outcomes,).
        """
        ...


@dataclass(kw_only=True)
class BoTorchTestProblem(ParamBasedTestProblem):
    """
    Class for generating data from a BoTorch ``BaseTestProblem``.

    Args:
        botorch_problem: The BoTorch ``BaseTestProblem``.
        modified_bounds: The bounds that are used by the Ax search space
            while optimizing the problem. If different from the bounds of the
            test problem, we project the parameters into the test problem
            bounds before evaluating the test problem.
            For example, if the test problem is defined on [0, 1] but the Ax
            search space is integers in [0, 10], an Ax parameter value of
            5 will correspond to 0.5 while evaluating the test problem.
            If modified bounds are not provided, the test problem will be
            evaluated using the raw parameter values.
    """

    botorch_problem: BaseTestProblem
    modified_bounds: list[tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if (
            self.botorch_problem.noise_std is not None
            or getattr(self.botorch_problem, "constraint_noise_std", None) is not None
        ):
            raise ValueError(
                "noise should be set on the `BenchmarkRunner`, not the test function."
            )
        self.botorch_problem = self.botorch_problem.to(dtype=torch.double)

    def tensorize_params(self, params: Mapping[str, int | float]) -> torch.Tensor:
        X = torch.tensor(
            list(islice(params.values(), self.botorch_problem.dim)),
            dtype=torch.double,
        )

        if self.modified_bounds is not None:
            # Normalize from modified bounds to unit cube.
            unit_X = normalize(
                X, torch.tensor(self.modified_bounds, dtype=torch.double).T
            )
            # Unnormalize from unit cube to original problem bounds.
            X = unnormalize(unit_X, self.botorch_problem.bounds)
        return X

    # pyre-fixme [14]: inconsistent override
    def evaluate_true(self, params: Mapping[str, float | int]) -> torch.Tensor:
        x = self.tensorize_params(params=params)
        objectives = self.botorch_problem(x).view(-1)
        if isinstance(self.botorch_problem, ConstrainedBaseTestProblem):
            constraints = self.botorch_problem.evaluate_slack_true(x).view(-1)
            return torch.cat([objectives, constraints], dim=-1)
        return objectives


@dataclass(kw_only=True)
class ParamBasedTestProblemRunner(BenchmarkRunner):
    """
    A Runner for evaluating `ParamBasedTestProblem`s.

    Given a trial, the Runner will use its `test_problem` to evaluate the
    problem noiselessly for each arm in the trial, and then add noise as
    specified by the `noise_std`. It will return
    metadata including the outcome names and values of metrics.

    Args:
        outcome_names: The names of the outcomes returned by the problem.
        search_space_digest: Used to extract target fidelity and task.
        test_problem: A ``ParamBasedTestProblem`` from which to generate
            deterministic data before adding noise.
        noise_std: The standard deviation of the noise added to the data. Can be
            a list to be per-metric.
    """

    test_problem: ParamBasedTestProblem
    noise_std: float | list[float] | dict[str, float] = 0.0

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

    def get_Y_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        """Evaluates the test problem.

        Returns:
            An `m`-dim tensor of ground truth (noiseless) evaluations.
        """
        return torch.atleast_1d(self.test_problem.evaluate_true(params=params))
