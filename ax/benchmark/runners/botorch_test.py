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
from ax.exceptions.core import UnsupportedError
from botorch.test_functions.multi_objective import MultiObjectiveTestProblem
from botorch.test_functions.synthetic import BaseTestProblem, ConstrainedBaseTestProblem
from botorch.utils.transforms import normalize, unnormalize
from torch import Tensor


@dataclass(kw_only=True)
class ParamBasedTestProblem(ABC):
    """
    The basic Ax class for generating deterministic data to benchmark against.

    (Noise - if desired - is added by the runner.)
    """

    num_objectives: int

    @abstractmethod
    def evaluate_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        """Evaluate noiselessly."""
        ...

    def evaluate_slack_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support constraints."
        )


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
        num_objectives: The number of objectives.
    """

    botorch_problem: BaseTestProblem
    modified_bounds: list[tuple[float, float]] | None = None
    num_objectives: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.botorch_problem, MultiObjectiveTestProblem):
            self.num_objectives = self.botorch_problem.num_objectives
        if self.botorch_problem.noise_std is not None:
            raise ValueError(
                "noise_std should be set on the runner, not the test problem."
            )
        if getattr(self.botorch_problem, "constraint_noise_std", None) is not None:
            raise ValueError(
                "constraint_noise_std should be set on the runner, not the test "
                "problem."
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
        return self.botorch_problem(x)

    # pyre-fixme [14]: inconsistent override
    def evaluate_slack_true(self, params: Mapping[str, float | int]) -> torch.Tensor:
        if not isinstance(self.botorch_problem, ConstrainedBaseTestProblem):
            raise UnsupportedError(
                "`evaluate_slack_true` is only supported when the BoTorch "
                "problem is a `ConstrainedBaseTestProblem`."
            )
        # todo: could return x so as to not recompute
        # or could do both methods together, track indices of outcomes,
        # and only negate the non-constraints
        x = self.tensorize_params(params=params)
        return self.botorch_problem.evaluate_slack_true(x)


@dataclass(kw_only=True)
class ParamBasedTestProblemRunner(BenchmarkRunner):
    """
    A Runner for evaluating `ParamBasedTestProblem`s.

    Given a trial, the Runner will use its `test_problem` to evaluate the
    problem noiselessly for each arm in the trial, and then add noise as
    specified by the `noise_std` and `constraint_noise_std`. It will return
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
    noise_std: float | list[float] | None = None
    constraint_noise_std: float | list[float] | None = None

    @property
    def _is_constrained(self) -> bool:
        return isinstance(self.test_problem, BoTorchTestProblem) and isinstance(
            self.test_problem.botorch_problem, ConstrainedBaseTestProblem
        )

    def get_noise_stds(self) -> None | float | dict[str, float]:
        noise_std = self.noise_std
        noise_std_dict: dict[str, float] = {}
        num_obj = self.test_problem.num_objectives

        # populate any noise_stds for constraints
        if self._is_constrained:
            constraint_noise_std = self.constraint_noise_std
            if isinstance(constraint_noise_std, list):
                for i, cns in enumerate(constraint_noise_std, start=num_obj):
                    if cns is not None:
                        noise_std_dict[self.outcome_names[i]] = cns
            elif constraint_noise_std is not None:
                noise_std_dict[self.outcome_names[num_obj]] = constraint_noise_std

        # if none of the constraints are subject to noise, then we may return
        # a single float or None for the noise level

        if not noise_std_dict and not isinstance(noise_std, list):
            return noise_std  # either a float or None

        if isinstance(noise_std, list):
            if not len(noise_std) == num_obj:
                # this shouldn't be possible due to validation upon construction
                # of the multi-objective problem, but better safe than sorry
                raise ValueError(
                    "Noise std must have length equal to number of objectives."
                )
        else:
            noise_std = [noise_std for _ in range(num_obj)]

        for i, noise_std_ in enumerate(noise_std):
            if noise_std_ is not None:
                noise_std_dict[self.outcome_names[i]] = noise_std_

        return noise_std_dict

    def get_Y_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        """Evaluates the test problem.

        Returns:
            A `batch_shape x m`-dim tensor of ground truth (noiseless) evaluations.
        """
        Y_true = self.test_problem.evaluate_true(params).view(-1)
        if self._is_constrained:
            # Convention: Concatenate objective and black box constraints. `view()`
            # makes the inputs 1d, so the resulting `Y_true` are also 1d.
            Y_true = torch.cat(
                [Y_true, self.test_problem.evaluate_slack_true(params).view(-1)],
                dim=-1,
            )
        return Y_true
