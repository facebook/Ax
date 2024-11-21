# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import islice

import torch
from ax.benchmark.benchmark_test_function import BenchmarkTestFunction
from botorch.test_functions.synthetic import BaseTestProblem, ConstrainedBaseTestProblem
from botorch.utils.transforms import normalize, unnormalize


@dataclass(kw_only=True)
class BoTorchTestFunction(BenchmarkTestFunction):
    """
    Class for generating data from a BoTorch ``BaseTestProblem``.

    Args:
        outcome_names: Names of outcomes. Should have the same length as the
            dimension of the test function, including constraints.
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

    outcome_names: Sequence[str]
    botorch_problem: BaseTestProblem
    modified_bounds: Sequence[tuple[float, float]] | None = None

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
