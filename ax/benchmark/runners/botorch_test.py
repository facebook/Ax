# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from ax.benchmark.runners.base import BenchmarkRunner
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TParamValue
from ax.utils.common.base import Base
from ax.utils.common.equality import equality_typechecker
from botorch.test_functions.synthetic import BaseTestProblem, ConstrainedBaseTestProblem
from botorch.utils.transforms import normalize, unnormalize
from torch import Tensor


@dataclass(kw_only=True)
class ParamBasedTestProblem(ABC):
    """
    Similar to a BoTorch test problem, but evaluated using an Ax
    TParameterization rather than a tensor.
    """

    num_objectives: int

    @abstractmethod
    def evaluate_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        """
        Evaluate noiselessly.

        This method should not depend on the value of `negate`, as negation is
        handled by the runner.
        """
        ...

    def evaluate_slack_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support constraints."
        )


@dataclass(kw_only=True)
class SyntheticProblemRunner(BenchmarkRunner, ABC):
    """A Runner for evaluating synthetic problems, either BoTorch
    `BaseTestProblem`s or Ax benchmarking `ParamBasedTestProblem`s.

    Given a trial, the Runner will evaluate the problem noiselessly for each
    arm in the trial, as well as return some metadata about the underlying
    problem such as the noise_std.

    Args:
        test_problem: A BoTorch `BaseTestProblem` or Ax `ParamBasedTestProblem`.
        outcome_names: The names of the outcomes returned by the problem.
        modified_bounds: The bounds that are used by the Ax search space
            while optimizing the problem. If different from the bounds of the
            test problem, we project the parameters into the test problem
            bounds before evaluating the test problem.
            For example, if the test problem is defined on [0, 1] but the Ax
            search space is integers in [0, 10], an Ax parameter value of
            5 will correspond to 0.5 while evaluating the test problem.
            If modified bounds are not provided, the test problem will be
            evaluated using the raw parameter values.
        search_space_digest: Used to extract target fidelity and task.
    """

    test_problem: BaseTestProblem | ParamBasedTestProblem
    modified_bounds: list[tuple[float, float]] | None = None
    constraint_noise_std: float | list[float] | None = None
    noise_std: float | list[float] | None = None
    negate: bool = False

    @property
    def _is_constrained(self) -> bool:
        return isinstance(self.test_problem, ConstrainedBaseTestProblem)

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


@dataclass(kw_only=True)
class BotorchTestProblemRunner(SyntheticProblemRunner):
    """
    A `SyntheticProblemRunner` for BoTorch `BaseTestProblem`s.

    Args:
        test_problem: A BoTorch `BaseTestProblem`.
        outcome_names: The names of the outcomes returned by the problem.
        modified_bounds: The bounds that are used by the Ax search space
            while optimizing the problem. If different from the bounds of the
            test problem, we project the parameters into the test problem
            bounds before evaluating the test problem.
            For example, if the test problem is defined on [0, 1] but the Ax
            search space is integers in [0, 10], an Ax parameter value of
            5 will correspond to 0.5 while evaluating the test problem.
            If modified bounds are not provided, the test problem will be
            evaluated using the raw parameter values.
        search_space_digest: Used to extract target fidelity and task.
    """

    test_problem: BaseTestProblem

    def __post_init__(self, search_space_digest: SearchSpaceDigest | None) -> None:
        super().__post_init__(search_space_digest=search_space_digest)
        if self.test_problem.noise_std is not None:
            raise ValueError(
                "noise_std should be set on the runner, not the test problem."
            )
        if (
            hasattr(self.test_problem, "constraint_noise_std")
            and self.test_problem.constraint_noise_std is not None
        ):
            raise ValueError(
                "constraint_noise_std should be set on the runner, not the test "
                "problem."
            )
        if self.test_problem.negate:
            raise ValueError(
                "negate should be set on the runner, not the test problem."
            )
        self.test_problem = self.test_problem.to(dtype=torch.double)

    def get_Y_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        """
        Convert the arm to a tensor and evaluate it on the base test problem.

        Convert the tensor to original bounds -- only if modified bounds were
        provided -- and evaluates the test problem. See the docstring for
        `modified_bounds` in `BotorchTestProblemRunner.__init__` for details.

        Args:
            params: Parameterization to evaluate. It will be converted to a
                `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                test problem.

        Returns:
            A `batch_shape x m`-dim tensor of ground truth (noiseless) evaluations.
        """
        X = torch.tensor(
            [value for _key, value in [*params.items()][: self.test_problem.dim]],
            dtype=torch.double,
        )

        if self.modified_bounds is not None:
            # Normalize from modified bounds to unit cube.
            unit_X = normalize(
                X, torch.tensor(self.modified_bounds, dtype=torch.double).T
            )
            # Unnormalize from unit cube to original problem bounds.
            X = unnormalize(unit_X, self.test_problem.bounds)

        Y_true = self.test_problem.evaluate_true(X).view(-1)
        # `BaseTestProblem.evaluate_true()` does not negate the outcome
        if self.negate:
            Y_true = -Y_true

        if self._is_constrained:
            # Convention: Concatenate objective and black box constraints. `view()`
            # makes the inputs 1d, so the resulting `Y_true` are also 1d.
            Y_true = torch.cat(
                [Y_true, self.test_problem.evaluate_slack_true(X).view(-1)],
                dim=-1,
            )

        return Y_true

    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        """
        Compare equality by comparing dicts, except for `test_problem`.

        Dataclasses are compared by comparing the results of calling asdict on
        them. However, equality checks don't work as needed with BoTorch test
        problems, e.g. Branin() == Branin() is False. To get around that, the
        test problem is stripped from the dictionary. This doesn't make the
        check less sensitive, as long as the problem has not been modified,
        because the test problem class and keyword arguments will still be
        compared.
        """
        if not isinstance(other, type(self)):
            return False
        self_as_dict = asdict(self)
        other_as_dict = asdict(other)
        self_as_dict.pop("test_problem")
        other_as_dict.pop("test_problem")
        return (self_as_dict == other_as_dict) and (
            type(self.test_problem) is type(other.test_problem)
        )


@dataclass(kw_only=True)
class ParamBasedTestProblemRunner(SyntheticProblemRunner):
    """
    A `SyntheticProblemRunner` for `ParamBasedTestProblem`s. See
    `SyntheticProblemRunner` for more information.
    """

    test_problem: ParamBasedTestProblem

    def get_Y_true(self, params: Mapping[str, TParamValue]) -> Tensor:
        """Evaluates the test problem.

        Returns:
            A `batch_shape x m`-dim tensor of ground truth (noiseless) evaluations.
        """
        Y_true = self.test_problem.evaluate_true(params).view(-1)
        # `ParamBasedTestProblem.evaluate_true()` does not negate the outcome
        if self.negate:
            Y_true = -Y_true
        return Y_true
