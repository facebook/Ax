# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from ax.benchmark.runners.base import BenchmarkRunner
from ax.core.arm import Arm
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TParameterization
from ax.utils.common.base import Base
from ax.utils.common.equality import equality_typechecker
from ax.utils.common.serialization import TClassDecoderRegistry, TDecoderRegistry
from botorch.test_functions.synthetic import BaseTestProblem, ConstrainedBaseTestProblem
from botorch.utils.transforms import normalize, unnormalize
from pyre_extensions import assert_is_instance
from torch import Tensor


@dataclass(kw_only=True)
class ParamBasedTestProblem(ABC):
    """
    Similar to a BoTorch test problem, but evaluated using an Ax
    TParameterization rather than a tensor.
    """

    num_objectives: int
    optimal_value: float
    # Constraints could easily be supported similar to BoTorch test problems,
    # but haven't been hooked up.
    _is_constrained: bool = False
    constraint_noise_std: Optional[Union[float, list[float]]] = None
    noise_std: Optional[Union[float, list[float]]] = None
    negate: bool = False

    @abstractmethod
    def evaluate_true(self, params: TParameterization) -> Tensor: ...

    def evaluate_slack_true(self, params: TParameterization) -> Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support constraints."
        )

    # pyre-fixme: Missing parameter annotation [2]: Parameter `other` must have
    # a type other than `Any`.
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.__class__.__name__ == other.__class__.__name__


class SyntheticProblemRunner(BenchmarkRunner, ABC):
    """A Runner for evaluating synthetic problems, either BoTorch
    `BaseTestProblem`s or Ax benchmarking `ParamBasedTestProblem`s.

    Given a trial, the Runner will evaluate the problem noiselessly for each
    arm in the trial, as well as return some metadata about the underlying
    problem such as the noise_std.
    """

    test_problem: Union[BaseTestProblem, ParamBasedTestProblem]
    _is_constrained: bool
    _test_problem_class: type[Union[BaseTestProblem, ParamBasedTestProblem]]
    _test_problem_kwargs: Optional[dict[str, Any]]

    def __init__(
        self,
        *,
        test_problem_class: type[Union[BaseTestProblem, ParamBasedTestProblem]],
        test_problem_kwargs: dict[str, Any],
        outcome_names: list[str],
        modified_bounds: Optional[list[tuple[float, float]]] = None,
        search_space_digest: SearchSpaceDigest | None = None,
    ) -> None:
        """Initialize the test problem runner.

        Args:
            test_problem_class: A BoTorch `BaseTestProblem` class or Ax
                `ParamBasedTestProblem` class.
            test_problem_kwargs: The keyword arguments used for initializing the
                test problem.
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
        super().__init__(search_space_digest=search_space_digest)
        self._test_problem_class = test_problem_class
        self._test_problem_kwargs = test_problem_kwargs
        self.test_problem = (
            # pyre-fixme: Invalid class instantiation [45]: Cannot instantiate
            # abstract class with abstract method `evaluate_true`.
            test_problem_class(**test_problem_kwargs)
        )
        if isinstance(self.test_problem, BaseTestProblem):
            self.test_problem = self.test_problem.to(dtype=torch.double)
        # A `ConstrainedBaseTestProblem` is a type of `BaseTestProblem`; a
        # `ParamBasedTestProblem` is never constrained.
        self._is_constrained: bool = isinstance(
            self.test_problem, ConstrainedBaseTestProblem
        )
        self._is_moo: bool = self.test_problem.num_objectives > 1
        self.outcome_names = outcome_names
        self._modified_bounds = modified_bounds

    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        if not isinstance(other, type(self)):
            return False

        return (
            self.test_problem.__class__.__name__
            == other.test_problem.__class__.__name__
        )

    def get_noise_stds(self) -> Union[None, float, dict[str, float]]:
        noise_std = self.test_problem.noise_std
        noise_std_dict: dict[str, float] = {}
        num_obj = 1 if not self._is_moo else self.test_problem.num_objectives

        # populate any noise_stds for constraints
        if self._is_constrained:
            constraint_noise_std = self.test_problem.constraint_noise_std
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

    @classmethod
    # pyre-fixme [2]: Parameter `obj` must have a type other than `Any``
    def serialize_init_args(cls, obj: Any) -> dict[str, Any]:
        """Serialize the properties needed to initialize the runner.
        Used for storage.
        """
        runner = assert_is_instance(obj, cls)

        return {
            "test_problem_module": runner._test_problem_class.__module__,
            "test_problem_class_name": runner._test_problem_class.__name__,
            "test_problem_kwargs": runner._test_problem_kwargs,
            "outcome_names": runner.outcome_names,
            "modified_bounds": runner._modified_bounds,
        }

    @classmethod
    def deserialize_init_args(
        cls,
        args: dict[str, Any],
        decoder_registry: Optional[TDecoderRegistry] = None,
        class_decoder_registry: Optional[TClassDecoderRegistry] = None,
    ) -> dict[str, Any]:
        """Given a dictionary, deserialize the properties needed to initialize the
        runner. Used for storage.
        """

        module = importlib.import_module(args["test_problem_module"])

        return {
            "test_problem_class": getattr(module, args["test_problem_class_name"]),
            "test_problem_kwargs": args["test_problem_kwargs"],
            "outcome_names": args["outcome_names"],
            "modified_bounds": args["modified_bounds"],
        }


class BotorchTestProblemRunner(SyntheticProblemRunner):
    """
    A `SyntheticProblemRunner` for BoTorch `BaseTestProblem`s.

    Args:
        test_problem_class: A BoTorch `BaseTestProblem` class.
        test_problem_kwargs: The keyword arguments used for initializing the
            test problem.
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

    def __init__(
        self,
        *,
        test_problem_class: type[BaseTestProblem],
        test_problem_kwargs: dict[str, Any],
        outcome_names: list[str],
        modified_bounds: Optional[list[tuple[float, float]]] = None,
        search_space_digest: SearchSpaceDigest | None = None,
    ) -> None:
        super().__init__(
            test_problem_class=test_problem_class,
            test_problem_kwargs=test_problem_kwargs,
            outcome_names=outcome_names,
            modified_bounds=modified_bounds,
            search_space_digest=search_space_digest,
        )
        self.test_problem: BaseTestProblem = self.test_problem.to(dtype=torch.double)
        self._is_constrained: bool = isinstance(
            self.test_problem, ConstrainedBaseTestProblem
        )

    def get_Y_true(self, arm: Arm) -> Tensor:
        """
        Convert the arm to a tensor and evaluate it on the base test problem.

        Convert the tensor to original bounds -- only if modified bounds were
        provided -- and evaluates the test problem. See the docstring for
        `modified_bounds` in `BotorchTestProblemRunner.__init__` for details.

        Args:
            arm: Arm to evaluate. It will be converted to a
                `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                test problem.

        Returns:
            A `batch_shape x m`-dim tensor of ground truth (noiseless) evaluations.
        """
        X = torch.tensor(
            [
                value
                for _key, value in [*arm.parameters.items()][: self.test_problem.dim]
            ],
            dtype=torch.double,
        )

        if self._modified_bounds is not None:
            # Normalize from modified bounds to unit cube.
            unit_X = normalize(
                X, torch.tensor(self._modified_bounds, dtype=torch.double).T
            )
            # Unnormalize from unit cube to original problem bounds.
            X = unnormalize(unit_X, self.test_problem.bounds)

        Y_true = self.test_problem.evaluate_true(X).view(-1)
        # `BaseTestProblem.evaluate_true()` does not negate the outcome
        if self.test_problem.negate:
            Y_true = -Y_true

        if self._is_constrained:
            # Convention: Concatenate objective and black box constraints. `view()`
            # makes the inputs 1d, so the resulting `Y_true` are also 1d.
            Y_true = torch.cat(
                [Y_true, self.test_problem.evaluate_slack_true(X).view(-1)],
                dim=-1,
            )

        return Y_true


class ParamBasedTestProblemRunner(SyntheticProblemRunner):
    """
    A `SyntheticProblemRunner` for `ParamBasedTestProblem`s. See
    `SyntheticProblemRunner` for more information.
    """

    # This could easily be supported, but hasn't been hooked up
    _is_constrained: bool = False

    def __init__(
        self,
        *,
        test_problem_class: type[ParamBasedTestProblem],
        test_problem_kwargs: dict[str, Any],
        outcome_names: list[str],
        modified_bounds: Optional[list[tuple[float, float]]] = None,
        search_space_digest: SearchSpaceDigest | None = None,
    ) -> None:
        if modified_bounds is not None:
            raise NotImplementedError(
                f"modified_bounds is not supported for {test_problem_class.__name__}"
            )
        super().__init__(
            test_problem_class=test_problem_class,
            test_problem_kwargs=test_problem_kwargs,
            outcome_names=outcome_names,
            modified_bounds=modified_bounds,
            search_space_digest=search_space_digest,
        )
        self.test_problem: ParamBasedTestProblem = self.test_problem

    def get_Y_true(self, arm: Arm) -> Tensor:
        """Evaluates the test problem.

        Returns:
            A `batch_shape x m`-dim tensor of ground truth (noiseless) evaluations.
        """
        Y_true = self.test_problem.evaluate_true(arm.parameters).view(-1)
        # `ParamBasedTestProblem.evaluate_true()` does not negate the outcome
        if self.test_problem.negate:
            Y_true = -Y_true
        return Y_true
