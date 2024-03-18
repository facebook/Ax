# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import importlib
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union

import torch
from ax.benchmark.runners.base import BenchmarkRunner
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.utils.common.base import Base
from ax.utils.common.equality import equality_typechecker
from ax.utils.common.serialization import TClassDecoderRegistry, TDecoderRegistry
from ax.utils.common.typeutils import checked_cast
from botorch.test_functions.base import BaseTestProblem, ConstrainedBaseTestProblem
from botorch.test_functions.multi_objective import MultiObjectiveTestProblem
from botorch.utils.transforms import normalize, unnormalize
from torch import Tensor


class BotorchTestProblemRunner(BenchmarkRunner):
    """A Runner for evaluating Botorch BaseTestProblems.

    Given a trial the Runner will evaluate the BaseTestProblem.forward method for each
    arm in the trial, as well as return some metadata about the underlying Botorch
    problem such as the noise_std. We compute the full result on the Runner (as opposed
    to the Metric as is typical in synthetic test problems) because the BoTorch problem
    computes all metrics in one stacked tensor in the MOO case, and we wish to avoid
    recomputation per metric.
    """

    test_problem: BaseTestProblem
    _is_constrained: bool
    _test_problem_class: Type[BaseTestProblem]
    _test_problem_kwargs: Optional[Dict[str, Any]]

    def __init__(
        self,
        test_problem_class: Type[BaseTestProblem],
        test_problem_kwargs: Dict[str, Any],
        outcome_names: List[str],
        modified_bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Initialize the test problem runner.

        Args:
            test_problem_class: The BoTorch test problem class.
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
        """

        self._test_problem_class = test_problem_class
        self._test_problem_kwargs = test_problem_kwargs

        # pyre-fixme [45]: Invalid class instantiation
        self.test_problem = test_problem_class(**test_problem_kwargs).to(
            dtype=torch.double
        )
        self._is_constrained: bool = isinstance(
            self.test_problem, ConstrainedBaseTestProblem
        )
        self._is_moo: bool = isinstance(self.test_problem, MultiObjectiveTestProblem)
        self._outcome_names = outcome_names
        self._modified_bounds = modified_bounds

    @property
    def outcome_names(self) -> List[str]:
        return self._outcome_names

    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        if not isinstance(other, BotorchTestProblemRunner):
            return False

        return (
            self.test_problem.__class__.__name__
            == other.test_problem.__class__.__name__
        )

    def get_noise_stds(self) -> Union[None, float, Dict[str, float]]:
        noise_std = self.test_problem.noise_std
        noise_std_dict: Dict[str, float] = {}
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

    def get_Y_true(self, arm: Arm) -> Tensor:
        """Converts X to original bounds -- only if modified bounds were provided --
        and evaluates the test problem. See `__init__` docstring for details.

        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
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

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
        return {TrialStatus.COMPLETED: {t.index for t in trials}}

    @classmethod
    # pyre-fixme [2]: Parameter `obj` must have a type other than `Any``
    def serialize_init_args(cls, obj: Any) -> Dict[str, Any]:
        """Serialize the properties needed to initialize the runner.
        Used for storage.
        """
        runner = checked_cast(BotorchTestProblemRunner, obj)

        return {
            "test_problem_module": runner._test_problem_class.__module__,
            "test_problem_class_name": runner._test_problem_class.__name__,
            "test_problem_kwargs": runner._test_problem_kwargs,
            "outcome_names": runner._outcome_names,
            "modified_bounds": runner._modified_bounds,
        }

    @classmethod
    def deserialize_init_args(
        cls,
        args: Dict[str, Any],
        decoder_registry: Optional[TDecoderRegistry] = None,
        class_decoder_registry: Optional[TClassDecoderRegistry] = None,
    ) -> Dict[str, Any]:
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
