# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type

import torch
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.runner import Runner
from ax.utils.common.base import Base
from ax.utils.common.equality import equality_typechecker
from ax.utils.common.typeutils import checked_cast
from botorch.test_functions.base import BaseTestProblem
from botorch.utils.transforms import normalize, unnormalize
from torch import Tensor


class BotorchTestProblemRunner(Runner):
    """A Runner for evaluation Botorch BaseTestProblems.
    Given a trial the Runner will evaluate the BaseTestProblem.forward method for each
    arm in the trial, as well as return some metadata about the underlying Botorch
    problem such as the noise_std. We compute the full result on the Runner (as opposed
    to the Metric as is typical in synthetic test problems) because the BoTorch problem
    computes all metrics in one stacked tensor in the MOO case, and we wish to avoid
    recomputation per metric.
    """

    test_problem: BaseTestProblem

    _test_problem_class: Type[BaseTestProblem]
    _test_problem_kwargs: Optional[Dict[str, Any]]

    def __init__(
        self,
        test_problem_class: Type[BaseTestProblem],
        test_problem_kwargs: Dict[str, Any],
        modified_bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Initialize the test problem runner.

        Args:
            test_problem_class: The BoTorch test problem class.
            test_problem_kwargs: The keyword arguments used for initializing the
                test problem.
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
        self.test_problem = test_problem_class(**test_problem_kwargs)
        self._modified_bounds = modified_bounds

    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        if not isinstance(other, BotorchTestProblemRunner):
            return False

        return (
            self.test_problem.__class__.__name__
            == other.test_problem.__class__.__name__
        )

    def evaluate_with_original_bounds(self, X: Tensor) -> Tensor:
        """Converts X to original bounds -- only if modified bounds were provided --
        and evaluates the test problem. See `__init__` docstring for details.
        """
        if self._modified_bounds is not None:
            # Normalize from modified bounds to unit cube.
            unit_X = normalize(X, torch.tensor(self._modified_bounds).T)
            # Unnormalize from unit cube to original problem bounds.
            X = unnormalize(unit_X, self.test_problem.bounds)
        return self.test_problem(X)

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        return {
            "Ys": {
                arm.name: self.evaluate_with_original_bounds(
                    torch.tensor(
                        [
                            value
                            for _key, value in [*arm.parameters.items()][
                                : self.test_problem.dim
                            ]
                        ]
                    )
                ).tolist()
                for arm in trial.arms
            },
        }

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
            "modified_bounds": runner._modified_bounds,
        }

    @classmethod
    def deserialize_init_args(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        """Given a dictionary, deserialize the properties needed to initialize the
        runner. Used for storage.
        """

        module = importlib.import_module(args["test_problem_module"])

        return {
            "test_problem_class": getattr(module, args["test_problem_class_name"]),
            "test_problem_kwargs": args["test_problem_kwargs"],
            "modified_bounds": args["modified_bounds"],
        }
