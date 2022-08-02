# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from typing import Any, Dict, Iterable, Set

import torch
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.runner import Runner
from ax.utils.common.base import Base
from ax.utils.common.equality import equality_typechecker
from ax.utils.common.typeutils import checked_cast
from botorch.test_functions.base import BaseTestProblem


class BotorchTestProblemRunner(Runner):
    """A Runner for evaluation Botorch BaseTestProblems.
    Given a trial the Runner will evaluate the BaseTestProblem.forward method for each
    arm in the trial, as well as return some metadata about the underlying Botorch
    problem such as the noise_std. We compute the full result on the Runner (as opposed
    to the Metric as is typical in synthetic test problems) because the BoTorch problem
    computes all metrics in one stacked tensor in the MOO case, and we wish to avoid
    recomputation per metric.
    """

    def __init__(self, test_problem: BaseTestProblem) -> None:
        self.test_problem = test_problem

    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        if not isinstance(other, BotorchTestProblemRunner):
            return False

        return (
            self.test_problem.__class__.__name__
            == other.test_problem.__class__.__name__
        )

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        return {
            "Ys": {
                arm.name: self.test_problem.forward(
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
    def serialize_init_args(cls, obj: Any) -> Dict[str, Any]:
        """Serialize the properties needed to initialize the runner.
        Used for storage.
        """
        runner = checked_cast(BotorchTestProblemRunner, obj)

        return {
            "test_problem_module": runner.test_problem.__module__,
            "test_problem_class_name": runner.test_problem.__class__.__name__,
        }

    @classmethod
    def deserialize_init_args(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        """Given a dictionary, deserialize the properties needed to initialize the
        runner. Used for storage.
        """
        module = importlib.import_module(args["test_problem_module"])

        return {"test_problem": getattr(module, args["test_problem_class_name"])()}
