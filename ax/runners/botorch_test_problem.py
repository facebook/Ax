# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Iterable, Any, Dict

import torch
from ax.core.base_trial import TrialStatus, BaseTrial
from ax.core.runner import Runner
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

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        return {
            "Ys": {
                arm.name: self.test_problem.forward(
                    torch.tensor([value for _key, value in arm.parameters.items()])
                ).tolist()
                for arm in trial.arms
            },
        }

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
        return {TrialStatus.COMPLETED: {t.index for t in trials}}
