#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from ax.core.base_trial import TrialStatus
from ax.runners.single_running_trial_mixin import SingleRunningTrialMixin
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class SyntheticRunnerWithSingleRunningTrial(SingleRunningTrialMixin, SyntheticRunner):
    ...


class SingleRunningTrialMixinTest(TestCase):
    def test_single_running_trial_mixin(self) -> None:
        runner = SyntheticRunnerWithSingleRunningTrial()
        exp = get_branin_experiment(with_trial=True, with_batch=True)
        exp.runner = runner
        trials = exp.trials.values()
        for trial in trials:
            trial.assign_runner()
            trial.run()
        trial_statuses = runner.poll_trial_status(trials=trials)
        self.assertEqual(trial_statuses[TrialStatus.COMPLETED], {0})
        self.assertEqual(trial_statuses[TrialStatus.RUNNING], {1})
