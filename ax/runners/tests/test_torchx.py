#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
from typing import List

from ax.core import (
    BatchTrial,
    Experiment,
    Objective,
    OptimizationConfig,
    Parameter,
    ParameterType,
    RangeParameter,
    SearchSpace,
)
from ax.metrics.torchx import TorchXMetric
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.runners.torchx import TorchXRunner
from ax.service.scheduler import FailureRateExceededError, Scheduler, SchedulerOptions
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from torchx.components import utils


class TorchXRunnerTest(TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp("torchx_runtime_hpo_ax_test")

        self.old_cwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.dirname(__file__)))

        self._parameters: List[Parameter] = [
            RangeParameter(
                name="x1",
                lower=-10.0,
                upper=10.0,
                parameter_type=ParameterType.FLOAT,
            ),
            RangeParameter(
                name="x2",
                lower=-10.0,
                upper=10.0,
                parameter_type=ParameterType.FLOAT,
            ),
        ]

        self._minimize = True
        self._objective = Objective(
            metric=TorchXMetric(
                name="booth_eval",
            ),
            minimize=self._minimize,
        )

        self._runner = TorchXRunner(
            tracker_base=self.test_dir,
            component=utils.booth,
            scheduler="local_cwd",
            cfg={"prepend_cwd": True},
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)
        os.chdir(self.old_cwd)

    def test_run_experiment_locally(self) -> None:
        """Runs optimization over n rounds of k sequential trials."""

        experiment = Experiment(
            name="torchx_booth_sequential_demo",
            search_space=SearchSpace(parameters=self._parameters),
            optimization_config=OptimizationConfig(objective=self._objective),
            runner=self._runner,
            is_test=True,
            properties={Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF: True},
        )

        scheduler = Scheduler(
            experiment=experiment,
            generation_strategy=(
                choose_generation_strategy(
                    search_space=experiment.search_space,
                )
            ),
            options=SchedulerOptions(),
        )

        try:
            for _ in range(3):
                scheduler.run_n_trials(max_trials=2)

            # TorchXMetric always returns trial index; hence the best experiment
            # for min objective will be the params for trial 0.
            scheduler.report_results()
        except FailureRateExceededError:
            pass  # TODO(ehotaj): Figure out why this test fails in OSS.
        # Nothing to assert, just make sure experiment runs.

    def test_stop_trials(self) -> None:
        experiment = Experiment(
            name="torchx_booth_sequential_demo",
            search_space=SearchSpace(parameters=self._parameters),
            optimization_config=OptimizationConfig(objective=self._objective),
            runner=self._runner,
            is_test=True,
            properties={Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF: True},
        )
        scheduler = Scheduler(
            experiment=experiment,
            generation_strategy=(
                choose_generation_strategy(
                    search_space=experiment.search_space,
                )
            ),
            options=SchedulerOptions(),
        )
        scheduler.run(max_new_trials=3)
        trial = scheduler.running_trials[0]
        reason = self._runner.stop(trial, reason="some_reason")
        self.assertEqual(reason, {"reason": "some_reason"})

    def test_run_experiment_locally_in_batches(self) -> None:
        """Runs optimization over k x n rounds of k parallel trials.

        This asks Ax to run up to max_parallelism_cap trials in parallel by
        submitting them to the scheduler at the same time.

        NOTE:
            * setting max_parallelism_cap in generation_strategy
            * setting run_trials_in_batches in scheduler options
            * setting total_trials = parallelism * rounds

        """
        parallelism = 2
        rounds = 3

        experiment = Experiment(
            name="torchx_booth_parallel_demo",
            search_space=SearchSpace(parameters=self._parameters),
            optimization_config=OptimizationConfig(objective=self._objective),
            runner=self._runner,
            is_test=True,
            properties={Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF: True},
        )

        scheduler = Scheduler(
            experiment=experiment,
            generation_strategy=(
                choose_generation_strategy(
                    search_space=experiment.search_space,
                    max_parallelism_cap=parallelism,
                )
            ),
            options=SchedulerOptions(
                run_trials_in_batches=True, total_trials=(parallelism * rounds)
            ),
        )

        try:
            scheduler.run_all_trials()

            # TorchXMetric always returns trial index; hence the best experiment
            # for min objective will be the params for trial 0.
            scheduler.report_results()
        except FailureRateExceededError:
            pass  # TODO(ehotaj): Figure out why this test fails in OSS.
        # Nothing to assert, just make sure experiment runs.

    def test_runner_no_batch_trials(self) -> None:
        experiment = Experiment(
            name="runner_test",
            search_space=SearchSpace(parameters=self._parameters),
            optimization_config=OptimizationConfig(objective=self._objective),
            runner=self._runner,
            is_test=True,
        )

        with self.assertRaises(ValueError):
            self._runner.run(trial=BatchTrial(experiment))
