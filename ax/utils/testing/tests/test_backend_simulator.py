#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import Mock, patch

from ax.core.base_trial import TrialStatus
from ax.utils.common.testutils import TestCase
from ax.utils.testing.backend_simulator import BackendSimulator, BackendSimulatorOptions
from ax.utils.testing.utils_testing_stubs import get_backend_simulator_with_trials


class BackendSimulatorTest(TestCase):
    @patch("ax.utils.testing.backend_simulator.time.time")
    def test_backend_simulator(self, time_mock: Mock) -> None:
        time_mock.return_value = 0.0
        dt = 0.001
        options = BackendSimulatorOptions(max_concurrency=2)

        # test init
        sim = BackendSimulator(options=options)
        options = sim.options
        self.assertEqual(options.max_concurrency, 2)
        self.assertEqual(options.time_scaling, 1.0)
        self.assertEqual(options.failure_rate, 0.0)
        self.assertEqual(sim.num_queued, 0)
        self.assertEqual(sim.num_running, 0)
        self.assertEqual(sim.num_failed, 0)
        self.assertEqual(sim.num_completed, 0)

        # test run trial
        sim.run_trial(0, dt)
        self.assertEqual(sim.num_queued, 0)
        self.assertEqual(sim.num_running, 1)
        sim.run_trial(1, dt)
        self.assertEqual(sim.num_queued, 0)
        self.assertEqual(sim.num_running, 2)
        sim.run_trial(2, dt)
        self.assertEqual(sim.num_queued, 1)
        self.assertEqual(sim.num_running, 2)
        status = sim.status()
        self.assertEqual(status.queued, [2])
        self.assertEqual(status.running, [0, 1])
        self.assertEqual(status.failed, [])
        self.assertEqual(status.completed, [])

        # "Wait" some time
        time_mock.return_value += 1.5 * dt
        sim.update()
        self.assertEqual(sim.num_queued, 0)
        self.assertEqual(sim.num_running, 1)
        self.assertEqual(sim.num_failed, 0)
        self.assertEqual(sim.num_completed, 2)

        # let time pass and update
        time_mock.return_value += 10 * dt
        sim.update()
        self.assertEqual(sim.num_queued, 0)
        self.assertEqual(sim.num_running, 0)
        self.assertEqual(sim.num_failed, 0)
        self.assertEqual(sim.num_completed, 3)

        # test failure rate
        options = BackendSimulatorOptions(max_concurrency=2, failure_rate=1.0)
        sim3 = BackendSimulator(options=options)
        sim3.run_trial(0, dt)
        self.assertEqual(sim3.num_queued, 0)
        self.assertEqual(sim3.num_running, 0)
        self.assertEqual(sim3.num_failed, 1)
        self.assertEqual(sim3.num_completed, 0)

    def test_backend_simulator_internal_clock(self) -> None:
        sim = get_backend_simulator_with_trials()
        self.assertEqual(len(sim.all_trials), 3)
        self.assertEqual(sim.time, 0.0)
        self.assertEqual(sim.num_queued, 1)
        self.assertEqual(
            sim.lookup_trial_index_status(trial_index=0), TrialStatus.RUNNING
        )
        self.assertEqual(
            sim.lookup_trial_index_status(trial_index=1), TrialStatus.RUNNING
        )
        self.assertEqual(
            sim.lookup_trial_index_status(trial_index=2), TrialStatus.STAGED
        )

        sim.update()
        self.assertEqual(sim.num_completed, 1)
        self.assertEqual(sim.num_running, 2)
        self.assertEqual(
            sim.lookup_trial_index_status(trial_index=1), TrialStatus.COMPLETED
        )

        sim.update()
        self.assertEqual(sim.num_completed, 2)
        self.assertEqual(sim.num_running, 1)
        self.assertEqual(
            sim.lookup_trial_index_status(trial_index=0), TrialStatus.COMPLETED
        )

        sim.stop_trial(trial_index=2)
        sim.update()
        self.assertEqual(
            sim.lookup_trial_index_status(trial_index=2), TrialStatus.COMPLETED
        )
        self.assertEqual(
            # pyre-fixme[16]: Optional type has no attribute `sim_completed_time`.
            sim.get_sim_trial_by_index(trial_index=2).sim_completed_time,
            2.0,
        )
        with self.assertRaisesRegex(ValueError, "Trial 100 not found in simulator"):
            sim.lookup_trial_index_status(trial_index=100)
