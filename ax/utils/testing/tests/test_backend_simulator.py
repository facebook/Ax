#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch, Mock

from ax.core.base_trial import TrialStatus
from ax.utils.common.testutils import TestCase
from ax.utils.testing.backend_simulator import BackendSimulator, BackendSimulatorOptions


class BackendSimulatorTest(TestCase):
    @patch("ax.utils.testing.backend_simulator.time.time")
    def test_backend_simulator(self, time_mock: Mock):
        time_mock.return_value = 0.0
        dt = 0.001
        options = BackendSimulatorOptions(max_concurrency=2)

        # test init
        sim = BackendSimulator(options=options)
        self.assertEqual(sim.max_concurrency, 2)
        self.assertEqual(sim.time_scaling, 1.0)
        self.assertEqual(sim.failure_rate, 0.0)
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

        # extract state for later use
        state = sim.state()

        # let time pass and update
        time_mock.return_value += 10 * dt
        sim.update()
        self.assertEqual(sim.num_queued, 0)
        self.assertEqual(sim.num_running, 0)
        self.assertEqual(sim.num_failed, 0)
        self.assertEqual(sim.num_completed, 3)

        # test reset
        sim.max_concurrency = 3
        sim.time_scaling = 2.0
        sim.failure_rate, 0.5
        sim.reset()
        self.assertEqual(sim.max_concurrency, 2)
        self.assertEqual(sim.time_scaling, 1.0)
        self.assertEqual(sim.failure_rate, 0.0)
        self.assertEqual(sim.num_queued, 0)
        self.assertEqual(sim.num_running, 0)
        self.assertEqual(sim.num_failed, 0)
        self.assertEqual(sim.num_completed, 0)

        # test load state
        sim2 = BackendSimulator.from_state(state)
        self.assertEqual(sim2.max_concurrency, 2)
        self.assertEqual(sim2.time_scaling, 1.0)
        self.assertEqual(sim2.failure_rate, 0.0)
        self.assertEqual(sim2.num_queued, 0)
        self.assertEqual(sim2.num_running, 1)
        self.assertEqual(sim2.num_failed, 0)
        self.assertEqual(sim2.num_completed, 2)
        sim2.update()
        self.assertEqual(sim2.num_queued, 0)
        self.assertEqual(sim2.num_running, 0)
        self.assertEqual(sim2.num_failed, 0)
        self.assertEqual(sim2.num_completed, 3)

        # test failure rate
        options = BackendSimulatorOptions(max_concurrency=2, failure_rate=1.0)
        sim3 = BackendSimulator(options=options)
        sim3.run_trial(0, dt)
        self.assertEqual(sim3.num_queued, 0)
        self.assertEqual(sim3.num_running, 0)
        self.assertEqual(sim3.num_failed, 1)
        self.assertEqual(sim3.num_completed, 0)

    def test_backend_simulator_internal_clock(self):
        options = BackendSimulatorOptions(
            internal_clock=0.0, use_update_as_start_time=True, max_concurrency=2
        )
        sim = BackendSimulator(options=options)
        sim.run_trial(0, 2)
        sim.run_trial(1, 1)
        sim.run_trial(2, 10)
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
            sim.get_sim_trial_by_index(trial_index=2).sim_completed_time, 2.0
        )
