#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.utils.testing.backend_simulator import BackendSimulator, BackendSimulatorOptions


def get_backend_simulator_with_trials() -> BackendSimulator:
    options = BackendSimulatorOptions(
        internal_clock=0.0, use_update_as_start_time=True, max_concurrency=2
    )
    sim = BackendSimulator(options=options)
    sim.run_trial(0, 2)
    sim.run_trial(1, 1)
    sim.run_trial(2, 10)
    return sim
