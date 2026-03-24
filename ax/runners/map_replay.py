#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.runner import Runner
from ax.metrics.map_replay import MapDataReplayState


STARTED_KEY = "replay_started"


class MapDataReplayRunner(Runner):
    """A runner that determines trial statuses from a shared
    ``MapDataReplayState`` and advances replay progression on each poll.

    This runner does not actually 'run' anything.
    """

    def __init__(self, replay_state: MapDataReplayState) -> None:
        self._replay_state: MapDataReplayState = replay_state

    def run(self, trial: BaseTrial) -> dict[str, Any]:
        return {STARTED_KEY: True}

    def stop(self, trial: BaseTrial, reason: str | None = None) -> dict[str, Any]:
        return {}

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        result = defaultdict(set)
        for t in trials:
            if not t.run_metadata.get(STARTED_KEY, False):
                result[TrialStatus.CANDIDATE].add(t.index)
            elif not self._replay_state.has_trial_data(trial_index=t.index):
                result[TrialStatus.ABANDONED].add(t.index)
            elif not self._replay_state.is_trial_complete(trial_index=t.index):
                self._replay_state.advance_trial(trial_index=t.index)
                result[TrialStatus.RUNNING].add(t.index)
            else:
                result[TrialStatus.COMPLETED].add(t.index)
        return result
