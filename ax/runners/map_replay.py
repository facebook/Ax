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
from ax.metrics.map_replay import MapDataReplayMetric


STARTED_KEY = "replay_started"


class MapDataReplayRunner(Runner):
    """A runner that uses a `MapDataReplayMetric` to determine trial statuses.
    This runner does not actually 'run' anything."""

    def __init__(self, replay_metric: MapDataReplayMetric) -> None:
        self.replay_metric: MapDataReplayMetric = replay_metric

    def run(self, trial: BaseTrial) -> dict[str, Any]:
        return {STARTED_KEY: True}

    def stop(self, trial: BaseTrial, reason: str | None = None) -> dict[str, Any]:
        return {}

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> dict[TrialStatus, set[int]]:
        result = defaultdict(set)
        # For each trial, if it hasn't been started yet by this runner,
        # then mark is as a CANDIDATE. If there is no replay data
        # associated with that trial at all, mark is FAILED. Otherwise,
        # depending on whether or not there is more data available,
        # mark it either RUNNING or COMPLETED.
        for t in trials:
            if not t.run_metadata.get(STARTED_KEY, False):
                result[TrialStatus.CANDIDATE].add(t.index)
            elif not self.replay_metric.has_trial_data(t.index):
                result[TrialStatus.ABANDONED].add(t.index)
            elif self.replay_metric.more_replay_available(t.index):
                result[TrialStatus.RUNNING].add(t.index)
            else:
                result[TrialStatus.COMPLETED].add(t.index)
        return result
