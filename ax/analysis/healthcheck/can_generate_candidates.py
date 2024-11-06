# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
from datetime import datetime
from typing import Optional

import pandas as pd
from ax.analysis.analysis import AnalysisCardLevel

from ax.analysis.healthcheck.healthcheck_analysis import (
    HealthcheckAnalysis,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from pyre_extensions import none_throws


class CanGenerateCandidatesAnalysis(HealthcheckAnalysis):
    REASON_PREFIX: str = "This experiment cannot generate candidates.\nREASON: "
    LAST_RUN_TEMPLATE: str = "\n\nLAST TRIAL RUN: {days} day(s) ago"

    def __init__(
        self, can_generate_candidates: bool, reason: str, days_till_fail: int
    ) -> None:
        """
        Args:
            can_generate_candidates: Whether the experiment can generate candidates.  If
                True, the status is automatically set to PASS. If False, this
                ``Analysis`` will check when last trial was run and compare it to the
                threshold of ``days_till_fail``.
            reason: The reason why the experiment cannot generate candidates, or
                statement that it can.
            days_till_fail: The number of days since the last trial was run before
                the status is set to FAIL.
        """
        self.can_generate_candidates = can_generate_candidates
        self.reason = reason
        self.days_till_fail = days_till_fail

    def compute(
        self,
        experiment: Optional[Experiment] = None,
        generation_strategy: GenerationStrategyInterface | None = None,
    ) -> HealthcheckAnalysisCard:
        status = HealthcheckStatus.PASS
        subtitle = self.reason
        title_status = "Success"
        level = AnalysisCardLevel.LOW
        if not self.can_generate_candidates:
            subtitle = f"{self.REASON_PREFIX}{self.reason}"
            most_recent_run_time = max(
                [
                    t.time_run_started
                    for t in none_throws(experiment).trials.values()
                    if t.time_run_started is not None
                ],
                default=None,
            )
            if most_recent_run_time is None:
                status = HealthcheckStatus.FAIL
                level = AnalysisCardLevel.HIGH
                title_status = "Failure"
            else:
                days_since_last_run = (datetime.now() - most_recent_run_time).days
                if days_since_last_run > self.days_till_fail:
                    status = HealthcheckStatus.FAIL
                    level = AnalysisCardLevel.HIGH
                    title_status = "Failure"
                else:
                    status = HealthcheckStatus.WARNING
                    level = AnalysisCardLevel.MID
                    title_status = "Warning"
                subtitle += self.LAST_RUN_TEMPLATE.format(days=days_since_last_run)

        return HealthcheckAnalysisCard(
            name="CanGenerateCandidates",
            title=f"Ax Candidate Generation {title_status}",
            blob=json.dumps(
                {
                    "status": status,
                }
            ),
            subtitle=subtitle,
            df=pd.DataFrame(
                {
                    "status": [status],
                    "reason": [self.reason],
                }
            ),
            level=level,
        )
