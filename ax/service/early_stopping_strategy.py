#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment


class BaseEarlyStoppingStrategy:
    """Interface for heuristics that halt trials early, typically based on early
    results from that trial."""

    def should_stop_trial_early(
        self,
        trial_index: int,
        experiment: Experiment,
        **kwargs: Dict[str, Any],
    ) -> Optional[TrialStatus]:
        """Decide whether to complete a trial before evaluation is fully concluded.

        Typical examples include stopping a machine learning model's training, or
        halting the gathering of samples before some planned number are collected.


        Args:
            trial_index: index of candidate trial to stop early.
            experiment: Experiment that contains the trial and other contextual data,

        Returns:
            Suggested new status for this Trial. `None` means no suggested update.
        """
        return None
