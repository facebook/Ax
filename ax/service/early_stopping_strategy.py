#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Set

from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment


class BaseEarlyStoppingStrategy:
    """Interface for heuristics that halt trials early, typically based on early
    results from that trial."""

    def should_stop_trials_early(
        self,
        trial_indices: Set[int],
        experiment: Experiment,
        **kwargs: Dict[str, Any],
    ) -> Dict[int, Optional[TrialStatus]]:
        """Decide whether to complete trials before evaluation is fully concluded.

        Typical examples include stopping a machine learning model's training, or
        halting the gathering of samples before some planned number are collected.


        Args:
            trial_indices: Indices of candidate trials to stop early.
            experiment: Experiment that contains the trials and other contextual data.

        Returns:
            Dictionary mapping from trial index to suggested new status. `None` means
            no suggested update.
        """
        return {i: None for i in trial_indices}
