#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from ax.core.experiment import Experiment


class BaseEarlyStoppingStrategy:
    """Interface for heuristics that halt trials early, typically based on early
    results from that trial."""

    def should_early_stop_trial(
        self,
        trial_index: int,
        experiment: Experiment,
        **kwargs: Dict[str, Any],
    ) -> bool:
        """Decide whether to complete a trial before evaluation is fully concluded.

        Typical examples include stopping a machine learning model's training, or
        halting the gathering of samples before some planned number are collected.


        Args:
            trial_index: index of candidate trial to stop early.
            experiment: Experiment that contains the trial and other contextual data,

        Returns:
            Whether to early-stop this trial.
        """
        return False
