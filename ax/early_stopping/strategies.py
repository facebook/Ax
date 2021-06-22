#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Dict, Set

import numpy as np
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.exceptions.core import UnsupportedError
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none

logger = get_logger(__name__)


class BaseEarlyStoppingStrategy(ABC):
    """Interface for heuristics that halt trials early, typically based on early
    results from that trial."""

    @abstractmethod
    def should_stop_trials_early(
        self,
        trial_indices: Set[int],
        experiment: Experiment,
        **kwargs: Dict[str, Any],
    ) -> Set[int]:
        """Decide whether to complete trials before evaluation is fully concluded.

        Typical examples include stopping a machine learning model's training, or
        halting the gathering of samples before some planned number are collected.


        Args:
            trial_indices: Indices of candidate trials to stop early.
            experiment: Experiment that contains the trials and other contextual data.

        Returns:
            Set of trial indices that should be early stopped. Empty set means
            no suggested update.
        """
        pass  # pragma: nocover


class PercentileEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
    """Implements the strategy of stopping a trial if its performance
    falls below that of other trials at the same step."""

    def __init__(
        self,
        percentile_threshold: float = 50.0,
        min_progression: float = 0.1,
        min_curves: float = 5,
    ) -> None:
        """Construct a PercentileEarlyStoppingStrategy instance.

        Args:
            percentile_threshold: Falling below this threshold compared to other trials
                at the same step will stop the run. Must be between 0.0 and 100.0.
                e.g. if percentile_threshold=25.0, the bottom 25% of trials are stopped.
            min_progression: Only stop trials if the latest progression value
                (e.g. timestamp) is greater than this threshold. Prevents stopping
                prematurely before enough data is gathered to make a decision.
            min_curves: Minimum number of trial curves that need to be available to
                make a stopping decision.
        """
        self.percentile_threshold = percentile_threshold
        self.min_progression = min_progression
        self.min_curves = min_curves

    def should_stop_trials_early(
        self,
        trial_indices: Set[int],
        experiment: Experiment,
        **kwargs: Dict[str, Any],
    ) -> Set[int]:
        """Stop a trial if its performance is in the bottom `percentile_threshold`
        of the trials at the same step.

        Args:
            trial_indices: Indices of candidate trials to stop early.
            experiment: Experiment that contains the trials and other contextual data.

        Returns:
            Set of trial indices that should be early stopped. Empty set means
            no suggested update.
        """
        if experiment.optimization_config is None:
            raise UnsupportedError(  # pragma: no cover
                "Experiment must have an optimization config in order to use an "
                "early stopping strategy."
            )

        optimization_config = not_none(experiment.optimization_config)
        objective_name = optimization_config.objective.metric.name
        minimize = optimization_config.objective.minimize

        data = experiment.lookup_data()
        if data.df.empty:
            raise ValueError("PercentileEarlyStoppingStrategy expects non-empty data.")

        if not isinstance(data, MapData):
            raise ValueError(
                "PercentileEarlyStoppingStrategy expects MapData, but the "
                f"data attached to experiment is of type {type(data)}."
            )

        map_keys = data.map_keys
        if len(map_keys) > 1:
            raise ValueError(  # pragma: no cover
                "PercentileEarlyStoppingStrategy expects MapData with a single "
                "map key, but the data attached to the experiment has multiple: "
                f"{data.map_keys}."
            )
        map_key = map_keys[0]

        df = data.df
        df = df[df.metric_name == objective_name]

        last_progression = df[map_key].max()
        if last_progression < self.min_progression:
            logger.info(
                f"Most recent progression ({last_progression}) is less than the "
                "specified minimum progression for early stopping "
                f"({self.min_progression}). "
                "Not early stopping any trials."
            )
            return set()

        # TODO: Apply smoothing to account for the case when different
        # trials report results at different progressions.
        data_at_last_progression = df[df[map_key] == last_progression]

        if len(data_at_last_progression) < self.min_curves:
            logger.info(
                f"The number of trials with data ({len(data_at_last_progression)}) "
                "is less than the specified minimum number for early stopping "
                f"({self.min_curves}). "
                "Not early stopping any trials."
            )
            return set()

        objective_value_at_last_progression = data_at_last_progression["mean"]
        percentile_threshold = (
            100.0 - self.percentile_threshold if minimize else self.percentile_threshold
        )
        percentile_value = np.percentile(
            objective_value_at_last_progression, percentile_threshold
        )
        worse_than_percentile_rows = (
            objective_value_at_last_progression > percentile_value
            if minimize
            else objective_value_at_last_progression < percentile_value
        )
        return set(data_at_last_progression[worse_than_percentile_rows]["trial_index"])
