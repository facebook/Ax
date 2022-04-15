#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from ax.core.experiment import Experiment
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.exceptions.core import UnsupportedError
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none

logger = get_logger(__name__)


class ThresholdEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
    """Implements the strategy of stopping a trial if its performance
    doesn't reach a pre-specified threshold by a certain progression."""

    def __init__(
        self,
        metric_names: Optional[Iterable[str]] = None,
        seconds_between_polls: int = 60,
        true_objective_metric_name: Optional[str] = None,
        metric_threshold: float = 0.2,
        min_progression: float = 10,
        trial_indices_to_ignore: Optional[List[int]] = None,
    ) -> None:
        """Construct a ThresholdEarlyStoppingStrategy instance.

        Args
            metric_names: A (length-one) list of name of the metric to observe. If
                None will default to the objective metric on the Experiment's
                OptimizationConfig.
            seconds_between_polls: How often to poll the early stopping metric to
                evaluate whether or not the trial should be early stopped.
            true_objective_metric_name: The actual objective to be optimized; used in
                situations where early stopping uses a proxy objective (such as training
                loss instead of eval loss) for stopping decisions.
            metric_threshold: The metric threshold that a trial needs to reach by
                min_progression in order not to be stopped.
            min_progression: Only stop trials if the latest progression value
                (e.g. timestamp) is greater than this threshold.
            trial_indices_to_ignore: Trial indices that should not be early stopped.
        """
        super().__init__(
            metric_names=metric_names,
            seconds_between_polls=seconds_between_polls,
            true_objective_metric_name=true_objective_metric_name,
        )
        self.metric_threshold = metric_threshold
        self.min_progression = min_progression
        self.trial_indices_to_ignore = trial_indices_to_ignore

        if metric_names is not None and len(list(metric_names)) > 1:
            raise UnsupportedError(
                "ThresholdEarlyStoppingStrategy only supports a single metric. Use "
                "LogicalEarlyStoppingStrategy to compose early stopping strategies "
                "with multiple metrics."
            )

    def should_stop_trials_early(
        self,
        trial_indices: Set[int],
        experiment: Experiment,
        **kwargs: Dict[str, Any],
    ) -> Dict[int, Optional[str]]:
        """Stop a trial if its performance doesn't reach a pre-specified threshold
        by `min_progression`.

        Args:
            trial_indices: Indices of candidate trials to consider for early stopping.
            experiment: Experiment that contains the trials and other contextual data.

        Returns:
            A dictionary mapping trial indices that should be early stopped to
            (optional) messages with the associated reason. An empty dictionary
            means no suggested updates to any trial's status.
        """
        data = self._check_validity_and_get_data(experiment=experiment)
        if data is None:
            # don't stop any trials if we don't get data back
            return {}

        if self.metric_names is None:
            optimization_config = not_none(experiment.optimization_config)
            metric_name = optimization_config.objective.metric.name
        else:
            metric_name = list(self.metric_names)[0]

        map_key = next(iter(data.map_keys))
        minimize = optimization_config.objective.minimize
        df = data.map_df
        df_objective = df[df["metric_name"] == metric_name]
        decisions = {
            trial_index: self.should_stop_trial_early(
                trial_index=trial_index,
                experiment=experiment,
                df=df_objective,
                map_key=map_key,
                minimize=minimize,
            )
            for trial_index in trial_indices
        }
        return {
            trial_index: reason
            for trial_index, (should_stop, reason) in decisions.items()
            if should_stop
        }

    def should_stop_trial_early(
        self,
        trial_index: int,
        experiment: Experiment,
        df: pd.DataFrame,
        map_key: str,
        minimize: bool,
    ) -> Tuple[bool, Optional[str]]:
        """Stop a trial if its performance doesn't reach a pre-specified threshold
        by `min_progression`.

        Args:
            trial_index: Indices of candidate trial to stop early.
            experiment: Experiment that contains the trials and other contextual data.
            df: Dataframe of partial results for the objective metric.
            map_key: Name of the column of the dataset that indicates progression.
            minimize: Whether objective value is being minimized.

        Returns:
            A tuple `(should_stop, reason)`, where `should_stop` is `True` iff the
            trial should be stopped, and `reason` is an (optional) string providing
            information on why the trial should or should not be stopped.
        """
        logger.info(f"Considering trial {trial_index} for early stopping.")

        # check for ignored indices
        if self.trial_indices_to_ignore is not None:
            if trial_index in self.trial_indices_to_ignore:
                return self._log_and_return_trial_ignored(
                    logger=logger, trial_index=trial_index
                )

        # check for no data
        df_trial = df[df["trial_index"] == trial_index].dropna(subset=["mean"])
        if df_trial.empty:
            return self._log_and_return_no_data(logger=logger, trial_index=trial_index)

        # check for min progression
        trial_last_progression = df_trial[map_key].max()
        logger.info(
            f"Last progression of Trial {trial_index} is {trial_last_progression}."
        )
        if trial_last_progression < self.min_progression:
            return self._log_and_return_min_progression(
                logger=logger,
                trial_index=trial_index,
                trial_last_progression=trial_last_progression,
                min_progression=self.min_progression,
            )

        # threshold early stopping logic
        data_at_last_progression = df_trial[
            df_trial[map_key] == trial_last_progression
        ]["mean"].iloc[0]
        logger.info(
            "Early stopping objective at last progression is:\n"
            f"{data_at_last_progression}."
        )
        should_early_stop = (
            data_at_last_progression > self.metric_threshold
            if minimize
            else data_at_last_progression < self.metric_threshold
        )
        comp = "worse" if should_early_stop else "better"
        reason = (
            f"Trial objective value {data_at_last_progression} is {comp} than "
            f"the metric threshold {self.metric_threshold:}."
        )
        logger.info(
            f"Early stopping decision for {trial_index}: {should_early_stop}. "
            f"Reason: {reason}"
        )
        return should_early_stop, reason
