#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.early_stopping.utils import align_partial_results
from ax.exceptions.core import UnsupportedError
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none

logger = get_logger(__name__)


class PercentileEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
    """Implements the strategy of stopping a trial if its performance
    falls below that of other trials at the same step."""

    def __init__(
        self,
        metric_names: Optional[Iterable[str]] = None,
        seconds_between_polls: int = 60,
        true_objective_metric_name: Optional[str] = None,
        percentile_threshold: float = 50.0,
        min_progression: float = 0.1,
        min_curves: float = 5,
        trial_indices_to_ignore: Optional[List[int]] = None,
    ) -> None:
        """Construct a PercentileEarlyStoppingStrategy instance.

        Args:
            metric_names: A (length-one) list of name of the metric to observe. If
                None will default to the objective metric on the Experiment's
                OptimizationConfig.
            seconds_between_polls: How often to poll the early stopping metric to
                evaluate whether or not the trial should be early stopped.
            true_objective_metric_name: The actual objective to be optimized; used in
                situations where early stopping uses a proxy objective (such as training
                loss instead of eval loss) for stopping decisions.
            percentile_threshold: Falling below this threshold compared to other trials
                at the same step will stop the run. Must be between 0.0 and 100.0.
                e.g. if percentile_threshold=25.0, the bottom 25% of trials are stopped.
                Note that "bottom" here is determined based on performance, not
                absolute values; if `minimize` is False, then "bottom" actually refers
                to the top trials in terms of metric value.
            min_progression: Only stop trials if the latest progression value
                (e.g. timestamp, epochs, training data used) is greater than this
                threshold. Prevents stopping prematurely before enough data is gathered
                to make a decision. The default value (10) is reasonable when we want
                early stopping to start after 10 epochs.
            min_curves: There must be `min_curves` number of completed trials and
                `min_curves` number of trials with curve data to make a stopping
                decision (i.e., even if there are enough completed trials but not all
                of them are correctly returning data, then do not apply early stopping).
            trial_indices_to_ignore: Trial indices that should not be early stopped.
        """
        super().__init__(
            metric_names=metric_names,
            seconds_between_polls=seconds_between_polls,
            true_objective_metric_name=true_objective_metric_name,
        )

        self.percentile_threshold = percentile_threshold
        self.min_progression = min_progression
        self.min_curves = min_curves
        self.trial_indices_to_ignore = trial_indices_to_ignore

        if metric_names is not None and len(list(metric_names)) > 1:
            raise UnsupportedError(
                "PercentileEarlyStoppingStrategy only supports a single metric. Use "
                "LogicalEarlyStoppingStrategy to compose early stopping strategies "
                "with multiple metrics."
            )

    def should_stop_trials_early(
        self,
        trial_indices: Set[int],
        experiment: Experiment,
        **kwargs: Dict[str, Any],
    ) -> Dict[int, Optional[str]]:
        """Stop a trial if its performance is in the bottom `percentile_threshold`
        of the trials at the same step.

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
            minimize = optimization_config.objective.minimize
        else:
            metric_name = list(self.metric_names)[0]
            minimize = experiment.metrics[metric_name].lower_is_better or False

        map_key = next(iter(data.map_keys))
        df = data.map_df
        try:
            metric_to_aligned_means, _ = align_partial_results(
                df=df,
                progr_key=map_key,
                metrics=[metric_name],
            )
        except Exception as e:
            logger.warning(
                f"Encountered exception while aligning data: {e}. "
                "Not early stopping any trials."
            )
            return {}

        aligned_means = metric_to_aligned_means[metric_name]
        decisions = {
            trial_index: self.should_stop_trial_early(
                trial_index=trial_index,
                experiment=experiment,
                df=aligned_means,
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
        minimize: bool,
    ) -> Tuple[bool, Optional[str]]:
        """Stop a trial if its performance is in the bottom `percentile_threshold`
        of the trials at the same step.

        Args:
            trial_index: Indices of candidate trial to stop early.
            experiment: Experiment that contains the trials and other contextual data.
            df: Dataframe of partial results after applying interpolation,
                filtered to objective metric.
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
        if trial_index not in df or len(not_none(df[trial_index].dropna())) == 0:
            return self._log_and_return_no_data(logger=logger, trial_index=trial_index)

        # check for min progression
        trial_last_progression = not_none(df[trial_index].dropna()).index.max()
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

        # dropna() here will exclude trials that have not made it to the
        # last progression of the trial under consideration, and therefore
        # can't be included in the comparison
        data_at_last_progression = df.loc[trial_last_progression].dropna()
        logger.info(
            "Early stopping objective at last progression is:\n"
            f"{data_at_last_progression}."
        )

        # check for enough completed trials
        num_completed = len(experiment.trial_indices_by_status[TrialStatus.COMPLETED])
        if num_completed < self.min_curves:
            return self._log_and_return_completed_trials(
                logger=logger, num_completed=num_completed, min_curves=self.min_curves
            )

        # check for enough number of trials with data
        if len(data_at_last_progression) < self.min_curves:
            return self._log_and_return_num_trials_with_data(
                logger=logger,
                trial_index=trial_index,
                trial_last_progression=trial_last_progression,
                num_trials_with_data=len(data_at_last_progression),
                min_curves=self.min_curves,
            )

        # percentile early stopping logic
        percentile_threshold = (
            100.0 - self.percentile_threshold if minimize else self.percentile_threshold
        )
        percentile_value = np.percentile(data_at_last_progression, percentile_threshold)
        trial_objective_value = data_at_last_progression[trial_index]
        should_early_stop = (
            trial_objective_value > percentile_value
            if minimize
            else trial_objective_value < percentile_value
        )
        comp = "worse" if should_early_stop else "better"
        reason = (
            f"Trial objective value {trial_objective_value} is {comp} than "
            f"{percentile_threshold:.1f}-th percentile ({percentile_value}) "
            "across comparable trials."
        )
        logger.info(
            f"Early stopping decision for {trial_index}: {should_early_stop}. "
            f"Reason: {reason}"
        )
        return should_early_stop, reason
