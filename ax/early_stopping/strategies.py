#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.early_stopping.utils import align_partial_results
from ax.exceptions.core import UnsupportedError
from ax.utils.common.base import Base
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none

logger = get_logger(__name__)


class BaseEarlyStoppingStrategy(ABC, Base):
    """Interface for heuristics that halt trials early, typically based on early
    results from that trial."""

    def __init__(
        self,
        seconds_between_polls: int = 60,
        true_objective_metric_name: Optional[str] = None,
    ) -> None:
        """A BaseEarlyStoppingStrategy class.

        Args:
            seconds_between_polls: How often to poll the early stopping metric to
                evaluate whether or not the trial should be early stopped.
            true_objective_metric_name: The actual objective to be optimized; used in
                situations where early stopping uses a proxy objective (such as training
                loss instead of eval loss) for stopping decisions.
        """
        self._seconds_between_polls = seconds_between_polls
        self._true_objective_metric_name = true_objective_metric_name

    @abstractmethod
    def should_stop_trials_early(
        self,
        trial_indices: Set[int],
        experiment: Experiment,
        **kwargs: Dict[str, Any],
    ) -> Dict[int, Optional[str]]:
        """Decide whether to complete trials before evaluation is fully concluded.

        Typical examples include stopping a machine learning model's training, or
        halting the gathering of samples before some planned number are collected.


        Args:
            trial_indices: Indices of candidate trials to stop early.
            experiment: Experiment that contains the trials and other contextual data.

        Returns:
            A dictionary mapping trial indices that should be early stopped to
            (optional) messages with the associated reason.
        """
        pass  # pragma: nocover

    @property
    def true_objective_metric_name(self) -> Optional[str]:
        return self._true_objective_metric_name

    @true_objective_metric_name.setter
    def true_objective_metric_name(self, true_objective_metric_name: Optional[str]):
        self._true_objective_metric_name = true_objective_metric_name

    def _check_validity_and_get_data(self, experiment: Experiment) -> Optional[MapData]:
        """Validity checks and returns the `MapData` used for early stopping."""
        if experiment.optimization_config is None:
            raise UnsupportedError(  # pragma: no cover
                "Experiment must have an optimization config in order to use an "
                "early stopping strategy."
            )

        optimization_config = not_none(experiment.optimization_config)
        objective_name = optimization_config.objective.metric.name

        data = experiment.lookup_data()
        if data.df.empty:
            logger.info(
                f"{self.__class__.__name__} received empty data. "
                "Not stopping any trials."
            )
            return None
        if objective_name not in set(data.df["metric_name"]):
            logger.info(
                f"{self.__class__.__name__} did not receive data "
                "from the objective metric. Not stopping any trials."
            )
            return None

        if not isinstance(data, MapData):
            logger.info(
                f"{self.__class__.__name__} expects MapData, but the "
                f"data attached to experiment is of type {type(data)}. "
                "Not stopping any trials."
            )
            return None

        data = checked_cast(MapData, data)
        map_keys = data.map_keys
        if len(list(map_keys)) > 1:
            logger.info(
                f"{self.__class__.__name__} expects MapData with a single "
                "map key, but the data attached to the experiment has multiple: "
                f"{data.map_keys}. Not stopping any trials."
            )
            return None
        return data

    @property
    def seconds_between_polls(self) -> int:
        return self._seconds_between_polls

    @seconds_between_polls.setter
    def seconds_between_polls(self, seconds_between_polls: int) -> None:
        if seconds_between_polls < 0:
            raise ValueError("`seconds_between_polls may not be less than 0")

        self._seconds_between_polls = seconds_between_polls


class PercentileEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
    """Implements the strategy of stopping a trial if its performance
    falls below that of other trials at the same step."""

    def __init__(
        self,
        seconds_between_polls: int = 60,
        true_objective_metric_name: Optional[str] = None,
        percentile_threshold: float = 50.0,
        min_progression: float = 0.1,
        min_curves: float = 5,
        trial_indices_to_ignore: Optional[List[int]] = None,
    ) -> None:
        """Construct a PercentileEarlyStoppingStrategy instance.

        Args:
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
            seconds_between_polls=seconds_between_polls,
            true_objective_metric_name=true_objective_metric_name,
        )

        self.percentile_threshold = percentile_threshold
        self.min_progression = min_progression
        self.min_curves = min_curves
        self.trial_indices_to_ignore = trial_indices_to_ignore

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

        optimization_config = not_none(experiment.optimization_config)
        objective_name = optimization_config.objective.metric.name

        map_key = next(iter(data.map_keys))
        minimize = optimization_config.objective.minimize
        df = data.map_df
        try:
            metric_to_aligned_means, _ = align_partial_results(
                df=df,
                progr_key=map_key,
                metrics=[objective_name],
            )
        except Exception as e:
            logger.warning(
                f"Encountered exception while aligning data: {e}. "
                "Not early stopping any trials."
            )
            return {}

        aligned_means = metric_to_aligned_means[objective_name]
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
                return _log_and_return_trial_ignored(trial_index=trial_index)

        # check for no data
        if trial_index not in df or len(not_none(df[trial_index].dropna())) == 0:
            return _log_and_return_no_data(trial_index=trial_index)

        # check for min progression
        trial_last_progression = not_none(df[trial_index].dropna()).index.max()
        logger.info(
            f"Last progression of Trial {trial_index} is {trial_last_progression}."
        )
        if trial_last_progression < self.min_progression:
            return _log_and_return_min_progression(
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
            return _log_and_return_completed_trials(
                num_completed=num_completed, min_curves=self.min_curves
            )

        # check for enough number of trials with data
        if len(data_at_last_progression) < self.min_curves:
            return _log_and_return_num_trials_with_data(
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


class ThresholdEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
    """Implements the strategy of stopping a trial if its performance
    doesn't reach a pre-specified threshold by a certain progression."""

    def __init__(
        self,
        true_objective_metric_name: Optional[str] = None,
        metric_threshold: float = 0.2,
        min_progression: float = 10,
        trial_indices_to_ignore: Optional[List[int]] = None,
    ) -> None:
        """Construct a ThresholdEarlyStoppingStrategy instance.

        Args:
            true_objective_metric_name: The actual objective to be optimized; used in
                situations where early stopping uses a proxy objective (such as training
                loss instead of eval loss) for stopping decisions.
            metric_threshold: The metric threshold that a trial needs to reach by
                min_progression in order not to be stopped.
            min_progression: Only stop trials if the latest progression value
                (e.g. timestamp) is greater than this threshold.
            trial_indices_to_ignore: Trial indices that should not be early stopped.
        """
        super().__init__(true_objective_metric_name=true_objective_metric_name)

        self.metric_threshold = metric_threshold
        self.min_progression = min_progression
        self.trial_indices_to_ignore = trial_indices_to_ignore

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

        optimization_config = not_none(experiment.optimization_config)
        objective_name = optimization_config.objective.metric.name

        map_key = next(iter(data.map_keys))
        minimize = optimization_config.objective.minimize
        df = data.map_df
        df_objective = df[df["metric_name"] == objective_name]
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
                return _log_and_return_trial_ignored(trial_index=trial_index)

        # check for no data
        df_trial = df[df["trial_index"] == trial_index].dropna(subset=["mean"])
        if df_trial.empty:
            return _log_and_return_no_data(trial_index=trial_index)

        # check for min progression
        trial_last_progression = df_trial[map_key].max()
        logger.info(
            f"Last progression of Trial {trial_index} is {trial_last_progression}."
        )
        if trial_last_progression < self.min_progression:
            return _log_and_return_min_progression(
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


def _log_and_return_trial_ignored(trial_index: int) -> Tuple[bool, str]:
    """Helper function for logging/constructing a reason when a trial
    should be ignored."""
    logger.info(
        f"Trial {trial_index} should be ignored and not considered "
        "for early stopping."
    )
    return False, "Specified as a trial to be ignored for early stopping."


def _log_and_return_no_data(trial_index: int) -> Tuple[bool, str]:
    """Helper function for logging/constructing a reason when there is no data."""
    logger.info(
        f"There is not yet any data associated with trial {trial_index}. "
        "Not early stopping this trial."
    )
    return False, "No data available to make an early stopping decision."


def _log_and_return_min_progression(
    trial_index: int, trial_last_progression: float, min_progression: float
) -> Tuple[bool, str]:
    """Helper function for logging/constructing a reason when min progression
    is not yet reached."""
    reason = (
        f"Most recent progression ({trial_last_progression}) is less than "
        "the specified minimum progression for early stopping "
        f"({min_progression}). "
    )
    logger.info(f"Trial {trial_index}'s m{reason[1:]} Not early stopping this trial.")
    return False, reason


def _log_and_return_completed_trials(
    num_completed: int, min_curves: float
) -> Tuple[bool, str]:
    """Helper function for logging/constructing a reason when min number of
    completed trials is not yet reached."""
    logger.info(
        f"The number of completed trials ({num_completed}) is less than "
        "the minimum number of curves needed for early stopping "
        f"({min_curves}). Not early stopping this trial."
    )
    reason = (
        f"Need {min_curves} completed trials, but only {num_completed} "
        "completed trials so far."
    )
    return False, reason


def _log_and_return_num_trials_with_data(
    trial_index: int,
    trial_last_progression: float,
    num_trials_with_data: int,
    min_curves: float,
) -> Tuple[bool, str]:
    """Helper function for logging/constructing a reason when min number of
    trials with data is not yet reached."""
    logger.info(
        f"The number of trials with data ({num_trials_with_data}) "
        f"at trial {trial_index}'s last progression ({trial_last_progression}) "
        "is less than the specified minimum number for early stopping "
        f"({min_curves}). Not early stopping this trial."
    )
    reason = (
        f"Number of trials with data ({num_trials_with_data}) at "
        f"last progression ({trial_last_progression}) is less than the "
        f"specified minimum number for early stopping ({min_curves})."
    )
    return False, reason
