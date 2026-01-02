#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from logging import Logger
from typing import cast

import pandas as pd
from ax.adapter.data_utils import _maybe_normalize_map_key
from ax.core.batch_trial import BatchTrial

from ax.core.data import Data, MAP_KEY
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective
from ax.core.trial_status import TrialStatus
from ax.early_stopping.utils import (
    _interval_boundary,
    align_partial_results,
    estimate_early_stopping_savings,
)
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.generation_strategy.generation_node import GenerationNode
from ax.utils.common.base import Base
from ax.utils.common.logger import get_logger
from pyre_extensions import none_throws

logger: Logger = get_logger(__name__)


# Kwargs removed from early stopping strategies that should be discarded for
# backwards compatibility when loading old strategies.
REMOVED_EARLY_STOPPING_STRATEGY_KWARGS: set[str] = {"trial_indices_to_ignore"}


class BaseEarlyStoppingStrategy(ABC, Base):
    """Interface for heuristics that halt trials early, typically based on early
    results from that trial."""

    def __init__(
        self,
        metric_signatures: Iterable[str] | None = None,
        min_progression: float | None = None,
        max_progression: float | None = None,
        min_curves: int | None = None,
        normalize_progressions: bool = False,
        interval: float | None = None,
        check_safe: bool = False,
    ) -> None:
        """A BaseEarlyStoppingStrategy class.

        Args:
            metric_signatures: The names of the metrics the strategy will interact with.
                If no metric names are provided, considers the objective metric(s).
            min_progression: Only stop trials if the latest progression value
                (e.g. timestamp, epochs, training data used) is greater than this
                threshold. Prevents stopping prematurely before enough data is gathered
                to make a decision.
            max_progression: Do not stop trials that have passed `max_progression`.
                Useful if we prefer finishing a trial that are already near completion.
            min_curves: Trials will not be stopped until a number of trials
                `min_curves` have completed with curve data attached. That is, if
                `min_curves` trials are completed but their curve data was not
                successfully retrieved, further trials may not be early-stopped.
            normalize_progressions: If True, normalizes the progression values
                for each metric to the [0, 1] range using the observed minimum and
                maximum progression values for that metric. This transformation maps
                the original progression range [`min_prog`, `max_prog`] to [0, 1]
                via (x - min_prog) / (max_prog - min_prog). Useful when progression
                values have arbitrary scales or when you want to specify
                `min_progression` and `max_progression` in a normalized [0, 1]
                space. Note: At least one trial should have completed (i.e.,
                `min_curves` > 0) to ensure reliable estimates of the progression
                range.
            interval: If specified, throttles early-stopping checks by only
                evaluating trials when they cross interval boundaries. Boundaries are
                defined at `min_progression + k * interval` for k = 0, 1, 2, etc.
                A trial is eligible if it's being checked for the first time, or if
                its progression has crossed a boundary since the last check. This
                prevents premature stopping decisions when the orchestrator polls
                frequently. For example, with `interval=10` and `min_progression=0`,
                boundaries are at 0, 10, 20, 30, etc. A trial at progression 15 is
                eligible on first check. If checked again at progression 18, it's not
                eligible (both in interval [10, 20)). Once it reaches progression 21,
                it's eligible again (crossed into interval [20, 30)).
            check_safe: If True, applies the relevant safety checks to gate
                early-stopping when it is likely to be harmful. If False (default),
                bypasses the safety check and directly applies early-stopping decisions.
        """
        # Validate interval
        if interval is not None and not interval > 0:
            raise UserInputError(f"Option `interval` must be positive (got {interval})")

        # Validate min_progression and max_progression
        if min_progression is not None and min_progression < 0:
            raise UserInputError(
                f"Option `min_progression` must be nonnegative (got {min_progression})"
            )

        if min_progression is not None and max_progression is not None:
            if min_progression >= max_progression:
                raise UserInputError(
                    f"Expect min_progression < max_progression, got "
                    f"min_progression={min_progression}, "
                    f"max_progression={max_progression}"
                )

        self.metric_signatures = metric_signatures
        self.min_progression = min_progression
        self.max_progression = max_progression
        self.min_curves = min_curves
        self.normalize_progressions = normalize_progressions
        self.interval = interval
        self.check_safe = check_safe
        # Track the last progression value where each trial was checked
        self._last_check_progressions: dict[int, float] = {}

    @abstractmethod
    def _should_stop_trials_early(
        self,
        trial_indices: set[int],
        experiment: Experiment,
        current_node: GenerationNode | None = None,
    ) -> dict[int, str | None]:
        """Decide whether to complete trials before evaluation is fully concluded.

        Typical examples include stopping a machine learning model's training, or
        halting the gathering of samples before some planned number are collected.

        Args:
            trial_indices: Indices of candidate trials to stop early.
            experiment: Experiment that contains the trials and other contextual data.
            current_node: The current ``GenerationNode`` on the ``GenerationStrategy``
                used to generate trials for the ``Experiment``. Early stopping
                strategies may utilize components of the current node when making
                stopping decisions.

        Returns:
            A dictionary mapping trial indices that should be early stopped to
            (optional) messages with the associated reason.
        """
        pass

    @abstractmethod
    def _is_harmful(
        self,
        trial_indices: set[int],
        experiment: Experiment,
    ) -> bool:
        """Determine if applying early stopping would be harmful to the experiment,
        e.g. if there are clear risks of prematurely eliminating optimal trials.

        Args:
            trial_indices: Indices of candidate trials to stop early.
            experiment: Experiment that contains the trials and other contextual data.

        Returns:
            True if early stopping should not be applied, False otherwise.
        """
        pass

    def should_stop_trials_early(
        self,
        trial_indices: set[int],
        experiment: Experiment,
        current_node: GenerationNode | None = None,
    ) -> dict[int, str | None]:
        """Decide whether trials should be stopped before evaluation is fully concluded.
        This method identifies trials that should be stopped based on early signals that
        are indicative of final performance. Early stopping is not applied if doing so
        would risk prematurely eliminating potentially optimal trials.

        Args:
            trial_indices: Indices of candidate trials to stop early.
            experiment: Experiment that contains the trials and other contextual data.
            current_node: The current ``GenerationNode`` on the ``GenerationStrategy``
                used to generate trials for the ``Experiment``. Early stopping
                strategies may utilize components of the current node when making
                stopping decisions.

        Returns:
            A dictionary mapping trial indices that should be early stopped to
            (optional) messages with the associated reason. Returns an empty
            dictionary if early stopping would be harmful (when safety check is
            enabled).
        """
        if self.check_safe and self._is_harmful(
            trial_indices=trial_indices,
            experiment=experiment,
        ):
            return {}
        return self._should_stop_trials_early(
            trial_indices=trial_indices,
            experiment=experiment,
            current_node=current_node,
        )

    def estimate_early_stopping_savings(self, experiment: Experiment) -> float:
        """Estimate early stopping savings using progressions of the MapMetric present
        on the EarlyStoppingConfig as a proxy for resource usage.

        Args:
            experiment: The experiment containing the trials and metrics used
                to estimate early stopping savings.

        Returns:
            The estimated resource savings as a fraction of total resource usage (i.e.
            0.11 estimated savings indicates we would expect the experiment to have used
            11% more resources without early stopping present)
        """
        if not experiment.lookup_data().has_step_column:
            return 0.0

        return estimate_early_stopping_savings(experiment=experiment)

    def _lookup_and_validate_data(
        self, experiment: Experiment, metric_signatures: list[str]
    ) -> Data | None:
        """Looks up and validates the `Data` used for early stopping that
        is associated with `metric_signatures`. This function also handles normalizing
        progressions.
        """
        data = experiment.lookup_data()
        if data.df.empty:
            logger.info(
                f"{self.__class__.__name__} received empty data. "
                "Not stopping any trials."
            )
            return None
        for metric_signature in metric_signatures:
            if metric_signature not in set(data.df["metric_signature"]):
                logger.info(
                    f"{self.__class__.__name__} did not receive data from the "
                    f"objective metric `{metric_signature}`. Not stopping any trials."
                )
                return None

        if not data.has_step_column:
            logger.info(
                f"{self.__class__.__name__} expects the data attached to the "
                f"to have a column '{MAP_KEY}' (representing progression), but "
                "it does not. Not stopping any trials."
            )
            return None

        full_df = data.full_df
        full_df = full_df[full_df["metric_signature"].isin(metric_signatures)]

        # Drop rows with NaN values in MAP_KEY column to prevent issues in
        # align_partial_results which uses MAP_KEY as the pivot index
        nan_mask = full_df[MAP_KEY].isna()
        if nan_mask.any():
            num_nan_rows = nan_mask.sum()
            nan_trial_indices = full_df.loc[nan_mask, "trial_index"].unique().tolist()
            logger.warning(
                f"Dropped {num_nan_rows} row(s) with NaN values in the progression "
                f"column ('{MAP_KEY}') for trial(s) {nan_trial_indices}."
            )
            full_df = full_df[~nan_mask]

        if self.normalize_progressions:
            full_df = _maybe_normalize_map_key(df=full_df)
        return Data(df=full_df)

    @staticmethod
    def _log_and_return_no_data(
        logger: logging.Logger, trial_index: int, metric_name: str
    ) -> tuple[bool, str]:
        """Helper function for logging/constructing a reason when there is no data."""
        logger.info(
            f"There is not yet any data associated with trial {trial_index} and "
            f"metric {metric_name}. Not early stopping this trial."
        )
        return False, "No data available to make an early stopping decision."

    @staticmethod
    def _log_and_return_progression_range(
        logger: logging.Logger,
        trial_index: int,
        trial_last_progression: float,
        min_progression: float | None,
        max_progression: float | None,
        metric_name: str,
    ) -> tuple[bool, str]:
        """Helper function for logging/constructing a reason when min progression
        is not yet reached."""
        reason = (
            f"Most recent progression ({trial_last_progression}) that is available for "
            f"metric {metric_name} falls out of the "
            f"min/max_progression range ({min_progression}, {max_progression})."
        )
        logger.info(
            f"Trial {trial_index}'s m{reason[1:]} Not early stopping this trial."
        )
        return False, reason

    @staticmethod
    def _log_and_return_interval_boundary(
        logger: logging.Logger,
        trial_index: int,
        prev_progression: float,
        curr_progression: float,
        interval: float,
        min_progression: float,
    ) -> tuple[bool, str]:
        """Helper function for logging/constructing a reason when interval
        boundary has not been crossed."""
        # Calculate the interval boundaries
        curr_boundary = _interval_boundary(prev_progression, min_progression, interval)
        next_boundary = curr_boundary + interval

        reason = (
            f"Trial has not crossed an interval boundary. "
            f"Progressed from {prev_progression:.2f} to {curr_progression:.2f}; "
            f"both are in the same interval [{curr_boundary:.2f}, "
            f"{next_boundary:.2f}). "
            f"Must reach progression {next_boundary:.2f} to be eligible for "
            f"early stopping."
        )
        logger.info(f"Trial {trial_index} not eligible for early stopping: {reason}")
        return False, reason

    def is_eligible_any(
        self,
        trial_indices: set[int],
        experiment: Experiment,
        df: pd.DataFrame,
    ) -> bool:
        """Perform a series of default checks for a set of trials `trial_indices` and
        determine if at least one of them is eligible for further stopping logic:
            1. Check that at least `self.min_curves` trials have completed`
            2. Check that at least one trial has reached `self.min_progression`
        Returns a boolean indicating if all checks are passed.

        This is useful for some situations where if no trials are eligible for stopping,
        then we can skip costly steps, such as model fitting, that occur before
        individual trials are considered for stopping.
        """
        # check for batch trials
        for idx, trial in experiment.trials.items():
            if isinstance(trial, BatchTrial):
                # In particular, align_partial_results requires a 1-1 mapping between
                # trial indices and arm names, which is not the case for batch trials.
                # See align_partial_results for more details.
                raise ValueError(
                    f"Trial {idx} is a BatchTrial, which is not yet supported by "
                    "early stopping strategies."
                )

        # check that there are sufficient completed trials
        num_completed = len(experiment.trial_indices_by_status[TrialStatus.COMPLETED])
        if self.min_curves is not None and num_completed < self.min_curves:
            return False

        # check that at least one trial has reached `self.min_progression`
        df_trials = df[df["trial_index"].isin(trial_indices)].dropna(subset=["mean"])
        any_last_prog = 0
        if not df_trials[MAP_KEY].empty:
            any_last_prog = df_trials[MAP_KEY].max()

        if self.min_progression is not None and any_last_prog < self.min_progression:
            # No trials have reached `self.min_progression`, not stopping any trials
            return False

        return True

    def is_eligible(
        self,
        trial_index: int,
        experiment: Experiment,
        df: pd.DataFrame,
    ) -> tuple[bool, str | None]:
        """Perform a series of default checks for a specific trial `trial_index` and
        determines whether it is eligible for further stopping logic:
            1. Check that `df` contains data for the trial `trial_index`
            2. Check that the trial has reached `self.min_progression`
            3. Check that the trial hasn't surpassed `self.max_progression`
            4. Check that the trial has progressed sufficiently since the last
               early-stopping decision (based on `self.interval`)
        Returns two elements: a boolean indicating if all checks are passed and a
        str indicating the reason that early stopping is not applied (None if all
        checks pass).

        Args:
            trial_index: The index of the trial to check.
            experiment: The experiment containing the trial.
            df: A dataframe containing the time-dependent metrics for the trial.
                NOTE: `df` should only contain data with `metric_signature` fields that
                are associated with the early stopping strategy. This is usually done
                automatically in `_check_validity_and_get_data`. `is_eligible` might
                otherwise return False even though the trial is eligible, if there are
                secondary tracking metrics that are in `df` but shouldn't be considered
                in the early stopping decision.

        Returns:
            A tuple of two elements: a boolean indicating if the trial is eligible and
                an optional string indicating any reason for ineligiblity.
        """
        # Check eligibility of each metric.
        for metric_signature, metric_df in df.groupby("metric_signature"):
            # check for no data
            metric_name = experiment.signature_to_metric[metric_signature].name
            df_trial = metric_df[metric_df["trial_index"] == trial_index]
            df_trial = df_trial.dropna(subset=["mean"])
            if df_trial.empty:
                return self._log_and_return_no_data(
                    logger=logger, trial_index=trial_index, metric_name=metric_name
                )

            # check for min/max progression
            trial_last_prog = df_trial[MAP_KEY].max()
            if (
                self.min_progression is not None
                and trial_last_prog < self.min_progression
            ) or (
                self.max_progression is not None
                and trial_last_prog > self.max_progression
            ):
                return self._log_and_return_progression_range(
                    logger=logger,
                    trial_index=trial_index,
                    trial_last_progression=trial_last_prog,
                    min_progression=self.min_progression,
                    max_progression=self.max_progression,
                    metric_name=metric_name,
                )

            # check for progression interval
            if self.interval is not None:
                last_check_progression = self._last_check_progressions.get(trial_index)
                if last_check_progression is None:
                    logger.debug(
                        f"Trial {trial_index} is being checked for early stopping "
                        f"for the first time at progression {trial_last_prog}."
                    )
                elif not self._has_crossed_interval_boundary(
                    prev_progression=last_check_progression,
                    curr_progression=trial_last_prog,
                ):
                    min_progression = (
                        0.0 if self.min_progression is None else self.min_progression
                    )
                    return self._log_and_return_interval_boundary(
                        logger=logger,
                        trial_index=trial_index,
                        prev_progression=last_check_progression,
                        curr_progression=trial_last_prog,
                        interval=none_throws(self.interval),
                        min_progression=min_progression,
                    )
                # Update the last checked progression for this trial
                self._last_check_progressions[trial_index] = trial_last_prog

        return True, None

    def _has_crossed_interval_boundary(
        self, prev_progression: float, curr_progression: float
    ) -> bool:
        """Check if the trial has crossed an interval boundary.

        This prevents drift by checking if we've passed a boundary defined by
        the interval parameter, rather than just checking if enough progression
        has occurred since the last check.

        Args:
            prev_progression: The progression value when trial was last checked.
            curr_progression: The current progression value.

        Returns:
            True if we've crossed an interval boundary, False otherwise.
        """
        min_progression = 0.0 if self.min_progression is None else self.min_progression
        interval = none_throws(self.interval)
        curr_interval_boundary = cast(
            float,
            _interval_boundary(
                progression=curr_progression,
                min_progression=min_progression,
                interval=interval,
            ),
        )
        # We've crossed a boundary if the last check was before the current boundary
        return prev_progression < curr_interval_boundary

    def _default_objective_and_direction(
        self, experiment: Experiment
    ) -> tuple[str, bool]:
        objectives_to_directions = self._all_objectives_and_directions(
            experiment=experiment
        )
        # if it is a multi-objective optimization problem, infer as first objective
        # although it is recommended to specify metric names explicitly.
        return next(iter(objectives_to_directions.items()))

    def _all_objectives_and_directions(self, experiment: Experiment) -> dict[str, bool]:
        """A dictionary mapping metric signatures to corresponding directions, i.e.
        a Boolean indicating whether the objective is minimized, for each objective in
        the experiment or in `self.metric_signatures`, if specified.

        Args:
            experiment: The experiment containing the optimization config.

        Returns: A dictionary mapping metric names to a Boolean indicating whether
            the objective is being minimized.
        """
        if self.metric_signatures is None:
            logger.debug(
                "No metric signatures specified. "
                "Defaulting to the objective metric(s).",
                stacklevel=2,
            )
            optimization_config = none_throws(experiment.optimization_config)
            objective = optimization_config.objective
            objectives = (
                objective.objectives
                if isinstance(objective, MultiObjective)
                else [objective]
            )
            directions = {}
            for objective in objectives:
                metric_signature = objective.metric.signature
                directions[metric_signature] = objective.minimize

        else:
            metric_signatures = list(self.metric_signatures)
            directions = {}
            metrics_without_direction = []
            for metric_signature in metric_signatures:
                metric = experiment.signature_to_metric[metric_signature]
                if metric.lower_is_better is None:
                    metrics_without_direction.append(metric_signature)
                else:
                    directions[metric_signature] = metric.lower_is_better

            if metrics_without_direction:
                raise UnsupportedError(
                    "Metrics used for early stopping must specify lower_is_better. "
                    f"The following metrics do not specify lower_is_better: "
                    f"{metrics_without_direction}."
                )

        return directions

    def _prepare_aligned_data(
        self, experiment: Experiment, metric_signatures: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """Get raw experiment data and align it for early stopping evaluation.

        Args:
            experiment: Experiment that contains the trials and other contextual data.
            metric_signatures: List of metric signatures to include in the aligned data.

        Returns:
            A tuple of (long_df, multilevel_wide_df) where:
            - long_df: The raw Data-style dataframe (long format) before
                interpolation
            - multilevel_wide_df: Hierarchical wide dataframe (indexed by progression)
              with first level ["mean", "sem"] and second level metric signatures
            Returns None if data cannot be retrieved or aligned.
        """
        data = self._lookup_and_validate_data(
            experiment=experiment, metric_signatures=metric_signatures
        )
        if data is None:
            return None

        try:
            multilevel_wide_df = align_partial_results(
                df=(long_df := data.full_df),
                metrics=metric_signatures,
            )
        except Exception as e:
            logger.warning(
                f"Encountered exception while aligning data: {e}. "
                "Cannot proceed with early stopping."
            )
            return None

        return long_df, multilevel_wide_df


class ModelBasedEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
    """A base class for model based early stopping strategies. Includes
    a helper function for processing Data into arrays."""

    def __init__(
        self,
        metric_signatures: Iterable[str] | None = None,
        min_progression: float | None = None,
        max_progression: float | None = None,
        min_curves: int | None = None,
        normalize_progressions: bool = False,
        min_progression_modeling: float | None = None,
        interval: float | None = None,
        check_safe: bool = False,
    ) -> None:
        """A ModelBasedEarlyStoppingStrategy class.

        Args:
            metric_signatures: The names of the metrics the strategy will interact with.
                If no metric names are provided the objective metric is assumed.
            min_progression: Only stop trials if the latest progression value
                (e.g. timestamp, epochs, training data used) is greater than this
                threshold. Prevents stopping prematurely before enough data is gathered
                to make a decision.
            max_progression: Do not stop trials that have passed `max_progression`.
                Useful if we prefer finishing a trial that are already near completion.
            min_curves: Trials will not be stopped until a number of trials
                `min_curves` have completed with curve data attached. That is, if
                `min_curves` trials are completed but their curve data was not
                successfully retrieved, further trials may not be early-stopped.
            normalize_progressions: If True, normalizes the progression values
                for each metric to the [0, 1] range using the observed minimum and
                maximum progression values for that metric. This transformation maps
                the original progression range [`min_prog`, `max_prog`] to [0, 1]
                via (x - min_prog) / (max_prog - min_prog). Useful when progression
                values have arbitrary scales or when you want to specify
                `min_progression` and `max_progression` in a normalized [0, 1]
                space. Note: At least one trial should have completed (i.e.,
                `min_curves` > 0) to ensure reliable estimates of the progression
                range.
            min_progression_modeling: If set, this will exclude progressions that are
                below this threshold from being used or modeling. Useful when early data
                is not reliable or excessively noisy.
            interval: If specified, throttles early-stopping checks by only
                evaluating trials when they cross interval boundaries. Boundaries are
                defined at `min_progression + k * interval` for k = 0, 1, 2, etc.
                A trial is eligible if it's being checked for the first time, or if
                its progression has crossed a boundary since the last check. This
                prevents premature stopping decisions when the orchestrator polls
                frequently. For example, with `interval=10` and `min_progression=0`,
                boundaries are at 0, 10, 20, 30, etc. A trial at progression 15 is
                eligible on first check. If checked again at progression 18, it's not
                eligible (both in interval [10, 20)). Once it reaches progression 21,
                it's eligible again (crossed into interval [20, 30)).
        """
        super().__init__(
            metric_signatures=metric_signatures,
            min_progression=min_progression,
            max_progression=max_progression,
            min_curves=min_curves,
            normalize_progressions=normalize_progressions,
            interval=interval,
            check_safe=check_safe,
        )
        self.min_progression_modeling = min_progression_modeling

    def _lookup_and_validate_data(
        self, experiment: Experiment, metric_signatures: list[str]
    ) -> Data | None:
        """Looks up and validates the `Data` used for early stopping that
        is associated with `metric_signatures`. This function also handles normalizing
        progressions.
        """
        map_data = super()._lookup_and_validate_data(
            experiment=experiment, metric_signatures=metric_signatures
        )
        if map_data is not None and self.min_progression_modeling is not None:
            full_df = map_data.full_df
            full_df = full_df[full_df[MAP_KEY] >= self.min_progression_modeling]
            map_data = Data(df=full_df)
        return map_data

    def get_training_data(
        self,
        experiment: Experiment,
        map_data: Data,
        max_training_size: int | None = None,
        outcomes: Sequence[str] | None = None,
        parameters: list[str] | None = None,
    ) -> None:
        # Deprecated in Ax 1.1.0, so should be removed in Ax 1.2.0+.
        raise DeprecationWarning(
            "`ModelBasedEarlyStoppingStrategy.get_training_data` is deprecated. "
            "Subclasses should either extract the training data manually, "
            "or rely on the fitted surrogates available in the current generation "
            "node that is passed into `should_stop_trials_early`."
        )
