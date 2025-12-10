#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Iterable
from logging import Logger

import pandas as pd
from ax.core.experiment import Experiment
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.early_stopping.utils import _is_worse
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.generation_strategy.generation_node import GenerationNode
from ax.utils.common.logger import get_logger

logger: Logger = get_logger(__name__)


class PercentileEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
    """Implements the strategy of stopping a trial if its performance
    falls below that of other trials at the same step."""

    def __init__(
        self,
        metric_signatures: Iterable[str] | None = None,
        percentile_threshold: float = 50.0,
        min_progression: float | None = 10,
        max_progression: float | None = None,
        min_curves: int | None = 5,
        normalize_progressions: bool = False,
        n_best_trials_to_complete: int | None = None,
        interval: float | None = None,
        patience: float = 0.0,
        check_safe: bool = False,
    ) -> None:
        """Construct a PercentileEarlyStoppingStrategy instance.

        Args:
            metric_signatures: A (length-one) list of signatures of the metric to
                observe. If None will default to the objective metric on the
                Experiment's OptimizationConfig.
            percentile_threshold: Falling below this threshold compared to other trials
                at the same step will stop the run. Must be between 0.0 and 100.0.
                e.g. if percentile_threshold=25.0, the bottom 25% of trials are stopped.
                Note that "bottom" here is determined based on performance, not
                absolute values; if `minimize` is False, then "bottom" actually refers
                to the top trials in terms of metric value.
            min_progression: Only stop trials if the latest progression value
                (i.e. "step") is greater than this
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
            n_best_trials_to_complete: If specified, guarantees that the top
                `n_best_trials_to_complete` trials (based on current objective value
                at the last progression) will never be early stopped, even if they
                fall below the percentile threshold. This ensures that the best
                performing trials are allowed to run to completion.
            interval: Throttles early-stopping evaluation to occur only when
                trials cross interval boundaries (at min_progression + k * interval,
                k = 0, 1, 2...). Prevents premature stopping decisions when the
                orchestrator (ex, GAIN) polls frequently.
            patience: If non-zero, requires that a trial underperforms the percentile
                threshold consistently across all steps in the patience window
                [step - patience, step] before stopping. This helps avoid stopping
                trials with noisy curves. If 0, the original behavior is used
                (checking only the latest step). The patience is measured in training
                progressions, so irregular spacing is handled naturally. Must be
                non-negative.
            check_safe: If True, applies the relevant safety checks to gate
                early-stopping when it is likely to be harmful. If False (default),
                bypasses the safety check and directly applies early-stopping decisions.
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

        if patience < 0:
            raise UserInputError(f"patience must be non-negative, got {patience}.")

        self.percentile_threshold = percentile_threshold
        self.n_best_trials_to_complete = n_best_trials_to_complete
        self.patience = patience

        if metric_signatures is not None and len(list(metric_signatures)) > 1:
            raise UnsupportedError(
                "PercentileEarlyStoppingStrategy only supports a single metric. Use "
                "LogicalEarlyStoppingStrategy to compose early stopping strategies "
                "with multiple metrics."
            )

    def _is_harmful(
        self,
        trial_indices: set[int],
        experiment: Experiment,
    ) -> bool:
        return False

    def _should_stop_trials_early(
        self,
        trial_indices: set[int],
        experiment: Experiment,
        current_node: GenerationNode | None = None,
    ) -> dict[int, str | None]:
        """Stop a trial if its performance is in the bottom `percentile_threshold`
        of the trials at the same step.

        Args:
            trial_indices: Indices of candidate trials to consider for early stopping.
            experiment: Experiment that contains the trials and other contextual data.
            current_node: The current ``GenerationNode`` on the ``GenerationStrategy``
                used to generate trials for the ``Experiment``. Early stopping
                strategies may utilize components of the current node when making
                stopping decisions.

        Returns:
            A dictionary mapping trial indices that should be early stopped to
            (optional) messages with the associated reason. An empty dictionary
            means no suggested updates to any trial's status.
        """
        metric_signature, minimize = self._default_objective_and_direction(
            experiment=experiment
        )
        maybe_aligned_dataframes = self._prepare_aligned_data(
            experiment=experiment, metric_signatures=[metric_signature]
        )
        if maybe_aligned_dataframes is None:
            return {}

        long_df, multilevel_wide_df = maybe_aligned_dataframes
        wide_df = multilevel_wide_df["mean"][metric_signature]

        # default checks on `min_progression` and `min_curves`; if not met, don't do
        # early stopping at all and return {}
        if not self.is_eligible_any(
            trial_indices=trial_indices, experiment=experiment, df=long_df
        ):
            return {}

        decisions = {
            trial_index: self._should_stop_trial_early(
                trial_index=trial_index,
                experiment=experiment,
                wide_df=wide_df,
                long_df=long_df,
                minimize=minimize,
            )
            for trial_index in trial_indices
        }
        return {
            trial_index: reason
            for trial_index, (should_stop, reason) in decisions.items()
            if should_stop
        }

    def _should_stop_trial_early(
        self,
        trial_index: int,
        experiment: Experiment,
        wide_df: pd.DataFrame,
        long_df: pd.DataFrame,
        minimize: bool,
    ) -> tuple[bool, str | None]:
        """Stop a trial if its performance is in the bottom `percentile_threshold`
        of the trials at the same step. With patience, requires consistent
        underperformance across the patience window.

        Args:
            trial_index: Indices of candidate trial to stop early.
            experiment: Experiment that contains the trials and other contextual data.
            wide_df: Dataframe of partial results after applying interpolation,
                filtered to objective metric (wide format, non-hierarchical).
            long_df: The original MapData dataframe (long format, before interpolation).
            minimize: Whether objective value is being minimized.

        Returns:
            A tuple `(should_stop, reason)`, where `should_stop` is `True` iff the
            trial should be stopped, and `reason` is an (optional) string providing
            information on why the trial should or should not be stopped.
        """

        stopping_eligible, reason = self.is_eligible(
            trial_index=trial_index, experiment=experiment, df=long_df
        )
        if not stopping_eligible:
            return False, reason

        # Find the latest progression with a recorded value
        # for the trial under consideration
        window_end = wide_df[trial_index].last_valid_index()

        # Define evaluation window [window_end - patience, window_end]
        # When patience=0, this is just a single point [window_end]
        window_start = window_end - self.patience
        # Ensure window_start respects min_progression to avoid including
        # data from progressions where early stopping should not be evaluated
        window_start = max(window_start, self.min_progression or float("-inf"))

        window_selector = (wide_df.index >= window_start) & (
            wide_df.index <= window_end
        )
        window_values = wide_df[window_selector]
        window_active_trials = window_values.notna()
        # Series of lists of active trial indices for each progression in the window
        window_active_trial_indices: pd.Series = window_active_trials.apply(
            lambda mask: window_values.columns[mask].tolist(),
            axis=1,
        )
        window_num_active_trials: pd.Series = window_active_trials.sum(axis=1)

        # Verify that sufficiently many trials have data at each progression in
        # the patience window. Note: `is_eligible_any` in `should_stop_trials_early`
        # already checks that at least `min_curves` trials have completed and uses
        # `align_partial_results` to interpolate missing values. This condition
        # should only trigger if `align_partial_results` fails or if this method
        # is called with non-aligned data.
        if self.min_curves is not None:
            is_insufficient = window_num_active_trials < self.min_curves
            if is_insufficient.any():
                return False, (
                    f"Insufficiently many trials with data at progressions in window "
                    f"[{window_start:.2f}, {window_end:.2f}]\n"
                    f"- Progressions: {window_values[is_insufficient].index.tolist()}\n"
                    f"- Number of trials: "
                    f"{window_num_active_trials[is_insufficient].tolist()}\n"
                    f"- Trial indices: "
                    f"{window_active_trial_indices[is_insufficient].tolist()}\n"
                    f"- Minimum required: {self.min_curves}"
                )

        # Check if trial is in top n_best_trials_to_complete
        # and should be protected from early stopping
        if self.n_best_trials_to_complete is not None:
            # method='dense' assigns same rank to ties
            window_ranks = window_values.rank(
                method="dense", axis=1, ascending=minimize
            )
            # Trials with rank <= n_best_trials_to_complete are in the top N
            best_criterion = window_ranks <= self.n_best_trials_to_complete
            if best_criterion[trial_index].any():
                # Create a Series of dictionaries that map progressions in
                # the window to the indices of best trials for that progression.
                window_best_trials = window_values.apply(
                    lambda row: row[best_criterion.loc[row.name]].to_dict(), axis=1
                )
                reason = (
                    f"Trial {trial_index} is among the top-"
                    f"{self.n_best_trials_to_complete} trials at one or more "
                    f"progressions in window [{window_start:.2f}, {window_end:.2f}] "
                    "so will not be early-stopped.\n"
                    f"- Progressions: {window_values.index.tolist()}\n"
                    f"- Best trials: {window_best_trials.tolist()}"
                )
                logger.info(reason)
                return False, reason

        # Calculate the percentile threshold for each progression.
        # For minimization problems, we flip the percentile (e.g., 25th percentile
        # becomes 75th percentile) to identify the worst-performing trials.
        q = self.percentile_threshold / 100.0
        q = 1 - q if minimize else q
        window_thresholds = window_values.quantile(q=q, axis=1)
        trial_window_values = window_values[trial_index]

        # Determine if this trial underperforms relative to the threshold at each
        # progression
        underperforms = _is_worse(
            trial_window_values, window_thresholds, minimize=minimize
        )
        # Trial should be stopped if it underperforms at *all* progression in the window
        should_early_stop = underperforms.all()

        # Build the percentile threshold message that explains performance
        # relative to threshold
        reason = (
            f"Trial objective values at progressions in "
            f"[{window_start:.2f}, {window_end:.2f}] are"
            f"{'' if should_early_stop else ' not'} all worse than "
            f"{self.percentile_threshold:.1f}-th percentile across comparable "
            f"trials \n"
            f"- Progressions: {window_values.index.tolist()}\n"
            f"- Underperforms: {underperforms.tolist()}\n"
            f"- Trial objective values: {trial_window_values.tolist()}\n"
            f"- Thresholds: {window_thresholds.tolist()}\n"
            f"- Number of trials: {window_num_active_trials.tolist()}\n"
            f"- Trial indices: {window_active_trial_indices.tolist()}"
        )

        if should_early_stop:
            logger.info(f"Early stopping trial {trial_index}: {reason}.")

        return should_early_stop, reason
