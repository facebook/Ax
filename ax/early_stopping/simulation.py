#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Iterable
from dataclasses import dataclass
from heapq import nsmallest

import numpy as np
import numpy.typing as npt
import pandas as pd
from ax.early_stopping.utils import _interval_boundary


@dataclass
class EarlyStoppingSimulationResult:
    """Result of early-stopping vulnerability simulation."""

    # Whether the best eligible trial would have been stopped
    best_stopped: bool

    # Index of the best eligible trial
    best_trial_index: int | None

    # Progression where best trial would have been stopped (if best_stopped is True)
    best_stop_progression: float | None = None


def _get_interval_progressions(
    progressions: npt.NDArray,
    min_progression: float,
    interval: float | None,
) -> npt.NDArray:
    """Filter progressions to only include those at interval boundaries.

    Interval boundaries are at min_progression + k * interval for k = 0, 1, 2, ...
    For each interval, we select the first progression that falls within it.

    Args:
        progressions: Array of progression values to filter.
        min_progression: The minimum progression (start of first interval).
        interval: The interval size. If None, returns progressions unchanged.

    Returns:
        Array of progressions, one per interval that contains data.
    """
    if interval is None or len(progressions) == 0:
        return progressions
    boundaries = _interval_boundary(
        progression=progressions,
        min_progression=min_progression,
        interval=interval,
    )
    # Select first progression in each interval (where boundary changes)
    return progressions[np.r_[True, np.diff(boundaries) != 0]]


def _check_patience_window(
    wide_df: pd.DataFrame,
    trial_indices: set[int],
    progression: float,
    patience: float,
    min_progression: float,
    quantile: float,
    minimize: bool,
    n_best_trials_to_complete: int | None,
    reference_trial_indices: set[int] | None = None,
) -> pd.Series:
    """Check which trials underperform at all progressions in the patience window.

    The patience window is defined as [progression - patience, progression]. A trial
    is considered to be stopped only if it underperforms (relative to the quantile
    threshold) at ALL progressions within this window.

    Args:
        wide_df: DataFrame with progressions as index and trial indices as columns.
        trial_indices: Set of trial indices to check.
        progression: The current progression (end of the patience window).
        patience: The patience value defining the window size.
        min_progression: The minimum progression (window start is clamped to this).
        quantile: The quantile threshold in [0, 1] range (already adjusted for
            minimize direction).
        minimize: Whether we're minimizing the metric.
        n_best_trials_to_complete: If specified, trials in the top N at any
            progression in the window are protected from stopping.
        reference_trial_indices: Set of trial indices to use for computing quantile
            thresholds. If None, uses all columns in wide_df.

    Returns:
        Boolean Series indexed by trial indices, True if trial should be stopped.
    """
    trial_cols = [*trial_indices]
    if not trial_indices:
        return pd.Series(dtype=bool, index=trial_cols)

    window_start = progression - patience
    window_selector = (wide_df.index >= window_start) & (wide_df.index <= progression)
    window_values = wide_df.loc[window_selector]

    # Check n_best_trials_to_complete protection
    if n_best_trials_to_complete is not None:
        # Rank against reference trials + trials being checked (the "world")
        rank_cols = (
            [*reference_trial_indices.union(trial_indices)]
            if reference_trial_indices is not None
            else wide_df.columns
        )
        # method='dense' assigns same rank to ties
        window_ranks = window_values[rank_cols].rank(
            method="dense", axis=1, ascending=minimize
        )
        # Protect trials that are in top K at any progression in window
        is_protected = (window_ranks[trial_cols] <= n_best_trials_to_complete).any(
            axis=0
        )
    else:
        is_protected = pd.Series(False, index=trial_cols)

    # Calculate threshold at each progression in the window
    ref_cols = (
        [*reference_trial_indices]
        if reference_trial_indices is not None
        else wide_df.columns
    )
    window_thresholds = window_values[ref_cols].quantile(q=quantile, axis=1)

    # Determine if each trial underperforms at each progression (vectorized)
    # Compare each column (trial) against the threshold series
    underperforms = (
        window_values[trial_cols].gt(window_thresholds, axis=0)
        if minimize
        else window_values[trial_cols].lt(window_thresholds, axis=0)
    )

    # A trial should be stopped only if it underperforms at ALL progressions
    # AND is not protected by n_best_trials_to_complete
    should_stop = underperforms.all(axis=0) & ~is_protected

    return should_stop


def best_trial_vulnerable(
    wide_df: pd.DataFrame,
    minimize: bool,
    completed_trials: Iterable[int],
    percentile_threshold: float = 50.0,
    min_progression: float | None = 10,
    max_progression: float | None = None,
    min_curves: int | None = 5,
    patience: float = 0.0,
    interval: float | None = None,
    n_best_trials_to_complete: int | None = None,
) -> EarlyStoppingSimulationResult:
    """Simulate early stopping to check if the best trial would be stopped.

    This function simulates the behavior of PercentileEarlyStoppingStrategy to
    determine if the globally best trial (based on final objective value) would
    have been prematurely stopped. It supports the same configuration options
    as the real strategy, including patience windows, interval throttling, and
    best trial protection.

    Args:
        wide_df: DataFrame with progressions as index and trial indices as columns.
            Values are the objective metric at each progression for each trial.
        minimize: Whether we're minimizing the metric.
        completed_trials: Indices of trials that have completed.
        percentile_threshold: Percentile threshold for early stopping. Trials
            falling below this threshold (relative to other trials at the same
            progression) are candidates for stopping. For example, if
            percentile_threshold=25.0, the bottom 25% of trials are stopped.
        min_progression: Minimum progression before early stopping is applied.
            Trials are not evaluated for stopping until they reach this value.
        max_progression: Maximum progression where early stopping is applied.
            Trials past this progression are not stopped.
        min_curves: Minimum number of trials needed before early stopping begins.
            The first min_curves completed trials form a protected "startup" set
            that can never be stopped.
        patience: If non-zero, requires that a trial underperforms the percentile
            threshold consistently across all progressions in the patience window
            [progression - patience, progression] before stopping. This helps avoid
            stopping trials with noisy curves. If 0, only the current progression
            is checked. Must be non-negative.
        interval: Throttles early-stopping evaluation to occur only when trials
            cross interval boundaries (at min_progression + k * interval, k=0,1,2...).
            If None, evaluation occurs at every progression.
        n_best_trials_to_complete: If specified, protects the top N trials (based
            on current objective value) from being stopped at each evaluation point.
            When combined with patience, a trial is protected if it is in the top N
            at ANY progression within the patience window.

    Returns:
        EarlyStoppingSimulationResult with information about whether the best trial
        would have been stopped, including the stopping progression and threshold
        if applicable.
    """
    all_trials = set(wide_df.columns)

    # Find the best trial from ALL trials
    # We want to check if the strategy would stop the overall best trial
    eventual_scores = wide_df.sort_index(axis=0).apply(
        lambda col: col.dropna().iloc[-1],
        axis=0,
    )
    best_trial_index = (
        eventual_scores.idxmin() if minimize else eventual_scores.idxmax()
    )

    # Get first min_curves trials, but exclude the best trial to avoid
    # circular dependency (we don't want the best trial contributing to
    # the threshold that determines if it's stopped)
    startup_trials = set(
        nsmallest(
            n=(min_curves or 0),
            iterable=set(completed_trials) - {best_trial_index},
        )
    )

    # eligible trials: can be stopped (all except protected startup trials)
    eligible_trials = all_trials - startup_trials
    # active trials are eligible trials excluding the best
    active_trials = eligible_trials - {best_trial_index}

    # Adjust percentile for minimization: for minimization, we flip the percentile
    # (e.g., 25th percentile becomes 75th) to identify the worst-performing trials.
    # Convert to quantile (0-1 range) for use with quantile/nanpercentile functions.
    quantile = percentile_threshold / 100.0
    quantile = 1 - quantile if minimize else quantile

    if min_progression is None:
        min_progression = 0.0
    if max_progression is None:
        max_progression = float("inf")

    start, stop = wide_df.index.searchsorted(
        [min_progression, max_progression], side="left"
    )
    progressions = wide_df.index[start:stop].to_numpy()
    progressions = _get_interval_progressions(progressions, min_progression, interval)

    for progression in progressions:
        if len(active_trials) == 0:
            break
        reference_trials = startup_trials | active_trials

        # Check if best trial should be stopped (against reference trials)
        stop_selector = _check_patience_window(
            wide_df=wide_df,
            trial_indices={best_trial_index},
            progression=progression,
            patience=patience,
            min_progression=min_progression,
            quantile=quantile,
            minimize=minimize,
            n_best_trials_to_complete=n_best_trials_to_complete,
            reference_trial_indices=reference_trials,
        )
        stop_best = stop_selector[best_trial_index]
        if stop_best:
            return EarlyStoppingSimulationResult(
                best_stopped=True,
                best_trial_index=best_trial_index,
                best_stop_progression=progression,
            )

        # Check which active trials should be stopped (for simulation purposes)
        stop_selector = _check_patience_window(
            wide_df=wide_df,
            trial_indices=active_trials,
            progression=progression,
            patience=patience,
            min_progression=min_progression,
            quantile=quantile,
            minimize=minimize,
            n_best_trials_to_complete=n_best_trials_to_complete,
            reference_trial_indices=reference_trials,
        )
        trials_to_stop = set(stop_selector[stop_selector].index)
        active_trials -= trials_to_stop

    return EarlyStoppingSimulationResult(
        best_stopped=False,
        best_trial_index=best_trial_index,
    )
