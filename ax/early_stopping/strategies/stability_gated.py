#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Iterable
from dataclasses import dataclass
from logging import Logger
from math import ceil, floor
from typing import cast

import numpy as np
import pandas as pd
from ax.core.experiment import Experiment
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.generation_strategy.generation_node import GenerationNode
from ax.utils.common.logger import get_logger
from pyre_extensions import override as overrides

logger: Logger = get_logger(__name__)


@dataclass(frozen=True)
class _CheckpointDecisionContext:
    """Per-checkpoint SGES state shared by all candidate trial decisions.

    This context is built once per checkpoint and reused across candidate
    trials. It records which trials are comparable, which are protected by
    top-k or recent-best guards, the checkpoint leader, and the stability
    threshold used to decide whether a candidate is flat enough to stop.
    """

    comparable_trial_indices: set[int]
    protected_trial_indices: set[int]
    leader_trial_index: int
    leader_avg: float
    stability_threshold: float | None


class StabilityGatedEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
    """Stops trials with stable, consistently worse learning-curve performance.

    This strategy compares interpolated learning curves at fixed progression
    checkpoints. A trial is stopped only after it is stable, worse than the
    current leader by a configured gap, and not protected by the current top-k
    or recent-best safeguards for several consecutive checkpoints.
    """

    def __init__(
        self,
        *,
        metric_signatures: Iterable[str] | None = None,
        min_progression: float | None = 0.1,
        max_progression: float | None = None,
        min_curves: int | None = 1,
        normalize_progressions: bool = True,
        check_interval: float = 0.1,
        window_size: int = 2,
        variance_threshold: float | None = None,
        gap_threshold: float | None = None,
        stability_count: int = 4,
        top_k_fraction: float = 0.5,
        min_top_k: int = 2,
        recent_best_lookback: int = 15,
    ) -> None:
        """Initialize the stability-gated early-stopping strategy.

        Args:
            metric_signatures: Optional iterable containing the single metric
                signature to use for early stopping. If omitted, the strategy
                uses the experiment's objective metric. SGES compares one
                learning curve at a time; use LogicalEarlyStoppingStrategy to
                combine multiple SGES instances for multiple metrics.
            min_progression: Minimum latest progression a candidate trial must
                reach before SGES can stop it. If `normalize_progressions` is
                true, set this in normalized [0, 1] progress units. Otherwise,
                set it in raw progression units such as examples trained.
            max_progression: Optional latest progression cutoff after which a
                trial is no longer stopped. Set this when late-stage trials are
                cheap enough or valuable enough to finish once they are near
                completion.
            min_curves: Minimum number of curves that must have data at
                min_progression before any candidate can be stopped. Increase
                this for noisier experiments so each decision has enough peer
                curves for comparison.
            normalize_progressions: Whether to min-max normalize progression
                values before checkpointing. When true, `min_progression`,
                `max_progression`, and `check_interval` are interpreted in
                normalized [0, 1] progress units. When false, those parameters
                are interpreted in raw progression units.
            check_interval: Distance between SGES decision checkpoints in
                the same progression units as `min_progression`. Match this to
                the metric reporting cadence where possible; using a much
                smaller interval than the real reporting cadence creates
                interpolated checkpoints from sparse data and can make smoke
                tests look more decisive than live behavior.
            window_size: Number of adjacent checkpoint values used to compute
                the moving average and moving variance. Larger windows smooth
                noisier curves but delay decisions.
            variance_threshold: Maximum moving variance for a candidate trial
                to count as stable at a checkpoint. If None, SGES infers the
                threshold from peer moving variances at each checkpoint, which
                avoids baking in metric-scale-specific defaults. Set an
                explicit value when the metric scale is known and a fixed
                stability tolerance is desired.
            gap_threshold: Required moving-average gap between the current
                leader and candidate trial. For minimize metrics, the candidate
                must be higher by this amount; for maximize metrics, lower by
                this amount. If None, any positive gap to the leader counts as
                worse, while top-k and recent-best protections still gate
                stopping. Set an explicit value when the metric scale is known
                and a minimum effect size is desired.
            stability_count: Number of consecutive checkpoints for which a
                candidate must be stable, worse, and unprotected before being
                stopped. Increase this to reduce premature stops at the cost of
                later savings.
            top_k_fraction: Fraction of comparable trials, ranked by moving
                average at each checkpoint, that are protected from stopping.
                Use this to keep a moving elite set alive while poorer stable
                trials are stopped.
            min_top_k: Minimum number of ranked trials protected regardless of
                top_k_fraction. Set this above zero to avoid stopping near-best
                trials in small experiments.
            recent_best_lookback: Number of recent checkpoint leaders to
                protect. This keeps trials that recently led the experiment
                from being stopped immediately after a ranking change. This is
                separate from `stability_count`: stability_count requires a
                candidate to remain stoppable for consecutive checkpoints,
                while recent-best protection gives a grace period to trials
                that were recently leaders. Set to zero to disable recent-best
                protection.
        """
        metric_signatures_list = (
            None if metric_signatures is None else list(metric_signatures)
        )
        if metric_signatures_list is not None and len(metric_signatures_list) > 1:
            raise UnsupportedError(
                "StabilityGatedEarlyStoppingStrategy only supports a single metric. "
                "Use LogicalEarlyStoppingStrategy to compose early stopping "
                "strategies with multiple metrics."
            )

        super().__init__(
            metric_signatures=metric_signatures_list,
            min_progression=min_progression,
            max_progression=max_progression,
            min_curves=min_curves,
            normalize_progressions=normalize_progressions,
            interval=None,
            check_safe=False,
        )

        if check_interval <= 0:
            raise UserInputError(
                f"check_interval must be positive, got {check_interval}."
            )
        if window_size < 1:
            raise UserInputError(f"window_size must be at least 1, got {window_size}.")
        if variance_threshold is not None and variance_threshold < 0:
            raise UserInputError(
                f"variance_threshold must be non-negative, got {variance_threshold}."
            )
        if gap_threshold is not None and gap_threshold < 0:
            raise UserInputError(
                f"gap_threshold must be non-negative, got {gap_threshold}."
            )
        if stability_count < 1:
            raise UserInputError(
                f"stability_count must be at least 1, got {stability_count}."
            )
        if top_k_fraction < 0 or top_k_fraction > 1:
            raise UserInputError(
                f"top_k_fraction must be in [0, 1], got {top_k_fraction}."
            )
        if min_top_k < 0:
            raise UserInputError(f"min_top_k must be non-negative, got {min_top_k}.")
        if recent_best_lookback < 0:
            raise UserInputError(
                "recent_best_lookback must be non-negative, "
                f"got {recent_best_lookback}."
            )

        self.check_interval = check_interval
        self.window_size = window_size
        self.variance_threshold = variance_threshold
        self.gap_threshold = gap_threshold
        self.stability_count = stability_count
        self.top_k_fraction = top_k_fraction
        self.min_top_k = min_top_k
        self.recent_best_lookback = recent_best_lookback

    @overrides
    def _is_harmful(
        self,
        trial_indices: set[int],
        experiment: Experiment,
    ) -> bool:
        """Return whether SGES should be skipped for safety reasons."""
        return False

    @overrides
    def _should_stop_trials_early(
        self,
        trial_indices: set[int],
        experiment: Experiment,
        current_node: GenerationNode | None = None,
    ) -> dict[int, str | None]:
        """Evaluate SGES for all eligible candidate trials in one experiment."""
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

        if not self.is_eligible_any(
            trial_indices=trial_indices, experiment=experiment, df=long_df
        ):
            return {}

        current_progression_by_trial: dict[int, float] = {}
        for trial_index in trial_indices:
            if trial_index not in wide_df.columns:
                continue
            current_progression = wide_df[trial_index].last_valid_index()
            if current_progression is None:
                continue
            current_progression_by_trial[trial_index] = float(current_progression)
        if not current_progression_by_trial:
            return {}

        checkpoint_df = self._checkpoint_values(
            wide_df=wide_df,
            current_progression=max(current_progression_by_trial.values()),
        )
        if checkpoint_df.empty:
            return {}

        moving_avg_df = cast(
            pd.DataFrame,
            checkpoint_df.rolling(
                window=self.window_size, min_periods=self.window_size
            ).mean(),
        )
        moving_var_df = cast(
            pd.DataFrame,
            checkpoint_df.rolling(
                window=self.window_size, min_periods=self.window_size
            ).var(ddof=0),
        )
        context_by_checkpoint = self._checkpoint_contexts(
            moving_avg_df=moving_avg_df,
            moving_var_df=moving_var_df,
            minimize=minimize,
        )

        decisions = {
            trial_index: self._should_stop_trial_early(
                trial_index=trial_index,
                experiment=experiment,
                long_df=long_df,
                current_progression=current_progression_by_trial.get(trial_index),
                moving_avg_df=moving_avg_df,
                moving_var_df=moving_var_df,
                context_by_checkpoint=context_by_checkpoint,
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
        long_df: pd.DataFrame,
        current_progression: float | None,
        moving_avg_df: pd.DataFrame,
        moving_var_df: pd.DataFrame,
        context_by_checkpoint: dict[float, _CheckpointDecisionContext],
        minimize: bool,
    ) -> tuple[bool, str | None]:
        """Evaluate one trial against SGES stability, gap, and protection gates."""
        stopping_eligible, reason = self.is_eligible(
            trial_index=trial_index, experiment=experiment, df=long_df
        )
        if not stopping_eligible:
            return False, reason

        if current_progression is None:
            return False, "No data available to make an early stopping decision."

        consecutive_count = 0
        latest_reason = "Not enough comparable stable checkpoint data."

        for checkpoint in moving_avg_df.index:
            checkpoint_float = float(checkpoint)
            if checkpoint_float > current_progression:
                break

            context = context_by_checkpoint.get(checkpoint_float)
            if context is None or trial_index not in context.comparable_trial_indices:
                consecutive_count = 0
                continue

            trial_avg = float(moving_avg_df.loc[checkpoint, trial_index])
            trial_var = float(moving_var_df.loc[checkpoint, trial_index])
            gap = (
                trial_avg - context.leader_avg
                if minimize
                else context.leader_avg - trial_avg
            )
            stable = (
                context.stability_threshold is not None
                and trial_var <= context.stability_threshold
            )
            worse = gap > (0.0 if self.gap_threshold is None else self.gap_threshold)
            trial_protected = trial_index in context.protected_trial_indices

            if stable and worse and not trial_protected:
                consecutive_count += 1
            else:
                consecutive_count = 0

            latest_reason = (
                f"Trial {trial_index} has SGES count {consecutive_count}/"
                f"{self.stability_count} at checkpoint {checkpoint:.2f}. "
                f"moving_avg={trial_avg:.8g}, moving_var={trial_var:.8g}, "
                f"stability_threshold={context.stability_threshold}, "
                f"gap={gap:.8g}, leader_trial={context.leader_trial_index}, "
                f"leader_moving_avg={context.leader_avg:.8g}, "
                f"stable={stable}, worse={worse}, protected={trial_protected}."
            )

        should_stop = consecutive_count >= self.stability_count
        logger.debug(
            f"SGES early stopping decision for trial {trial_index}: "
            f"{should_stop}. {latest_reason}"
        )
        return should_stop, latest_reason

    def _checkpoint_contexts(
        self,
        moving_avg_df: pd.DataFrame,
        moving_var_df: pd.DataFrame,
        minimize: bool,
    ) -> dict[float, _CheckpointDecisionContext]:
        """Build per-checkpoint leader and protection context once per poll."""
        context_by_checkpoint: dict[float, _CheckpointDecisionContext] = {}
        recent_best: list[int] = []

        for checkpoint in moving_avg_df.index:
            avg_row = moving_avg_df.loc[checkpoint].dropna()
            var_row = moving_var_df.loc[checkpoint].dropna()
            comparable_trial_indices = avg_row.index.intersection(var_row.index)
            if len(comparable_trial_indices) < 2:
                continue

            ordered_avgs = avg_row.loc[comparable_trial_indices].sort_values(
                ascending=minimize
            )
            leader_trial_index = int(ordered_avgs.index[0])
            leader_avg = float(ordered_avgs.iloc[0])
            recent_best.append(leader_trial_index)
            if self.recent_best_lookback == 0:
                recent_best = []
            else:
                recent_best = recent_best[-self.recent_best_lookback :]

            top_k = max(
                self.min_top_k,
                ceil(self.top_k_fraction * len(comparable_trial_indices)),
            )
            top_k = min(top_k, len(comparable_trial_indices))
            protected_trial_indices = {int(i) for i in ordered_avgs.index[:top_k]}
            protected_trial_indices.update(recent_best)

            context_by_checkpoint[float(checkpoint)] = _CheckpointDecisionContext(
                comparable_trial_indices={int(i) for i in comparable_trial_indices},
                protected_trial_indices=protected_trial_indices,
                leader_trial_index=leader_trial_index,
                leader_avg=leader_avg,
                stability_threshold=self._stability_threshold(
                    variances=var_row.loc[comparable_trial_indices]
                ),
            )

        return context_by_checkpoint

    def _stability_threshold(self, variances: pd.Series) -> float | None:
        """Return the explicit or peer-inferred stability threshold."""
        if self.variance_threshold is not None:
            return self.variance_threshold

        non_null_variances = variances.dropna()
        if non_null_variances.empty:
            return None
        return float(non_null_variances.median())

    def _checkpoint_values(
        self, wide_df: pd.DataFrame, current_progression: float
    ) -> pd.DataFrame:
        """Interpolate learning curves onto SGES progression checkpoints."""
        min_progression = 0.0 if self.min_progression is None else self.min_progression
        if current_progression < min_progression:
            return pd.DataFrame()

        num_intervals = floor(
            (current_progression - min_progression) / self.check_interval + 1e-12
        )
        checkpoints = np.array(
            [
                min_progression + interval_index * self.check_interval
                for interval_index in range(num_intervals + 1)
            ],
            dtype=float,
        )
        if len(checkpoints) == 0:
            return pd.DataFrame()

        values: dict[int, np.ndarray] = {}
        for trial_index in wide_df.columns:
            trial_curve = wide_df[trial_index].dropna()
            if len(trial_curve) < 2:
                continue
            progressions = trial_curve.index.astype(float).to_numpy()
            means = trial_curve.astype(float).to_numpy()
            values[int(trial_index)] = np.interp(
                checkpoints,
                progressions,
                means,
                left=np.nan,
                right=np.nan,
            )

        return pd.DataFrame(values, index=checkpoints)
