#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Iterable
from logging import Logger

import numpy as np
import pandas as pd
from ax.core.experiment import Experiment
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.early_stopping.utils import align_partial_results
from ax.exceptions.core import UnsupportedError
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
        trial_indices_to_ignore: list[int] | None = None,
        normalize_progressions: bool = False,
        n_best_trials_to_complete: int | None = None,
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
            trial_indices_to_ignore: Trial indices that should not be early stopped.
            normalize_progressions: Normalizes the progression column of the MapData df
                by dividing by the max. If the values were originally in [0, `prog_max`]
                (as we would expect), the transformed values will be in [0, 1]. Useful
                for inferring the max progression and allows `min_progression` to be
                specified in the transformed space. IMPORTANT: Typically, `min_curves`
                should be > 0 to ensure that at least one trial has completed and that
                we have a reliable approximation for `prog_max`.
            n_best_trials_to_complete: If specified, guarantees that the top
                `n_best_trials_to_complete` trials (based on current objective value
                at the last progression) will never be early stopped, even if they
                fall below the percentile threshold. This ensures that the best
                performing trials are allowed to run to completion.
        """
        super().__init__(
            metric_signatures=metric_signatures,
            trial_indices_to_ignore=trial_indices_to_ignore,
            min_progression=min_progression,
            max_progression=max_progression,
            min_curves=min_curves,
            normalize_progressions=normalize_progressions,
        )

        self.percentile_threshold = percentile_threshold
        self.n_best_trials_to_complete = n_best_trials_to_complete

        if metric_signatures is not None and len(list(metric_signatures)) > 1:
            raise UnsupportedError(
                "PercentileEarlyStoppingStrategy only supports a single metric. Use "
                "LogicalEarlyStoppingStrategy to compose early stopping strategies "
                "with multiple metrics."
            )

    def should_stop_trials_early(
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
        data = self._check_validity_and_get_data(
            experiment=experiment, metric_signatures=[metric_signature]
        )
        if data is None:
            # don't stop any trials if we don't get data back
            return {}

        df = data.map_df

        # default checks on `min_progression` and `min_curves`; if not met, don't do
        # early stopping at all and return {}
        if not self.is_eligible_any(
            trial_indices=trial_indices, experiment=experiment, df=df
        ):
            return {}

        try:
            aligned_df = align_partial_results(
                df=df,
                metrics=[metric_signature],
            )
        except Exception as e:
            logger.warning(
                f"Encountered exception while aligning data: {e}. "
                "Not early stopping any trials."
            )
            return {}

        metric_to_aligned_means = aligned_df["mean"]
        aligned_means = metric_to_aligned_means[metric_signature]
        decisions = {
            trial_index: self._should_stop_trial_early(
                trial_index=trial_index,
                experiment=experiment,
                df=aligned_means,
                df_raw=df,
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
        df: pd.DataFrame,
        df_raw: pd.DataFrame,
        minimize: bool,
    ) -> tuple[bool, str | None]:
        """Stop a trial if its performance is in the bottom `percentile_threshold`
        of the trials at the same step.

        Args:
            trial_index: Indices of candidate trial to stop early.
            experiment: Experiment that contains the trials and other contextual data.
            df: Dataframe of partial results after applying interpolation,
                filtered to objective metric.
            df_raw: The original MapData dataframe (before interpolation).
            minimize: Whether objective value is being minimized.

        Returns:
            A tuple `(should_stop, reason)`, where `should_stop` is `True` iff the
            trial should be stopped, and `reason` is an (optional) string providing
            information on why the trial should or should not be stopped.
        """

        stopping_eligible, reason = self.is_eligible(
            trial_index=trial_index, experiment=experiment, df=df_raw
        )
        if not stopping_eligible:
            return False, reason

        # Extract the metric curve for the trial under consideration
        trial_series = df[trial_index]
        # Find the latest progression with a recorded value for this trial
        trial_latest_prog = trial_series.last_valid_index()

        # Get objective values for all trials at this progression
        objective_latest_prog = df.loc[trial_latest_prog]
        # Filter to trials that have reached this progression (exclude NaN values)
        ref_selector = objective_latest_prog.notna()
        ref_objectives_latest_prog = objective_latest_prog[ref_selector]
        ref_trial_indices = objective_latest_prog.index[ref_selector]

        # Verify that sufficiently many trials have data at this progression.
        # Note: `is_eligible_any` in `should_stop_trials_early` already checks
        # that at least `min_curves` trials have completed and uses
        # `align_partial_results` to interpolate missing values. This condition
        # should only trigger if `align_partial_results` fails or if this method
        # is called with non-aligned data.
        if (
            self.min_curves is not None
            and len(ref_trial_indices) < self.min_curves  # pyre-ignore[58]
        ):
            return False, (
                f"Number of trials with data ({len(ref_trial_indices)}: "
                f"{sorted(ref_trial_indices.tolist())}) at "
                f"latest progression ({trial_latest_prog}) is less than the "
                f"specified minimum number for early stopping ({self.min_curves})."
            )

        # Calculate the percentile threshold value from reference trials.
        # For minimization problems, we flip the percentile (e.g., 25th percentile
        # becomes 75th percentile) to identify the worst-performing trials.
        percentile_threshold = (
            100.0 - self.percentile_threshold if minimize else self.percentile_threshold
        )
        ref_threshold_value = np.percentile(
            ref_objectives_latest_prog,
            q=percentile_threshold,
        )
        trial_objective_value = objective_latest_prog[trial_index]
        # Determine if this trial should be stopped based on its performance
        # relative to the threshold
        should_early_stop = (
            trial_objective_value > ref_threshold_value
            if minimize
            else trial_objective_value < ref_threshold_value
        )

        # Build the percentile threshold message that explains performance
        # relative to threshold
        comp = "worse" if should_early_stop else "better"
        percentile_reason = (
            f"Trial objective value {trial_objective_value:.3f} is {comp} than "
            f"{percentile_threshold:.1f}-th percentile ({ref_threshold_value:.3f}) "
            f"across comparable trials at progression {trial_latest_prog} "
            f"(calculated from {len(ref_trial_indices)} trials: "
            f"{sorted(ref_trial_indices.tolist())})."
        )

        # Check if trial is in top n_best_trials_to_complete
        # and should be protected from early stopping
        if should_early_stop and self.n_best_trials_to_complete is not None:
            # Rank trials by objective value at last progression
            sorted_values = ref_objectives_latest_prog.sort_values(ascending=minimize)
            best_trial_values = sorted_values.head(self.n_best_trials_to_complete)
            best_trial_indices = set(best_trial_values.index)
            if trial_index in best_trial_indices:
                # Get the worst (last) value among the top trials
                worst_of_best_value = best_trial_values.iloc[-1]
                worst_of_best_index = best_trial_values.index[-1]

                top_trials_reason = (
                    f"Trial {trial_index} is in top-"
                    f"{self.n_best_trials_to_complete} trials "
                    f"(top trials: {best_trial_values.index.tolist()} "
                    f"with objective values: {best_trial_values.tolist()}; "
                    f"worst of top trials: trial {worst_of_best_index} "
                    f"with value {worst_of_best_value}) and will not be "
                    f"early stopped despite falling below percentile threshold. "
                    f"{percentile_reason}"
                )
                logger.info(top_trials_reason)
                return False, top_trials_reason

        if should_early_stop:
            logger.info(f"Early stopping trial {trial_index}: {percentile_reason}.")

        return should_early_stop, percentile_reason
