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
from pyre_extensions import none_throws

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

        # dropna() here will exclude trials that have not made it to the
        # last progression of the trial under consideration, and therefore
        # can't be included in the comparison
        df_trial = none_throws(df[trial_index].dropna())
        trial_last_prog = df_trial.index.max()
        data_at_last_progression = df.loc[trial_last_prog].dropna()

        # Check that enough trials have data at the last progression. Note that
        # `is_eligible_any` is called in `should_stop_trials_early`, and checks that
        # at least `min_curves` trials have completed, and uses `align_partial_results`
        # to fill in results for each metric and progression. Therefore, the following
        # condition should only be triggered when `align_partial_results` encounters an
        # exception or `should_stop_trial_early` is called without the aligned data.
        if (
            self.min_curves is not None
            and len(data_at_last_progression) < self.min_curves  # pyre-ignore[58]
        ):
            return False, (
                f"Number of trials with data ({len(data_at_last_progression)}) at "
                f"last progression ({trial_last_prog}) is less than the "
                f"specified minimum number for early stopping ({self.min_curves})."
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

        if should_early_stop:
            logger.info(f"Early stopping trial {trial_index}: {reason}.")

        return should_early_stop, reason
