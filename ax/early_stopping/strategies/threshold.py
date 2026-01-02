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
from ax.core.map_data import MAP_KEY
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.exceptions.core import UnsupportedError
from ax.generation_strategy.generation_node import GenerationNode
from ax.utils.common.logger import get_logger

logger: Logger = get_logger(__name__)


class ThresholdEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
    """Implements the strategy of stopping a trial if its performance
    doesn't reach a pre-specified threshold by a certain progression."""

    def __init__(
        self,
        metric_signatures: Iterable[str] | None = None,
        metric_threshold: float = 0.2,
        min_progression: float | None = 10,
        max_progression: float | None = None,
        min_curves: int | None = 5,
        normalize_progressions: bool = False,
        check_safe: bool = False,
    ) -> None:
        """Construct a ThresholdEarlyStoppingStrategy instance.

        Args
            metric_signatures: A (length-one) list of signatures of the metric
                to observe. If None will default to the objective metric on the
                 Experiment's OptimizationConfig.
            metric_threshold: The metric threshold that a trial needs to reach by
                min_progression in order not to be stopped.
            min_progression: Only stop trials if the latest progression value
                (e.g. timestamp, epochs, training data used) is worse than this
                threshold. Prevents stopping prematurely before enough data is gathered
                to make a decision.
            max_progression: Do not stop trials that have passed `max_progression`.
                Useful if we prefer finishing a trial that are already near completion.
            min_curves: Trials will not be stopped until a number of trials
                `min_curves` have completed with curve data attached. That is, if
                `min_curves` trials are completed but their curve data was not
                successfully retrieved, further trials may not be early-stopped.
            normalize_progressions: Normalizes the progression column of the Data df
                by dividing by the max. If the values were originally in [0, `prog_max`]
                (as we would expect), the transformed values will be in [0, 1]. Useful
                for inferring the max progression and allows `min_progression` to be
                specified in the transformed space. IMPORTANT: Typically, `min_curves`
                should be > 0 to ensure that at least one trial has completed and that
                we have a reliable approximation for `prog_max`.
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
            check_safe=check_safe,
        )
        self.metric_threshold = metric_threshold

        if metric_signatures is not None and len(list(metric_signatures)) > 1:
            raise UnsupportedError(
                "ThresholdEarlyStoppingStrategy only supports a single metric. Use "
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
        """Stop a trial if its performance doesn't reach a pre-specified threshold
        by `min_progression`.

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
        data = self._lookup_and_validate_data(
            experiment=experiment, metric_signatures=[metric_signature]
        )
        if data is None:
            # don't stop any trials if we don't get data back
            return {}

        df = data.full_df

        # default checks on `min_progression` and `min_curves`; if not met, don't do
        # early stopping at all and return {}
        if not self.is_eligible_any(
            trial_indices=trial_indices, experiment=experiment, df=df
        ):
            return {}

        df_objective = df[df["metric_signature"] == metric_signature]
        decisions = {
            trial_index: self._should_stop_trial_early(
                trial_index=trial_index,
                experiment=experiment,
                df=df_objective,
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
        minimize: bool,
    ) -> tuple[bool, str | None]:
        """Stop a trial if its performance doesn't reach a pre-specified threshold
        by `min_progression`.

        Args:
            trial_index: Indices of candidate trial to stop early.
            experiment: Experiment that contains the trials and other contextual data.
            df: Dataframe of partial results for the objective metric.
            minimize: Whether objective value is being minimized.

        Returns:
            A tuple `(should_stop, reason)`, where `should_stop` is `True` iff the
            trial should be stopped, and `reason` is an (optional) string providing
            information on why the trial should or should not be stopped.
        """
        logger.info(f"Considering trial {trial_index} for early stopping.")

        stopping_eligible, reason = self.is_eligible(
            trial_index=trial_index, experiment=experiment, df=df
        )
        if not stopping_eligible:
            return False, reason

        # threshold early stopping logic
        df_trial = df[df["trial_index"] == trial_index].dropna(subset=["mean"])
        trial_last_prog = df_trial[MAP_KEY].max()
        data_last_prog = df_trial[df_trial[MAP_KEY] == trial_last_prog]["mean"].iloc[0]
        logger.info(
            "Early stopping objective at last progression is:\n" f"{data_last_prog}."
        )
        should_early_stop = (
            data_last_prog > self.metric_threshold
            if minimize
            else data_last_prog < self.metric_threshold
        )
        comp = "worse" if should_early_stop else "better"
        reason = (
            f"Trial objective value {data_last_prog} is {comp} than "
            f"the metric threshold {self.metric_threshold:}."
        )
        logger.info(
            f"Early stopping decision for {trial_index}: {should_early_stop}. "
            f"Reason: {reason}"
        )
        return should_early_stop, reason
