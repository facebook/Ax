#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from ax.core.experiment import Experiment
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.exceptions.core import UnsupportedError
from ax.utils.common.logger import get_logger

logger: Logger = get_logger(__name__)


class ThresholdEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
    """Implements the strategy of stopping a trial if its performance
    doesn't reach a pre-specified threshold by a certain progression."""

    def __init__(
        self,
        metric_names: Optional[Iterable[str]] = None,
        seconds_between_polls: int = 300,
        metric_threshold: float = 0.2,
        min_progression: Optional[float] = 10,
        max_progression: Optional[float] = None,
        min_curves: Optional[int] = 5,
        trial_indices_to_ignore: Optional[List[int]] = None,
        true_objective_metric_name: Optional[str] = None,
        normalize_progressions: bool = False,
    ) -> None:
        """Construct a ThresholdEarlyStoppingStrategy instance.

        Args
            metric_names: A (length-one) list of name of the metric to observe. If
                None will default to the objective metric on the Experiment's
                OptimizationConfig.
            seconds_between_polls: How often to poll the early stopping metric to
                evaluate whether or not the trial should be early stopped.
            metric_threshold: The metric threshold that a trial needs to reach by
                min_progression in order not to be stopped.
            min_progression: Only stop trials if the latest progression value
                (e.g. timestamp, epochs, training data used) is greater than this
                threshold. Prevents stopping prematurely before enough data is gathered
                to make a decision.
            max_progression: Do not stop trials that have passed `max_progression`.
                Useful if we prefer finishing a trial that are already near completion.
            min_curves: There must be `min_curves` number of completed trials and
                `min_curves` number of trials with curve data to make a stopping
                decision (i.e., even if there are enough completed trials but not all
                of them are correctly returning data, then do not apply early stopping).
            trial_indices_to_ignore: Trial indices that should not be early stopped.
            true_objective_metric_name: The actual objective to be optimized; used in
                situations where early stopping uses a proxy objective (such as training
                loss instead of eval loss) for stopping decisions.
            normalize_progressions: Normalizes the progression column of the MapData df
                by dividing by the max. If the values were originally in [0, `prog_max`]
                (as we would expect), the transformed values will be in [0, 1]. Useful
                for inferring the max progression and allows `min_progression` to be
                specified in the transformed space. IMPORTANT: Typically, `min_curves`
                should be > 0 to ensure that at least one trial has completed and that
                we have a reliable approximation for `prog_max`.
        """
        super().__init__(
            metric_names=metric_names,
            seconds_between_polls=seconds_between_polls,
            true_objective_metric_name=true_objective_metric_name,
            min_progression=min_progression,
            max_progression=max_progression,
            min_curves=min_curves,
            trial_indices_to_ignore=trial_indices_to_ignore,
            normalize_progressions=normalize_progressions,
        )
        self.metric_threshold = metric_threshold

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
        metric_name, minimize = self._default_objective_and_direction(
            experiment=experiment
        )
        data = self._check_validity_and_get_data(
            experiment=experiment, metric_names=[metric_name]
        )
        if data is None:
            # don't stop any trials if we don't get data back
            return {}

        map_key = next(iter(data.map_keys))
        df = data.map_df

        # default checks on `min_progression` and `min_curves`; if not met, don't do
        # early stopping at all and return {}
        if not self.is_eligible_any(
            trial_indices=trial_indices, experiment=experiment, df=df, map_key=map_key
        ):
            return {}

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

        stopping_eligible, reason = self.is_eligible(
            trial_index=trial_index, experiment=experiment, df=df, map_key=map_key
        )
        if not stopping_eligible:
            return False, reason

        # threshold early stopping logic
        df_trial = df[df["trial_index"] == trial_index].dropna(subset=["mean"])
        trial_last_prog = df_trial[map_key].max()
        data_last_prog = df_trial[df_trial[map_key] == trial_last_prog]["mean"].iloc[0]
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
