# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final

import pandas as pd
from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.healthcheck.healthcheck_analysis import (
    create_healthcheck_analysis_card,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.analysis.utils import validate_experiment
from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.map_metric import MapMetric
from ax.early_stopping.experiment_replay import replay_experiment
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.early_stopping.strategies.percentile import PercentileEarlyStoppingStrategy
from ax.early_stopping.utils import estimate_early_stopping_savings
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.utils.early_stopping import get_early_stopping_metrics
from pyre_extensions import none_throws, override

DEFAULT_MIN_SAVINGS_THRESHOLD = 0.01  # 1% threshold
DEFAULT_MAP_KEY = "step"
MAX_PENDING_TRIALS_DEFAULT = 5


@final
class EarlyStoppingAnalysis(Analysis):
    """
    Healthcheck that evaluates early stopping status and potential savings using
    experiment replay.

    This analysis serves two purposes:
    1. When early stopping is enabled: Reports status (trials stopped) and
       estimated resource savings
    2. When early stopping is not enabled: Suggests enabling early stopping if
       significant savings are possible

    Status Logic:
    - PASS: Early stopping is enabled and functioning correctly or not enabled and not
            applicable or no significant savings expected
    - WARNING: Early stopping is not enabled but could be beneficial
    - FAIL: Early stopping is enabled but misconfigured

    The healthcheck evaluates:
    - Number of early stopped trials vs total completed trials
    - Estimated resource savings from early stopping
    - Whether the experiment could benefit from early stopping
    """

    def __init__(
        self,
        early_stopping_strategy: BaseEarlyStoppingStrategy | None = None,
        map_key: str = DEFAULT_MAP_KEY,
        min_savings_threshold: float = DEFAULT_MIN_SAVINGS_THRESHOLD,
        max_pending_trials: int = MAX_PENDING_TRIALS_DEFAULT,
    ) -> None:
        """
        Args:
            early_stopping_strategy: The early stopping strategy being used
                (None if not enabled)
            map_key: The key in MapData representing the progression metric
                (e.g., "step", "epochs"). Default is "step".
            min_savings_threshold: Minimum savings threshold to suggest early
                stopping. Default is 0.01 (1% savings).
            max_pending_trials: Maximum number of pending trials for replay
                orchestrator. Default is 5.
        """
        self.early_stopping_strategy = early_stopping_strategy
        self.map_key = map_key
        self.min_savings_threshold = min_savings_threshold
        self.max_pending_trials = max_pending_trials

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        """
        EarlyStoppingAnalysis requires an experiment.
        """
        return validate_experiment(experiment=experiment)

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> HealthcheckAnalysisCard:
        experiment = none_throws(experiment)

        # Validation: Early stopping requires MapData
        data = experiment.lookup_data()
        if not isinstance(data, MapData):
            return create_healthcheck_analysis_card(
                name=self.__class__.__name__,
                title="Early Stopping Healthcheck",
                subtitle=(
                    "Early stopping requires MapData. "
                    "This experiment does not have MapData, so early stopping "
                    "cannot be used."
                ),
                df=pd.DataFrame(),
                status=HealthcheckStatus.PASS,
            )

        # Auto-detect if early stopping was used by checking for early stopped trials
        n_stopped = len(experiment.trial_indices_by_status[TrialStatus.EARLY_STOPPED])

        # If a strategy was explicitly provided, use it
        if self.early_stopping_strategy is not None:
            return self._report_early_stopping_status(experiment)

        # If no strategy provided but trials were early stopped, create a default
        # strategy to enable proper savings calculation
        if n_stopped > 0:
            # Create a default PercentileEarlyStoppingStrategy for calculations
            self.early_stopping_strategy = PercentileEarlyStoppingStrategy()
            return self._report_early_stopping_status(experiment)

        # If early stopping is not enabled, check if it should be (nudge)
        return self._report_early_stopping_nudge(experiment)

    def _report_early_stopping_status(
        self, experiment: Experiment
    ) -> HealthcheckAnalysisCard:
        """Report early stopping status when it's enabled."""
        # Get metrics used by the early stopping strategy
        target_ess_metric_names = get_early_stopping_metrics(
            experiment=experiment,
            early_stopping_strategy=self.early_stopping_strategy,
        )

        # Return early if no metrics found
        if len(target_ess_metric_names) == 0:
            strategy_type = type(self.early_stopping_strategy).__name__
            df = pd.DataFrame(
                [
                    {"Property": "Strategy Type", "Value": strategy_type},
                    {"Property": "Metrics Found", "Value": "0"},
                    {
                        "Property": "Action Required",
                        "Value": "Configure metric_names in strategy",
                    },
                ]
            )
            return create_healthcheck_analysis_card(
                name=self.__class__.__name__,
                title="Early Stopping Healthcheck",
                subtitle=(
                    "Early stopping is enabled but no metrics were found. "
                    "This may indicate a configuration issue."
                ),
                df=df,
                status=HealthcheckStatus.FAIL,
            )

        # Log warning if multiple metrics (multi-objective case)
        if len(target_ess_metric_names) > 1:
            multi_objective_note = (
                f"\n\nNote: Although {len(target_ess_metric_names)} metrics are "
                f"being used for early stopping, '{target_ess_metric_names[0]}' "
                f"is being used to compute resource savings. "
            )
        else:
            multi_objective_note = ""

        # Validate the target metric is a MapMetric with map_data
        target_metric = experiment.metrics[target_ess_metric_names[0]]
        if not (isinstance(target_metric, MapMetric) and target_metric.has_map_data):
            actual_type = type(target_metric).__name__
            df = pd.DataFrame(
                [
                    {"Property": "Metric Name", "Value": target_ess_metric_names[0]},
                    {"Property": "Current Type", "Value": actual_type},
                    {"Property": "Required Type", "Value": "MapMetric"},
                    {
                        "Property": "Action Required",
                        "Value": "Change metric to MapMetric type",
                    },
                ]
            )
            return create_healthcheck_analysis_card(
                name=self.__class__.__name__,
                title="Early Stopping Healthcheck",
                subtitle=(
                    f"The metric '{target_ess_metric_names[0]}' used for early "
                    f"stopping is not a MapMetric with map data. Early stopping "
                    f"requires time-series data."
                ),
                df=df,
                status=HealthcheckStatus.FAIL,
            )

        # Calculate savings
        savings = 0
        if ess := self.early_stopping_strategy:
            savings = ess.estimate_early_stopping_savings(experiment=experiment)

        # Construct the message
        n_stopped = len(experiment.trial_indices_by_status[TrialStatus.EARLY_STOPPED])
        n_completed = len(experiment.trial_indices_by_status[TrialStatus.COMPLETED])
        n_ran = n_stopped + n_completed

        ess_msg = (
            f"Throughout this experiment, {n_stopped} trials were early stopped, "
            f"out of a total of {n_ran} trials. "
        )

        if savings > 0:
            ess_msg += (
                f"The capacity savings (computed using {self.map_key}) are "
                f"estimated to be {savings * 100:.0f}%."
            )
        else:
            ess_msg += (
                "Capacity savings are not yet available. Either no trials have "
                "been early stopped, or no trials have completed (which is "
                "required to estimate savings). Check back once more trials are "
                "completed and/or early stopped."
            )

        # Add multi-objective note if applicable
        subtitle = ess_msg + multi_objective_note

        # Create dataframe with details
        df = pd.DataFrame(
            [
                {"Property": "Early Stopped Trials", "Value": str(n_stopped)},
                {"Property": "Completed Trials", "Value": str(n_completed)},
                {"Property": "Total Trials", "Value": str(n_ran)},
                {
                    "Property": "Target Metric",
                    "Value": target_ess_metric_names[0],
                },
                {
                    "Property": "Estimated Savings",
                    "Value": f"{savings * 100:.0f}%" if savings > 0 else "N/A",
                },
            ]
        )

        return create_healthcheck_analysis_card(
            name=self.__class__.__name__,
            title="Capacity savings due to early stopping",
            subtitle=subtitle,
            df=df,
            status=HealthcheckStatus.PASS,
        )

    def _report_early_stopping_nudge(
        self, experiment: Experiment
    ) -> HealthcheckAnalysisCard:
        """Check if early stopping should be suggested (nudge) by estimating
        hypothetical savings using replay logic."""
        # Get map metrics from the experiment
        map_metrics = self._get_map_metrics(experiment)

        if not map_metrics:
            # No compatible map metrics found - early stopping cannot be used
            return create_healthcheck_analysis_card(
                name=self.__class__.__name__,
                title="Early Stopping Healthcheck",
                subtitle=(
                    "Early stopping is not applicable: no compatible map "
                    "metrics found."
                ),
                df=pd.DataFrame(),
                status=HealthcheckStatus.PASS,
            )

        # Estimate hypothetical savings for compatible metrics using replay
        metric_to_savings = self._estimate_hypothetical_savings_with_replay(
            experiment=experiment, map_metrics=map_metrics
        )

        if not metric_to_savings:
            # No significant savings detected
            return create_healthcheck_analysis_card(
                name=self.__class__.__name__,
                title="Early Stopping Healthcheck",
                subtitle=(
                    "Early stopping is not enabled. "
                    "While this experiment has MapData-compatible metrics, "
                    "we did not detect significant potential savings at this "
                    "time.\n\n"
                    "This could be because:\n"
                    "- The experiment hasn't run enough trials yet\n"
                    "- Trials have similar performance curves\n"
                    "- The current trial distribution doesn't show clear "
                    "underperformers"
                ),
                df=pd.DataFrame(),
                status=HealthcheckStatus.PASS,
            )

        # Found significant potential savings - nudge the user
        best_metric_name = max(metric_to_savings, key=metric_to_savings.get)
        best_savings = metric_to_savings[best_metric_name]

        subtitle = (
            "This sweep uses metrics that are **compatible with early stopping**! "
            "Using early stopping could have saved you both capacity and "
            "optimization wall time. For example, we estimate that using early "
            f"stopping on the '{best_metric_name}' metric could have provided "
            f"{best_savings:.0f}% capacity savings, with "
            "[no regression in optimization performance]"
            "(https://fb.workplace.com/notes/784094016018868). See "
            "[this tutorial](https://www.internalfb.com/intern/anp/view/?id=2713396) "
            "for instructions on how to turn on early stopping."
        )

        # Create detailed metrics table
        metric_rows = [
            {
                "Metric Name": metric_name,
                "Estimated Savings": f"{savings:.1f}%",
            }
            for metric_name, savings in sorted(
                metric_to_savings.items(), key=lambda x: x[1], reverse=True
            )
        ]
        df = pd.DataFrame(metric_rows)

        title = (
            f"{best_savings:.0f}% potential capacity savings if you turn on "
            f"early stopping feature"
        )

        return create_healthcheck_analysis_card(
            name=self.__class__.__name__,
            title=title,
            subtitle=subtitle,
            df=df,
            status=HealthcheckStatus.WARNING,
            potential_savings=best_savings,
            best_metric=best_metric_name,
        )

    def _get_map_metrics(self, experiment: Experiment) -> list[MapMetric]:
        """Get list of MapMetrics from the experiment, sorted with objectives first."""
        map_metrics = [
            m for m in experiment.metrics.values() if isinstance(m, MapMetric)
        ]

        # Sort so that objective metrics appear first
        if experiment.optimization_config is not None:
            metric_names = [
                m.name for m in experiment.optimization_config.objective.metrics
            ]
            map_metrics.sort(
                key=lambda e: e.name in metric_names,
                reverse=True,
            )
        return map_metrics

    def _estimate_hypothetical_savings_with_replay(
        self, experiment: Experiment, map_metrics: list[MapMetric]
    ) -> dict[str, float]:
        """
        Estimate hypothetical early stopping savings for each map metric using
        replay infrastructure.

        This is the accurate method that replays the experiment with early stopping
        enabled to calculate actual savings.

        Args:
            experiment: The experiment to analyze
            map_metrics: List of MapMetrics to analyze

        Returns:
            Dictionary mapping metric names to estimated savings percentages
            (only includes metrics where savings > min_savings_threshold)
        """
        metric_to_savings: dict[str, float] = {}

        MAX_REPLAYS = 3
        MAX_REPLAY_TRIALS = 50
        REPLAY_NUM_POINTS_PER_CURVE = 20
        REPLAY_PERCENTILE_THRESHOLD = 65
        REPLAY_MIN_PROGRESSION_FRAC = 0.4
        REPLAY_MIN_CURVES = 5

        # Limit to first few metrics to avoid expensive computation
        for map_metric in map_metrics[:MAX_REPLAYS]:
            try:
                # Create replayed experiment with early stopping
                replayed_experiment = replay_experiment(
                    historical_experiment=experiment,
                    num_samples_per_curve=REPLAY_NUM_POINTS_PER_CURVE,
                    max_replay_trials=MAX_REPLAY_TRIALS,
                    metric=map_metric,
                    max_pending_trials=self.max_pending_trials,
                    early_stopping_strategy=PercentileEarlyStoppingStrategy(
                        min_curves=REPLAY_MIN_CURVES,
                        min_progression=REPLAY_MIN_PROGRESSION_FRAC,
                        percentile_threshold=REPLAY_PERCENTILE_THRESHOLD,
                        normalize_progressions=True,
                    ),
                )

                if replayed_experiment is not None:
                    savings = estimate_early_stopping_savings(
                        experiment=replayed_experiment
                    )

                    # Only include if savings exceed threshold (> 1%)
                    if savings > self.min_savings_threshold:
                        metric_to_savings[map_metric.name] = 100 * savings

            except Exception:
                # Skip metrics that fail replay
                continue

        return metric_to_savings
