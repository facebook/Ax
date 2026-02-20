# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final, Literal

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
from ax.core.map_metric import MapMetric
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.early_stopping.dispatch import get_default_ess_or_none
from ax.early_stopping.experiment_replay import (
    estimate_hypothetical_early_stopping_savings,
    MAX_CONCURRENT_TRIALS,
    MIN_SAVINGS_THRESHOLD,
)
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.early_stopping.utils import (
    EARLY_STOPPING_NUDGE_MSG,
    EARLY_STOPPING_NUDGE_TITLE,
    EARLY_STOPPING_SAVINGS_TITLE,
    estimate_early_stopping_savings,
    format_early_stopping_savings_message,
)
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.utils.early_stopping import get_early_stopping_metrics
from pyre_extensions import none_throws, override


DEFAULT_EARLY_STOPPING_HEALTHCHECK_TITLE = "Early Stopping Healthcheck"

# Type alias for auto_early_stopping_config
AutoEarlyStoppingConfig = Literal["disabled", "standard"]


@final
class EarlyStoppingAnalysis(Analysis):
    """
    Healthcheck that evaluates early stopping status and potential savings using
    experiment replay and the number of early stopped trials vs total completed
    trials.

    This analysis serves two purposes:
    1. When early stopping is enabled: Reports status (trials stopped) and
       estimated resource savings
    2. When early stopping is not enabled: Suggests enabling early stopping if
       significant savings are possible

    Status Logic:
    - PASS: Early stopping is enabled and functioning correctly. Not enabled
      but no significant savings expected
    - WARNING: Early stopping is not enabled but could be beneficial, or
      early stopping was used but no strategy was provided for a MOO/constrained
      experiment. (For unconstrained single-objective problems,
      a default strategy will be used if not provided.)
    - FAIL: Early stopping is enabled but misconfigured

    Note on Multi-Objective and Constrained Experiments:
        For multi-objective (MOO) or constrained experiments, users should pass
        the original early_stopping_strategy used during the experiment for
        accurate reporting. The default early stopping strategy
        (PercentileEarlyStoppingStrategy) is only automatically applied to
        single-objective unconstrained experiments. If no strategy is provided
        for MOO/constrained experiments that have early stopped trials, this analysis
        will report a warning indicating that the original strategy should be provided.
    """

    def __init__(
        self,
        early_stopping_strategy: BaseEarlyStoppingStrategy | None = None,
        min_savings_threshold: float = MIN_SAVINGS_THRESHOLD,
        max_pending_trials: int = MAX_CONCURRENT_TRIALS,
        auto_early_stopping_config: AutoEarlyStoppingConfig | None = None,
        nudge_additional_info: str | None = None,
    ) -> None:
        """
        Args:
            early_stopping_strategy: The early stopping strategy being used
                (None if not enabled). The original strategy used during
                the experiment needs to be passed for multi-objective or constrained
                experiments, for accurate reporting. If not provided, a
                default early stopping strategy will only be used for
                single-objective unconstrained experiments.
            min_savings_threshold: Minimum savings threshold to suggest early
                stopping. Default is 0.1 (10% savings).
            max_pending_trials: Maximum number of concurrent trials for replay
                orchestrator. Default is 5.
            auto_early_stopping_config: A string for configuring automated early
                stopping strategy.
                Set to "disabled" to indicate ESS was explicitly disabled.
                Set to "standard" to indicate ESS was explicitly enabled using
                the default strategy for eligible experiments.
                When None, the healthcheck will use the presence of
                early_stopping_strategy or early stopped trials to infer status.
            nudge_additional_info: Optional additional information to append to
                the nudge subtitle. This can be used by callers to include
                context-specific information when early stopping is not enabled
                but could be beneficial.
        """
        self.early_stopping_strategy = early_stopping_strategy
        self.min_savings_threshold = min_savings_threshold
        self.max_pending_trials = max_pending_trials
        self.auto_early_stopping_config = auto_early_stopping_config
        self.nudge_additional_info = nudge_additional_info

    def _create_card(
        self,
        subtitle: str,
        status: HealthcheckStatus,
        df: pd.DataFrame | None = None,
        title: str = DEFAULT_EARLY_STOPPING_HEALTHCHECK_TITLE,
        **kwargs: float | str,
    ) -> HealthcheckAnalysisCard:
        """Helper to create healthcheck analysis cards with common parameters.

        Args:
            subtitle: The subtitle/description for the card.
            status: The healthcheck status (PASS, WARNING, FAIL).
            df: Optional DataFrame with details. Defaults to empty DataFrame.
            title: The card title. Defaults to "Early Stopping Healthcheck".
            **kwargs: Additional arguments passed to create_healthcheck_analysis_card.

        Returns:
            A HealthcheckAnalysisCard with the specified parameters.
        """
        return create_healthcheck_analysis_card(
            name=self.__class__.__name__,
            title=title,
            subtitle=subtitle,
            df=df if df is not None else pd.DataFrame(),
            status=status,
            **kwargs,
        )

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        """
        Validates that the experiment is suitable for early stopping analysis.

        Returns a validation error message (fails validation) if:
        - No experiment is provided
        - The experiment has no trials
        - The experiment has no data

        Returns None (passes validation) only when all requirements are met.
        """
        return validate_experiment(
            experiment=experiment, require_trials=True, require_data=True
        )

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> HealthcheckAnalysisCard:
        experiment = none_throws(experiment)

        # Calculate number of trials early stopped till now
        n_stopped = len(experiment.trial_indices_by_status[TrialStatus.EARLY_STOPPED])

        # Handle auto_early_stopping_config if provided
        # This takes precedence over early_stopping_strategy inference
        if self.auto_early_stopping_config == "disabled":
            return self._report_early_stopping_disabled(experiment=experiment)

        if self.auto_early_stopping_config == "standard":
            if self.early_stopping_strategy is None:
                self.early_stopping_strategy = get_default_ess_or_none(experiment)

        # If early stopping was used (strategy provided or trials were stopped),
        # report status. Note: n_stopped > 0 without a strategy can happen when
        # the experiment is loaded from storage where trials were previously early
        # stopped, but the ESS object wasn't persisted or passed to this analysis.
        if self.early_stopping_strategy is not None or n_stopped > 0:
            return self._report_early_stopping_status(
                experiment=experiment, n_stopped=n_stopped
            )

        # If early stopping is not enabled, check if it should be
        return self._report_early_stopping_nudge(experiment)

    def _report_early_stopping_disabled(
        self, experiment: Experiment
    ) -> HealthcheckAnalysisCard:
        """Report that early stopping was explicitly disabled.

        This method is called when auto_early_stopping_config is set to "disabled",
        while computing the early stopping healthcheck where users can explicitly
        opt-out of automated early stopping.

        Args:
            experiment: The experiment to analyze.
        """
        problem_type = self._get_problem_type(experiment)
        n_trials = len(experiment.trials)

        df = pd.DataFrame(
            [
                {"Property": "Early Stopping Status", "Value": "Disabled"},
                {
                    "Property": "Configuration",
                    "Value": "auto_early_stopping_config='disabled'",
                },
                {"Property": "Problem Type", "Value": problem_type},
                {"Property": "Total Trials", "Value": str(n_trials)},
            ]
        )

        return self._create_card(
            subtitle=(
                "Early stopping was explicitly disabled via "
                "auto_early_stopping_config='disabled'. No early stopping analysis "
                "will be performed. To enable early stopping, set "
                "auto_early_stopping_config='standard' or configure an "
                "early_stopping_strategy_config."
            ),
            status=HealthcheckStatus.PASS,
            df=df,
        )

    def _report_early_stopping_status(
        self, experiment: Experiment, n_stopped: int
    ) -> HealthcheckAnalysisCard:
        """Report early stopping status when it's enabled.

        Args:
            experiment: The experiment to analyze.
            n_stopped: Number of early stopped trials
        """
        # Use provided strategy or use get_default_ess_or_none
        # PercentileEarlyStoppingStrategy for single-objective unconstrained
        # problems only.
        if self.early_stopping_strategy is None:
            self.early_stopping_strategy = get_default_ess_or_none(experiment)

        # For MOO/constrained experiments, get_default_ess_or_none returns None.
        # In this case, warn the user to provide the original strategy for
        # accurate reporting
        if self.early_stopping_strategy is None:
            df = pd.DataFrame(
                [
                    {"Property": "Early Stopped Trials", "Value": str(n_stopped)},
                    {
                        "Property": "Problem Type",
                        "Value": self._get_problem_type(experiment),
                    },
                    {
                        "Property": "Action Required",
                        "Value": "Pass the original early_stopping_strategy to "
                        "EarlyStoppingAnalysis for accurate reporting",
                    },
                ]
            )
            return self._create_card(
                subtitle=(
                    f"This experiment has {n_stopped} early stopped trial(s), but no "
                    "early stopping strategy was provided. For multi-objective or "
                    "constrained experiments, please pass the original "
                    "early_stopping_strategy used during the experiment to "
                    "EarlyStoppingAnalysis for accurate reporting of metrics and "
                    "savings."
                ),
                status=HealthcheckStatus.WARNING,
                df=df,
            )

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
            return self._create_card(
                subtitle=(
                    "Early stopping is enabled but no metrics were found. "
                    "This may indicate a configuration issue. Please ensure that "
                    "the metrics used by the early stopping strategy are correctly "
                    "configured in the experiment."
                ),
                status=HealthcheckStatus.FAIL,
                df=df,
            )

        # Find a suitable MapMetric from the target metrics
        # If the first metric is not a MapMetric, try other metrics in the list
        target_metric_name = None
        for metric_name in target_ess_metric_names:
            metric = experiment.metrics[metric_name]
            if isinstance(metric, MapMetric) and metric.has_map_data:
                target_metric_name = metric_name
                break

        # If no suitable MapMetric found, return failure
        if target_metric_name is None:
            metric_types = [
                f"'{name}' ({type(experiment.metrics[name]).__name__})"
                for name in target_ess_metric_names
            ]
            df = pd.DataFrame(
                [
                    {"Property": "Metrics Checked", "Value": ", ".join(metric_types)},
                    {"Property": "Required Type", "Value": "MapMetric"},
                    {
                        "Property": "Action Required",
                        "Value": "Change at least one metric to MapMetric type",
                    },
                ]
            )
            return self._create_card(
                subtitle=(
                    f"None of the {len(target_ess_metric_names)} metrics used for "
                    f"early stopping are MapMetrics with map data. Early stopping "
                    f"requires time-series data."
                ),
                status=HealthcheckStatus.FAIL,
                df=df,
            )

        # Calculate savings
        savings = estimate_early_stopping_savings(experiment=experiment)

        # Construct the message
        n_completed = len(experiment.trial_indices_by_status[TrialStatus.COMPLETED])
        n_failed = len(experiment.trial_indices_by_status[TrialStatus.FAILED])
        n_running = len(experiment.trial_indices_by_status[TrialStatus.RUNNING])
        n_ran = n_stopped + n_completed + n_failed + n_running

        # Add multi-objective note if applicable
        subtitle = format_early_stopping_savings_message(
            n_stopped=n_stopped, n_ran=n_ran, savings=savings
        )
        if len(target_ess_metric_names) > 1:
            subtitle += (
                f"\n\nNote: Although {len(target_ess_metric_names)} metrics are "
                f"being used for early stopping, '{target_metric_name}' is being "
                f"used to compute resource savings."
            )

        # Create dataframe with details
        df = pd.DataFrame(
            [
                {"Property": "Early Stopped Trials", "Value": str(n_stopped)},
                {"Property": "Completed Trials", "Value": str(n_completed)},
                {"Property": "Failed Trials", "Value": str(n_failed)},
                {"Property": "Running Trials", "Value": str(n_running)},
                {"Property": "Total Trials", "Value": str(n_ran)},
                {
                    "Property": "Target Metric",
                    "Value": target_metric_name,
                },
                {
                    "Property": "Estimated Savings",
                    "Value": f"{savings * 100:.0f}%" if savings > 0 else "N/A",
                },
            ]
        )

        return self._create_card(
            title=EARLY_STOPPING_SAVINGS_TITLE,
            subtitle=subtitle,
            df=df,
            status=HealthcheckStatus.PASS,
        )

    def _report_early_stopping_nudge(
        self, experiment: Experiment
    ) -> HealthcheckAnalysisCard:
        """Check if early stopping should be suggested (nudge) by estimating
        hypothetical savings using replay logic.

        Only applicable for single-objective unconstrained experiments where a
        default early stopping strategy is available.
        """
        opt_config = none_throws(experiment.optimization_config)
        metric = next(iter(opt_config.objective.metrics))
        try:
            savings = estimate_hypothetical_early_stopping_savings(
                experiment=experiment,
                metric=metric,
                max_pending_trials=self.max_pending_trials,
            )
        except Exception as e:
            # Exception is raised when estimate_hypothetical_early_stopping_savings
            # cannot compute savings. This happens for:
            # - Multi-objective or constrained experiments (no default ESS)
            # - Experiments without MapMetric data
            # - Experiment replay failures
            return self._create_card(
                subtitle=(
                    f"Early stopping is not enabled. Automatic savings estimation "
                    f"is unavailable for this experiment: {e} To use "
                    f"early stopping, configure a strategy manually."
                ),
                status=HealthcheckStatus.PASS,
            )

        if savings < self.min_savings_threshold:
            return self._create_card(
                subtitle=(
                    "Early stopping is not enabled. We did not detect "
                    "significant potential savings at this time.\n\n"
                    "This could be because:\n"
                    "- The experiment hasn't run enough trials yet\n"
                    "- Trials have similar performance curves\n"
                    "- The current trial distribution doesn't show clear "
                    "underperformers"
                ),
                status=HealthcheckStatus.PASS,
            )

        # Found significant potential savings - nudge the user
        savings_pct = 100 * savings

        subtitle = EARLY_STOPPING_NUDGE_MSG.format(
            metric_name=metric.name, savings=savings_pct
        )

        # Append additional info if provided
        if self.nudge_additional_info:
            subtitle += f" {self.nudge_additional_info}"

        # Create detailed metrics table
        df = pd.DataFrame(
            [
                {
                    "Metric Name": metric.name,
                    "Estimated Savings": f"{savings_pct:.1f}%",
                }
            ]
        )

        title = EARLY_STOPPING_NUDGE_TITLE.format(savings=savings_pct)

        return self._create_card(
            title=title,
            subtitle=subtitle,
            df=df,
            status=HealthcheckStatus.WARNING,
            potential_savings=savings_pct,
            best_metric=metric.name,
        )

    def _get_problem_type(self, experiment: Experiment) -> str:
        """Get a human-readable description of the problem type."""
        opt_config = experiment.optimization_config
        if opt_config is None:
            return "No optimization config"

        if isinstance(opt_config, MultiObjectiveOptimizationConfig):
            return "Multi-objective"
        if len(opt_config.outcome_constraints) > 0:
            return f"Constrained ({len(opt_config.outcome_constraints)} constraints)"
        return "Single-objective unconstrained"

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
