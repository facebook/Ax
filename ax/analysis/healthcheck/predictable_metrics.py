# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final

import pandas as pd
from ax.adapter.base import Adapter
from ax.adapter.random import RandomAdapter
from ax.analysis.analysis import Analysis
from ax.analysis.healthcheck.healthcheck_analysis import (
    create_healthcheck_analysis_card,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.analysis.utils import validate_experiment
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.utils.report_utils import warn_if_unpredictable_metrics
from pyre_extensions import none_throws, override

# Constants for consistent messaging
HEALTHCHECK_TITLE: str = "Predictable Metrics"
HEALTHCHECK_DESCRIPTION: str = (
    "The predictable metrics health check evaluates whether metrics can "
    "be reliably predicted by the model. "
)
DEFAULT_MODEL_FIT_THRESHOLD = 0.1


@final
class PredictableMetricsAnalysis(Analysis):
    """
    Healthcheck that warns if any metrics are unpredictable by the model.

    A metric is considered unpredictable when the model's cross-validation
    R² (coefficient of determination) score falls below the threshold. Low R²
    indicates the model cannot reliably predict the metric values.
    Common causes include:

        - High measurement noise in the metric data
        - Insufficient data to learn the parameter-metric relationship
        - Metrics that depend on factors outside the search space

    Status Logic:
        - PASS: All metrics are predictable.
        - WARNING: Some metrics are unpredictable and may need attention.
    """

    def __init__(
        self,
        model_fit_threshold: float = DEFAULT_MODEL_FIT_THRESHOLD,
        guidance_message: str | None = None,
    ) -> None:
        """
        Args:
            model_fit_threshold: Minimum model fit score for a metric to be
                considered predictable. Default is 0.1. Metrics scoring below
                this threshold are flagged as unpredictable. The score is
                computed via cross-validation: 1.0 means perfect predictions,
                0.0 means predictions are no better than the average, and
                negative values indicate poor model fit.
            guidance_message: Optional message to show when unpredictable
                metrics are found, providing context or next steps.
        """
        self.model_fit_threshold = model_fit_threshold
        self.guidance_message = guidance_message

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        """
        Validate that the analysis can be computed for the given state.

        Args:
            experiment: The experiment to validate.
            generation_strategy: The generation strategy to validate.
            adapter: Optional adapter to use for predictions.

        Returns:
            An error message if the state is invalid, or None if valid.
        """
        # Use standard validation utility for experiment
        experiment_validation = validate_experiment(
            experiment=experiment,
        )
        if experiment_validation is not None:
            return experiment_validation

        if generation_strategy is None:
            return "PredictableMetricsAnalysis requires a GenerationStrategy."

        # Resolve adapter from generation strategy if not provided
        resolved_adapter = adapter
        if resolved_adapter is None:
            resolved_adapter = generation_strategy.adapter
            if resolved_adapter is None:
                # Fit the current generation node to get an adapter
                generation_strategy._curr._fit(experiment=none_throws(experiment))
                resolved_adapter = generation_strategy.adapter

        # RandomAdapter has no model to evaluate
        if isinstance(resolved_adapter, RandomAdapter):
            return (
                "PredictableMetricsAnalysis is not applicable when using a "
                "RandomAdapter because there is no model to evaluate."
            )

        return None

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> HealthcheckAnalysisCard:
        """
        Compute the predictable metrics healthcheck analysis.

        This method evaluates whether metrics in the experiment are predictable
        by the model using cross-validation. Metrics with model fit scores below
        the threshold are flagged as unpredictable.

        Args:
            experiment: The ``Experiment`` to analyze. Must not be None.
            generation_strategy: The ``GenerationStrategy`` used for the
                experiment. Must not be None.
            adapter: Optional ``Adapter`` to use for predictions. If None,
                the adapter from the generation strategy will be used.

        Returns:
            A ``HealthcheckAnalysisCard`` with status PASS if all metrics are
            predictable, or WARNING if some metrics are unpredictable.

        Note:
            This method assumes ``validate_applicable_state`` has already been
            called and returned None (i.e., the state is valid).
        """

        experiment = none_throws(experiment)
        generation_strategy = none_throws(generation_strategy)

        warning_message = warn_if_unpredictable_metrics(
            experiment=experiment,
            generation_strategy=generation_strategy,
            model_fit_threshold=self.model_fit_threshold,
        )

        if warning_message is not None:
            subtitle = HEALTHCHECK_DESCRIPTION + warning_message
            if self.guidance_message is not None:
                subtitle += "\n\n" + self.guidance_message
            return create_healthcheck_analysis_card(
                name=self.__class__.__name__,
                title=f"{HEALTHCHECK_TITLE} Warning",
                subtitle=subtitle.strip(),
                df=pd.DataFrame(),
                status=HealthcheckStatus.WARNING,
            )

        return create_healthcheck_analysis_card(
            name=self.__class__.__name__,
            title=f"{HEALTHCHECK_TITLE} Success",
            subtitle=HEALTHCHECK_DESCRIPTION + "All metrics are predictable.",
            df=pd.DataFrame(),
            status=HealthcheckStatus.PASS,
        )
