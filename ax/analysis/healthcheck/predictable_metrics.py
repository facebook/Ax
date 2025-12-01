# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final

import pandas as pd
from ax.adapter.base import Adapter
from ax.adapter.cross_validation import compute_model_fit_metrics_from_adapter
from ax.adapter.random import RandomAdapter
from ax.analysis.analysis import Analysis
from ax.analysis.healthcheck.healthcheck_analysis import (
    create_healthcheck_analysis_card,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import none_throws, override

HEALTHCHECK_DESCRIPTION: str = (
    "The predictable metrics health check evaluates whether metrics can "
    "be reliably predicted by the model. "
)


@final
class PredictableMetricsAnalysis(Analysis):
    """
    Healthcheck that warns if any metrics are unpredictable by the model.

    A metric is considered unpredictable if its coefficient of determination (R²)
    is below a specified threshold. Predictability is evaluated using
    cross-validation on generalization data.

    Status Logic:
    - PASS: All metrics are predictable (R² >= threshold)
    - WARNING: Some or all metrics are unpredictable (R² < threshold)
    """

    def __init__(
        self,
        model_fit_threshold: float = 0.0,
        guidance_message: str | None = None,
    ) -> None:
        """
        Args:
            model_fit_threshold: Minimum R² value for a metric to be considered
                predictable. Default is 0.0, meaning metrics with R² < 0.0 are
                flagged as unpredictable. Checks all metrics in the optimization
                config (or all experiment metrics if no optimization config).
            guidance_message: Optional guidance message to append to the warning
                subtitle when unpredictable metrics are found. If None, no
                additional guidance is provided.
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
        if experiment is None:
            return "PredictableMetricsAnalysis requires an Experiment."
        if generation_strategy is None:
            return "PredictableMetricsAnalysis requires a GenerationStrategy."
        return None

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> HealthcheckAnalysisCard:
        experiment = none_throws(experiment)
        generation_strategy = none_throws(generation_strategy)

        # Get adapter (refit if necessary)
        if adapter is None:
            adapter = generation_strategy.adapter
            if adapter is None:
                generation_strategy._curr._fit(experiment=experiment)
                adapter = generation_strategy.adapter

        adapter = none_throws(adapter)

        # Skip check for RandomAdapter (no model to evaluate)
        if isinstance(adapter, RandomAdapter):
            return create_healthcheck_analysis_card(
                name=self.__class__.__name__,
                title="Predictable Metrics Warning",
                subtitle=(
                    HEALTHCHECK_DESCRIPTION
                    + "Current adapter is RandomAdapter. Cannot evaluate metric "
                    "predictability as no model is being used."
                ),
                df=pd.DataFrame(),
                status=HealthcheckStatus.WARNING,
            )

        # Compute model fit metrics
        model_fit_dict = compute_model_fit_metrics_from_adapter(
            adapter=adapter,
            generalization=True,  # Use generalization metrics for evaluation
            untransform=False,
        )
        fit_quality_dict = model_fit_dict["coefficient_of_determination"]

        # Determine which metrics to check
        if experiment.optimization_config is None:
            metric_names = list(experiment.metrics.keys())
        else:
            metric_names = list(
                none_throws(experiment.optimization_config).metrics.keys()
            )

        # Find unpredictable metrics
        unpredictable_metrics: dict[str, float] = {
            k: v
            for k, v in fit_quality_dict.items()
            if k in metric_names and v < self.model_fit_threshold
        }

        # If all metrics are predictable, return PASS
        if not unpredictable_metrics:
            return create_healthcheck_analysis_card(
                name=self.__class__.__name__,
                title="Predictable Metrics Success",
                subtitle=HEALTHCHECK_DESCRIPTION + "All metrics are predictable.",
                df=pd.DataFrame(),
                status=HealthcheckStatus.PASS,
            )

        # Build warning message for unpredictable metrics
        subtitle = (
            HEALTHCHECK_DESCRIPTION
            + f"The following metric(s) are behaving unpredictably (R² < "
            f"{self.model_fit_threshold:.2f}) and may be noisy, misconfigured, "
            "or may not vary reliably as a function of your parameters:\n\n"
        )

        for metric_name in sorted(unpredictable_metrics.keys()):
            r_squared = unpredictable_metrics[metric_name]
            subtitle += f"- `{metric_name}` (R² = {r_squared:.4f})\n"

        # Add optional guidance message
        if self.guidance_message is not None:
            subtitle += self.guidance_message

        return create_healthcheck_analysis_card(
            name=self.__class__.__name__,
            title="Predictable Metrics Warning",
            subtitle=subtitle.strip(),
            df=pd.DataFrame(
                [
                    {"Metric": name, "R²": round(r2, 4)}
                    for name, r2 in sorted(unpredictable_metrics.items())
                ]
            ),
            status=HealthcheckStatus.WARNING,
        )
