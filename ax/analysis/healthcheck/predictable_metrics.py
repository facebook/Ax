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
from ax.core.experiment import Experiment
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.utils.report_utils import warn_if_unpredictable_metrics
from pyre_extensions import none_throws, override

HEALTHCHECK_DESCRIPTION: str = (
    "The predictable metrics health check evaluates whether metrics can "
    "be reliably predicted by the model. "
)


@final
class PredictableMetricsAnalysis(Analysis):
    """
    Healthcheck that warns if any metrics are unpredictable by the model.

    Unpredictable metrics may indicate noisy data, misconfigured metrics, or
    metrics that don't vary reliably with the parameters being optimized.

    Status Logic:
    - PASS: All metrics are predictable.
    - WARNING: Some metrics are unpredictable and may need attention.
    """

    def __init__(
        self,
        model_fit_threshold: float = 0.0,
        guidance_message: str | None = None,
    ) -> None:
        """
        Args:
            model_fit_threshold: Minimum model fit score for a metric to be
                considered predictable. Default is 0.0. Metrics scoring below
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
        if experiment is None:
            return "PredictableMetricsAnalysis requires an Experiment."
        if generation_strategy is None:
            return "PredictableMetricsAnalysis requires a GenerationStrategy."
        # Temporary: adapter evolves from RandomAdapter to model-based as GS progresses.
        if adapter is None:
            adapter = generation_strategy.adapter
            if adapter is None:
                generation_strategy._curr._fit(experiment=experiment)
                adapter = generation_strategy.adapter
        if isinstance(adapter, RandomAdapter):
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
                title="Predictable Metrics Warning",
                subtitle=subtitle.strip(),
                df=pd.DataFrame(),
                status=HealthcheckStatus.WARNING,
            )

        return create_healthcheck_analysis_card(
            name=self.__class__.__name__,
            title="Predictable Metrics Success",
            subtitle=HEALTHCHECK_DESCRIPTION + "All metrics are predictable.",
            df=pd.DataFrame(),
            status=HealthcheckStatus.PASS,
        )
