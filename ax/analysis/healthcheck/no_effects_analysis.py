# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Sequence

from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel
from ax.analysis.healthcheck.healthcheck_analysis import (
    HealthcheckAnalysis,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.core.auxiliary import AuxiliaryExperimentPurpose
from ax.core.experiment import Experiment
from ax.exceptions.core import AxError, UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from ax.utils.stats.no_effects import check_experiment_effects_per_metric
from pyre_extensions import override


class TestOfNoEffectAnalysis(HealthcheckAnalysis):
    """
    Analysis for checking whether a randomization test can show that there are any
    effects whatsoever. This test is performed independently on each metric
    and it is based on the Welch's test for testing whether the means across
    groups are identical assuming unequal variances across groups.
    """

    def __init__(self, no_effect_alpha: float = 0.05) -> None:
        r"""
        Args:
            no_effect_alpha: The confidence level at which to reject the
                null hypothesis of no experimental effects.

        """
        self.no_effect_alpha = no_effect_alpha

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[HealthcheckAnalysisCard]:
        r"""
        Compute the test of no effect separately for all metrics in the
        experiment. If objective metrics are found to not have effects,
        we return a card with a warning status and message.

        Args:
            experiment: Ax experiment.
            generation_strategy: Ax generation strategy.
            adapter: Ax modelbridge adapter

        Returns:
            A HealthcheckAnalysisCard object deatailing which metrics we don't
            detect any effect for.

        """
        status = HealthcheckStatus.PASS
        subtitle = "Effects are observed for all objective metrics."
        title_status = "Success"
        level = AnalysisCardLevel.LOW
        category = AnalysisCardCategory.DIAGNOSTIC

        if experiment is None:
            raise UserInputError("TestOfNoEffectAnalysis requires an Experiment.")
        if (
            pe_experiments := experiment.auxiliary_experiments_by_purpose.get(
                AuxiliaryExperimentPurpose.PE_EXPERIMENT
            )
        ) is not None and len(pe_experiments) > 0:
            objective_names = pe_experiments[0].experiment.parameters.keys()
        else:
            if (opt_config := experiment.optimization_config) is not None:
                objective_names = opt_config.objective.metric_names
            else:
                raise UserInputError(
                    (
                        "TestOfNoEffectAnalysis requires an optimization config "
                        "to be set on the experiment."
                    )
                )

        data = experiment.lookup_data()
        if data.df.empty:
            raise AxError(
                (
                    "TestOfNoEffectAnalysis requires data to be attached "
                    "to the experiment. "
                )
            )

        df_tone = check_experiment_effects_per_metric(
            data=data, objective_names=set(objective_names)
        )
        metrics_tone = df_tone.groupby("metric_name")["has_effect"].sum() > 0
        metrics_with_effects = [i for i in metrics_tone.index if metrics_tone[i]]
        objectives_without_effects = set()
        if "is_objective" in df_tone.columns:
            objectives_with_data = set(df_tone[df_tone["is_objective"]]["metric_name"])
            objectives_without_effects = objectives_with_data.difference(
                set(metrics_with_effects)
            )

        if len(objectives_without_effects) > 0:
            status = HealthcheckStatus.WARNING
            formatted_metrics = "<br>".join(objectives_without_effects)
            subtitle = (
                "The test of no effect checks to see whether the objective "
                "metrics of an experiment have had any detectable effects. Based "
                "on the experiment data thus far, no effects have been detected at a "
                f"{(1.0 - self.no_effect_alpha) * 100:.0f}% confidence level for "
                f"the following metrics: <br><br> {formatted_metrics}"
            )
            title_status = "Warning"

        return [
            self._create_healthcheck_analysis_card(
                title=f"Ax Test of No Effect {title_status}",
                subtitle=subtitle,
                df=df_tone,
                level=level,
                status=status,
                category=category,
            ),
        ]
