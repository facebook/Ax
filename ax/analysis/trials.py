# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import final

from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.plotly.arm_effects import ArmEffectsPlot
from ax.analysis.utils import extract_relevant_adapter, validate_experiment
from ax.core.analysis_card import AnalysisCardGroup
from ax.core.base_trial import BaseTrial
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import override

TRIALS_CARDGROUP_TITLE = "Trial-Level Analyses"
TRIALS_CARDGROUP_SUBTITLE = (
    "This analysis provides detailed information about the individual trials in the "
    "experiment. It contains visualizations specifically computed for each trial."
)


@final
class AllTrialsAnalysis(Analysis):
    """
    An Analysis that provides detailed information about all trials in an experiment.

    AllTrialsAnalysis serves as a container for trial-level analyses, organizing them
    into separate card groups. Each child in the card group represents the output of
    TrialAnalysis for a specific trial in the experiment.
    """

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        return validate_experiment(
            experiment=experiment,
            require_trials=True,
            require_data=False,
        )

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardGroup:
        # Ensure we have an Experiment provided by the user to extract the relevant
        # metric names from.
        if experiment is None:
            raise UserInputError("TrialsAnalysis requires an Experiment.")

        trials = experiment.trials.values()

        return AnalysisCardGroup(
            name="AllTrialsAnalysis",
            title=TRIALS_CARDGROUP_TITLE,
            subtitle=TRIALS_CARDGROUP_SUBTITLE,
            children=[
                TrialAnalysis(trial=trial).compute_or_error_card(
                    experiment=experiment,
                    generation_strategy=generation_strategy,
                    adapter=adapter,
                )
                for trial in trials
                if isinstance(trial, BatchTrial)
            ],
        )


@final
class TrialAnalysis(Analysis):
    """
    An Analysis that provides detailed information about a specific trial.

    TrialAnalysis computes and organizes various analyses specific to a single trial,
    such as predicted effects plots.
    """

    @override
    def __init__(
        self,
        trial: BaseTrial,
    ) -> None:
        self.trial = trial

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        return validate_experiment(
            experiment=experiment,
            require_trials=True,
            require_data=False,
        )

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCardGroup:
        # Ensure we have an Experiment provided by the user to extract the relevant
        # metric names from.
        if experiment is None:
            raise UserInputError("TrialAnalysis requires an Experiment.")

        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        # If the Experiment has an OptimizationConfig set, extract the objective and
        # constraint names.
        objective_names = []
        constraint_names = []
        if (optimization_config := experiment.optimization_config) is not None:
            objective_names = optimization_config.objective.metric_names
            constraint_names = [
                constraint.metric.name
                for constraint in optimization_config.outcome_constraints
            ]

        relativize = experiment.status_quo is not None and isinstance(
            self.trial, BatchTrial
        )
        analyses = [
            ArmEffectsPlot(
                metric_name=metric_name,
                use_model_predictions=True,
                relativize=relativize,
                trial_index=self.trial.index,
            )
            for metric_name in [*objective_names, *constraint_names]
            if self.trial.status == TrialStatus.CANDIDATE
        ]
        return AnalysisCardGroup(
            name=f"{self.trial.index}",
            title=f"Trial {self.trial.index}",
            subtitle=(
                "Trial-level visualizations computed specifically for this trial."
            ),
            children=[
                analysis.compute_or_error_card(
                    experiment=experiment,
                    generation_strategy=generation_strategy,
                    adapter=relevant_adapter,
                )
                for analysis in analyses
            ],
        )
