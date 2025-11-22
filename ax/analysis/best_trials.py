# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import final, Sequence

from ax.adapter.base import Adapter

from ax.analysis.analysis import Analysis
from ax.analysis.analysis_card import AnalysisCard
from ax.analysis.summary import Summary
from ax.analysis.utils import validate_experiment
from ax.core.experiment import Experiment
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import ScalarizedOutcomeConstraint
from ax.core.trial_status import NON_STALE_STATUSES, TrialStatus
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.utils.best_point import (
    get_best_by_raw_objective_with_trial_index,
    get_best_parameters_from_model_predictions_with_trial_index,
    get_pareto_optimal_parameters,
)
from pyre_extensions import override


@final
class BestTrials(Analysis):
    """
    High-level summary of the best trial(s) in the Experiment with one row per arm.
    Any values missing at compute time will be represented as None. Columns where
    every value is None will be omitted by default.

    For single-objective optimization (SOO), this analysis identifies the best trial
    while for multi-objective optimization (MOO), this analysis identifies the Pareto
    frontier trials.

    The DataFrame computed will contain one row per best arm and the following columns:
        - trial_index: The trial index of the arm
        - arm_name: The name of the arm
        - trial_status: The status of the trial (e.g. RUNNING, SUCCEDED, FAILED)
        - failure_reason: The reason for the failure, if applicable
        - generation_node: The name of the ``GenerationNode`` that generated the arm
        - **METADATA: Any metadata associated with the trial, as specified by the
            Experiment's runner.run_metadata_report_keys field
        - **METRIC_NAME: The observed mean of the metric specified, for each metric
        - **PARAMETER_NAME: The parameter value for the arm, for each parameter
     Args:
        trial_statuses: If specified, only include trials with this status.
        omit_empty_columns: If True, omit columns where every value is None.
        use_model_predictions: If True, use model predictions for best trial selection
            instead of raw observations. This is useful in noisy settings where model
            predictions can help filter out observation noise.
    """

    def __init__(
        self,
        trial_statuses: Sequence[TrialStatus] | None = None,
        omit_empty_columns: bool = True,
        use_model_predictions: bool = False,
    ) -> None:
        self.trial_statuses: Sequence[TrialStatus] = (
            trial_statuses if trial_statuses is not None else list(NON_STALE_STATUSES)
        )
        self.omit_empty_columns = omit_empty_columns
        self.use_model_predictions = use_model_predictions

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
            require_data=True,
        )

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCard:
        if experiment is None:
            raise UserInputError("`BestTrials` analysis requires an `Experiment` input")

        if generation_strategy is None:
            raise UserInputError(
                "`BestTrials` analysis requires a `GenerationStrategy` input"
            )

        if experiment.optimization_config is None:
            raise UserInputError(
                "`BestTrials` analysis requires an `OptimizationConfig`. "
                "Ensure the `Experiment` has an `optimization_config` set."
            )
        optimization_config = experiment.optimization_config

        # Check for ScalarizedOutcomeConstraints, which are not currently supported
        if any(
            isinstance(oc, ScalarizedOutcomeConstraint)
            for oc in optimization_config.outcome_constraints
        ):
            raise UserInputError(
                "`BestTrials` does not currently support experiments with "
                "`ScalarizedOutcomeConstraints`. Please use experiments with "
                "regular `OutcomeConstraint` objects."
            )

        trial_indices = self._get_best_trial_indices(
            experiment=experiment,
            generation_strategy=generation_strategy,
            optimization_config=optimization_config,
        )

        if not trial_indices:
            raise UserInputError(
                "No best trial(s) could be identified. This could be due to "
                "insufficient data or no trials meeting the optimization criteria."
            )

        # Use Summary analysis to compute the dataframe for the best trials to ensure
        # consistency in formatting.
        summary = Summary(
            trial_indices=trial_indices,
            trial_statuses=self.trial_statuses,
            omit_empty_columns=self.omit_empty_columns,
        )
        summary_card = summary.compute(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        is_moo = optimization_config.is_moo_problem
        title_suffix = "Pareto Frontier Trials" if is_moo else "Best Trial"
        subtitle_prefix = "Pareto Frontier trials" if is_moo else "Best trial"

        # Extract the relativization info from the summary subtitle
        relativization_info = (
            "Metric results are relativized against status quo."
            if "relativized" in summary_card.subtitle
            else ""
        )

        return self._create_analysis_card(
            title=(
                f"{title_suffix} for "
                f"{experiment.name if experiment.has_name else 'Experiment'}"
            ),
            subtitle=(
                f"{subtitle_prefix} in this experiment. {relativization_info}".strip()
            ),
            df=summary_card.df,
        )

    def _get_best_trial_indices(
        self,
        experiment: Experiment,
        generation_strategy: GenerationStrategy,
        optimization_config: OptimizationConfig,
    ) -> list[int]:
        """Get the trial indices of the best trial(s) based on optimization type."""
        if optimization_config.is_moo_problem:
            # For MOO, get the Pareto optimal parameters
            pareto_optimal = get_pareto_optimal_parameters(
                experiment=experiment,
                generation_strategy=generation_strategy,
                optimization_config=optimization_config,
                trial_indices=None,
                use_model_predictions=self.use_model_predictions,
            )
            return list(pareto_optimal.keys())
        else:
            # For SOO, get the best trial
            if self.use_model_predictions:
                best_trial_result = (
                    get_best_parameters_from_model_predictions_with_trial_index(
                        experiment=experiment,
                        adapter=generation_strategy.adapter,
                        optimization_config=optimization_config,
                        trial_indices=None,
                    )
                )
            else:
                best_trial_result = get_best_by_raw_objective_with_trial_index(
                    experiment=experiment,
                    optimization_config=optimization_config,
                    trial_indices=None,
                )
            if best_trial_result is None:
                return []
            trial_index, _, _ = best_trial_result
            return [trial_index]
