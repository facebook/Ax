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
from ax.core.trial_status import TrialStatus
from ax.exceptions.core import DataRequiredError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.utils.best_point import (
    get_best_by_raw_objective_with_trial_index,
    get_best_parameters_from_model_predictions_with_trial_index,
    get_pareto_optimal_parameters,
)
from pyre_extensions import none_throws, override

# Subtitle constants for reuse
_MODEL_PREDICTIONS_DESCRIPTION = (
    "based on model predictions. The model predictions apply shrinkage for "
    "noise and adjust for non-stationarity, making them more representative "
    "of reproducible effects."
)
_RAW_OBSERVATIONS_DESCRIPTION = (
    "based on raw observations. This reflects actual measured performance "
    "during execution."
)
_PARETO_DESCRIPTION = (
    "These trials represent optimal trade-offs between competing objectives. "
    "Use this to understand the available trade-offs and select a trial that "
    "best balances your optimization goals."
)
_SOO_DESCRIPTION = (
    "This trial achieved the optimal objective value and represents the "
    "recommended configuration for your optimization goal."
)


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
        - trial_status: The status of the trial (e.g. RUNNING, SUCCEEDED, FAILED)
        - failure_reason: The reason for the failure, if applicable
        - generation_node: The name of the ``GenerationNode`` that generated the arm
        - **METADATA: Any metadata associated with the trial, as specified by the
            Experiment's runner.run_metadata_report_keys field
        - **METRIC_NAME: The observed mean of the metric specified, for each metric
        - **PARAMETER_NAME: The parameter value for the arm, for each parameter
    Args:
        trial_statuses: If specified, only include trials with this status.
        use_model_predictions: If True, use model predictions for best trial selection
            instead of raw observations. This is useful in noisy settings where model
            predictions can help filter out observation noise.
    """

    def __init__(
        self,
        trial_statuses: Sequence[TrialStatus] | None = None,
        use_model_predictions: bool = False,
    ) -> None:
        self.trial_statuses: Sequence[TrialStatus] = (
            [TrialStatus.COMPLETED] if trial_statuses is None else trial_statuses
        )
        self.use_model_predictions = use_model_predictions

    @override
    def validate_applicable_state(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> str | None:
        # Basic experiment validation
        error = validate_experiment(
            experiment=experiment,
            require_trials=True,
            require_data=True,
        )
        if error is not None:
            return error

        # Validate optimization config exists
        if experiment is None or experiment.optimization_config is None:
            return (
                "`BestTrials` analysis requires an `OptimizationConfig`. "
                "Ensure the `Experiment` has an `optimization_config` set to compute "
                "this analysis."
            )

        # Check for trials with required status
        eligible_trial_indices = [
            trial.index
            for trial in experiment.trials.values()
            if trial.status in self.trial_statuses
        ]
        if not eligible_trial_indices:
            status_names = [s.name for s in self.trial_statuses]
            return f"No trials found with status in {status_names}."

        optimization_config = experiment.optimization_config

        # Validate GenerationStrategy is present when using model predictions or MOO
        if self.use_model_predictions or optimization_config.is_moo_problem:
            if generation_strategy is None:
                return (
                    "`BestTrials` analysis requires a `GenerationStrategy` input "
                    "when using model predictions or for multi-objective "
                    "optimization problems."
                )

        return None

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> AnalysisCard:
        exp = none_throws(experiment)
        optimization_config = none_throws(exp.optimization_config)

        # Filter trials by status before finding best trials
        eligible_trial_indices = [
            trial.index
            for trial in exp.trials.values()
            if trial.status in self.trial_statuses
        ]

        trial_indices = self._get_best_trial_indices(
            experiment=exp,
            generation_strategy=generation_strategy,
            optimization_config=optimization_config,
            trial_indices=eligible_trial_indices,
        )

        if not trial_indices:
            raise DataRequiredError(
                "No best trial(s) could be identified. This could be due to "
                "insufficient data or no trials meeting the optimization criteria."
            )

        # Use Summary analysis to compute the dataframe for the best trials to ensure
        # consistency in formatting.
        summary = Summary(trial_indices=trial_indices)
        summary_card = summary.compute(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        is_moo = optimization_config.is_moo_problem

        # Build descriptive subtitle based on optimization type and prediction method
        prediction_method = (
            _MODEL_PREDICTIONS_DESCRIPTION
            if self.use_model_predictions
            else _RAW_OBSERVATIONS_DESCRIPTION
        )

        if is_moo:
            title_prefix = "Pareto Frontier Trials"
            subtitle = (
                f"Displays trials on the Pareto frontier {prediction_method} "
                f"No trial is strictly better across all objectives. "
                f"{_PARETO_DESCRIPTION}"
            )
        else:
            title_prefix = "Best Trial"
            subtitle = (
                f"Displays the trial with the best objective value "
                f"{prediction_method} {_SOO_DESCRIPTION}"
            )

        # Add trial status context
        status_names = ", ".join(s.name for s in self.trial_statuses)
        subtitle += f" Only considering {status_names} trials."

        # Add relativization context
        if "relativized" in summary_card.subtitle:
            subtitle += " Metric values are shown relative to the status quo baseline."

        return self._create_analysis_card(
            title=(f"{title_prefix} for Experiment"),
            subtitle=subtitle,
            df=summary_card.df,
        )

    def _get_best_trial_indices(
        self,
        experiment: Experiment,
        generation_strategy: GenerationStrategy | None,
        optimization_config: OptimizationConfig,
        trial_indices: list[int],
    ) -> list[int]:
        """Get the trial indices of the best trial(s) based on optimization type.

        Note: All best point methods used here only consider in-sample points
        (arms that have already been run on the experiment). They do not return
        parameters that have not been tried yet.

        Args:
            experiment: The experiment to get best trials from.
            generation_strategy: The generation strategy, required for MOO or
                model predictions.
            optimization_config: The optimization config for the experiment.
            trial_indices: The trial indices to consider when finding best trials.
                This should be pre-filtered by trial status.

        Returns:
            A list of trial indices representing the best trial(s). For SOO, this
            contains at most one index. For MOO, this contains all Pareto optimal
            trial indices.
        """
        if optimization_config.is_moo_problem:
            # For MOO, get the Pareto optimal parameters.
            pareto_optimal = get_pareto_optimal_parameters(
                experiment=experiment,
                generation_strategy=none_throws(generation_strategy),
                optimization_config=optimization_config,
                trial_indices=trial_indices,
                use_model_predictions=self.use_model_predictions,
            )
            return list(pareto_optimal.keys())
        else:
            # For SOO, get the best trial
            if self.use_model_predictions:
                best_trial_result = (
                    get_best_parameters_from_model_predictions_with_trial_index(
                        experiment=experiment,
                        adapter=none_throws(generation_strategy).adapter,
                        optimization_config=optimization_config,
                        trial_indices=trial_indices,
                    )
                )
            else:
                best_trial_result = get_best_by_raw_objective_with_trial_index(
                    experiment=experiment,
                    optimization_config=optimization_config,
                    trial_indices=trial_indices,
                )
            return [best_trial_result[0]] if best_trial_result is not None else []
