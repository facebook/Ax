#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.experiment import Experiment
from ax.core.optimization_config import OptimizationConfig
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.utils.best_point import (
    get_best_by_raw_objective_with_trial_index,
    get_best_parameters_from_model_predictions_with_trial_index,
    get_pareto_optimal_parameters,
)
from pyre_extensions import none_throws

BASELINE_ARM_NAME = "baseline_arm"


def get_best_trial_indices(
    experiment: Experiment,
    optimization_config: OptimizationConfig,
    generation_strategy: GenerationStrategy | None = None,
    trial_indices: list[int] | None = None,
    use_model_predictions: bool = False,
) -> list[int]:
    """Get the trial indices of the best trial(s) based on optimization type.

    Note: All best point methods used here only consider in-sample points
    (arms that have already been run on the experiment). They do not return
    parameters that have not been tried yet.

    Args:
        experiment: The experiment to get best trials from.
        optimization_config: The optimization config for the experiment.
        generation_strategy: The generation strategy, required for MOO or
            model predictions.
        trial_indices: The trial indices to consider when finding best trials.
            This should be pre-filtered by trial status.
            If None, considers all trials.
        use_model_predictions: If True, use model predictions instead of raw
            observations for best trial selection.

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
            use_model_predictions=use_model_predictions,
        )
        return list(pareto_optimal.keys())
    else:
        # For SOO, get the best trial
        if use_model_predictions:
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


def select_baseline_name_default_first_trial(
    experiment: Experiment, baseline_arm_name: str | None
) -> tuple[str, bool]:
    """
    Choose a baseline arm from arms on the experiment. Logic:
    1. If ``baseline_arm_name`` provided, validate that arm exists
       and return that arm name.
    2. If ``experiment.status_quo`` is set, return its arm name.
    3. If there is at least one trial on the experiment, use the
       first trial's first arm as the baseline.
    4. Error if 1-3 all don't apply.

    Returns:
        Tuple:
            baseline arm name (str)
            true when baseline selected from first arm of experiment (bool)
        raise ValueError if no valid baseline found
    """

    arms_dict = experiment.arms_by_name

    if baseline_arm_name:
        if baseline_arm_name not in arms_dict:
            raise ValueError(f"Arm by name {baseline_arm_name=} not found.")
        return baseline_arm_name, False

    if experiment.status_quo and none_throws(experiment.status_quo).name in arms_dict:
        baseline_arm_name = none_throws(experiment.status_quo).name
        return baseline_arm_name, False

    if (
        experiment.trials
        and experiment.trials[0].arms
        and experiment.trials[0].arms[0].name in arms_dict
    ):
        baseline_arm_name = experiment.trials[0].arms[0].name
        return baseline_arm_name, True

    raise UserInputError("Could not find valid baseline arm.")
