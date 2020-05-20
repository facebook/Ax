#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from typing import Any, Dict, Type

from ax.benchmark.benchmark_problem import BenchmarkProblem, SimpleBenchmarkProblem
from ax.core import ObservationFeatures
from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import ChoiceParameter, FixedParameter, RangeParameter
from ax.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ax.core.search_space import SearchSpace
from ax.core.simple_experiment import SimpleExperiment
from ax.core.trial import Trial
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.transforms.base import Transform
from ax.runners.synthetic import SyntheticRunner
from ax.storage.transform_registry import TRANSFORM_REGISTRY
from ax.storage.utils import get_object_properties


def experiment_to_dict(experiment: Experiment) -> Dict[str, Any]:
    """Convert Ax experiment to a dictionary."""
    return {
        "__type": experiment.__class__.__name__,
        "name": experiment._name,
        "description": experiment.description,
        "experiment_type": experiment.experiment_type,
        "search_space": experiment.search_space,
        "optimization_config": experiment.optimization_config,
        "tracking_metrics": list(experiment._tracking_metrics.values()),
        "runner": experiment.runner,
        "status_quo": experiment.status_quo,
        "time_created": experiment.time_created,
        "trials": experiment.trials,
        "is_test": experiment.is_test,
        "data_by_trial": experiment.data_by_trial,
    }


def simple_experiment_to_dict(experiment: SimpleExperiment) -> Dict[str, Any]:
    """Convert AE simple experiment to a dictionary."""
    return experiment_to_dict(experiment)


def batch_to_dict(batch: BatchTrial) -> Dict[str, Any]:
    """Convert Ax batch to a dictionary."""
    return {
        "__type": batch.__class__.__name__,
        "index": batch.index,
        "trial_type": batch.trial_type,
        "ttl_seconds": batch.ttl_seconds,
        "status": batch.status,
        "status_quo": batch.status_quo,
        "status_quo_weight_override": batch._status_quo_weight_override,
        "time_created": batch.time_created,
        "time_completed": batch.time_completed,
        "time_staged": batch.time_staged,
        "time_run_started": batch.time_run_started,
        "abandoned_reason": batch.abandoned_reason,
        "run_metadata": batch.run_metadata,
        "generator_run_structs": batch.generator_run_structs,
        "runner": batch.runner,
        "abandoned_arms_metadata": batch._abandoned_arms_metadata,
        "num_arms_created": batch._num_arms_created,
        "optimize_for_power": batch.optimize_for_power,
        "generation_step_index": batch._generation_step_index,
    }


def trial_to_dict(trial: Trial) -> Dict[str, Any]:
    """Convert Ax trial to a dictionary."""
    return {
        "__type": trial.__class__.__name__,
        "index": trial.index,
        "trial_type": trial.trial_type,
        "ttl_seconds": trial.ttl_seconds,
        "status": trial.status,
        "time_created": trial.time_created,
        "time_completed": trial.time_completed,
        "time_staged": trial.time_staged,
        "time_run_started": trial.time_run_started,
        "abandoned_reason": trial.abandoned_reason,
        "run_metadata": trial.run_metadata,
        "generator_run": trial.generator_run,
        "runner": trial.runner,
        "num_arms_created": trial._num_arms_created,
        "generation_step_index": trial._generation_step_index,
    }


def range_parameter_to_dict(parameter: RangeParameter) -> Dict[str, Any]:
    """Convert Ax range parameter to a dictionary."""
    return {
        "__type": parameter.__class__.__name__,
        "name": parameter.name,
        "parameter_type": parameter.parameter_type,
        "lower": parameter.lower,
        "upper": parameter.upper,
        "log_scale": parameter.log_scale,
        "digits": parameter.digits,
        "is_fidelity": parameter.is_fidelity,
        "target_value": parameter.target_value,
    }


def choice_parameter_to_dict(parameter: ChoiceParameter) -> Dict[str, Any]:
    """Convert Ax choice parameter to a dictionary."""
    return {
        "__type": parameter.__class__.__name__,
        "is_ordered": parameter.is_ordered,
        "is_task": parameter.is_task,
        "name": parameter.name,
        "parameter_type": parameter.parameter_type,
        "values": parameter.values,
        "is_fidelity": parameter.is_fidelity,
        "target_value": parameter.target_value,
    }


def fixed_parameter_to_dict(parameter: FixedParameter) -> Dict[str, Any]:
    """Convert Ax fixed parameter to a dictionary."""
    return {
        "__type": parameter.__class__.__name__,
        "name": parameter.name,
        "parameter_type": parameter.parameter_type,
        "value": parameter.value,
        "is_fidelity": parameter.is_fidelity,
        "target_value": parameter.target_value,
    }


def order_parameter_constraint_to_dict(
    parameter_constraint: OrderConstraint,
) -> Dict[str, Any]:
    """Convert Ax order parameter constraint to a dictionary."""
    return {
        "__type": parameter_constraint.__class__.__name__,
        "lower_name": parameter_constraint.lower_parameter.name,
        "upper_name": parameter_constraint.upper_parameter.name,
    }


def sum_parameter_constraint_to_dict(
    parameter_constraint: SumConstraint,
) -> Dict[str, Any]:
    """Convert Ax sum parameter constraint to a dictionary."""
    return {
        "__type": parameter_constraint.__class__.__name__,
        "parameter_names": parameter_constraint._parameter_names,
        "is_upper_bound": parameter_constraint._is_upper_bound,
        # SumParameterConstraint constructor takes in absolute value of
        # the bound and transforms it based on the is_upper_bound value
        "bound": abs(parameter_constraint._bound),
    }


def parameter_constraint_to_dict(
    parameter_constraint: ParameterConstraint,
) -> Dict[str, Any]:
    """Convert Ax sum parameter constraint to a dictionary."""
    return {
        "__type": parameter_constraint.__class__.__name__,
        "constraint_dict": parameter_constraint.constraint_dict,
        "bound": parameter_constraint.bound,
    }


def arm_to_dict(arm: Arm) -> Dict[str, Any]:
    """Convert Ax arm to a dictionary."""
    return {
        "__type": arm.__class__.__name__,
        "parameters": arm.parameters,
        "name": arm._name,
    }


def search_space_to_dict(search_space: SearchSpace) -> Dict[str, Any]:
    """Convert Ax search space to a dictionary."""
    return {
        "__type": search_space.__class__.__name__,
        "parameters": list(search_space.parameters.values()),
        "parameter_constraints": search_space.parameter_constraints,
    }


def metric_to_dict(metric: Metric) -> Dict[str, Any]:
    """Convert Ax metric to a dictionary."""
    properties = get_object_properties(object=metric)
    properties["__type"] = metric.__class__.__name__
    return properties


def objective_to_dict(objective: Objective) -> Dict[str, Any]:
    """Convert Ax objective to a dictionary."""
    return {
        "__type": objective.__class__.__name__,
        "metric": objective.metric,
        "minimize": objective.minimize,
    }


def multi_objective_to_dict(objective: MultiObjective) -> Dict[str, Any]:
    """Convert Ax objective to a dictionary."""
    return {
        "__type": objective.__class__.__name__,
        "metrics": objective.metrics,
        "weights": objective.weights,
        "minimize": objective.minimize,
    }


def scalarized_objective_to_dict(objective: ScalarizedObjective) -> Dict[str, Any]:
    """Convert Ax objective to a dictionary."""
    return {
        "__type": objective.__class__.__name__,
        "metrics": objective.metrics,
        "weights": objective.weights,
        "minimize": objective.minimize,
    }


def outcome_constraint_to_dict(outcome_constraint: OutcomeConstraint) -> Dict[str, Any]:
    """Convert Ax outcome constraint to a dictionary."""
    return {
        "__type": outcome_constraint.__class__.__name__,
        "metric": outcome_constraint.metric,
        "op": outcome_constraint.op,
        "bound": outcome_constraint.bound,
        "relative": outcome_constraint.relative,
    }


def optimization_config_to_dict(
    optimization_config: OptimizationConfig,
) -> Dict[str, Any]:
    """Convert Ax optimization config to a dictionary."""
    return {
        "__type": optimization_config.__class__.__name__,
        "objective": optimization_config.objective,
        "outcome_constraints": optimization_config.outcome_constraints,
    }


def generator_run_to_dict(generator_run: GeneratorRun) -> Dict[str, Any]:
    """Convert Ax generator run to a dictionary."""
    gr = generator_run
    cand_metadata = gr.candidate_metadata_by_arm_signature
    return {
        "__type": gr.__class__.__name__,
        "arms": gr.arms,
        "weights": gr.weights,
        "optimization_config": gr.optimization_config,
        "search_space": gr.search_space,
        "time_created": gr.time_created,
        "model_predictions": gr.model_predictions,
        "best_arm_predictions": gr.best_arm_predictions,
        "generator_run_type": gr.generator_run_type,
        "index": gr.index,
        "fit_time": gr.fit_time,
        "gen_time": gr.gen_time,
        "model_key": gr._model_key,
        "model_kwargs": gr._model_kwargs,
        "bridge_kwargs": gr._bridge_kwargs,
        "gen_metadata": gr._gen_metadata,
        "model_state_after_gen": gr._model_state_after_gen,
        "generation_step_index": gr._generation_step_index,
        "candidate_metadata_by_arm_signature": cand_metadata,
    }


def runner_to_dict(runner: SyntheticRunner) -> Dict[str, Any]:
    """Convert Ax synthetic runner to a dictionary."""
    properties = get_object_properties(object=runner)
    properties["__type"] = runner.__class__.__name__
    return properties


def data_to_dict(data: Data) -> Dict[str, Any]:
    """Convert Ax data to a dictionary."""
    return {
        "__type": data.__class__.__name__,
        "df": data.df,
        "description": data.description,
    }


def transform_type_to_dict(transform_type: Type[Transform]) -> Dict[str, Any]:
    """Convert a transform class to a dictionary."""
    return {
        "__type": "Type[Transform]",
        "index_in_registry": TRANSFORM_REGISTRY[transform_type],
        "transform_type": f"{transform_type}",
    }


def generation_strategy_to_dict(
    generation_strategy: GenerationStrategy,
) -> Dict[str, Any]:
    if generation_strategy.uses_non_registered_models:
        raise ValueError(  # pragma: no cover
            "Generation strategies that use custom models provided through "
            "callables cannot be serialized and stored."
        )
    return {
        "__type": generation_strategy.__class__.__name__,
        "db_id": generation_strategy._db_id,
        "name": generation_strategy.name,
        "steps": generation_strategy._steps,
        "curr_index": generation_strategy._curr.index,
        "generator_runs": generation_strategy._generator_runs,
        "had_initialized_model": generation_strategy.model is not None,
        "experiment": generation_strategy._experiment,
    }


def observation_features_to_dict(obs_features: ObservationFeatures) -> Dict[str, Any]:
    """Converts Ax observation features to a dictionary"""
    return {
        "__type": obs_features.__class__.__name__,
        "parameters": obs_features.parameters,
        "trial_index": obs_features.trial_index,
        "start_time": obs_features.start_time,
        "end_time": obs_features.end_time,
        "random_split": obs_features.random_split,
        "metadata": obs_features.metadata,
    }


def benchmark_problem_to_dict(benchmark_problem: BenchmarkProblem) -> Dict[str, Any]:
    """Converts an Ax benchmark problem to a serializable dictionary."""
    if isinstance(benchmark_problem, SimpleBenchmarkProblem):
        if benchmark_problem.uses_synthetic_function:
            function_name = benchmark_problem.f.name  # pyre-ignore[16]
            f = None
        else:
            function_name = benchmark_problem.f.__name__  # pyre-ignore[16]
            f = pickle.dumps(benchmark_problem.f, 0).decode()
        return {
            "__type": benchmark_problem.__class__.__name__,
            "uses_synthetic_function": benchmark_problem.uses_synthetic_function,
            "function_name": function_name,
            # If the benchamrk problem uses a custom callable, pickle it.
            "f": f,
            "name": benchmark_problem.name,
            "domain": benchmark_problem.domain,
            "minimize": benchmark_problem.minimize,
            "noise_sd": benchmark_problem.noise_sd,
            "evaluate_suggested": benchmark_problem.evaluate_suggested,
            "optimal_value": benchmark_problem.optimal_value,
        }
    elif isinstance(benchmark_problem, BenchmarkProblem):
        properties = get_object_properties(object=benchmark_problem)
        properties["__type"] = benchmark_problem.__class__.__name__
        return properties
    else:  # pragma: no cover
        raise ValueError(f"Expected benchmark problem, got: {benchmark_problem}.")
