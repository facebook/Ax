#!/usr/bin/env python3

from typing import Any, Dict

from ae.lazarus.ae.core.batch_trial import BatchTrial
from ae.lazarus.ae.core.condition import Condition
from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.core.generator_run import GeneratorRun
from ae.lazarus.ae.core.metric import Metric
from ae.lazarus.ae.core.objective import Objective
from ae.lazarus.ae.core.optimization_config import OptimizationConfig
from ae.lazarus.ae.core.outcome_constraint import OutcomeConstraint
from ae.lazarus.ae.core.parameter import ChoiceParameter, FixedParameter, RangeParameter
from ae.lazarus.ae.core.parameter_constraint import (
    OrderConstraint,
    ParameterConstraint,
    SumConstraint,
)
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.core.trial import Trial
from ae.lazarus.ae.metrics.branin import BraninMetric
from ae.lazarus.ae.runners.synthetic import SyntheticRunner


def experiment_to_dict(experiment: Experiment) -> Dict[str, Any]:
    """Convert AE experiment to a dictionary."""
    return {
        "__type": experiment.__class__.__name__,
        "name": experiment.name,
        "description": experiment.description,
        "experiment_type": experiment.experiment_type,
        "search_space": experiment.search_space,
        "optimization_config": experiment.optimization_config,
        "tracking_metrics": list(experiment.metrics.values()),
        "runner": experiment.runner,
        "status_quo": experiment.status_quo,
        "time_created": experiment.time_created,
        "trials": experiment.trials,
        "is_test": experiment.is_test,
    }


def batch_to_dict(batch: BatchTrial) -> Dict[str, Any]:
    """Convert AE batch to a dictionary."""
    return {
        "__type": batch.__class__.__name__,
        "index": batch.index,
        "trial_type": batch.trial_type,
        "status": batch.status,
        "status_quo": batch.status_quo,
        "status_quo_weight": batch._status_quo_weight,
        "time_created": batch.time_created,
        "time_completed": batch.time_completed,
        "time_staged": batch.time_staged,
        "time_run_started": batch.time_run_started,
        "abandoned_reason": batch.abandoned_reason,
        "run_metadata": batch.run_metadata,
        "generator_run_structs": batch.generator_run_structs,
        "runner": batch.runner,
        "abandoned_conditions_metadata": batch._abandoned_conditions_metadata,
        "num_conditions_created": batch._num_conditions_created,
    }


def trial_to_dict(trial: Trial) -> Dict[str, Any]:
    """Convert AE trial to a dictionary."""
    return {
        "__type": trial.__class__.__name__,
        "index": trial.index,
        "trial_type": trial.trial_type,
        "status": trial.status,
        "time_created": trial.time_created,
        "time_completed": trial.time_completed,
        "time_staged": trial.time_staged,
        "time_run_started": trial.time_run_started,
        "abandoned_reason": trial.abandoned_reason,
        "run_metadata": trial.run_metadata,
        "generator_run": trial.generator_run,
        "runner": trial.runner,
        "num_conditions_created": trial._num_conditions_created,
    }


def range_parameter_to_dict(parameter: RangeParameter) -> Dict[str, Any]:
    """Convert AE range parameter to a dictionary."""
    return {
        "__type": parameter.__class__.__name__,
        "name": parameter.name,
        "parameter_type": parameter.parameter_type,
        "lower": parameter.lower,
        "upper": parameter.upper,
        "log_scale": parameter.log_scale,
        "digits": parameter.digits,
    }


def choice_parameter_to_dict(parameter: ChoiceParameter) -> Dict[str, Any]:
    """Convert AE choice parameter to a dictionary."""
    return {
        "__type": parameter.__class__.__name__,
        "is_ordered": parameter.is_ordered,
        "is_task": parameter.is_task,
        "name": parameter.name,
        "parameter_type": parameter.parameter_type,
        "values": parameter.values,
    }


def fixed_parameter_to_dict(parameter: FixedParameter) -> Dict[str, Any]:
    """Convert AE fixed parameter to a dictionary."""
    return {
        "__type": parameter.__class__.__name__,
        "name": parameter.name,
        "parameter_type": parameter.parameter_type,
        "value": parameter.value,
    }


def order_parameter_constraint_to_dict(
    parameter_constraint: OrderConstraint
) -> Dict[str, Any]:
    """Convert AE order parameter constraint to a dictionary."""
    return {
        "__type": parameter_constraint.__class__.__name__,
        "lower_name": parameter_constraint.lower_name,
        "upper_name": parameter_constraint.upper_name,
    }


def sum_parameter_constraint_to_dict(
    parameter_constraint: SumConstraint
) -> Dict[str, Any]:
    """Convert AE sum parameter constraint to a dictionary."""
    return {
        "__type": parameter_constraint.__class__.__name__,
        "parameter_names": parameter_constraint._parameter_names,
        "is_upper_bound": parameter_constraint._is_upper_bound,
        # SumParameterConstraint constructor takes in absolute value of
        # the bound and transforms it based on the is_upper_bound value
        "bound": abs(parameter_constraint._bound),
    }


def parameter_constraint_to_dict(
    parameter_constraint: ParameterConstraint
) -> Dict[str, Any]:
    """Convert AE sum parameter constraint to a dictionary."""
    return {
        "__type": parameter_constraint.__class__.__name__,
        "constraint_dict": parameter_constraint.constraint_dict,
        "bound": parameter_constraint.bound,
    }


def condition_to_dict(condition: Condition) -> Dict[str, Any]:
    """Convert AE condition to a dictionary."""
    return {
        "__type": condition.__class__.__name__,
        "params": condition.params,
        "name": condition._name,
    }


def search_space_to_dict(search_space: SearchSpace) -> Dict[str, Any]:
    """Convert AE search space to a dictionary."""
    return {
        "__type": search_space.__class__.__name__,
        "parameters": list(search_space.parameters.values()),
        "parameter_constraints": search_space.parameter_constraints,
    }


def metric_to_dict(metric: Metric) -> Dict[str, Any]:
    """Convert AE metric to a dictionary."""
    return {
        "__type": metric.__class__.__name__,
        "name": metric.name,
        "lower_is_better": metric.lower_is_better,
    }


def objective_to_dict(objective: Objective) -> Dict[str, Any]:
    """Convert AE objective to a dictionary."""
    return {
        "__type": objective.__class__.__name__,
        "metric": objective.metric,
        "minimize": objective.minimize,
    }


def outcome_constraint_to_dict(outcome_constraint: OutcomeConstraint) -> Dict[str, Any]:
    """Convert AE outcome constraint to a dictionary."""
    return {
        "__type": outcome_constraint.__class__.__name__,
        "metric": outcome_constraint.metric,
        "op": outcome_constraint.op,
        "bound": outcome_constraint.bound,
        "relative": outcome_constraint.relative,
    }


def optimization_config_to_dict(
    optimization_config: OptimizationConfig
) -> Dict[str, Any]:
    """Convert AE optimization config to a dictionary."""
    return {
        "__type": optimization_config.__class__.__name__,
        "objective": optimization_config.objective,
        "outcome_constraints": optimization_config.outcome_constraints,
    }


def generator_run_to_dict(generator_run: GeneratorRun) -> Dict[str, Any]:
    """Convert AE generator run to a dictionary."""
    return {
        "__type": generator_run.__class__.__name__,
        "conditions": generator_run.conditions,
        "weights": generator_run.weights,
        "optimization_config": generator_run.optimization_config,
        "search_space": generator_run.search_space,
        "time_created": generator_run.time_created,
        "model_predictions": generator_run.model_predictions,
        "best_condition_predictions": generator_run.best_condition_predictions,
        "generator_run_type": generator_run.generator_run_type,
        "index": generator_run.index,
        "fit_time": generator_run.fit_time,
        "gen_time": generator_run.gen_time,
    }


def synthetic_runner_to_dict(runner: SyntheticRunner) -> Dict[str, Any]:
    """Convert AE synthetic runner to a dictionary."""
    return {"__type": runner.__class__.__name__}


# TODO this should be generalized to handle arbitrary subclasses of SyntheticMetric
def branin_metric_to_dict(metric: BraninMetric) -> Dict[str, Any]:
    """Convert AE Branin metric to a dictionary."""
    return {
        "__type": metric.__class__.__name__,
        "name": metric.name,
        "param_names": metric.param_names,
        "noise_sd": metric.noise_sd,
        "lower_is_better": metric.lower_is_better,
    }
