#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Iterable, NamedTuple, Set, Tuple

import numpy as np
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective
from ax.core.optimization_config import OptimizationConfig
from ax.core.trial import Trial
from ax.core.types import ComparisonOp
from ax.utils.common.typeutils import not_none


TArmTrial = Tuple[str, int]


# --------------------------- Data intergrity utils. ---------------------------


class MissingMetrics(NamedTuple):
    objective: Dict[str, Set[TArmTrial]]
    outcome_constraints: Dict[str, Set[TArmTrial]]
    tracking_metrics: Dict[str, Set[TArmTrial]]


def get_missing_metrics(
    data: Data, optimization_config: OptimizationConfig
) -> MissingMetrics:
    """Return all arm_name, trial_index pairs, for which some of the
    observatins of optimization config metrics are missing.

    Args:
        data: Data to search.
        optimization_config: provides metric_names to search for.

    Returns:
        A NamedTuple(missing_objective, Dict[str, missing_outcome_constraint])
    """
    objective = optimization_config.objective
    if isinstance(objective, MultiObjective):  # pragma: no cover
        objective_metric_names = [m.name for m in objective.metrics]
    else:
        objective_metric_names = [optimization_config.objective.metric.name]

    outcome_constraints_metric_names = [
        outcome_constraint.metric.name
        for outcome_constraint in optimization_config.outcome_constraints
    ]
    missing_objectives = {
        objective_metric_name: _get_missing_arm_trial_pairs(data, objective_metric_name)
        for objective_metric_name in objective_metric_names
    }
    missing_outcome_constraints = get_missing_metrics_by_name(
        data, outcome_constraints_metric_names
    )
    all_metric_names = set(data.df["metric_name"])
    optimization_config_metric_names = set(missing_objectives.keys()).union(
        outcome_constraints_metric_names
    )
    missing_tracking_metric_names = all_metric_names.difference(
        optimization_config_metric_names
    )
    missing_tracking_metrics = get_missing_metrics_by_name(
        data=data, metric_names=missing_tracking_metric_names
    )
    return MissingMetrics(
        objective={k: v for k, v in missing_objectives.items() if len(v) > 0},
        outcome_constraints={
            k: v for k, v in missing_outcome_constraints.items() if len(v) > 0
        },
        tracking_metrics={
            k: v for k, v in missing_tracking_metrics.items() if len(v) > 0
        },
    )


def get_missing_metrics_by_name(
    data: Data, metric_names: Iterable[str]
) -> Dict[str, Set[TArmTrial]]:
    """Return all arm_name, trial_index pairs missing some observations of
    specified metrics.

    Args:
        data: Data to search.
        metric_names: list of metrics to search for.

    Returns:
        A Dict[str, missing_metrics], one entry for each metric_name.
    """
    missing_metrics = {
        metric_name: _get_missing_arm_trial_pairs(data=data, metric_name=metric_name)
        for metric_name in metric_names
    }
    return missing_metrics


def _get_missing_arm_trial_pairs(data: Data, metric_name: str) -> Set[TArmTrial]:
    """Return arm_name and trial_index pairs missing a specified metric."""
    metric_df = data.df[data.df.metric_name == metric_name]
    present_metric_df = metric_df[metric_df["mean"].notnull()]
    arm_trial_pairs = set(zip(data.df["arm_name"], data.df["trial_index"]))
    arm_trial_pairs_with_metric = set(
        zip(present_metric_df["arm_name"], present_metric_df["trial_index"])
    )
    missing_arm_trial_pairs = arm_trial_pairs.difference(arm_trial_pairs_with_metric)
    return missing_arm_trial_pairs


# -------------------- Experiment result extraction utils. ---------------------


def best_feasible_objective(  # pragma: no cover
    optimization_config: OptimizationConfig, values: Dict[str, np.ndarray]
) -> np.ndarray:
    """Compute the best feasible objective value found by each iteration.

    Args:
        optimization_config: Optimization config.
        values: Dictionary from metric name to array of value at each
            iteration. If optimization config contains outcome constraints, values
            for them must be present in `values`.

    Returns: Array of cumulative best feasible value.
    """
    # Get objective at each iteration
    objective = optimization_config.objective
    f = values[objective.metric.name]
    # Set infeasible points to have infinitely bad values
    infeas_val = np.Inf if objective.minimize else -np.Inf
    for oc in optimization_config.outcome_constraints:
        if oc.relative:
            raise ValueError(  # pragma: no cover
                "Benchmark aggregation does not support relative constraints"
            )
        g = values[oc.metric.name]
        feas = g <= oc.bound if oc.op == ComparisonOp.LEQ else g >= oc.bound
        f[~feas] = infeas_val

    # Get cumulative best
    minimize = objective.minimize
    accumulate = np.minimum.accumulate if minimize else np.maximum.accumulate
    return accumulate(f)


def get_model_times(experiment: Experiment) -> Tuple[float, float]:  # pragma: no cover
    """Get total times spent fitting the model and generating candidates in the
    course of the experiment.
    """
    fit_time = 0.0
    gen_time = 0.0
    for trial in experiment.trials.values():
        if isinstance(trial, BatchTrial):  # pragma: no cover
            gr = trial._generator_run_structs[0].generator_run
        elif isinstance(trial, Trial):
            gr = not_none(trial.generator_run)
        else:
            raise ValueError("Unexpected trial type")  # pragma: no cover
        if gr.fit_time is not None:
            fit_time += not_none(gr.fit_time)
        if gr.gen_time is not None:
            gen_time += not_none(gr.gen_time)
    return fit_time, gen_time
