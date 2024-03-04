#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import defaultdict
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import numpy as np
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.objective import MultiObjective

from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig

from ax.core.trial import Trial
from ax.core.types import ComparisonOp
from pyre_extensions import none_throws

TArmTrial = Tuple[str, int]

# Threshold for switching to pending points extraction based on trial status.
MANY_TRIALS_IN_EXPERIMENT = 100


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
    if isinstance(objective, MultiObjective):
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


def best_feasible_objective(
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
            raise ValueError(
                "Benchmark aggregation does not support relative constraints"
            )
        g = values[oc.metric.name]
        feas = g <= oc.bound if oc.op == ComparisonOp.LEQ else g >= oc.bound
        f[~feas] = infeas_val

    # Get cumulative best
    minimize = objective.minimize
    accumulate = np.minimum.accumulate if minimize else np.maximum.accumulate
    return accumulate(f)


def _extract_generator_run(trial: BaseTrial) -> GeneratorRun:
    if isinstance(trial, BatchTrial):
        if len(trial.generator_run_structs) > 1:
            raise NotImplementedError(
                "Run time is not supported with multiple generator runs per trial."
            )
        return trial._generator_run_structs[0].generator_run
    if isinstance(trial, Trial):
        return none_throws(trial.generator_run)
    raise ValueError("Unexpected trial type")


def get_model_trace_of_times(
    experiment: Experiment,
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    Get time spent fitting the model and generating candidates during each trial.
    Not cumulative.

    Returns:
        List of fit times, list of gen times.
    """
    generator_runs = [
        _extract_generator_run(trial=trial) for trial in experiment.trials.values()
    ]
    fit_times = [gr.fit_time for gr in generator_runs]
    gen_times = [gr.gen_time for gr in generator_runs]
    return fit_times, gen_times


def get_model_times(experiment: Experiment) -> Tuple[float, float]:
    """Get total times spent fitting the model and generating candidates in the
    course of the experiment.
    """
    fit_times, gen_times = get_model_trace_of_times(experiment)
    fit_time = sum((t for t in fit_times if t is not None))
    gen_time = sum((t for t in gen_times if t is not None))
    return fit_time, gen_time


# -------------------- Pending observations extraction utils. ---------------------


def extract_pending_observations(
    experiment: Experiment,
    include_out_of_design_points: bool = False,
) -> Optional[Dict[str, List[ObservationFeatures]]]:
    """Computes a list of pending observation features (corresponding to:
    - arms that have been generated and run in the course of the experiment,
    but have not been completed with data,
    - arms that have been abandoned or belong to abandoned trials).

    This function dispatches to:
    - ``get_pending_observation_features`` if experiment is using
    ``BatchTrial``-s or has fewer than 100 trials,
    - ``get_pending_observation_features_based_on_trial_status`` if
    experiment is using  ``Trial``-s and has more than 100 trials.

    ``get_pending_observation_features_based_on_trial_status`` is a faster
    way to compute pending observations, but it is not guaranteed to be
    accurate for ``BatchTrial`` settings and makes assumptions, e.g.
    arms in ``COMPLETED`` trial never being pending. See docstring of
    that function for more details.

    NOTE: Pending observation features are passed to the model to
    instruct it to not generate the same points again.
    """
    if len(experiment.trials) >= MANY_TRIALS_IN_EXPERIMENT and all(
        isinstance(t, Trial) for t in experiment.trials.values()
    ):
        return get_pending_observation_features_based_on_trial_status(
            experiment=experiment,
            include_out_of_design_points=include_out_of_design_points,
        )

    return get_pending_observation_features(
        experiment=experiment, include_out_of_design_points=include_out_of_design_points
    )


def get_pending_observation_features(
    experiment: Experiment,
    *,
    include_out_of_design_points: bool = False,
) -> Optional[Dict[str, List[ObservationFeatures]]]:
    """Computes a list of pending observation features (corresponding to:
    - arms that have been generated and run in the course of the experiment,
    but have not been completed with data,
    - arms that have been abandoned or belong to abandoned trials).

    NOTE: Pending observation features are passed to the model to
    instruct it to not generate the same points again.

    Args:
        experiment: Experiment, pending features on which we seek to compute.
        include_out_of_design_points: By default, this function will not include
            "out of design" points (those that are not in the search space) among
            the pending points. This is because pending points are generally used to
            help the model avoid re-suggesting the same points again. For points
            outside of the search space, this will not happen, so they typically do
            not need to be included. However, if the user wants to include them,
            they can be included by setting this flag to ``True``.

    Returns:
        An optional mapping from metric names to a list of observation features,
        pending for that metric (i.e. do not have evaluation data for that metric).
        If there are no pending features for any of the metrics, return is None.
    """

    def _is_in_design(arm: Arm) -> bool:
        return experiment.search_space.check_membership(parameterization=arm.parameters)

    pending_features = {metric_name: [] for metric_name in experiment.metrics}

    def create_observation_feature(
        arm: Arm,
        trial_index: int,
        trial: BaseTrial,
    ) -> Optional[ObservationFeatures]:
        if not include_out_of_design_points and not _is_in_design(arm=arm):
            return None
        return ObservationFeatures.from_arm(
            arm=arm,
            trial_index=trial_index,
            metadata=trial._get_candidate_metadata(arm_name=arm.name),
        )

    # Note that this assumes that if a metric appears in fetched data, the trial is
    # not pending for the metric. Where only the most recent data matters, this will
    # work, but may need to add logic to check previously added data objects, too.
    for trial_index, trial in experiment.trials.items():
        if trial.status.is_deployed:
            metric_names_in_data = set(trial.lookup_data().df.metric_name.values)
        else:
            metric_names_in_data = set()

        for metric_name in experiment.metrics:
            if metric_name not in pending_features:
                pending_features[metric_name] = []

            if trial.status.is_abandoned or (
                trial.status.is_deployed
                and metric_name not in metric_names_in_data
                and trial.arms is not None
            ):
                for arm in trial.arms:
                    if feature := create_observation_feature(
                        arm=arm,
                        trial_index=trial_index,
                        trial=trial,
                    ):
                        pending_features[metric_name].append(feature)

            # Also add abandoned arms as pending for all metrics.
            if isinstance(trial, BatchTrial):
                for arm in trial.abandoned_arms:
                    if feature := create_observation_feature(
                        arm=arm,
                        trial_index=trial_index,
                        trial=trial,
                    ):
                        pending_features[metric_name].append(feature)

    return pending_features if any(x for x in pending_features.values()) else None


# TODO: allow user to pass search space which overrides that on the experiment
# (to use for the `include_out_of_design_points` check)
def get_pending_observation_features_based_on_trial_status(
    experiment: Experiment,
    include_out_of_design_points: bool = False,
) -> Optional[Dict[str, List[ObservationFeatures]]]:
    """A faster analogue of ``get_pending_observation_features`` that makes
    assumptions about trials in experiment in order to speed up extraction
    of pending points.

    Assumptions:

    * All arms in all trials in ``STAGED,`` ``RUNNING`` and ``ABANDONED`` statuses
      are to be considered pending for all outcomes.
    * All arms in all trials in other statuses are to be considered not pending for
      all outcomes.

    This entails:

    * No actual data-fetching for trials to determine whether arms in them are pending
      for specific outcomes.
    * Even if data is present for some outcomes in ``RUNNING`` trials, their arms will
      still be considered pending for those outcomes.

    NOTE: This function should not be used to extract pending features in field
    experiments, where arms in running trials should not be considered pending if
    there is data for those arms.

    Args:
        experiment: Experiment, pending features on which we seek to compute.

    Returns:
        An optional mapping from metric names to a list of observation features,
        pending for that metric (i.e. do not have evaluation data for that metric).
        If there are no pending features for any of the metrics, return is None.
    """
    pending_features = defaultdict(list)
    for status in [TrialStatus.STAGED, TrialStatus.RUNNING, TrialStatus.ABANDONED]:
        for trial in experiment.trials_by_status[status]:
            for metric_name in experiment.metrics:
                for arm in trial.arms:
                    if (
                        not include_out_of_design_points
                        and not experiment.search_space.check_membership(arm.parameters)
                    ):
                        continue
                    pending_features[metric_name].append(
                        ObservationFeatures.from_arm(
                            arm=arm,
                            trial_index=trial.index,
                            metadata=trial._get_candidate_metadata(arm_name=arm.name),
                        )
                    )
    return dict(pending_features) if any(x for x in pending_features.values()) else None


def extend_pending_observations(
    experiment: Experiment,
    pending_observations: Dict[str, List[ObservationFeatures]],
    generator_run: GeneratorRun,
) -> None:
    """Extend given pending observations dict (from metric name to observations
    that are pending for that metric), with arms in a given generator run.
    """
    for m in experiment.metrics:
        if m not in pending_observations:
            pending_observations[m] = []
        pending_observations[m].extend(
            ObservationFeatures.from_arm(a) for a in generator_run.arms
        )
