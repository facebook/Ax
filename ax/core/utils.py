#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Callable, Iterable
from copy import deepcopy
from datetime import datetime
from functools import wraps
from logging import Logger
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.map_metric import MapMetric
from ax.core.objective import MultiObjective
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.trial import Trial
from ax.core.types import ComparisonOp
from ax.exceptions.core import AxError, UserInputError
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from pyre_extensions import none_throws

logger: Logger = get_logger(__name__)
TArmTrial = tuple[str, int]

# Threshold for switching to pending points extraction based on trial status.
MANY_TRIALS_IN_EXPERIMENT = 100


# --------------------------- Data integrity utils. ---------------------------


class MissingMetrics(NamedTuple):
    objective: dict[str, set[TArmTrial]]
    outcome_constraints: dict[str, set[TArmTrial]]
    tracking_metrics: dict[str, set[TArmTrial]]


def get_missing_metrics(
    data: Data, optimization_config: OptimizationConfig
) -> MissingMetrics:
    """Return all arm_name, trial_index pairs, for which some of the
    observations of optimization config metrics are missing.

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
) -> dict[str, set[TArmTrial]]:
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


def _get_missing_arm_trial_pairs(data: Data, metric_name: str) -> set[TArmTrial]:
    """Return arm_name and trial_index pairs missing a specified metric."""
    metric_df = data.df[data.df.metric_name == metric_name]
    present_metric_df = metric_df[metric_df["mean"].notnull()]
    arm_trial_pairs = set(zip(data.df["arm_name"], data.df["trial_index"]))
    arm_trial_pairs_with_metric = set(
        zip(present_metric_df["arm_name"], present_metric_df["trial_index"])
    )
    missing_arm_trial_pairs = arm_trial_pairs.difference(arm_trial_pairs_with_metric)
    return missing_arm_trial_pairs


# ------------------- Utils shared by Client and BatchClient--------------------
def _maybe_update_trial_status_to_complete(
    experiment: Experiment,
    trial_index: int,
) -> None:
    """Check if a trial has all relevant metrics and mark it as completed.
    If the trial has all relevant metrics, mark it as completed.
    If the trial is missing metrics, mark it as failed.

    Args:
        experiment: The experiment to check.
        trial_index: The index of the trial to check.
    """
    if experiment.optimization_config is None:
        raise UserInputError(
            "Cannot attempt to mark a trial as failed without an optimization"
            " config on the expeirment"
        )
    optimization_config = experiment.optimization_config
    trial_data = experiment.lookup_data(trial_indices=[trial_index])
    missing_metrics = {*optimization_config.metrics.keys()} - {*trial_data.metric_names}

    # If all necessary metrics are present mark the trial as COMPLETED
    if len(missing_metrics) == 0:
        experiment.trials[trial_index].mark_completed()
        return

    # If any metrics are missing mark the trial as FAILED
    logger.warning(
        f"Trial {trial_index} marked completed but metrics "
        f"{missing_metrics} are missing, marking trial FAILED."
    )
    experiment.trials[trial_index].mark_failed(
        reason=f"{missing_metrics} are missing, marking trial FAILED."
    )


# -------------------- Experiment result extraction utils. ---------------------


def best_feasible_objective(
    optimization_config: OptimizationConfig,
    values: dict[str, npt.NDArray],
) -> npt.NDArray:
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
    infeas_val = np.inf if objective.minimize else -np.inf
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


def _extract_generator_runs(trial: BaseTrial) -> list[GeneratorRun]:
    if isinstance(trial, BatchTrial):
        return trial.generator_runs
    if isinstance(trial, Trial):
        return [none_throws(trial.generator_run)]
    raise ValueError("Unexpected trial type")


def get_model_trace_of_times(
    experiment: Experiment,
) -> tuple[list[float | None], list[float | None]]:
    """
    Get time spent fitting the model and generating candidates during each trial.
    Not cumulative.

    Returns:
        List of fit times, list of gen times.
    """
    generator_runs = [
        gr
        for trial in experiment.trials.values()
        for gr in _extract_generator_runs(trial=trial)
    ]
    fit_times = [gr.fit_time for gr in generator_runs]
    gen_times = [gr.gen_time for gr in generator_runs]
    return fit_times, gen_times


def get_model_times(experiment: Experiment) -> tuple[float, float]:
    """Get total times spent fitting the model and generating candidates in the
    course of the experiment.
    """
    fit_times, gen_times = get_model_trace_of_times(experiment)
    fit_time = sum(t for t in fit_times if t is not None)
    gen_time = sum(t for t in gen_times if t is not None)
    return fit_time, gen_time


# -------------------- Pending observations extraction utils. ---------------------


def extract_pending_observations(
    experiment: Experiment,
    include_out_of_design_points: bool = False,
) -> dict[str, list[ObservationFeatures]] | None:
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
) -> dict[str, list[ObservationFeatures]] | None:
    """Computes a list of pending observation features (corresponding to:
    - arms that have been generated in the course of the experiment,
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
    ) -> ObservationFeatures | None:
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

            if (
                trial.status.is_abandoned
                or trial.status.is_candidate
                or (
                    trial.status.is_deployed
                    and metric_name not in metric_names_in_data
                    and trial.arms is not None
                )
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
) -> dict[str, list[ObservationFeatures]] | None:
    """A faster analogue of ``get_pending_observation_features`` that makes
    assumptions about trials in experiment in order to speed up extraction
    of pending points.

    Assumptions:

    * All arms in all trials in ``CANDIDATE``, ``STAGED``, ``RUNNING`` and ``ABANDONED``
      statuses are to be considered pending for all outcomes.
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
    pending_features_list = []
    for status in [
        TrialStatus.CANDIDATE,
        TrialStatus.STAGED,
        TrialStatus.RUNNING,
        TrialStatus.ABANDONED,
    ]:
        for trial in experiment.trials_by_status[status]:
            for arm in trial.arms:
                if (
                    include_out_of_design_points
                    or experiment.search_space.check_membership(arm.parameters)
                ):
                    pending_features_list.append(
                        ObservationFeatures.from_arm(
                            arm=arm,
                            trial_index=trial.index,
                            metadata=trial._get_candidate_metadata(arm_name=arm.name),
                        )
                    )
    pending_features = {
        # Using deepcopy here to avoid issues due to in-place transforms.
        metric_name: deepcopy(pending_features_list)
        for metric_name in experiment.metrics
    }
    return pending_features if pending_features_list else None


def extend_pending_observations(
    experiment: Experiment,
    pending_observations: dict[str, list[ObservationFeatures]],
    generator_run: GeneratorRun,
) -> None:
    """Extend given pending observations dict (from metric name to observations
    that are pending for that metric), with arms in a given generator run.

    Note: This function performs this operation in-place for performance reasons.
    It is only used within the ``GenerationStrategy`` class, and is not intended
    for wide re-use. Please use caution when re-using this function.

    Args:
        experiment: Experiment, for which the generation strategy is producing
            ``GeneratorRun``s.
        pending_observations: Dict from metric name to pending observations for
            that metric, used to avoid resuggesting arms that will be explored soon.
        generator_run: ``GeneratorRun`` currently produced by the
            ``GenerationStrategy`` to add to the pending points.

    """
    for m in experiment.metrics:
        if m not in pending_observations:
            pending_observations[m] = []
        pending_observations[m].extend(
            ObservationFeatures.from_arm(a) for a in generator_run.arms
        )
    return


# -------------------- Get target trial utils. ---------------------


def get_target_trial_index(experiment: Experiment) -> int | None:
    """Get the index of the target trial in the ``Experiment``.

    Find the target trial, among the trials with data for status quo arm, giving
    priority in the following order:
        1. a running long-run trial. Note if there is a running long-run trial on the
            experiment without data, or if there is no data on the experiment, then
            this will return None.
        2. Most recent trial expecting data with running trials be considered the most
            recent.

    In the event of any ties, the tie breaking order is:
        a. longest running trial by duration
        b. trial with most arms
        c. arbitrary selection

    Args:
        experiment: The experiment associated with this ``GenerationStrategy``.

    Returns:
        The index of the target trial in the ``Experiment``.
    """
    # TODO: @mgarrard improve logic to include trial_obsolete_threshold that
    # takes into account the age of the trial, and consider more heavily weighting
    # long run trials.
    df = experiment.lookup_data().df
    status_quo = experiment.status_quo
    if df.empty or status_quo is None:
        return None
    # Filter to only trials with data for status quo arm.
    df = df[df["arm_name"] == status_quo.name]
    trial_indices_with_data = set(df.trial_index.unique())
    # only consider running trials with data
    running_trials = [
        trial
        for trial in experiment.trials_by_status[TrialStatus.RUNNING]
        if trial.index in trial_indices_with_data
    ]
    sorted_running_trials = _sort_trials(trials=running_trials, trials_are_running=True)
    # Priority 1: Any running long-run trial
    has_running_long_run_trial = any(
        trial.trial_type == Keys.LONG_RUN
        for trial in experiment.trials_by_status[TrialStatus.RUNNING]
    )
    if has_running_long_run_trial:
        # This returns a running long-run trial with data or None
        # if there are running long-run trials on the experiment, but
        # no data for that trial
        return next(
            (
                trial.index
                for trial in sorted_running_trials
                if trial.trial_type == Keys.LONG_RUN
            ),
            None,
        )

    # Priority 2: longest running currently running trial with data
    if len(sorted_running_trials) > 0:
        return sorted_running_trials[0].index

    # Priortiy 3: the longest running trial with data, discounting running trials
    # as we handled those above
    non_running_trial_indices_with_data = trial_indices_with_data - {
        t.index for t in running_trials
    }
    non_running_trials_with_data = [
        experiment.trials[i] for i in non_running_trial_indices_with_data
    ]
    sorted_non_running_trials_with_data = _sort_trials(
        trials=non_running_trials_with_data, trials_are_running=False
    )
    if len(sorted_non_running_trials_with_data) > 0:
        return sorted_non_running_trials_with_data[0].index

    return None


def _sort_trials(
    trials: list[BaseTrial],
    trials_are_running: bool,
) -> list[BaseTrial]:
    """Sort a list of trials by (1) duration of trial, (2) number of arms in trial.

    Args:
        trials: The trials to sort.
        trials_are_running: Whether the trials are running or not, used to determine
            the trial duration for sorting

    Returns:
        The sorted trials.
    """
    default_time_run_started = datetime.now()
    twelve_hours_in_secs = 12 * 60 * 60
    sorted_trials_expecting_data = sorted(
        trials,
        key=lambda t: (
            # First sorting criterion: trial duration, if a trial's duration is within
            # 12 hours of another trial, we consider them to be a tie
            int(
                (
                    # if the trial is running, we set end time to now for sorting ease
                    (
                        _time_trial_completed_safe(trial=t).timestamp()
                        if not trials_are_running
                        else default_time_run_started.timestamp()
                    )
                    - _time_trial_started_safe(
                        trial=t, default_time_run_started=default_time_run_started
                    ).timestamp()
                )
                // twelve_hours_in_secs
            ),
            # In event of a tie, we want the trial with the most arms
            +len(t.arms_by_name),
        ),
        reverse=True,
    )
    return sorted_trials_expecting_data


def _time_trial_started_safe(
    trial: BatchTrial, default_time_run_started: datetime
) -> datetime:
    """Not all RUNNING trials have ``time_run_started`` defined.
    This function accepts, but penalizes those trials by using a
    default ``time_run_started``, which moves them to the end of
    the sort because they would be running a very short time.

    Args:
        trial: The trial to check.
        default_time_run_started: The time to use if `time_run_started` is not defined.
            Do not use ``default_time_run_started=datetime.now()`` as it will be
            slightly different for each trial.  Instead set ``val = datetime.now()``
            and then pass ``val`` as the ``default_time_run_started`` argument.
    """
    return (
        trial.time_run_started
        if trial.time_run_started is not None
        else default_time_run_started
    )


def _time_trial_completed_safe(trial: BatchTrial) -> datetime:
    """Not all COMPLETED trials have `time_completed` defined.
    This functions accepts, but penalizes those trials by
    choosing epoch 0 as the completed time,
    which moves them to the end of the sort because
    they would be running a very short time."""
    return (
        trial.time_completed
        if trial.time_completed is not None
        else datetime.fromtimestamp(0)
    )


# -------------------- MapMetric related utils. ---------------------


def extract_map_keys_from_opt_config(
    optimization_config: OptimizationConfig,
) -> set[str]:
    """Extract names of the map keys of all map metrics from the optimization config.

    Args:
        optimization_config: Optimization config.

    Returns:
        A set of map keys.
    """
    map_metrics = {
        name: metric
        for name, metric in optimization_config.metrics.items()
        if isinstance(metric, MapMetric)
    }
    map_key_names = {m.map_key_info.key for m in map_metrics.values()}
    return map_key_names


# -------------------- Context manager and decorator utils. ---------------------


# pyre-ignore[3]: Allowing `Any` in this case
def batch_trial_only(msg: str | None = None) -> Callable[..., Any]:
    """A decorator to verify that the value passed to the `trial`
    argument to `func` is a `BatchTrial`.
    """

    # pyre-ignore[2,3]: Allowing `Any` in this case
    def batch_trial_only_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def _batch_trial_only(*args: Any, **kwargs: Any) -> Any:  # pyre-ignore[3]
            if "trial" not in kwargs:
                raise AxError(
                    f"Expected a keyword argument `trial` to `{func.__name__}`."
                )
            if not isinstance(kwargs["trial"], BatchTrial):
                message = msg or (
                    f"Expected the argument `trial` to `{func.__name__}` "
                    f"to be a `BatchTrial`, but got {type(kwargs['trial'])}."
                )
                raise AxError(message)
            return func(*args, **kwargs)

        return _batch_trial_only

    return batch_trial_only_decorator
