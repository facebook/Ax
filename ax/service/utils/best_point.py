#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from functools import reduce
from logging import Logger

import numpy as np

import pandas as pd
import torch
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.trial import Trial
from ax.core.types import ComparisonOp, TModelPredictArm, TParameterization
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from ax.modelbridge.cross_validation import (
    assess_model_fit,
    compute_diagnostics,
    cross_validate,
)
from ax.modelbridge.modelbridge_utils import (
    observed_pareto_frontier as observed_pareto,
    predicted_pareto_frontier as predicted_pareto,
)
from ax.modelbridge.registry import Generators
from ax.modelbridge.torch import TorchAdapter
from ax.modelbridge.transforms.derelativize import Derelativize
from ax.plot.pareto_utils import get_tensor_converter_model
from ax.utils.common.logger import get_logger
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from numpy import nan
from numpy.typing import NDArray
from pyre_extensions import assert_is_instance, none_throws

logger: Logger = get_logger(__name__)


def derelativize_opt_config(
    optimization_config: OptimizationConfig,
    experiment: Experiment,
    trial_indices: Iterable[int] | None = None,
) -> OptimizationConfig:
    tf = Derelativize(
        search_space=None, observations=None, config={"use_raw_status_quo": True}
    )
    optimization_config = tf.transform_optimization_config(
        optimization_config=optimization_config.clone(),
        modelbridge=get_tensor_converter_model(
            experiment=experiment,
            data=experiment.lookup_data(trial_indices=trial_indices),
        ),
        fixed_features=None,
    )
    return optimization_config


def get_best_raw_objective_point_with_trial_index(
    experiment: Experiment,
    optimization_config: OptimizationConfig | None = None,
    trial_indices: Iterable[int] | None = None,
) -> tuple[int, TParameterization, TModelPredictArm]:
    """Given an experiment, identifies the arm that had the best raw objective,
    based on the data fetched from the experiment.

    TModelPredictArm is of the form:
        ({metric_name: mean}, {metric_name_1: {metric_name_2: cov_1_2}})

    Note: This function will error with invalid inputs. If you would
    prefer for error logs rather than exceptions, use
    `get_best_by_raw_objective_with_trial_index`, which returns None if
    inputs are invalid.

    Args:
        experiment: Experiment, on which to identify best raw objective arm.
        optimization_config: Optimization config to use in place of the one stored
            on the experiment.
        trial_indices: Indices of trials for which to retrieve data. If None will
            retrieve data from all available trials.

    Returns:
        Tuple of trial index, parameterization, and model predictions for it.
    """
    optimization_config = optimization_config or experiment.optimization_config
    if optimization_config is None:
        raise UserInputError(
            "Cannot identify the best point without an optimization config, but no "
            "optimization config was provided on the experiment or as an argument."
        )
    if optimization_config.is_moo_problem:
        logger.warning(
            "get_best_raw_objective_point is deprecated for multi-objective "
            "optimization. This method will return an arbitrary point on the "
            "pareto frontier."
        )

    dat = experiment.lookup_data(trial_indices=trial_indices)
    if dat.df.empty:
        raise ValueError("Cannot identify best point if experiment contains no data.")
    if any(oc.relative for oc in optimization_config.all_constraints):
        optimization_config = derelativize_opt_config(
            optimization_config=optimization_config,
            experiment=experiment,
        )

    # Only COMPLETED trials should be considered when identifying the best point
    completed_indices = {
        t.index for t in experiment.trials_by_status[TrialStatus.COMPLETED]
    }
    if len(completed_indices) == 0:
        raise ValueError("Cannot identify best point if no trials are completed.")
    completed_df = dat.df[dat.df["trial_index"].isin(completed_indices)]

    is_feasible = _is_row_feasible(
        df=completed_df, optimization_config=optimization_config
    )
    if not is_feasible.any():
        raise ValueError(
            "No points satisfied all outcome constraints within 95 percent "
            "confidence interval."
        )
    feasible_df = completed_df.loc[is_feasible]

    is_in_design = feasible_df["arm_name"].apply(
        lambda arm_name: experiment.search_space.check_membership(
            parameterization=experiment.arms_by_name[arm_name].parameters
        )
    )

    if not is_in_design.any():
        raise ValueError("No feasible points are in the search space.")

    in_design_df = feasible_df.loc[is_in_design]
    value_by_arm_pull = get_trace_by_arm_pull_from_data(
        df=in_design_df,
        optimization_config=optimization_config,
        use_cumulative_best=False,
    )

    maximize = isinstance(optimization_config.objective, MultiObjective) or (
        not optimization_config.objective.minimize
    )
    best_row_idx = (
        value_by_arm_pull["value"].idxmax()
        if maximize
        else value_by_arm_pull["value"].idxmin()
    )
    best_row = value_by_arm_pull.loc[best_row_idx]

    best_arm = experiment.arms_by_name[best_row["arm_name"]]
    best_trial_index = int(best_row["trial_index"])
    objective_rows = dat.df.loc[
        (dat.df["arm_name"] == best_arm.name)
        & (dat.df["trial_index"] == best_trial_index)
    ]
    vals = {
        row["metric_name"]: (row["mean"], row["sem"])
        for _, row in objective_rows.iterrows()
    }
    predict_arm = _raw_values_to_model_predict_arm(values=vals)

    return best_trial_index, none_throws(best_arm).parameters, predict_arm


def _extract_best_arm_from_gr(
    gr: GeneratorRun,
    trials: Mapping[int, BaseTrial],
) -> tuple[int, TParameterization, TModelPredictArm | None] | None:
    """Extracts best arm predictions from a GeneratorRun, if available,
    and maps it to the trial index of the first trial that contains it.

    Args:
        gr: GeneratorRun, from which to extract best arm predictions.
        trials: Trials from the experiment, used to map the arm to a trial index.

    Returns:
        If the best arm or the best arm predictions are not available, returns
        None. Otherwise, returns a tuple of the trial index, parameterization,
        and model predictions for the best arm.
    """
    if gr.best_arm_predictions is None:
        return None

    best_arm, best_arm_predictions = gr.best_arm_predictions

    if best_arm is None:
        return None

    for trial_index, trial in trials.items():
        if best_arm in trial.arms:
            return trial_index, best_arm.parameters, best_arm_predictions


def _raw_values_to_model_predict_arm(
    values: dict[str, tuple[float, float]],
) -> TModelPredictArm:
    return (
        {k: v[0] for k, v in values.items()},  # v[0] is mean
        {k: {k: v[1] * v[1]} for k, v in values.items()},  # v[1] is sem
    )


def get_best_parameters_from_model_predictions_with_trial_index(
    experiment: Experiment,
    adapter: Adapter | None,
    optimization_config: OptimizationConfig | None = None,
    trial_indices: Iterable[int] | None = None,
) -> tuple[int, TParameterization, TModelPredictArm | None] | None:
    """Given an experiment, returns the best predicted parameterization and
    corresponding prediction.

    The best point & predictions are computed using the given ``Adapter``
    and its ``predict`` method (if implemented). If ``adapter`` is not a
    ``TorchAdapter``, the best point is extracted from the (first) generator run
    of the latest trial. If the latest trial doesn't have a generator run, returns
    None. If the model fit assessment returns bad fit for any of the metrics, this
    will fall back to returning the best point based on raw observations.

    TModelPredictArm is of the form:
        ({metric_name: mean}, {metric_name_1: {metric_name_2: cov_1_2}})

    Args:
        experiment: ``Experiment``, on which to identify best raw objective arm.
        adapter: The ``Adapter`` to use to get the model predictions. If None, the
            best point will be extracted from the generator run of the latest trial.
        optimization_config: Optional ``OptimizationConfig`` override, to use in place
            of the one stored on the experiment.
        trial_indices: Indices of trials for which to retrieve data. If None will
            retrieve data from all available trials.

    Returns:
        Tuple of trial index, parameterization, and model predictions for it.
    """
    optimization_config = optimization_config or experiment.optimization_config
    if optimization_config is None:
        raise ValueError(
            "Cannot identify the best point without an optimization config, but no "
            "optimization config was provided on the experiment or as an argument."
        )
    if optimization_config.is_moo_problem:
        logger.warning(
            "get_best_parameters_from_model_predictions is deprecated for "
            "multi-objective optimization configs. This method will return an "
            "arbitrary point on the pareto frontier."
        )
    gr = None
    data = experiment.lookup_data(trial_indices=trial_indices)
    # Extract the latest GR from the experiment.
    for _, trial in sorted(experiment.trials.items(), key=lambda x: x[0], reverse=True):
        if isinstance(trial, Trial):
            gr = trial.generator_run
        elif isinstance(trial, BatchTrial):
            if len(trial.generator_run_structs) > 0:
                # In theory batch_trial can have >1 gr, grab the first
                gr = trial.generator_run_structs[0].generator_run
        if gr is not None:
            break

    if not isinstance(adapter, TorchAdapter):
        if gr is None:
            return None
        return _extract_best_arm_from_gr(gr=gr, trials=experiment.trials)

    # Check to see if the adapter is worth using.
    cv_results = cross_validate(model=adapter)
    diagnostics = compute_diagnostics(result=cv_results)
    assess_model_fit_results = assess_model_fit(diagnostics=diagnostics)
    objective_name = optimization_config.objective.metric.name
    # If model fit is bad use raw results
    if objective_name in assess_model_fit_results.bad_fit_metrics_to_fisher_score:
        logger.warning("Model fit is poor; falling back on raw data for best point.")

        if not _is_all_noiseless(df=data.df, metric_name=objective_name):
            logger.warning(
                "Model fit is poor and data on objective metric "
                + f"{objective_name} is noisy; interpret best points "
                + "results carefully."
            )

        return get_best_by_raw_objective_with_trial_index(
            experiment=experiment,
            optimization_config=optimization_config,
            trial_indices=trial_indices,
        )

    res = adapter.model_best_point()
    if res is None:
        if gr is None:
            return None
        return _extract_best_arm_from_gr(gr=gr, trials=experiment.trials)

    best_arm, best_arm_predictions = res

    # Map the arm to the trial index of the first trial that contains it.
    for trial_index, trial in experiment.trials.items():
        if best_arm in trial.arms:
            return (
                trial_index,
                none_throws(best_arm).parameters,
                best_arm_predictions,
            )

    return None


def get_best_by_raw_objective_with_trial_index(
    experiment: Experiment,
    optimization_config: OptimizationConfig | None = None,
    trial_indices: Iterable[int] | None = None,
) -> tuple[int, TParameterization, TModelPredictArm] | None:
    """Given an experiment, identifies the arm that had the best raw objective,
    based on the data fetched from the experiment.

    TModelPredictArm is of the form:
        ({metric_name: mean}, {metric_name_1: {metric_name_2: cov_1_2}})

    This is a version of `get_best_raw_objective_point_with_trial_index` that
    logs errors rather than letting exceptions be raised.

    Args:
        experiment: Experiment, on which to identify best raw objective arm.
        optimization_config: Optimization config to use in place of the one stored
            on the experiment.
        trial_indices: Indices of trials for which to retrieve data. If None will
            retrieve data from all available trials.

    Returns:
        Tuple of trial index, parameterization, and model predictions for it.
    """
    try:
        result = get_best_raw_objective_point_with_trial_index(
            experiment=experiment,
            optimization_config=optimization_config,
            trial_indices=trial_indices,
        )
    except ValueError as err:
        logger.error(
            "Encountered error while trying to identify the best point: "
            f"'{err}'. Returning None."
        )
        return None
    return result


def get_pareto_optimal_parameters(
    experiment: Experiment,
    generation_strategy: GenerationStrategy,
    optimization_config: OptimizationConfig | None = None,
    trial_indices: Iterable[int] | None = None,
    use_model_predictions: bool = True,
) -> dict[int, tuple[TParameterization, TModelPredictArm]]:
    """Identifies the best parameterizations tried in the experiment so far,
    using model predictions if ``use_model_predictions`` is true and using
    observed values from the experiment otherwise. By default, uses model
    predictions to account for observation noise.

    NOTE: The format of this method's output is as follows:
    { trial_index --> (parameterization, (means, covariances) }, where means
    are a dictionary of form { metric_name --> metric_mean } and covariances
    are a nested dictionary of form
    { one_metric_name --> { another_metric_name: covariance } }.

    Args:
        experiment: Experiment, from which to find Pareto-optimal arms.
        generation_strategy: Generation strategy containing the modelbridge.
        optimization_config: Optimization config to use in place of the one stored
            on the experiment.
        trial_indices: Indices of trials for which to retrieve data. If None will
            retrieve data from all available trials.
        use_model_predictions: Whether to extract the Pareto frontier using
            model predictions or directly observed values. If ``True``,
            the metric means and covariances in this method's output will
            also be based on model predictions and may differ from the
            observed values.

    Returns:
        A mapping from trial index to the tuple of:
        - the parameterization of the arm in that trial,
        - two-item tuple of metric means dictionary and covariance matrix
            (model-predicted if ``use_model_predictions=True`` and observed
            otherwise).
    """
    optimization_config = optimization_config or experiment.optimization_config
    if optimization_config is None:
        raise ValueError(
            "Cannot identify the best point without an optimization config, but no "
            "optimization config was provided on the experiment or as an argument."
        )

    # Validate aspects of the experiment: that it is a MOO experiment and
    # that the current model can be used to produce the Pareto frontier.
    if not optimization_config.is_moo_problem:
        raise UnsupportedError(
            "Please use `get_best_parameters` for single-objective problems."
        )

    moo_optimization_config = assert_is_instance(
        optimization_config,
        MultiObjectiveOptimizationConfig,
    )

    # Use existing modelbridge if it supports MOO otherwise create a new MOO modelbridge
    # to use for Pareto frontier extraction.
    modelbridge = generation_strategy.model
    is_moo_modelbridge = (
        modelbridge
        and isinstance(modelbridge, TorchAdapter)
        and assert_is_instance(
            modelbridge,
            TorchAdapter,
        ).is_moo_problem
    )
    if is_moo_modelbridge:
        generation_strategy._curr._fit(experiment=experiment)
    else:
        modelbridge = Generators.BOTORCH_MODULAR(
            experiment=experiment,
            data=assert_is_instance(
                experiment.lookup_data(trial_indices=trial_indices),
                Data,
            ),
        )
    modelbridge = assert_is_instance(modelbridge, TorchAdapter)

    objective_thresholds_override = None
    # If objective thresholds are not specified in optimization config, infer them.
    if not moo_optimization_config.objective_thresholds:
        objective_thresholds_override = modelbridge.infer_objective_thresholds(
            search_space=experiment.search_space,
            optimization_config=optimization_config,
            fixed_features=None,
        )
        logger.info(
            f"Using inferred objective thresholds: {objective_thresholds_override}, "
            "as objective thresholds were not specified as part of the optimization "
            "configuration on the experiment."
        )

    pareto_util = predicted_pareto if use_model_predictions else observed_pareto
    pareto_optimal_observations = pareto_util(
        modelbridge=modelbridge,
        optimization_config=moo_optimization_config,
        objective_thresholds=objective_thresholds_override,
    )

    # Insert observations into OrderedDict in order of descending individual
    # hypervolume, formated as
    # { trial_index --> (parameterization, (means, covariances) }
    res: dict[int, tuple[TParameterization, TModelPredictArm]] = OrderedDict()
    for obs in pareto_optimal_observations:
        res[int(none_throws(obs.features.trial_index))] = (
            obs.features.parameters,
            (obs.data.means_dict, obs.data.covariance_matrix),
        )

    return res


def _is_row_feasible(
    df: pd.DataFrame,
    optimization_config: OptimizationConfig,
) -> pd.Series:
    """Return a series of boolean values indicating whether arms satisfy outcome
    constraints or not.

    Looks at all arm data collected and returns False for rows corresponding to arms in
    which one or more of their associated metrics' 95% confidence interval
    falls outside of any outcome constraint's bounds (i.e. we are 95% sure the
    bound is not satisfied), else True.
    """
    if len(optimization_config.all_constraints) < 1:
        return pd.Series([True] * len(df), index=df.index)

    name = df["metric_name"]

    # When SEM is NaN we should treat it as if it were 0
    sems = none_throws(df["sem"].fillna(0))

    # Bounds computed for 95% confidence interval on Normal distribution
    lower_bound = df["mean"] - sems * 1.96
    upper_bound = df["mean"] + sems * 1.96

    # Nested function from OC -> Mask for consumption in later map/reduce from
    # [OC] -> Mask. Constraint relativity is handled inside so long as relative bounds
    # are set in surrounding closure (which will occur in proper experiment setup).
    # pyre-fixme[53]: Captured variable `lower_bound` is not annotated.
    # pyre-fixme[53]: Captured variable `name` is not annotated.
    # pyre-fixme[53]: Captured variable `rel_lower_bound` is not annotated.
    # pyre-fixme[53]: Captured variable `rel_upper_bound` is not annotated.
    # pyre-fixme[53]: Captured variable `upper_bound` is not annotated.
    def oc_mask(oc: OutcomeConstraint) -> pd.Series:
        name_match_mask = name == oc.metric.name
        if oc.relative:
            logger.warning(
                f"Ignoring relative constraint {oc}. Derelativize "
                "OptimizationConfig before passing to `_is_row_feasible`."
            )
            return pd.Series(True, index=df.index)
        # Return True if metrics are different, or whether the confidence
        # interval is entirely not within the bound
        if oc.op == ComparisonOp.GEQ:
            return ~name_match_mask | (upper_bound >= float(oc.bound))
        else:
            return ~name_match_mask | (lower_bound <= float(oc.bound))

    mask = reduce(
        lambda left, right: left & right,
        map(oc_mask, optimization_config.all_constraints),
    )
    # Mark all rows corresponding to infeasible arms as infeasible.
    bad_arm_names = df[~mask]["arm_name"].tolist()
    return assert_is_instance(
        df["arm_name"].apply(lambda x: x not in bad_arm_names),
        pd.Series,
    )


def _is_all_noiseless(df: pd.DataFrame, metric_name: str) -> bool:
    """Noiseless is defined as SEM = 0 or SEM = NaN on a given metric (usually
    the objective).
    """

    name_mask = df["metric_name"] == metric_name
    df_metric_arms_sems = df[name_mask]["sem"]

    return ((df_metric_arms_sems == 0) | df_metric_arms_sems == nan).all()


def get_values_of_outcomes_single_or_scalarized_objective(
    df_wide: pd.DataFrame, objective: Objective
) -> NDArray:
    """
    Return a list with one entry for each row in `df_wide` according to the
    objective `objective` and whether the outcomes are feasible.

    Whether higher or lower is better depends on `objective.minimize` (no
    absolute values are taken here).

    The entry for any infeasible value will be infinity if the objective is to
    minimize and negative infinity if the objective is to maximize.

    Example:
    >>> objective = Objective(metric=Metric(name="m1"), minimize=True)
    >>> df_wide = pd.DataFrame.from_records(
    ...     [
    ...         {"m1": 2.0, "feasible": True},
    ...         {"m1": 1.0, "feasible": False},
    ...     ]
    ... )
    >>> get_value_of_outcomes_single_or_scalarized_objective(
    ...     df_wide=df_wide, objective=objective
    ... )
    np.array([2.0, inf])
    """
    if isinstance(objective, MultiObjective):
        raise ValueError(
            "MultiObjective is not supported. Use "
            "`get_hypervolume_of_outcomes_multi_objective`."
        )
    if isinstance(objective, ScalarizedObjective):
        value = df_wide[objective.metric_names].dot(objective.weights).to_numpy()
    else:
        value = df_wide[objective.metric.name].to_numpy()
    value = value.astype(np.float64)
    infeasible_idx = np.where(~df_wide["feasible"])[0]
    value[infeasible_idx] = float("inf") if objective.minimize else float("-inf")
    return value


def _compute_hv_trace(
    ref_point: torch.Tensor,
    metrics_tensor: torch.Tensor,
    is_feasible_array: NDArray,
    use_cumulative_hv: bool,
) -> list[float]:
    # Compute hypervolume of feasible points
    hvs = []
    ref_point = ref_point

    if use_cumulative_hv:
        partitioning = DominatedPartitioning(ref_point=ref_point)
        cumulative_hv = 0.0
        for i, is_feasible in enumerate(is_feasible_array):
            if not is_feasible:
                hvs.append(cumulative_hv)
            else:
                Y = metrics_tensor[[i], :]
                partitioning.update(Y=Y)
                cumulative_hv = partitioning.compute_hypervolume().item()
                hvs.append(cumulative_hv)
        return hvs

    for i, is_feasible in enumerate(is_feasible_array):
        if not is_feasible:
            hvs.append(0.0)
        else:
            Y = metrics_tensor[[i], :]
            partitioning = DominatedPartitioning(ref_point=ref_point, Y=Y)
            hvs.append(partitioning.compute_hypervolume().item())
    return hvs


# NOTE: we are ignoring `MultiObjective` weights here. these
# should likely be deprecated or removed
def get_hypervolume_trace_of_outcomes_multi_objective(
    df_wide: pd.DataFrame,
    optimization_config: MultiObjectiveOptimizationConfig,
    use_cumulative_hv: bool = True,
) -> list[float]:
    """
    Get hypervolume of the outcomes represented in `df_wide`.

    Args:
        df_wide: Dataframe with columns ["feasible"] + relevant
            metrics. This can come from reshaping the data that comes from `Data.df`.
        optimization_config: A multi-objective optimization config with a
            `MultiObjective` (not a `ScalarizedObjective`).
        use_cumulative_hv: If True, the hypervolume returned is the cumulative
            hypervolume of the points in each row. Otherwise, this is the
            hypervolume of each point.

    Returns:
        A list of hypervolumes, one for each row in `df_wide`.

    Example:
    >>> optimization_config = MultiObjectiveOptimizationConfig(
    ...     objective=MultiObjective(
    ...         objectives=[
    ...             Objective(metric=Metric(name="m1"), minimize=False),
    ...             Objective(metric=Metric(name="m2"), minimize=False),
    ...         ]
    ...     ),
    ... )
    >>> # Objective threshols will be inferred to be zero
    >>> df_wide = pd.DataFrame.from_records(
    ...     [
    ...         {"m1": 0.0, "m2": 0.0, "feasible": True},
    ...         {"m1": 1.0, "m2": 2.0, "feasible": True},
    ...         {"m1": 2.0, "m2": 1.0, "feasible": False},
    ...         {"m1": 3.0, "m2": 3.0, "feasible": True},
    ...     ]
    ... )
    >>> get_hypervolume_trace_of_outcomes_multi_objective(
    ...     df_wide=df_wide,
    ...     optimization_config=optimization_config,
    ...     use_cumulative_hv=True,
    ... )
    [0.0, 2.0, 2.0, 9.0]
    >>>
    >>> get_hypervolume_trace_of_outcomes_multi_objective(
    ...     df_wide=df_wide,
    ...     optimization_config=optimization_config,
    ...     use_cumulative_hv=False,
    ... )
    [0.0, 2.0, 0.0, 9.0]
    """
    objective = assert_is_instance(optimization_config.objective, MultiObjective)
    for obj in objective.objectives:
        if obj.minimize:
            df_wide[obj.metric.name] *= -1

    objective_thresholds = []
    objective_thresholds_dict = {
        threshold.metric.name: threshold
        for threshold in optimization_config.objective_thresholds
    }
    for obj in objective.objectives:
        metric_name = obj.metric.name
        if metric_name in objective_thresholds_dict:
            threshold = objective_thresholds_dict[metric_name]
            if threshold.relative:
                raise ValueError(
                    "Relative objective thresholds are not supported. Please "
                    "`Derelativize` the optimization config, or use "
                    "`get_trace`."
                )
            bound = threshold.bound
        else:
            metric_vals = df_wide[metric_name]
            bound = metric_vals.max() if obj.minimize else metric_vals.min()

        objective_thresholds.append(-bound if obj.minimize else bound)

    objective_thresholds = torch.tensor(objective_thresholds, dtype=torch.double)

    metrics_tensor = torch.from_numpy(df_wide[objective.metric_names].to_numpy())
    return _compute_hv_trace(
        ref_point=objective_thresholds,
        metrics_tensor=metrics_tensor,
        is_feasible_array=df_wide["feasible"].to_numpy(),
        use_cumulative_hv=use_cumulative_hv,
    )


def get_trace_by_arm_pull_from_data(
    df: pd.DataFrame,
    optimization_config: OptimizationConfig,
    use_cumulative_best: bool = True,
) -> pd.DataFrame:
    """
    Get a trace of the objective value or hypervolume of outcomes.

    An "arm pull" is the combination of a trial (index) and an arm. This
    function returns a single value for each arm pull, even if there are
    multiple arms per trial or if an arm is repeated in multiple trials.

    Args:
        df: Data in the format returned by ``Data.df``, with a separate row for
            each trial index-arm name-metric.
        optimization_config: ``OptimizationConfig`` to use to get the trace. Must
            not be in relative form.
        use_cumulative_best: If True, the trace will be the cumulative best
            objective. Otherwise, the trace will be the value of each point.

    Return:
        A DataFrame containing columns 'trial_index', 'arm_name', and "value",
        where "value" is the value of the outcomes attained.
    """
    if any(oc.relative for oc in optimization_config.all_constraints):
        raise ValueError(
            "Relativized optimization config not supported. Please "
            "`Derelativize` the optimization config, or use `get_trace`."
        )

    empty_result = pd.DataFrame(columns=["trial_index", "arm_name", "value"])

    if len(df) == 0:
        return empty_result

    # reshape data to wide, using only the metrics in the optimization config
    metrics = list(optimization_config.metrics.keys())

    df["row_feasible"] = _is_row_feasible(
        df=df, optimization_config=optimization_config
    )

    # Transform to a DataFrame with columns ["trial_index", "arm_name"] +
    # relevant metric names, and values being means.
    df_wide = (
        df[df["metric_name"].isin(metrics)]
        .set_index(["trial_index", "arm_name", "metric_name"])["mean"]
        .unstack(level="metric_name")
    )
    missing_metrics = [
        m for m in metrics if m not in df_wide.columns or df_wide[m].isnull().any()
    ]
    if len(missing_metrics) > 0:
        raise ValueError(
            "Some metrics are not present for all trials and arms. The "
            f"following are missing: {missing_metrics}."
        )
    if len(df_wide) == 0:
        return empty_result
    df_wide["feasible"] = df.groupby(["trial_index", "arm_name"])["row_feasible"].all()
    df_wide.reset_index(inplace=True)

    # MOO and *not* ScalarizedObjective
    if isinstance(optimization_config.objective, MultiObjective):
        optimization_config = assert_is_instance(
            optimization_config, MultiObjectiveOptimizationConfig
        )
        df_wide["value"] = get_hypervolume_trace_of_outcomes_multi_objective(
            df_wide=df_wide,
            optimization_config=optimization_config,
            use_cumulative_hv=use_cumulative_best,
        )
        return df_wide[["trial_index", "arm_name", "value"]]
    df_wide["value"] = get_values_of_outcomes_single_or_scalarized_objective(
        df_wide=df_wide, objective=optimization_config.objective
    )
    if df_wide["feasible"].any() and use_cumulative_best:
        min_or_max = (
            np.minimum if optimization_config.objective.minimize else np.maximum
        )
        df_wide["value"] = min_or_max.accumulate(df_wide["value"])
    return df_wide[["trial_index", "arm_name", "value"]]


def get_trace(
    experiment: Experiment,
    optimization_config: OptimizationConfig | None = None,
) -> list[float]:
    """Compute the optimization trace at each iteration.

    Given an experiment and an optimization config, compute the performance
    at each iteration. For multi-objective, the performance is computed as
    the hypervolume. For single objective, the performance is computed as
    the best observed objective value.

    Infeasible points (that violate constraints) do not contribute to
    improvements in the optimization trace. If the first trial(s) are infeasible,
    the trace can start at inf or -inf.

    An iteration here refers to a completed or early-stopped (batch) trial.
    There will be one performance metric in the trace for each iteration.

    Args:
        experiment: The experiment to get the trace for.
        optimization_config: Optimization config to use in place of the one
            stored on the experiment.

    Returns:
        A list of performance values at each iteration.
    """
    optimization_config = optimization_config or none_throws(
        experiment.optimization_config
    )
    df = experiment.lookup_data().df
    if len(df) == 0:
        return []
    # Get the names of the metrics in optimization config.
    metric_names = set(optimization_config.objective.metric_names)
    for cons in optimization_config.outcome_constraints:
        metric_names.update({cons.metric.name})
    metric_names = list(metric_names)

    # Don't compute results for status quo data (for compatibility with legacy behavior)
    trial_is_completed = df["trial_index"].map(
        {
            i: t.status in (TrialStatus.COMPLETED, TrialStatus.EARLY_STOPPED)
            for i, t in experiment.trials.items()
        }
    )
    idx = df["metric_name"].isin(metric_names) & trial_is_completed
    # Don't include status quo (for compatibility with legacy behavior)
    if (status_quo := experiment.status_quo) is not None:
        idx &= df["arm_name"] != status_quo.name
    df = df.loc[idx, :]
    if len(df) == 0:
        return []

    # Derelativize the optimization config.
    optimization_config = derelativize_opt_config(
        optimization_config=optimization_config,
        experiment=experiment,
    )

    # Get a value for each trial_index + arm
    value_by_arm_pull = get_trace_by_arm_pull_from_data(
        df=df,
        optimization_config=optimization_config,
        use_cumulative_best=True,
    )
    # Aggregate to trial level
    objective = optimization_config.objective
    maximize = isinstance(objective, MultiObjective) or not objective.minimize
    trial_grouped = value_by_arm_pull.groupby("trial_index")["value"]
    if maximize:
        value_by_trial = trial_grouped.max()
        cumulative_value = np.maximum.accumulate(value_by_trial)
    else:
        value_by_trial = trial_grouped.min()
        cumulative_value = np.minimum.accumulate(value_by_trial)

    return cumulative_value.tolist()
