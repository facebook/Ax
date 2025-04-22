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
from ax.core.observation import Observation
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
from ax.modelbridge.transforms.utils import (
    derelativize_optimization_config_with_raw_status_quo,
)
from ax.plot.pareto_utils import get_tensor_converter_model
from ax.utils.common.logger import get_logger
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from numpy import nan
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
) -> tuple[int, TParameterization, dict[str, tuple[float, float]]]:
    """Given an experiment, identifies the arm that had the best raw objective,
    based on the data fetched from the experiment.

    Args:
        experiment: Experiment, on which to identify best raw objective arm.
        optimization_config: Optimization config to use in place of the one stored
            on the experiment.
        trial_indices: Indices of trials for which to retrieve data. If None will
            retrieve data from all available trials.

    Returns:
        Tuple of (trial_index, parameterization, mapping from metric name to a
            tuple of the corresponding objective mean and SEM).
    """
    optimization_config = optimization_config or experiment.optimization_config
    if optimization_config is None:
        raise UserInputError(
            "Cannot identify the best point without an optimization config, but no "
            "optimization config was provided on the experiment or as an argument."
        )
    if optimization_config.is_moo_problem:
        raise ValueError(
            "get_best_raw_objective_point is not supported for multi-objective "
            "optimization."
        )

    dat = experiment.lookup_data(trial_indices=trial_indices)
    if dat.df.empty:
        raise ValueError("Cannot identify best point if experiment contains no data.")

    # Only COMPLETED trials should be considered when identifying the best point
    # TODO: dedup with logic from get_trace
    completed_indices = {
        t.index for t in experiment.trials_by_status[TrialStatus.COMPLETED]
    }
    if len(completed_indices) == 0:
        raise ValueError("Cannot identify best point if no trials are completed.")
    optimization_config = derelativize_opt_config(
        optimization_config=optimization_config,
        experiment=experiment,
        trial_indices=trial_indices,
    )
    completed_df = dat.df[dat.df["trial_index"].isin(completed_indices)]
    is_in_design = completed_df["arm_name"].apply(
        lambda arm_name: experiment.search_space.check_membership(
            parameterization=experiment.arms_by_name[arm_name].parameters
        )
    )

    if not is_in_design.any():
        raise ValueError("No feasible points are in the search space.")

    in_design_df = completed_df.loc[is_in_design]

    value_by_arm_pull = get_trace_by_arm_pull_from_data(
        df=in_design_df,
        optimization_config=optimization_config,
        use_cumulative_best=False,
    )
    value = value_by_arm_pull["value"]
    best_row_idx = (
        value.idxmin() if optimization_config.objective.minimize else value.idxmax()
    )

    best_arm_name = value_by_arm_pull["arm_name"].iloc[best_row_idx]
    best_trial_index = value_by_arm_pull["trial_index"].iloc[best_row_idx]

    objective_rows = completed_df.loc[
        (completed_df["arm_name"] == best_arm_name)
        & (completed_df["trial_index"] == best_trial_index)
    ]
    vals = {
        row["metric_name"]: (row["mean"], row["sem"])
        for _, row in objective_rows.iterrows()
    }
    best_arm = experiment.arms_by_name[best_arm_name]

    return best_trial_index, none_throws(best_arm).parameters, vals


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
) -> tuple[int, TParameterization, TModelPredictArm | None] | None:
    """Given an experiment, identifies the arm that had the best raw objective,
    based on the data fetched from the experiment.

    TModelPredictArm is of the form:
        ({metric_name: mean}, {metric_name_1: {metric_name_2: cov_1_2}})

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
        (
            trial_index,
            parameterization,
            values,
        ) = get_best_raw_objective_point_with_trial_index(
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
    return (
        trial_index,
        parameterization,
        _raw_values_to_model_predict_arm(values),
    )


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


# TODO: see if we can dedup this with derelativizse_opt_config
# Which is better because it doesn't ignore SQ values!
# This is only used in report_utils, so it can move there
def _derel_opt_config_wrapper(
    optimization_config: OptimizationConfig,
    modelbridge: Adapter | None = None,
    experiment: Experiment | None = None,
    observations: list[Observation] | None = None,
) -> OptimizationConfig:
    """Derelativize optimization_config using raw status-quo values"""

    # If optimization_config is already derelativized, return a copy.
    if not any(oc.relative for oc in optimization_config.all_constraints):
        return optimization_config.clone()

    if modelbridge is None and experiment is None:
        raise ValueError(
            "Must specify Adapter or Experiment when calling "
            "`_derel_opt_config_wrapper`."
        )
    elif not modelbridge:
        modelbridge = get_tensor_converter_model(
            experiment=none_throws(experiment),
            data=none_throws(experiment).lookup_data(),
        )
    else:  # Both modelbridge and experiment specified.
        logger.warning(
            "Adapter and Experiment provided to `_derel_opt_config_wrapper`. "
            "Ignoring the latter."
        )
    if not modelbridge.status_quo:
        raise ValueError(
            "`modelbridge` must have status quo if specified. If `modelbridge` is "
            "unspecified, `experiment` must have a status quo."
        )
    observations = observations or modelbridge.get_training_data()
    return derelativize_optimization_config_with_raw_status_quo(
        optimization_config=optimization_config,
        modelbridge=modelbridge,
        observations=observations,
    )


def get_value_of_outcomes_single_or_scalarized_objective(
    is_feasible: bool, metrics: Mapping[str, float], objective: Objective
) -> float:
    """Get the value of the outcomes and whether they are feasible."""
    if isinstance(objective, MultiObjective):
        raise ValueError(
            "MultiObjective is not supported. Use "
            "`get_hypervolume_of_outcomes_multi_objective`."
        )

    sign = -1 if objective.minimize else 1
    if not is_feasible:
        worst_value = float("-inf") * sign
        return worst_value

    if isinstance(objective, ScalarizedObjective):
        value = sum(
            (
                metrics[metric.name] * weight
                for metric, weight in zip(objective.metrics, objective.weights)
            )
        )
        return value
    # `Objective` case
    return metrics[objective.metric.name]


# NOTE: we are ignoring weights here. these should likely be deprecated or removed
def get_hypervolume_trace_of_outcomes_multi_objective(
    df_wide: pd.DataFrame,
    optimization_config: MultiObjectiveOptimizationConfig,
    use_cumulative_hv: bool = True,
) -> list[float]:
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
                    "Delrelativize the optimization config, or use `get_trace`."
                )
            bound = threshold.bound
        else:
            metric_vals = df_wide[metric_name]
            bound = metric_vals.max() if obj.minimize else metric_vals.min()

        objective_thresholds.append(bound)

    objective_thresholds = torch.tensor(objective_thresholds, dtype=torch.double)

    # Compute hypervolume of feasible points
    hvs = []
    ref_point = objective_thresholds
    if use_cumulative_hv:
        partitioning = DominatedPartitioning(ref_point=ref_point)
        cumulative_hv = 0.0
        for i, (_, row) in enumerate(
            df_wide[optimization_config.objective.metric_names].iterrows()
        ):
            if not df_wide["feasible"].iloc[i]:
                hvs.append(cumulative_hv)
            else:
                Y = torch.tensor(row.to_numpy()).unsqueeze(0)
                partitioning.update(Y=Y)
                cumulative_hv = partitioning.compute_hypervolume().item()
                hvs.append(cumulative_hv)
        return hvs

    for i, (_, row) in enumerate(
        df_wide[optimization_config.objective.metric_names].iterrows()
    ):
        if not df_wide.loc[i, "feasible"]:
            hvs.append(0.0)
        else:
            Y = torch.tensor(row.to_numpy()).unsqueeze(0)
            partitioning = DominatedPartitioning(ref_point=ref_point, Y=Y)
            hvs.append(partitioning.compute_hypervolume().item())
    return hvs


def get_trace_by_arm_pull_from_data(
    df: pd.DataFrame,
    optimization_config: OptimizationConfig,
    use_cumulative_best: bool = True,
) -> pd.DataFrame:
    """Get a trace of the objective value or hypervolume of outcomes.

    Args:
        df: Data in the format returned by ``Data.df``.
        optimization_config: OptimizationConfig to use to get the trace. Must
            not be in relative form.
        use_cumulative_best: If True, the trace will be the cumulative best
            objective. Otherwise, the trace will be the value of each point.

    Return:
        A DataFrame containing columns 'trial_index', 'arm_name', and "value",
        where "value" is the value of the outcomes attained.
    """
    if any((oc.relative for oc in optimization_config.all_constraints)):
        raise ValueError(
            "Relativized optimization config not supported. Please "
            "Delrelativize the optimization config, or use `get_trace`."
        )

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

    df_wide["value"] = df_wide.apply(
        lambda row: get_value_of_outcomes_single_or_scalarized_objective(
            is_feasible=row.loc["feasible"],
            metrics={
                name: val
                for name, val in row.items()
                if name in optimization_config.metrics
            },
            objective=optimization_config.objective,
        ),
        axis=1,
    )

    if use_cumulative_best:
        min_or_max = (
            np.minimum if optimization_config.objective.minimize else np.maximum
        )
        df_wide["value"] = min_or_max.accumulate(df_wide["value"])
    return df_wide[["trial_index", "arm_name", "value"]]


def get_trace_by_trial_index_from_data(
    df: pd.DataFrame,
    optimization_config: OptimizationConfig,
    use_cumulative_best: bool = True,
) -> dict[int, float]:
    value_by_arm_pull = get_trace_by_arm_pull_from_data(
        df=df,
        optimization_config=optimization_config,
        use_cumulative_best=use_cumulative_best,
    )
    objective = optimization_config.objective
    maximize = isinstance(objective, MultiObjective) or not objective.minimize
    trial_grouped = value_by_arm_pull.groupby("trial_index")["value"]
    value_by_trial = trial_grouped.max() if maximize else trial_grouped.min()

    if use_cumulative_best:
        accumulator = np.maximum if maximize else np.minimum
        trial_indices = value_by_trial.index.get_level_values("trial_index")
        return dict(zip(trial_indices, accumulator.accumulate(value_by_trial)))
    return value_by_trial.to_dict()


def get_trace(
    experiment: Experiment,
    optimization_config: OptimizationConfig | None = None,
    trial_indices: Iterable[int] | None = None,
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
    df = experiment.lookup_data(trial_indices=trial_indices).df
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
        trial_indices=trial_indices,
    )

    trial_to_value = get_trace_by_trial_index_from_data(
        df=df, optimization_config=optimization_config
    )
    return list(trial_to_value.values())
