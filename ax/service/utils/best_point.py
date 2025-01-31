#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import OrderedDict
from collections.abc import Iterable
from functools import reduce

from logging import Logger
from typing import Mapping

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
from ax.core.outcome_constraint import ObjectiveThreshold, OutcomeConstraint
from ax.core.trial import Trial
from ax.core.types import ComparisonOp, TModelPredictArm, TParameterization
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.cross_validation import (
    assess_model_fit,
    compute_diagnostics,
    cross_validate,
)
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.modelbridge_utils import (
    observed_pareto_frontier as observed_pareto,
    predicted_pareto_frontier as predicted_pareto,
)
from ax.modelbridge.registry import (
    get_model_from_generator_run,
    ModelRegistryBase,
    Models,
)
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.utils import (
    derelativize_optimization_config_with_raw_status_quo,
)
from ax.plot.pareto_utils import get_tensor_converter_model
from ax.utils.common.logger import get_logger
from numpy import nan
from pyre_extensions import assert_is_instance, none_throws
from torch import Tensor

logger: Logger = get_logger(__name__)


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
        Tuple of parameterization and a mapping from metric name to a tuple of
            the corresponding objective mean and SEM.
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
        if experiment.status_quo is not None:
            optimization_config = _derel_opt_config_wrapper(
                optimization_config=optimization_config,
                experiment=experiment,
            )
        else:
            logger.warning(
                "No status quo provided; relative constraints will be ignored."
            )

    # Only COMPLETED trials should be considered when identifying the best point
    completed_indices = {
        t.index for t in experiment.trials_by_status[TrialStatus.COMPLETED]
    }
    completed_df = dat.df[dat.df["trial_index"].isin(completed_indices)]

    feasible_df = _filter_feasible_rows(
        df=completed_df,
        optimization_config=optimization_config,
    )
    objective = optimization_config.objective
    best_row_helper = (
        _get_best_row_for_scalarized_objective
        if isinstance(objective, ScalarizedObjective)
        else _get_best_row_for_single_objective
    )
    # pyre-ignore Incompatible parameter type [6]
    best_row = best_row_helper(df=feasible_df, objective=objective)
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

    return best_trial_index, none_throws(best_arm).parameters, vals


def get_best_raw_objective_point(
    experiment: Experiment,
    optimization_config: OptimizationConfig | None = None,
    trial_indices: Iterable[int] | None = None,
) -> tuple[TParameterization, dict[str, tuple[float, float]]]:
    _, parameterization, vals = get_best_raw_objective_point_with_trial_index(
        experiment=experiment,
        optimization_config=optimization_config,
        trial_indices=trial_indices,
    )
    return parameterization, vals


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
    models_enum: type[ModelRegistryBase],
    optimization_config: OptimizationConfig | None = None,
    trial_indices: Iterable[int] | None = None,
) -> tuple[int, TParameterization, TModelPredictArm | None] | None:
    """Given an experiment, returns the best predicted parameterization and
    corresponding prediction.

    The best point & predictions are computed using the model from the
    (first) generator run of the latest trial. If the latest trial doesn't
    have a generator run, returns None. If the model from the latest trial
    is not TorchModelBridge or the model construction fails, this will
    return the best point & the predictions that were saved on the
    generator run (rather than re-computing them with latest data). If the
    model fit assessment returns bad fit for any of the metrics, this will
    fall back to returning the best point based on raw observations.

    Only some models return predictions. For instance GPEI does while Sobol does not.

    TModelPredictArm is of the form:
        ({metric_name: mean}, {metric_name_1: {metric_name_2: cov_1_2}})

    Args:
        experiment: Experiment, on which to identify best raw objective arm.
        models_enum: Registry of all models that may be in the experiment's
            generation strategy.
        optimization_config: Optimization config to use in place of the one stored
            on the experiment.
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
    for _, trial in sorted(experiment.trials.items(), key=lambda x: x[0], reverse=True):
        gr = None
        if isinstance(trial, Trial):
            gr = trial.generator_run
        elif isinstance(trial, BatchTrial):
            if len(trial.generator_run_structs) > 0:
                # In theory batch_trial can have >1 gr, grab the first
                gr = trial.generator_run_structs[0].generator_run

        if gr is not None:
            data = experiment.lookup_data(trial_indices=trial_indices)

            try:
                model = get_model_from_generator_run(
                    generator_run=gr,
                    experiment=experiment,
                    data=data,
                    models_enum=models_enum,
                )
            except ValueError:
                return _extract_best_arm_from_gr(gr=gr, trials=experiment.trials)

            # If model is not TorchModelBridge, just use the best arm from the
            # last good generator run
            if not isinstance(model, TorchModelBridge):
                return _extract_best_arm_from_gr(gr=gr, trials=experiment.trials)

            # Check to see if the model is worth using
            cv_results = cross_validate(model=model)
            diagnostics = compute_diagnostics(result=cv_results)
            assess_model_fit_results = assess_model_fit(diagnostics=diagnostics)
            objective_name = optimization_config.objective.metric.name
            # If model fit is bad use raw results
            if (
                objective_name
                in assess_model_fit_results.bad_fit_metrics_to_fisher_score
            ):
                logger.warning(
                    "Model fit is poor; falling back on raw data for best point."
                )

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

            res = model.model_best_point()
            if res is None:
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
            f"Encountered error while trying to identify the best point: {err}"
        )
        return None
    return (
        trial_index,
        parameterization,
        _raw_values_to_model_predict_arm(values),
    )


def get_best_by_raw_objective(
    experiment: Experiment,
    optimization_config: OptimizationConfig | None = None,
    trial_indices: Iterable[int] | None = None,
) -> tuple[TParameterization, TModelPredictArm | None] | None:
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
        Tuple of parameterization, and model predictions for it.
    """
    res = get_best_by_raw_objective_with_trial_index(
        experiment=experiment,
        optimization_config=optimization_config,
        trial_indices=trial_indices,
    )

    if res is None:
        return None

    _, parameterization, vals = res
    return parameterization, vals


def get_best_parameters_with_trial_index(
    experiment: Experiment,
    models_enum: type[ModelRegistryBase],
    optimization_config: OptimizationConfig | None = None,
    trial_indices: Iterable[int] | None = None,
) -> tuple[int, TParameterization, TModelPredictArm | None] | None:
    """Given an experiment, identifies the best arm.

    First attempts according to do so with models used in optimization and
    its corresponding predictions if available. Falls back to the best raw
    objective based on the data fetched from the experiment.

    TModelPredictArm is of the form:
        ({metric_name: mean}, {metric_name_1: {metric_name_2: cov_1_2}})

    Args:
        experiment: Experiment, on which to identify best raw objective arm.
        models_enum: Registry of all models that may be in the experiment's
            generation strategy.
        optimization_config: Optimization config to use in place of the one stored
            on the experiment.
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
            "get_best_parameters is deprecated for multi-objective optimization. "
            "This method will return an arbitrary point on the pareto frontier."
        )

    # Find latest trial which has a generator_run attached and get its predictions
    res = get_best_parameters_from_model_predictions_with_trial_index(
        experiment=experiment,
        models_enum=models_enum,
        optimization_config=optimization_config,
        trial_indices=trial_indices,
    )

    if res is not None:
        return res

    return get_best_by_raw_objective_with_trial_index(
        experiment=experiment, optimization_config=optimization_config
    )


def get_best_parameters(
    experiment: Experiment,
    models_enum: type[ModelRegistryBase],
    optimization_config: OptimizationConfig | None = None,
    trial_indices: Iterable[int] | None = None,
) -> tuple[TParameterization, TModelPredictArm | None] | None:
    """Given an experiment, identifies the best arm.

    First attempts according to do so with models used in optimization and
    its corresponding predictions if available. Falls back to the best raw
    objective based on the data fetched from the experiment.

    TModelPredictArm is of the form:
        ({metric_name: mean}, {metric_name_1: {metric_name_2: cov_1_2}})

    Args:
        experiment: Experiment, on which to identify best raw objective arm.
        models_enum: Registry of all models that may be in the experiment's
            generation strategy.
        optimization_config: Optimization config to use in place of the one stored
            on the experiment.
        trial_indices: Indices of trials for which to retrieve data. If None will
            retrieve data from all available trials.

    Returns:
        Tuple of parameterization and model predictions for it.
    """
    res = get_best_parameters_with_trial_index(
        experiment=experiment,
        models_enum=models_enum,
        optimization_config=optimization_config,
        trial_indices=trial_indices,
    )

    if res is None:
        return None

    _, parameterization, vals = res
    return parameterization, vals


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
        and isinstance(modelbridge, TorchModelBridge)
        and assert_is_instance(
            modelbridge,
            TorchModelBridge,
        ).is_moo_problem
    )
    if is_moo_modelbridge:
        generation_strategy._fit_current_model(data=None)
    else:
        modelbridge = Models.BOTORCH_MODULAR(
            experiment=experiment,
            data=assert_is_instance(
                experiment.lookup_data(trial_indices=trial_indices),
                Data,
            ),
        )
    modelbridge = assert_is_instance(
        modelbridge,
        TorchModelBridge,
    )

    # If objective thresholds are not specified in optimization config, extract
    # the inferred ones if possible or infer them anew if not.
    objective_thresholds_override = None
    if not moo_optimization_config.objective_thresholds:
        lgr = generation_strategy.last_generator_run
        if lgr and lgr.gen_metadata and "objective_thresholds" in lgr.gen_metadata:
            objective_thresholds_override = lgr.gen_metadata["objective_thresholds"]
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


def _get_best_row_for_scalarized_objective(
    df: pd.DataFrame,
    objective: ScalarizedObjective,
) -> pd.Series:
    df = df.copy()
    # First, add a weight column, setting 0.0 if the metric is not part
    # of the objective
    metric_to_weight = {
        m.name: objective.weights[i] for i, m in enumerate(objective.metrics)
    }
    df["weight"] = df["metric_name"].apply(lambda x: metric_to_weight.get(x) or 0.0)
    # Now, calculate the weighted linear combination via groupby,
    # filtering out NaN for missing data
    df["weighted_mean"] = df["mean"] * df["weight"]
    groupby_df = (
        df[["arm_name", "trial_index", "weighted_mean"]]
        .groupby(["arm_name", "trial_index"], as_index=False)
        .sum(min_count=1)
        .dropna()
    )
    if groupby_df.empty:
        raise ValueError("No data has been logged for scalarized objective.")
    return (
        groupby_df.loc[groupby_df["weighted_mean"].idxmin()]
        if objective.minimize
        else groupby_df.loc[groupby_df["weighted_mean"].idxmax()]
    )


def _get_best_row_for_single_objective(
    df: pd.DataFrame, objective: Objective
) -> pd.Series:
    objective_name = objective.metric.name
    objective_rows = df.loc[df["metric_name"] == objective_name]
    if objective_rows.empty:
        raise ValueError(f'No data has been logged for objective "{objective_name}".')
    return (
        objective_rows.loc[objective_rows["mean"].idxmin()]
        if objective.minimize
        else objective_rows.loc[objective_rows["mean"].idxmax()]
    )


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


def _filter_feasible_rows(
    df: pd.DataFrame,
    optimization_config: OptimizationConfig,
) -> pd.DataFrame:
    """Filter out arms that do not satisfy outcome constraints

    Looks at all arm data collected and removes rows corresponding to arms in
    which one or more of their associated metrics' 95% confidence interval
    falls outside of any outcome constraint's bounds (i.e. we are 95% sure the
    bound is not satisfied).
    """

    feasible = df.loc[_is_row_feasible(df=df, optimization_config=optimization_config)]
    if feasible.empty:
        raise ValueError(
            "No points satisfied all outcome constraints within 95 percent"
            "confidence interval."
        )

    return feasible


def _is_all_noiseless(df: pd.DataFrame, metric_name: str) -> bool:
    """Noiseless is defined as SEM = 0 or SEM = NaN on a given metric (usually
    the objective).
    """

    name_mask = df["metric_name"] == metric_name
    df_metric_arms_sems = df[name_mask]["sem"]

    return ((df_metric_arms_sems == 0) | df_metric_arms_sems == nan).all()


def _derel_opt_config_wrapper(
    optimization_config: OptimizationConfig,
    modelbridge: ModelBridge | None = None,
    experiment: Experiment | None = None,
    observations: list[Observation] | None = None,
) -> OptimizationConfig:
    """Derelativize optimization_config using raw status-quo values"""

    # If optimization_config is already derelativized, return a copy.
    if not any(oc.relative for oc in optimization_config.all_constraints):
        return optimization_config.clone()

    if modelbridge is None and experiment is None:
        raise ValueError(
            "Must specify ModelBridge or Experiment when calling "
            "`_derel_opt_config_wrapper`."
        )
    elif not modelbridge:
        modelbridge = get_tensor_converter_model(
            experiment=none_throws(experiment),
            data=none_throws(experiment).lookup_data(),
        )
    else:  # Both modelbridge and experiment specified.
        logger.warning(
            "ModelBridge and Experiment provided to `_derel_opt_config_wrapper`. "
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


def extract_Y_from_data(
    experiment: Experiment,
    metric_names: list[str],
    data: Data | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Converts the experiment observation data into a tensor.

    NOTE: This requires block design for observations. It will
    error out if any trial is missing data for any of the given
    metrics or if the data is missing the `trial_index`.

    Args:
        experiment: The experiment to extract the data from.
        metric_names: List of metric names to extract data for.
        data: An optional `Data` object to use instead of the
            experiment data. Note that the experiment must have
            a corresponding COMPLETED or EARLY_STOPPED trial for
            each `trial_index` in the `data`.

    Returns:
        A two-element Tuple containing a tensor of observed metrics and a
        tensor of trial_indices.
    """
    df = data.df if data is not None else experiment.lookup_data().df
    if len(df) == 0:
        y = torch.empty(0, len(metric_names), dtype=torch.double)
        indices = torch.empty(0, dtype=torch.long)
        return y, indices

    trials_to_use = []
    data_to_use = df[df["metric_name"].isin(metric_names)]

    for trial_idx, trial_data in data_to_use.groupby("trial_index"):
        trial = experiment.trials[trial_idx]
        if trial.status not in [TrialStatus.COMPLETED, TrialStatus.EARLY_STOPPED]:
            # Skip trials that are not completed or early stopped.
            continue
        trials_to_use.append(trial_idx)
        if trial_data[["metric_name", "arm_name"]].duplicated().any():
            raise UserInputError(
                "Trial data has more than one row per arm, metric pair. "
                f"Got\n\n{trial_data}\n\nfor trial {trial_idx}."
            )
        # We have already ensured that `trial_data` has no metrics not in
        # `metric_names` and that there are no duplicate metrics, so if
        # len(trial_data) < len(metric_names), the only possibility is that
        if len(trial_data) < len(metric_names):
            raise UserInputError(
                f"Trial {trial_idx} is missing data on metrics "
                f"{set(metric_names) - set(trial_data['metric_name'])}."
            )

    keeps = df["trial_index"].isin(trials_to_use)

    if not keeps.any():
        return torch.empty(0, len(metric_names), dtype=torch.double), torch.empty(
            0, dtype=torch.long
        )

    data_as_wide = pd.pivot_table(
        df[keeps],
        index=["trial_index", "arm_name"],
        columns="metric_name",
        values="mean",
    )[metric_names]

    means = torch.tensor(data_as_wide.to_numpy()).to(torch.double)
    trial_indices = torch.tensor(
        data_as_wide.reset_index()["trial_index"].to_numpy(), dtype=torch.long
    )
    return means, trial_indices


def _objective_threshold_from_nadir(
    experiment: Experiment,
    objective: Objective,
    optimization_config: MultiObjectiveOptimizationConfig | None = None,
) -> ObjectiveThreshold:
    """
    Find the worst value observed for each objective and create an ObjectiveThreshold
    with this as the bound.
    """

    logger.info(f"Inferring ObjectiveThreshold for {objective} using nadir point.")

    optimization_config = optimization_config or assert_is_instance(
        experiment.optimization_config,
        MultiObjectiveOptimizationConfig,
    )

    data_df = experiment.fetch_data().df

    mean = data_df[data_df["metric_name"] == objective.metric.name]["mean"]
    bound = max(mean) if objective.minimize else min(mean)
    op = ComparisonOp.LEQ if objective.minimize else ComparisonOp.GEQ

    return ObjectiveThreshold(
        metric=objective.metric, bound=bound, op=op, relative=False
    )


def fill_missing_thresholds_from_nadir(
    experiment: Experiment, optimization_config: OptimizationConfig
) -> list[ObjectiveThreshold]:
    r"""Get the objective thresholds from the optimization config and
    fill the missing thresholds based on the nadir point.

    Args:
        experiment: The experiment, whose data is used to calculate the nadir point.
        optimization_config: Optimization config to get the objective thresholds
            and the objective directions from.

    Returns:
        A list of objective thresholds, one for each objective in
        optimization config.
    """
    objectives = assert_is_instance(
        optimization_config.objective,
        MultiObjective,
    ).objectives
    optimization_config = assert_is_instance(
        optimization_config,
        MultiObjectiveOptimizationConfig,
    )
    provided_thresholds = {
        obj_t.metric.name: obj_t for obj_t in optimization_config.objective_thresholds
    }
    objective_thresholds = [
        (
            provided_thresholds[objective.metric.name]
            if objective.metric.name in provided_thresholds
            else _objective_threshold_from_nadir(
                experiment=experiment,
                objective=objective,
                optimization_config=optimization_config,
            )
        )
        for objective in objectives
    ]
    return objective_thresholds
