#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import reduce

from logging import Logger
from typing import Dict, Iterable, Optional, Tuple, Type

import pandas as pd
from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.objective import Objective, ScalarizedObjective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.trial import Trial
from ax.core.types import ComparisonOp, TModelPredictArm, TParameterization
from ax.exceptions.core import UnsupportedError, UserInputError
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
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none
from ax.utils.stats.statstools import relativize_data
from numpy import NaN

logger: Logger = get_logger(__name__)


def get_best_raw_objective_point_with_trial_index(
    experiment: Experiment,
    optimization_config: Optional[OptimizationConfig] = None,
    trial_indices: Optional[Iterable[int]] = None,
) -> Tuple[int, TParameterization, Dict[str, Tuple[float, float]]]:
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
    objective = optimization_config.objective
    feasible_df = _filter_feasible_rows(
        df=dat.df,
        optimization_config=optimization_config,
        status_quo=experiment.status_quo,
    )
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

    return best_trial_index, not_none(best_arm).parameters, vals


def get_best_raw_objective_point(
    experiment: Experiment,
    optimization_config: Optional[OptimizationConfig] = None,
    trial_indices: Optional[Iterable[int]] = None,
) -> Tuple[TParameterization, Dict[str, Tuple[float, float]]]:

    _, parameterization, vals = get_best_raw_objective_point_with_trial_index(
        experiment=experiment,
        optimization_config=optimization_config,
        trial_indices=trial_indices,
    )
    return parameterization, vals


def _gr_to_prediction_with_trial_index(
    idx: int, gr: GeneratorRun
) -> Optional[Tuple[int, TParameterization, Optional[TModelPredictArm]]]:
    if gr.best_arm_predictions is None:
        return None

    best_arm, best_arm_predictions = gr.best_arm_predictions

    if best_arm is None:
        return None

    return idx, best_arm.parameters, best_arm_predictions


def _raw_values_to_model_predict_arm(
    values: Dict[str, Tuple[float, float]]
) -> TModelPredictArm:
    return (
        {k: v[0] for k, v in values.items()},  # v[0] is mean
        {k: {k: v[1] * v[1]} for k, v in values.items()},  # v[1] is sem
    )


def get_best_parameters_from_model_predictions_with_trial_index(
    experiment: Experiment,
    models_enum: Type[ModelRegistryBase],
    optimization_config: Optional[OptimizationConfig] = None,
    trial_indices: Optional[Iterable[int]] = None,
) -> Optional[Tuple[int, TParameterization, Optional[TModelPredictArm]]]:
    """Given an experiment, returns the best predicted parameterization and
    corresponding prediction based on the most recent Trial with predictions. If no
    trials have predictions returns None.

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
    for idx, trial in sorted(
        experiment.trials.items(), key=lambda x: x[0], reverse=True
    ):
        gr = None
        if isinstance(trial, Trial):
            gr = trial.generator_run
        elif isinstance(trial, BatchTrial):
            if len(trial.generator_run_structs) > 0:
                # In theory batch_trial can have >1 gr, grab the first
                gr = trial.generator_run_structs[0].generator_run

        if gr is not None and gr.best_arm_predictions is not None:  # pragma: no cover
            data = experiment.lookup_data(trial_indices=trial_indices)

            try:
                model = get_model_from_generator_run(
                    generator_run=gr,
                    experiment=experiment,
                    data=data,
                    models_enum=models_enum,
                )
            except ValueError:
                return _gr_to_prediction_with_trial_index(idx, gr)

            # If model is not TorchModelBridge, just use the best arm frmo the
            # last good generator run
            if not isinstance(model, TorchModelBridge):
                return _gr_to_prediction_with_trial_index(idx, gr)

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
                return _gr_to_prediction_with_trial_index(idx, gr)

            best_arm, best_arm_predictions = res

            return idx, not_none(best_arm).parameters, best_arm_predictions

    return None


def get_best_parameters_from_model_predictions(
    experiment: Experiment,
    models_enum: Type[ModelRegistryBase],
    trial_indices: Optional[Iterable[int]] = None,
) -> Optional[Tuple[TParameterization, Optional[TModelPredictArm]]]:
    """Given an experiment, returns the best predicted parameterization and
    corresponding prediction based on the most recent Trial with predictions. If no
    trials have predictions returns None.

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
        Tuple of parameterization and model predictions for it.
    """
    res = get_best_parameters_from_model_predictions_with_trial_index(
        experiment=experiment, models_enum=models_enum, trial_indices=trial_indices
    )

    if res is None:
        return None

    _, parameterization, vals = res

    return parameterization, vals


def get_best_by_raw_objective_with_trial_index(
    experiment: Experiment,
    optimization_config: Optional[OptimizationConfig] = None,
    trial_indices: Optional[Iterable[int]] = None,
) -> Optional[Tuple[int, TParameterization, Optional[TModelPredictArm]]]:
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
    optimization_config: Optional[OptimizationConfig] = None,
    trial_indices: Optional[Iterable[int]] = None,
) -> Optional[Tuple[TParameterization, Optional[TModelPredictArm]]]:
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
    models_enum: Type[ModelRegistryBase],
    optimization_config: Optional[OptimizationConfig] = None,
    trial_indices: Optional[Iterable[int]] = None,
) -> Optional[Tuple[int, TParameterization, Optional[TModelPredictArm]]]:
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
    models_enum: Type[ModelRegistryBase],
    optimization_config: Optional[OptimizationConfig] = None,
    trial_indices: Optional[Iterable[int]] = None,
) -> Optional[Tuple[TParameterization, Optional[TModelPredictArm]]]:
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
    optimization_config: Optional[OptimizationConfig] = None,
    trial_indices: Optional[Iterable[int]] = None,
    use_model_predictions: bool = True,
) -> Dict[int, Tuple[TParameterization, TModelPredictArm]]:
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

    moo_optimization_config = checked_cast(
        MultiObjectiveOptimizationConfig, optimization_config
    )

    # Use existing modelbridge if it supports MOO otherwise create a new MOO modelbridge
    # to use for Pareto frontier extraction.
    modelbridge = generation_strategy.model
    is_moo_modelbridge = (
        modelbridge
        and isinstance(modelbridge, TorchModelBridge)
        and checked_cast(TorchModelBridge, modelbridge).is_moo_problem
    )
    if is_moo_modelbridge:
        generation_strategy._fit_or_update_current_model(data=None)
    else:
        modelbridge = Models.MOO(
            experiment=experiment,
            data=checked_cast(
                Data, experiment.lookup_data(trial_indices=trial_indices)
            ),
        )
    modelbridge = checked_cast(TorchModelBridge, modelbridge)

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

    # Extract the Pareto frontier and format it as follows:
    # { trial_index --> (parameterization, (means, covariances) }
    pareto_util = predicted_pareto if use_model_predictions else observed_pareto
    pareto_optimal_observations = pareto_util(
        modelbridge=modelbridge, objective_thresholds=objective_thresholds_override
    )
    return {
        int(not_none(obs.features.trial_index)): (
            obs.features.parameters,
            (obs.data.means_dict, obs.data.covariance_matrix),
        )
        for obs in pareto_optimal_observations
    }


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


def _filter_feasible_rows(
    df: pd.DataFrame,
    optimization_config: OptimizationConfig,
    status_quo: Optional[Arm],
) -> pd.DataFrame:
    """Filter out arms that do not satisfy outcome constraints

    Looks at all arm data collected and removes rows corresponding to arms in
    which one or more of their associated metrics' 95% confidence interval
    falls outside of any outcome constraint's bounds (i.e. we are 95% sure the
    bound is not satisfied).
    """
    if len(optimization_config.outcome_constraints) < 1:
        return df

    name = df["metric_name"]

    # When SEM is NaN we should treat it as if it were 0
    sems = not_none(df["sem"].fillna(0))

    # Bounds computed for 95% confidence interval on Normal distribution
    lower_bound = df["mean"] - sems * 1.96
    upper_bound = df["mean"] + sems * 1.96

    # Only compute relativization if some constraints are relative
    rel_df = None
    rel_lower_bound = None
    rel_upper_bound = None
    if status_quo is not None and any(
        oc.relative for oc in optimization_config.outcome_constraints
    ):
        # relativize_data expects all arms to come from the same trial, we need to
        # format the data as if it was.
        to_relativize = df.copy()
        to_relativize["trial_index"] = 0

        rel_df = relativize_data(
            data=Data(to_relativize), status_quo_name=status_quo.name
        ).df.append(
            {
                "arm_name": "status_quo",
                "metric_name": status_quo.name,
                "mean": 0,
                "sem": 0,
            },
            ignore_index=True,
        )
        rel_sems = not_none(rel_df["sem"].fillna(0))
        rel_lower_bound = rel_df["mean"] - rel_sems * 1.96
        rel_upper_bound = rel_df["mean"] + rel_sems * 1.96

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
            if rel_lower_bound is None or rel_upper_bound is None:
                logger.warning(
                    f"No status quo provided; relative constraint {oc} ignored."
                )
                return pd.Series(True, index=df.index)

            observed_lower_bound = rel_lower_bound
            observed_upper_bound = rel_upper_bound
        else:
            observed_lower_bound = lower_bound
            observed_upper_bound = upper_bound

        # Return True if metrics are different, or whether the confidence
        # interval is entirely not within the bound
        if oc.op == ComparisonOp.GEQ:
            return ~name_match_mask | (observed_upper_bound > oc.bound)
        else:
            return ~name_match_mask | (observed_lower_bound < oc.bound)

    mask = reduce(
        lambda left, right: left & right,
        map(oc_mask, optimization_config.outcome_constraints),
    )
    bad_arm_names = (
        df[~mask]["arm_name"].tolist()
        if rel_df is None
        else rel_df[~mask]["arm_name"].tolist()
    )
    feasible = df.loc[df["arm_name"].apply(lambda x: x not in bad_arm_names)]

    if feasible.empty:
        raise ValueError(
            "No points satisfied all outcome constraints within 95 percent"
            + "confidence interval"
        )

    return feasible


def _is_all_noiseless(df: pd.DataFrame, metric_name: str) -> bool:
    """Noiseless is defined as SEM = 0 or SEM = NaN on a given metric (usually
    the objective).
    """

    name_mask = df["metric_name"] == metric_name
    df_metric_arms_sems = df[name_mask]["sem"]

    return ((df_metric_arms_sems == 0) | df_metric_arms_sems == NaN).all()
