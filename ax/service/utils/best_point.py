#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from logging import Logger

import numpy as np
import pandas as pd
import torch
from ax.adapter.adapter_utils import (
    _get_adapter_training_data,
    get_pareto_frontier_and_configs,
    observed_pareto_frontier as observed_pareto,
    predicted_pareto_frontier as predicted_pareto,
)
from ax.adapter.base import Adapter
from ax.adapter.cross_validation import (
    assess_model_fit,
    compute_diagnostics,
    cross_validate,
)
from ax.adapter.registry import Generators, MBM_X_trans
from ax.adapter.torch import TorchAdapter
from ax.adapter.transforms.derelativize import Derelativize
from ax.core.auxiliary import AuxiliaryExperimentPurpose
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
    PreferenceOptimizationConfig,
)
from ax.core.outcome_constraint import ObjectiveThreshold, OutcomeConstraint
from ax.core.trial import Trial
from ax.core.types import ComparisonOp, TModelPredictArm, TParameterization
from ax.exceptions.core import DataRequiredError, UnsupportedError, UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generators.torch_base import TorchGenerator
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.preference.preference_utils import get_preference_adapter
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from numpy.typing import NDArray
from pyre_extensions import assert_is_instance, none_throws

logger: Logger = get_logger(__name__)


def derelativize_opt_config(
    optimization_config: OptimizationConfig,
    experiment: Experiment,
    trial_indices: Iterable[int] | None = None,
) -> OptimizationConfig:
    tf = Derelativize(search_space=None, config={"use_raw_status_quo": True})
    optimization_config = tf.transform_optimization_config(
        optimization_config=optimization_config.clone(),
        adapter=get_tensor_converter_adapter(
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

    is_feasible = is_row_feasible(
        df=completed_df,
        optimization_config=optimization_config,
    )
    is_na_mask = is_feasible.isna()
    if not is_feasible.any():
        msg = (
            "No points satisfied all outcome constraints within 95 percent "
            "confidence interval."
        )
        na_arms = completed_df[is_na_mask]["arm_name"].unique()
        if len(na_arms) > 0:
            msg += (
                f" The feasibility of {len(na_arms)} arm(s) could not be determined: "
                f"{na_arms}."
            )
        raise ValueError(msg)
    # For the sake of best point identification, we only care about feasible trials.
    # The distinction between infeasible and undetermined is not important.
    is_feasible[is_na_mask] = False
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
        experiment=experiment,
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
            if len(trial.generator_runs) > 0:
                # In theory batch_trial can have >1 gr, grab the first
                gr = trial.generator_runs[0]
        if gr is not None:
            break

    if not isinstance(adapter, TorchAdapter):
        if gr is None:
            return None
        return _extract_best_arm_from_gr(gr=gr, trials=experiment.trials)

    # Check to see if the adapter is worth using.
    cv_results = cross_validate(adapter=adapter)
    diagnostics = compute_diagnostics(result=cv_results)
    assess_model_fit_results = assess_model_fit(diagnostics=diagnostics)

    # For ScalarizedObjective, check model fit for all component metrics
    # For regular Objective, check model fit for the single objective metric
    if isinstance(optimization_config.objective, ScalarizedObjective):
        objective_metric_names = [
            metric.name for metric in optimization_config.objective.metrics
        ]
    else:
        objective_metric_names = [optimization_config.objective.metric.name]

    # If model fit is bad for any objective metric, use raw results
    bad_fit_objective_metrics = [
        name
        for name in objective_metric_names
        if name in assess_model_fit_results.bad_fit_metrics_to_fisher_score
    ]

    if bad_fit_objective_metrics:
        logger.warning(
            f"Model fit is poor for objective metric(s) {bad_fit_objective_metrics}; "
            "falling back on raw data for best point."
        )

        # Check if any of the objective metrics are noisy
        noisy_metrics = [
            name
            for name in bad_fit_objective_metrics
            if not _is_all_noiseless(df=data.df, metric_name=name)
        ]
        if noisy_metrics:
            logger.warning(
                f"Model fit is poor and data on objective metric(s) "
                f"{noisy_metrics} is noisy; interpret best points "
                "results carefully."
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
    except (ValueError, UserInputError, DataRequiredError) as err:
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
        generation_strategy: Generation strategy containing the adapter.
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

    # Use existing adapter if it supports MOO otherwise create a new MOO adapter
    # to use for Pareto frontier extraction.
    adapter = generation_strategy.adapter
    is_moo_adapter = (
        adapter
        and isinstance(adapter, TorchAdapter)
        and assert_is_instance(
            adapter,
            TorchAdapter,
        ).is_moo_problem
    )
    if is_moo_adapter:
        generation_strategy._curr._fit(experiment=experiment)
    else:
        adapter = Generators.BOTORCH_MODULAR(
            experiment=experiment,
            data=assert_is_instance(
                experiment.lookup_data(trial_indices=trial_indices),
                Data,
            ),
        )
    adapter = assert_is_instance(adapter, TorchAdapter)

    objective_thresholds_override = None
    # If objective thresholds are not specified in optimization config, infer them.
    if not moo_optimization_config.objective_thresholds:
        objective_thresholds_override = adapter.infer_objective_thresholds(
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
        adapter=adapter,
        optimization_config=moo_optimization_config,
        objective_thresholds=objective_thresholds_override,
    )

    # Insert observations into OrderedDict in order of descending individual
    # hypervolume, formatted as
    # { trial_index --> (parameterization, (means, covariances) }
    res: dict[int, tuple[TParameterization, TModelPredictArm]] = OrderedDict()
    for obs in pareto_optimal_observations:
        res[int(none_throws(obs.features.trial_index))] = (
            obs.features.parameters,
            (obs.data.means_dict, obs.data.covariance_matrix),
        )

    return res


def is_row_feasible(
    df: pd.DataFrame,
    optimization_config: OptimizationConfig,
    undetermined_value: bool | None = None,
) -> pd.Series:
    """Determine whether arms satisfy outcome constraints based on observed data.

    Evaluates each arm's feasibility by checking if its associated metrics' 95%
    confidence intervals satisfy all outcome constraints. Returns False for arms
    where we are 95% confident that at least one constraint is violated, True for
    arms that satisfy all constraints, and undetermined_value for arms where
    feasibility cannot be conclusively determined.

    Args:
        df: DataFrame of arm data with required columns: "metric_name", "mean",
            "sem", and "arm_name". Each row represents a metric observation for
            a specific arm.
        optimization_config: OptimizationConfig containing the outcome constraints
            to evaluate. Must have derelativized constraints.
        undetermined_value: Value to return for arms where feasibility cannot be
            determined due to missing data. Defaults to None.

    Returns:
        Series of boolean or None values indexed by df.index, where:
        - True: Arm satisfies all outcome constraints
        - False: Arm violates at least one outcome constraint (95% confidence)
        - undetermined_value: Feasibility cannot be determined (missing data or
          relative constraints present)
    """
    if len(optimization_config.all_constraints) < 1:
        return pd.Series([True] * len(df), index=df.index)

    relative_constraints = [
        c for c in optimization_config.all_constraints if c.relative
    ]
    if len(relative_constraints) > 0:
        logger.warning(
            f"Determining trial feasibility only supported with a derelativized "
            "OptimizationConfig, but found the following relative constraints: "
            f"{relative_constraints}. "
            f"Returning {undetermined_value} as the feasibility."
        )
        return pd.Series([undetermined_value for _ in df.index], index=df.index)

    name = df["metric_name"]

    # When SEM is NaN we should treat it as if it were 0
    sems = none_throws(df["sem"].fillna(0))
    # Bounds computed for 95% confidence interval on Normal distribution
    lower_bound = df["mean"] - sems * 1.96
    upper_bound = df["mean"] + sems * 1.96
    # TODO: Support scalarized outcome constraints by getting weights and scalarizing
    # the bounds here.

    # Nested function from OC -> Mask for consumption in later map/reduce from
    # [OC] -> Mask. Constraint relativity is handled inside so long as relative bounds
    # are set in surrounding closure (which will occur in proper experiment setup).
    def compute_feasibility_per_constraint(
        oc: OutcomeConstraint,
        lower_bound: pd.Series = lower_bound,
        upper_bound: pd.Series = upper_bound,
        name: pd.Series = name,
    ) -> pd.Series:
        name_match_mask = name == oc.metric.name
        # Return True if metrics are different, or whether the confidence
        # interval is entirely not within the bound
        if oc.op == ComparisonOp.GEQ:
            return ~name_match_mask | (upper_bound >= float(oc.bound))
        else:
            return ~name_match_mask | (lower_bound <= float(oc.bound))

    # Keep track of whether arms have mising values (NaNs) or rows.
    is_na_mask = df["mean"].isna()

    # If an arm doesn't have data for all constrained metrics, and the observed
    # constrained metric values are feasible, (in)feasibility cannot be determined
    # conclusively.
    has_missing_metric_mask = pd.Series([False] * len(df), index=df.index)
    constrained_metric_names = {
        oc.metric.name for oc in optimization_config.all_constraints
    }
    for arm_name, arm_group in df.groupby("arm_name"):
        metrics_for_arm = set(arm_group["metric_name"].unique())
        missing_metrics = constrained_metric_names - metrics_for_arm
        if missing_metrics:
            logger.warning(
                f"Arm {arm_name} is missing data for one or more constrained metrics: "
                f"{missing_metrics}."
            )
            has_missing_metric_mask = has_missing_metric_mask | (
                df["arm_name"] == arm_name
            )

    # Computing feasibility on a per-row (metric-arm-combination) basis.
    is_feasible_per_constraint_list = [
        compute_feasibility_per_constraint(oc=oc)
        for oc in optimization_config.all_constraints
    ]
    # stacking the feasibility masks for each constraint an checking if all are feasible
    is_feasible_mask = pd.DataFrame(is_feasible_per_constraint_list).all(axis=0)

    # can definititively determine infeasibility for all rows that are evaluated
    # infeasible (~is_feasible_mask) based on available data (~is_na_mask).
    infeasible_df = df[~is_feasible_mask & ~is_na_mask]
    infeasible_arm_names = set(infeasible_df["arm_name"].unique())
    # Can't determine feasibility for rows that are not definitively infeasible
    # and that have missing values (NaN or missing metrics).
    na_df = df[has_missing_metric_mask | is_na_mask]
    na_arm_names = set(na_df["arm_name"].unique())

    def tag_feasible_arms(
        x: str,
        infeasible_arm_names: set[str] = infeasible_arm_names,
        na_arm_names: set[str] = na_arm_names,
    ) -> bool | None:
        if x in infeasible_arm_names:
            return False
        elif x in na_arm_names:
            return undetermined_value
        return True

    return assert_is_instance(
        df["arm_name"].apply(tag_feasible_arms),
        pd.Series,
    )


def _is_all_noiseless(df: pd.DataFrame, metric_name: str) -> bool:
    """Noiseless is defined as SEM = 0 or SEM = NaN on a given metric (usually
    the objective).
    """

    name_mask = df["metric_name"] == metric_name
    df_metric_arms_sems = df[name_mask]["sem"]

    return ((df_metric_arms_sems == 0) | df_metric_arms_sems.isna()).all()


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

    metrics_tensor = torch.from_numpy(df_wide[objective.metric_names].to_numpy().copy())
    return _compute_hv_trace(
        ref_point=objective_thresholds,
        metrics_tensor=metrics_tensor,
        is_feasible_array=df_wide["feasible"].to_numpy(),
        use_cumulative_hv=use_cumulative_hv,
    )


def _compute_utility_from_preference_model(
    df_wide: pd.DataFrame,
    experiment: Experiment,
    optimization_config: PreferenceOptimizationConfig,
) -> NDArray:
    """Compute utility predictions for each arm using the learned preference model.

    This function accesses the PE_EXPERIMENT auxiliary experiment, fits a PairwiseGP
    model to the preference data, and uses it to predict utility values for each
    arm's metric values.

    Args:
        df_wide: DataFrame with columns for trial_index, arm_name, feasible,
            and metric values.
        experiment: The main experiment containing the PE_EXPERIMENT auxiliary.
        optimization_config: PreferenceOptimizationConfig specifying the preference
            profile to use.

    Returns:
        Array of utility predictions, one for each row in df_wide. Infeasible
        arms will have utility of negative infinity.

    Raises:
        DataRequiredError: If PE_EXPERIMENT has no data.
        UserInputError: If PE_EXPERIMENT is not found for the specified profile.
    """
    pref_profile_name = optimization_config.preference_profile_name

    # Find the PE_EXPERIMENT auxiliary experiment
    pe_aux_exp = experiment.find_auxiliary_experiment_by_name(
        purpose=AuxiliaryExperimentPurpose.PE_EXPERIMENT,
        auxiliary_experiment_name=pref_profile_name,
        raise_if_not_found=False,
    )

    if pe_aux_exp is None:
        raise UserInputError(
            f"Preference profile '{pref_profile_name}' not found in experiment "
            f"'{experiment.name}'. Cannot compute utility-based trace without "
            "a valid preference profile."
        )

    pe_experiment = pe_aux_exp.experiment
    pe_data = pe_experiment.lookup_data()

    if pe_data.df.empty:
        raise DataRequiredError(
            f"No preference data found in preference profile '{pref_profile_name}'. "
            "Update the preference profile or play the preference game before "
            "computing utility-based trace."
        )

    # Create adapter with fitted preference model
    adapter = get_preference_adapter(experiment=pe_experiment, data=pe_data)

    # Create ObservationFeatures for each arm with metric values as parameters
    observation_features = []
    for _, row in df_wide.iterrows():
        # Create parameters dict with metric names as keys and their values
        parameters = {
            metric_name: row[metric_name]
            for metric_name in optimization_config.objective.metric_names
        }
        obs_feat = ObservationFeatures(parameters=parameters)
        observation_features.append(obs_feat)

    # Predict utilities using the fitted preference model
    f_dict, _ = adapter.predict(
        observation_features=observation_features,
        use_posterior_predictive=False,
    )

    # Extract utility metric predictions
    # PE_EXPERIMENT always has a single metric: "pairwise_pref_query"
    utility_metric_name = Keys.PAIRWISE_PREFERENCE_QUERY.value
    utilities = np.array(f_dict[utility_metric_name])

    # Set infeasible arms to -inf (higher utility is better, so infeasible arms
    # should have the worst possible utility)
    infeasible_idx = np.where(~df_wide["feasible"])[0]
    utilities[infeasible_idx] = float("-inf")

    return utilities


def _prepare_data_for_trace(
    df: pd.DataFrame,
    optimization_config: OptimizationConfig,
) -> pd.DataFrame:
    """
    Prepare data for trace computation by adding feasibility information
    and reshaping to wide format.

    This function is shared between get_trace_by_arm_pull_from_data and
    get_is_feasible_trace.

    Args:
        df: Data in the format returned by ``Data.df``, with a separate row for
            each trial index-arm name-metric.
        optimization_config: ``OptimizationConfig`` to use to get the trace. Must
            not be in relative form.

    Return:
        A DataFrame with columns ["trial_index", "arm_name", "feasible"] +
        relevant metric names, where "feasible" indicates whether the arm
        satisfies all constraints.
    """
    # Add feasibility information
    df["row_feasible"] = is_row_feasible(
        df=df,
        optimization_config=optimization_config,
        # For the sake of this function, we only care about feasible trials. The
        # distinction between infeasible and undetermined is not important.
        undetermined_value=False,
    )

    # Get the metrics we need
    metrics = list(optimization_config.metrics.keys())

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
    df_wide["feasible"] = df.groupby(["trial_index", "arm_name"])["row_feasible"].all()
    df_wide.reset_index(inplace=True)

    return df_wide


def get_trace_by_arm_pull_from_data(
    df: pd.DataFrame,
    optimization_config: OptimizationConfig,
    use_cumulative_best: bool = True,
    experiment: Experiment | None = None,
) -> pd.DataFrame:
    """
    Get a trace of the objective value or hypervolume of outcomes.

    An "arm pull" is the combination of a trial (index) and an arm. This
    function returns a single value for each arm pull, even if there are
    multiple arms per trial or if an arm is repeated in multiple trials.

    For BOPE experiments, this function computes
    utility predictions using the learned preference model from the PE_EXPERIMENT
    auxiliary experiment.

    Args:
        df: Data in the format returned by ``Data.df``, with a separate row for
            each trial index-arm name-metric.
        optimization_config: ``OptimizationConfig`` to use to get the trace. Must
            not be in relative form.
        use_cumulative_best: If True, the trace will be the cumulative best
            objective. Otherwise, the trace will be the value of each point.
        experiment: Optional experiment object. Required for preference learning
            experiments to access the PE_EXPERIMENT auxiliary experiment.

    Return:
        A DataFrame containing columns 'trial_index', 'arm_name', and "value",
        where "value" is the value of the outcomes attained (or predicted utility
        for preference learning experiments).
    """
    if any(oc.relative for oc in optimization_config.all_constraints):
        raise ValueError(
            "Relativized optimization config not supported. Please "
            "`Derelativize` the optimization config, or use `get_trace`."
        )
    empty_result = pd.DataFrame(columns=["trial_index", "arm_name", "value"])
    if len(df) == 0:
        return empty_result

    df_wide = _prepare_data_for_trace(df=df, optimization_config=optimization_config)
    if len(df_wide) == 0:
        return empty_result

    # Handle preference learning experiments
    if experiment is not None and isinstance(
        optimization_config, PreferenceOptimizationConfig
    ):
        logger.info(
            f"Computing utility-based trace for preference learning experiment "
            f"using PE_EXPERIMENT '{optimization_config.preference_profile_name}'."
        )
        df_wide["value"] = _compute_utility_from_preference_model(
            df_wide=df_wide,
            experiment=experiment,
            optimization_config=optimization_config,
        )
        return df_wide[["trial_index", "arm_name", "value"]]

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
    include_status_quo: bool = False,
) -> list[float]:
    """Compute the optimization trace at each iteration.

    Given an experiment and an optimization config, compute the performance
    at each iteration. For multi-objective, the performance is computed as
    the hypervolume. For single objective, the performance is computed as
    the best observed objective value.

    For BOPE experiments, the utility of each trial is computed using
    the learned preference model from the PE_EXPERIMENT auxiliary experiment. The
    preference model is used to predict the utility of each trial's metric values,
    and the trace represents the best predicted utility over time.

    Infeasible points (that violate constraints) do not contribute to
    improvements in the optimization trace. If the first trial(s) are infeasible,
    the trace can start at inf or -inf.

    An iteration here refers to a completed or early-stopped (batch) trial.
    There will be one performance metric in the trace for each iteration.

    Args:
        experiment: The experiment to get the trace for.
        optimization_config: Optimization config to use in place of the one
            stored on the experiment.
        include_status_quo: If True, include status quo in the trace computation.
            If False (default), exclude status quo for compatibility with legacy
            behavior.

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
    if not include_status_quo and (status_quo := experiment.status_quo) is not None:
        idx &= df["arm_name"] != status_quo.name
    df = df.loc[idx, :]
    if len(df) == 0:
        return []

    # Derelativize the optimization config only if needed (i.e., if there are
    # relative constraints). This avoids unnecessary data pivoting that can
    # fail with duplicate indices.
    if any(oc.relative for oc in optimization_config.all_constraints):
        optimization_config = derelativize_opt_config(
            optimization_config=optimization_config,
            experiment=experiment,
        )

    # Get a value for each trial_index + arm
    value_by_arm_pull = get_trace_by_arm_pull_from_data(
        df=df,
        optimization_config=optimization_config,
        use_cumulative_best=True,
        experiment=experiment,
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


def get_tensor_converter_adapter(
    experiment: Experiment, data: Data | None = None
) -> TorchAdapter:
    """
    Constructs a minimal model for converting things to tensors.

    Model fitting will instantiate all of the transforms but will not do any
    expensive (i.e. GP) fitting beyond that. The model will raise an error if
    it is used for predicting or generating.

    Will work for any search space regardless of types of parameters.

    Args:
        experiment: Experiment.
        data: Data for fitting the model.

    Returns: A torch adapter with transforms set.
    """
    # Transforms is the minimal set that will work for converting any search
    # space to tensors.
    return TorchAdapter(
        experiment=experiment,
        data=data,
        generator=TorchGenerator(),
        transforms=MBM_X_trans,
    )


def infer_reference_point_from_experiment(
    experiment: Experiment, data: Data
) -> list[ObjectiveThreshold]:
    """This functions is a wrapper around ``infer_reference_point`` to find the nadir
    point from the pareto front of an experiment. Aside from converting experiment
    to tensors, this wrapper transforms back and forth the objectives of the experiment
    so that they are appropriately used by ``infer_reference_point``.

    Args:
        experiment: The experiment for which we want to infer the reference point.

    Returns:
        A list of objective thresholds representing the reference point.
    """
    if not experiment.is_moo_problem:
        raise ValueError(
            "This function works for MOO experiments only."
            f" Experiment {experiment.name} is single objective."
        )

    # Reading experiment data.
    adapter = get_tensor_converter_adapter(
        experiment=experiment,
        data=data,
    )
    obs_feats, obs_data, _ = _get_adapter_training_data(adapter=adapter)

    # Since objectives could have arbitrary orders in objective_thresholds and
    # further down the road `get_pareto_frontier_and_configs` arbitrarily changes the
    # orders of the objectives, we fix the objective orders here based on the
    # observation_data and maintain it throughout the flow.
    objective_orders = obs_data[0].metric_signatures

    # Defining a dummy reference point so that all observed points are considered
    # when calculating the Pareto front. Also, defining a multiplier to turn all
    # the objectives to be maximized. Note that the multiplier at this point
    # contains 0 for outcome_constraint metrics, but this will be dropped later.
    opt_config = assert_is_instance(
        experiment.optimization_config, MultiObjectiveOptimizationConfig
    )
    inferred_rp = _get_objective_thresholds(optimization_config=opt_config)
    multiplier = [0] * len(objective_orders)
    if len(opt_config.objective_thresholds) > 0:
        inferred_rp = deepcopy(opt_config.objective_thresholds)
    else:
        inferred_rp = []
        for objective in assert_is_instance(
            opt_config.objective, MultiObjective
        ).objectives:
            ot = ObjectiveThreshold(
                metric=objective.metric,
                bound=0.0,  # dummy value
                op=ComparisonOp.LEQ if objective.minimize else ComparisonOp.GEQ,
                relative=False,
            )
            inferred_rp.append(ot)
    for ot in inferred_rp:
        # In the following, we find the index of the objective in
        # `objective_orders`. If there is an objective that does not exist
        # in `obs_data`, a ValueError is raised.
        try:
            objective_index = objective_orders.index(ot.metric.signature)
        except ValueError:
            raise ValueError(
                f"Metric {ot.metric.signature} does not exist in `obs_data`."
            )

        if ot.op == ComparisonOp.LEQ:
            ot.bound = np.inf
            multiplier[objective_index] = -1
        else:
            ot.bound = -np.inf
            multiplier[objective_index] = 1

    # Finding the pareto frontier
    frontier_observations, f, obj_w, _ = get_pareto_frontier_and_configs(
        adapter=adapter,
        observation_features=obs_feats,
        observation_data=obs_data,
        objective_thresholds=inferred_rp,
        use_model_predictions=False,
    )
    if len(frontier_observations) == 0:
        outcome_constraints = opt_config._outcome_constraints
        if len(outcome_constraints) == 0:
            raise RuntimeError(
                "No frontier observations found in the experiment and no constraints "
                "are present. Please check the data of the experiment."
            )

        logger.warning(
            "No frontier observations found in the experiment. The likely cause is "
            "the absence of feasible arms in the experiment if a constraint is present."
            " Trying to find a reference point with the unconstrained objective values."
        )

        opt_config._outcome_constraints = []  # removing the constraints
        # getting the unconstrained pareto frontier
        frontier_observations, f, obj_w, _ = get_pareto_frontier_and_configs(
            adapter=adapter,
            observation_features=obs_feats,
            observation_data=obs_data,
            objective_thresholds=inferred_rp,
            use_model_predictions=False,
        )
        # restoring constraints
        opt_config._outcome_constraints = outcome_constraints

    # Need to reshuffle columns of `f` and `obj_w` to be consistent
    # with objective_orders.
    order = [
        objective_orders.index(metric_signature)
        for metric_signature in frontier_observations[0].data.metric_signatures
    ]
    f = f[:, order]
    obj_w = obj_w[:, order]

    # Dropping the columns related to outcome constraints.
    # obj_w is 2D (n_objectives, n_outcomes); collapse to a 1D mask.
    obj_w_mask = (obj_w != 0).any(dim=0)
    obj_col_indices = obj_w_mask.nonzero().view(-1)
    f = f[:, obj_col_indices]
    multiplier_tensor = torch.tensor(multiplier, dtype=f.dtype, device=f.device)
    multiplier_nonzero = multiplier_tensor[obj_col_indices]

    # Transforming all the objectives to be maximized.
    f_transformed = multiplier_nonzero * f

    # Finding nadir point.
    rp_raw = infer_reference_point(f_transformed)

    # Un-transforming the reference point.
    rp = multiplier_nonzero * rp_raw

    # Removing the non-objective metrics form the order.
    objective_orders_reduced = [
        x for (i, x) in enumerate(objective_orders) if multiplier[i] != 0
    ]

    for obj_threshold in inferred_rp:
        obj_threshold.bound = rp[
            objective_orders_reduced.index(obj_threshold.metric.signature)
        ].item()
    return inferred_rp


def _get_objective_thresholds(
    optimization_config: MultiObjectiveOptimizationConfig,
) -> list[ObjectiveThreshold]:
    """Get objective thresholds for an optimization config.

    This will return objective thresholds with dummy values if there are
    no objective thresholds on the optimization config.

    Args:
        optimization_config: Optimization config.

    Returns:
        List of objective thresholds.
    """
    if optimization_config.objective_thresholds is not None:
        return deepcopy(optimization_config.objective_thresholds)
    objective_thresholds = []
    for objective in assert_is_instance(
        optimization_config.objective, MultiObjective
    ).objectives:
        ot = ObjectiveThreshold(
            metric=objective.metric,
            bound=0.0,  # dummy value
            op=ComparisonOp.LEQ if objective.minimize else ComparisonOp.GEQ,
            relative=False,
        )
        objective_thresholds.append(ot)
    return objective_thresholds
