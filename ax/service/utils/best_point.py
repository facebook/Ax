#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import pandas as pd
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.optimization_config import (
    OptimizationConfig,
    MultiObjectiveOptimizationConfig,
)
from ax.core.trial import Trial
from ax.core.types import TModelPredictArm, TParameterization
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.modelbridge_utils import (
    predicted_pareto_frontier as predicted_pareto,
    observed_pareto_frontier as observed_pareto,
)
from ax.modelbridge.multi_objective_torch import MultiObjectiveTorchModelBridge
from ax.modelbridge.registry import Models
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none, checked_cast


logger = get_logger(__name__)


def get_best_raw_objective_point(
    experiment: Experiment, optimization_config: Optional[OptimizationConfig] = None
) -> Tuple[TParameterization, Dict[str, Tuple[float, float]]]:
    """Given an experiment, identifies the arm that had the best raw objective,
    based on the data fetched from the experiment.

    Args:
        experiment: Experiment, on which to identify best raw objective arm.
        optimization_config: Optimization config to use in absence or in place of
            the one stored on the experiment.

    Returns:
        Tuple of parameterization and a mapping from metric name to a tuple of
            the corresponding objective mean and SEM.
    """
    # pyre-ignore [16]
    if isinstance(experiment.optimization_config.objective, MultiObjective):
        logger.warn(
            "get_best_raw_objective_point is deprecated for multi-objective "
            "optimization. This method will return an arbitrary point on the "
            "pareto frontier."
        )
    opt_config = optimization_config or experiment.optimization_config
    assert opt_config is not None, (
        "Cannot identify the best point without an optimization config, but no "
        "optimization config was provided on the experiment or as an argument."
    )
    dat = experiment.fetch_data()
    if dat.df.empty:
        raise ValueError("Cannot identify best point if experiment contains no data.")
    objective = opt_config.objective
    if isinstance(objective, ScalarizedObjective):
        best_row = _get_best_row_for_scalarized_objective(dat.df, objective)
    else:
        best_row = _get_best_row_for_single_objective(dat.df, objective)
    # pyre-fixme[6]: Expected `str` for 1st param but got `Series`.
    best_arm = experiment.arms_by_name[best_row["arm_name"]]
    best_trial_index = best_row["trial_index"]
    objective_rows = dat.df.loc[
        (dat.df["arm_name"] == best_arm.name)
        & (dat.df["trial_index"] == best_trial_index)
    ]
    vals = {
        row["metric_name"]: (row["mean"], row["sem"])
        for _, row in objective_rows.iterrows()
    }
    return not_none(best_arm).parameters, vals


def get_best_from_model_predictions(
    experiment: Experiment,
) -> Optional[Tuple[TParameterization, Optional[TModelPredictArm]]]:
    """Given an experiment, returns the best predicted parameterization and corresponding
    prediction based on the most recent Trial with predictions. If no trials have
    predictions returns None.

    Only some models return predictions. For instance GPEI does while Sobol does not.

    TModelPredictArm is of the form:
        ({metric_name: mean}, {metric_name_1: {metric_name_2: cov_1_2}})

    Args:
        experiment: Experiment, on which to identify best raw objective arm.

    Returns:
        Tuple of parameterization and model predictions for it.
    """
    # pyre-ignore [16]
    if isinstance(experiment.optimization_config.objective, MultiObjective):
        logger.warn(
            "get_best_from_model_predictions is deprecated for multi-objective "
            "optimization configs. This method will return an arbitrary point on "
            "the pareto frontier."
        )
    for _, trial in sorted(experiment.trials.items(), key=lambda x: x[0], reverse=True):
        gr = None
        if isinstance(trial, Trial):
            gr = trial.generator_run
        elif isinstance(trial, BatchTrial):
            if len(trial.generator_run_structs) > 0:
                # In theory batch_trial can have >1 gr, grab the first
                gr = trial.generator_run_structs[0].generator_run

        if gr is not None and gr.best_arm_predictions is not None:  # pragma: no cover
            best_arm, best_arm_predictions = not_none(gr.best_arm_predictions)
            return not_none(best_arm).parameters, best_arm_predictions
    return None


def get_best_parameters(
    experiment: Experiment,
    use_model_predictions: bool = True,
) -> Optional[Tuple[TParameterization, Optional[TModelPredictArm]]]:
    """Given an experiment, identifies the best arm.

    First attempts according to do so with models used in optimization and
    its corresponding predictions if available. Falls back to the best raw
    objective based on the data fetched from the experiment.

    TModelPredictArm is of the form:
        ({metric_name: mean}, {metric_name_1: {metric_name_2: cov_1_2}})

    Args:
        experiment: Experiment, on which to identify best raw objective arm.
        use_model_predictions: Whether to extract the best point using
            model predictions or directly observed values. If ``True``,
            the metric means and covariances in this method's output will
            also be based on model predictions and may differ from the
            observed values.

    Returns:
        Tuple of parameterization and model predictions for it.
    """
    # pyre-ignore [16]
    if isinstance(experiment.optimization_config.objective, MultiObjective):
        logger.warn(
            "get_best_parameters is deprecated for multi-objective optimization. "
            "This method will return an arbitrary point on the pareto frontier."
        )
    # Find latest trial which has a generator_run attached and get its predictions
    if use_model_predictions:
        model_predictions = get_best_from_model_predictions(experiment=experiment)
        if model_predictions is not None:  # pragma: no cover
            return model_predictions
        logger.info(
            "Could not use model predictions to identify best point, will use raw "
            "objective values."
        )

    # Could not find through model, default to using raw objective.
    try:
        parameterization, values = get_best_raw_objective_point(experiment=experiment)
    except ValueError:
        return None
    return (
        parameterization,
        (
            {k: v[0] for k, v in values.items()},  # v[0] is mean
            {k: {k: v[1] * v[1]} for k, v in values.items()},  # v[1] is sem
        ),
    )


def get_pareto_optimal_parameters(
    experiment: Experiment,
    generation_strategy: GenerationStrategy,
    use_model_predictions: bool = True,
) -> Optional[Dict[int, Tuple[TParameterization, TModelPredictArm]]]:
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
        use_model_predictions: Whether to extract the Pareto frontier using
            model predictions or directly observed values. If ``True``,
            the metric means and covariances in this method's output will
            also be based on model predictions and may differ from the
            observed values.

    Returns:
        ``None`` if it was not possible to extract the Pareto frontier,
        otherwise a mapping from trial index to the tuple of:
        - the parameterization of the arm in that trial,
        - two-item tuple of metric means dictionary and covariance matrix
            (model-predicted if ``use_model_predictions=True`` and observed
            otherwise).
    """
    # Validate aspects of the experiment: that it is a MOO experiment and
    # that the current model can be used to produce the Pareto frontier.
    if not not_none(experiment.optimization_config).is_moo_problem:
        raise UnsupportedError(
            "Please use `get_best_parameters` for single-objective problems."
        )

    moo_optimization_config = checked_cast(
        MultiObjectiveOptimizationConfig, experiment.optimization_config
    )
    if moo_optimization_config.outcome_constraints:
        # TODO[drfreund]: Test this flow and remove error.
        raise NotImplementedError(
            "Support for outcome constraints is currently under development."
        )

    # Extract or instantiate modelbridge to use for Pareto frontier extraction.
    mb = generation_strategy.model
    if mb is None or not isinstance(mb, MultiObjectiveTorchModelBridge):
        logger.info(
            "Can only extract a Pareto frontier using a multi-objective model bridge"
            f", but currently used model bridge is: {mb} of type {type(mb)}. Will "
            "use `Models.MOO` instead to extract Pareto frontier."
        )
        mb = checked_cast(
            MultiObjectiveTorchModelBridge,
            Models.MOO(
                experiment=experiment, data=checked_cast(Data, experiment.lookup_data())
            ),
        )
    else:
        # Make sure the model is up-to-date with the most recent data.
        generation_strategy._set_or_update_current_model(data=None)

    # If objective thresholds are not specified in optimization config, extract
    # the inferred ones if possible or infer them anew if not.
    objective_thresholds_override = None
    if not moo_optimization_config.objective_thresholds:
        lgr = generation_strategy.last_generator_run
        if lgr and lgr.gen_metadata and "objective_thresholds" in lgr.gen_metadata:
            objective_thresholds_override = lgr.gen_metadata["objective_thresholds"]
        objective_thresholds_override = mb.infer_objective_thresholds(
            search_space=experiment.search_space,
            optimization_config=experiment.optimization_config,
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
        modelbridge=mb, objective_thresholds=objective_thresholds_override
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
) -> pd.DataFrame:
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
) -> pd.DataFrame:
    objective_name = objective.metric.name
    objective_rows = df.loc[df["metric_name"] == objective_name]
    if objective_rows.empty:
        raise ValueError(f'No data has been logged for objective "{objective_name}".')
    return (
        objective_rows.loc[objective_rows["mean"].idxmin()]
        if objective.minimize
        else objective_rows.loc[objective_rows["mean"].idxmax()]
    )
