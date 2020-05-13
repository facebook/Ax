#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import pandas as pd
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.objective import Objective, ScalarizedObjective
from ax.core.optimization_config import OptimizationConfig
from ax.core.trial import Trial
from ax.core.types import TModelPredictArm, TParameterization
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none


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
) -> Optional[Tuple[TParameterization, Optional[TModelPredictArm]]]:
    """Given an experiment, identifies the best arm.

    First attempts according to do so with models used in optimization and
    its corresponding predictions if available. Falls back to the best raw
    objective based on the data fetched from the experiment.

    TModelPredictArm is of the form:
        ({metric_name: mean}, {metric_name_1: {metric_name_2: cov_1_2}})

    Args:
        experiment: Experiment, on which to identify best raw objective arm.

    Returns:
        Tuple of parameterization and model predictions for it.
    """

    # Find latest trial which has a generator_run attached and get its predictions
    model_predictions = get_best_from_model_predictions(experiment=experiment)
    if model_predictions is not None:  # pragma: no cover
        return model_predictions

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


def _get_best_row_for_scalarized_objective(
    df: pd.DataFrame, objective: ScalarizedObjective
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
