#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Dict, Optional, Tuple

from ax.core.experiment import Experiment
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
    dat = experiment.fetch_data()
    if dat.df.empty:
        raise ValueError("Cannot identify best point if experiment contains no data.")
    opt_config = optimization_config or experiment.optimization_config
    objective_name = opt_config.objective.metric.name
    objective_rows = dat.df.loc[dat.df["metric_name"] == objective_name]
    if objective_rows.empty:
        raise ValueError('No data has been logged for objective "{objective_name}".')
    optimization_config = optimization_config or opt_config
    assert optimization_config is not None, (
        "Cannot identify the best point without an optimization config, but no "
        "optimization config was provided on the experiment or as an argument."
    )
    best_row = (
        objective_rows.loc[objective_rows["mean"].idxmin()]
        if opt_config.objective.minimize
        else objective_rows.loc[objective_rows["mean"].idxmax()]
    )
    best_arm = experiment.arms_by_name.get(best_row["arm_name"])
    objective_rows = dat.df.loc[
        (dat.df["arm_name"] == best_row["arm_name"])
        & (dat.df["trial_index"] == best_row["trial_index"])
    ]
    vals = {
        row["metric_name"]: (row["mean"], row["sem"])
        for _, row in objective_rows.iterrows()
    }
    return not_none(best_arm).parameters, vals


def get_best_from_model_predictions(
    experiment: Experiment
) -> Optional[Tuple[TParameterization, Optional[TModelPredictArm]]]:
    """Given an experiment, identifies the best arm according to the outputs
    of models used in optimization and its corresponding predictions if available.

    TModelPredictArm is of the form:
        ({metric_name: mean}, {metric_name_1: {metric_name_2: cov_1_2}})

    Args:
        experiment: Experiment, on which to identify best raw objective arm.

    Returns:
        Tuple of parameterization and model predictions for it.
    """
    for _, trial in sorted(
        list(experiment.trials.items()), key=lambda x: x[0], reverse=True
    ):
        if not isinstance(trial, Trial):
            logger.warn(
                "While looking for best arm based on model predictions, "
                "encontered a batch trial. Best arm model predictions are not "
                "supported for it yet, and will be disregarded."
            )
            continue
        gr = trial.generator_run
        if gr is not None and gr.best_arm_predictions is not None:
            best_arm, best_arm_predictions = gr.best_arm_predictions
            return not_none(best_arm).parameters, best_arm_predictions
    return None
