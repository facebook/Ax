#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import ScalarizedObjective
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import TParameterization
from ax.exceptions.core import AxError, UnsupportedError
from ax.modelbridge.registry import Models
from ax.models.torch.posterior_mean import get_PosteriorMean
from ax.utils.common.logger import get_logger
from ax.utils.stats.statstools import relativize


# type aliases
Mu = Dict[str, List[float]]
Cov = Dict[str, Dict[str, List[float]]]


class COLORS(enum.Enum):
    STEELBLUE = (128, 177, 211)
    CORAL = (251, 128, 114)
    TEAL = (141, 211, 199)
    PINK = (188, 128, 189)
    LIGHT_PURPLE = (190, 186, 218)
    ORANGE = (253, 180, 98)


logger = get_logger(__name__)


def rgba(rgb_tuple: Tuple[float], alpha: float = 1) -> str:
    """Convert RGB tuple to an RGBA string."""
    return "rgba({},{},{},{alpha})".format(*rgb_tuple, alpha=alpha)


class ParetoFrontierResults(NamedTuple):
    """Container for results from Pareto frontier computation."""

    param_dicts: List[TParameterization]
    means: Dict[str, List[float]]
    sems: Dict[str, List[float]]
    primary_metric: str
    secondary_metric: str
    absolute_metrics: List[str]
    outcome_constraints: Optional[List[OutcomeConstraint]]


def compute_pareto_frontier(
    experiment: Experiment,
    primary_objective: Metric,
    secondary_objective: Metric,
    data: Optional[Data] = None,
    outcome_constraints: Optional[List[OutcomeConstraint]] = None,
    absolute_metrics: Optional[List[str]] = None,
    num_points: int = 10,
    trial_index: Optional[int] = None,
    chebyshev: bool = True,
) -> ParetoFrontierResults:
    """Compute the Pareto frontier between two objectives. For experiments
    with batch trials, a trial index or data object must be provided.

    Args:
        experiment: The experiment to compute a pareto frontier for.
        primary_objective: The primary objective to optimize.
        secondary_objective: The secondary objective against which
            to trade off the primary objective.
        outcome_constraints: Outcome
            constraints to be respected by the optimization. Can only contain
            constraints on metrics that are not primary or secondary objectives.
        absolute_metrics: List of outcome metrics that
            should NOT be relativized w.r.t. the status quo (all other outcomes
            will be in % relative to status_quo).
        num_points: The number of points to compute on the
            Pareto frontier.
        chebyshev: Whether to use augmented_chebyshev_scalarization
            when computing Pareto Frontier points.

    Returns:
        ParetoFrontierResults: A NamedTuple with the following fields:
            - param_dicts: The parameter dicts of the
                points generated on the Pareto Frontier.
            - means: The posterior mean predictions of
                the model for each metric (same order as the param dicts).
            - sems: The posterior sem predictions of
                the model for each metric (same order as the param dicts).
            - primary_metric: The name of the primary metric.
            - secondary_metric: The name of the secondary metric.
            - absolute_metrics: List of outcome metrics that
                are NOT be relativized w.r.t. the status quo (all other metrics
                are in % relative to status_quo).
    """
    # TODO(jej): Implement using MultiObjectiveTorchModelBridge's _pareto_frontier
    model_gen_options = {
        "acquisition_function_kwargs": {"chebyshev_scalarization": chebyshev}
    }

    if (
        trial_index is None
        and data is None
        and any(isinstance(t, BatchTrial) for t in experiment.trials.values())
    ):
        raise UnsupportedError(
            "Must specify trial index or data for experiment with batch trials"
        )
    absolute_metrics = [] if absolute_metrics is None else absolute_metrics
    for metric in absolute_metrics:
        if metric not in experiment.metrics:
            raise ValueError(f"Model was not fit on metric `{metric}`")

    if outcome_constraints is None:
        outcome_constraints = []
    else:
        # ensure we don't constrain an objective
        _validate_outcome_constraints(
            outcome_constraints=outcome_constraints,
            primary_objective=primary_objective,
            secondary_objective=secondary_objective,
        )

    # build posterior mean model
    if not data:
        try:
            data = (
                experiment.trials[trial_index].fetch_data()
                if trial_index
                else experiment.fetch_data()
            )
        except Exception as e:
            logger.info(f"Could not fetch data from experiment or trial: {e}")

    oc = _build_new_optimization_config(
        weights=np.array([0.5, 0.5]),
        primary_objective=primary_objective,
        secondary_objective=secondary_objective,
        outcome_constraints=outcome_constraints,
    )
    model = Models.MOO(
        experiment=experiment,
        data=data,
        acqf_constructor=get_PosteriorMean,
        optimization_config=oc,
    )

    status_quo = experiment.status_quo
    if status_quo:
        try:
            status_quo_prediction = model.predict(
                [
                    ObservationFeatures(
                        parameters=status_quo.parameters,
                        # pyre-fixme [6]: Expected `Optional[np.int64]` for trial_index
                        trial_index=trial_index,
                    )
                ]
            )
        except ValueError as e:
            logger.warning(f"Could not predict OOD status_quo outcomes: {e}")
            status_quo = None
            status_quo_prediction = None
    else:
        status_quo_prediction = None

    param_dicts: List[TParameterization] = []

    # Construct weightings with linear angular spacing.
    # TODO: Verify whether 0, 1 weights cause problems because of subset_model.
    alpha = np.linspace(0 + 0.01, np.pi / 2 - 0.01, num_points)
    primary_weight = (-1 if primary_objective.lower_is_better else 1) * np.cos(alpha)
    secondary_weight = (-1 if secondary_objective.lower_is_better else 1) * np.sin(
        alpha
    )
    weights_list = np.stack([primary_weight, secondary_weight]).transpose()
    for weights in weights_list:
        outcome_constraints = outcome_constraints
        oc = _build_new_optimization_config(
            weights=weights,
            primary_objective=primary_objective,
            secondary_objective=secondary_objective,
            outcome_constraints=outcome_constraints,
        )
        # TODO: (jej) T64002590 Let this serve as a starting point for optimization.
        # ex. Add global spacing criterion. Implement on BoTorch side.
        # pyre-fixme [6]: Expected different type for model_gen_options
        run = model.gen(1, model_gen_options=model_gen_options, optimization_config=oc)
        param_dicts.append(run.arms[0].parameters)

    # Call predict on points to get their decomposed metrics.
    means, cov = model.predict(
        [ObservationFeatures(parameters) for parameters in param_dicts]
    )

    return _extract_pareto_frontier_results(
        param_dicts=param_dicts,
        means=means,
        variances=cov,
        primary_metric=primary_objective.name,
        secondary_metric=secondary_objective.name,
        absolute_metrics=absolute_metrics,
        outcome_constraints=outcome_constraints,
        status_quo_prediction=status_quo_prediction,
    )


def _extract_pareto_frontier_results(
    param_dicts: List[TParameterization],
    means: Mu,
    variances: Cov,
    primary_metric: str,
    secondary_metric: str,
    absolute_metrics: List[str],
    outcome_constraints: Optional[List[OutcomeConstraint]],
    status_quo_prediction: Optional[Tuple[Mu, Cov]],
) -> ParetoFrontierResults:
    """Extract prediction results into ParetoFrontierResults struture."""
    metrics = list(means.keys())
    means_out = {metric: m.copy() for metric, m in means.items()}
    sems_out = {metric: np.sqrt(v[metric]) for metric, v in variances.items()}

    # relativize predicted outcomes if requested
    primary_is_relative = primary_metric not in absolute_metrics
    secondary_is_relative = secondary_metric not in absolute_metrics
    # Relativized metrics require a status quo prediction
    if primary_is_relative or secondary_is_relative:
        if status_quo_prediction is None:
            raise AxError("Relativized metrics require a valid status quo prediction")
        sq_mean, sq_sem = status_quo_prediction

        for metric in metrics:
            if metric not in absolute_metrics and metric in sq_mean:
                means_out[metric], sems_out[metric] = relativize(
                    means_t=means_out[metric],
                    sems_t=sems_out[metric],
                    mean_c=sq_mean[metric][0],
                    sem_c=np.sqrt(sq_sem[metric][metric][0]),
                    as_percent=True,
                )

    return ParetoFrontierResults(
        param_dicts=param_dicts,
        means={metric: means for metric, means in means_out.items()},
        sems={metric: sems for metric, sems in sems_out.items()},
        primary_metric=primary_metric,
        secondary_metric=secondary_metric,
        absolute_metrics=absolute_metrics,
        outcome_constraints=outcome_constraints,
    )


def _validate_outcome_constraints(
    outcome_constraints: List[OutcomeConstraint],
    primary_objective: Metric,
    secondary_objective: Metric,
) -> None:
    """Validate that outcome constraints don't involve objectives."""
    objective_metrics = [primary_objective.name, secondary_objective.name]
    if outcome_constraints is not None:
        for oc in outcome_constraints:
            if oc.metric.name in objective_metrics:
                raise ValueError(
                    "Metric `{metric_name}` occurs in both outcome constraints "
                    "and objectives".format(metric_name=oc.metric.name)
                )


def _build_new_optimization_config(
    weights, primary_objective, secondary_objective, outcome_constraints=None
):
    obj = ScalarizedObjective(
        metrics=[primary_objective, secondary_objective],
        weights=weights,
        minimize=False,
    )
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=obj, outcome_constraints=outcome_constraints
    )
    return optimization_config
