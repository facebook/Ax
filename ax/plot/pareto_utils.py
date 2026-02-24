#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from copy import deepcopy
from itertools import combinations
from logging import Logger
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch
from ax.adapter.adapter_utils import observed_pareto_frontier
from ax.adapter.registry import Generators
from ax.adapter.transforms.derelativize import derelativize_bound
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import ScalarizedObjective
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import TParameterization
from ax.exceptions.core import AxError, UnsupportedError, UserInputError
from ax.service.utils.best_point import get_tensor_converter_adapter
from ax.utils.common.logger import get_logger
from ax.utils.stats.math_utils import relativize
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.utils.multi_objective import is_non_dominated

# type aliases
Mu = dict[str, list[float]]
Cov = dict[str, dict[str, list[float]]]


logger: Logger = get_logger(__name__)


def _extract_observed_pareto_2d(
    Y: npt.NDArray,
    reference_point: tuple[float, float] | None,
    minimize: bool | tuple[bool, bool] = True,
) -> npt.NDArray:
    if Y.shape[1] != 2:
        raise NotImplementedError("Currently only the 2-dim case is handled.")
    # If `minimize` is a bool, apply to both dimensions
    if isinstance(minimize, bool):
        minimize = (minimize, minimize)
    Y_copy = deepcopy(torch.from_numpy(Y).to())
    if reference_point:
        ref_point = torch.tensor(reference_point, dtype=Y_copy.dtype)
        for i in range(2):
            # Filter based on reference point
            Y_copy = (
                Y_copy[Y_copy[:, i] < ref_point[i]]
                if minimize[i]
                else Y_copy[Y_copy[:, i] > ref_point[i]]
            )
    for i in range(2):
        # Flip sign in each dimension based on minimize
        Y_copy[:, i] *= (-1) ** minimize[i]
    Y_pareto = Y_copy[is_non_dominated(Y_copy)]
    Y_pareto = Y_pareto[torch.argsort(input=Y_pareto[:, 0], descending=True)]
    for i in range(2):
        # Flip sign back
        Y_pareto[:, i] *= (-1) ** minimize[i]

    assert Y_pareto.shape[1] == 2  # Y_pareto should have two outcomes.
    return Y_pareto.detach().cpu().numpy()


class ParetoFrontierResults(NamedTuple):
    """Container for results from Pareto frontier computation.

    Fields are:
    - param_dicts: The parameter dicts of the points generated on the Pareto Frontier.
    - means: The posterior mean predictions of the model for each metric (same order as
    the param dicts). These must be as a percent change relative to status quo for
    any metric not listed in absolute_metrics.
    - sems: The posterior sem predictions of the model for each metric (same order as
    the param dicts). Also must be relativized wrt status quo for any metric not
    listed in absolute_metrics.
    - primary_metric: The name of the primary metric.
    - secondary_metric: The name of the secondary metric.
    - absolute_metrics: List of outcome metrics that are NOT be relativized w.r.t. the
    status quo. All other metrics are assumed to be given here as % relative to
    status_quo.
    - objective_thresholds: Threshold for each objective. Must be on the same scale as
    means, so if means is relativized it should be the relative value, otherwise it
    should be absolute.
    - arm_names: Optional list of arm names for each parameterization.
    """

    param_dicts: list[TParameterization]
    means: dict[str, list[float]]
    sems: dict[str, list[float]]
    primary_metric: str
    secondary_metric: str
    absolute_metrics: list[str]
    objective_thresholds: dict[str, float] | None
    arm_names: list[str | None] | None


def _extract_sq_data(
    experiment: Experiment, data: Data
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Returns sq_means and sq_sems, each a mapping from metric name to, respectively, mean
    and sem of the status quo arm. Empty dictionaries if no SQ arm.
    """
    sq_means = {}
    sq_sems = {}
    if experiment.status_quo is not None:
        # Extract SQ values
        sq_df = data.df[
            data.df["arm_name"] == experiment.status_quo.name  # pyre-ignore
        ]
        for metric, metric_df in sq_df.groupby("metric_name"):
            sq_means[metric] = metric_df["mean"].values[0]
            sq_sems[metric] = metric_df["sem"].values[0]
    return sq_means, sq_sems


def _relativize_values(
    means: list[float], sq_mean: float, sems: list[float], sq_sem: float
) -> tuple[list[float], list[float]]:
    """
    Relativize values, using delta method if SEMs provided, or just by relativizing
    means if not. Relativization is as percent.
    """
    if np.isnan(sq_sem) or np.isnan(sems).any():
        # Just relativize means
        means = [(mu / sq_mean - 1) * 100 for mu in means]
    else:
        # Use delta method
        means_arr, sems_arr = relativize(
            means_t=np.array(means),
            sems_t=np.array(sems),
            mean_c=sq_mean,
            sem_c=sq_sem,
            as_percent=True,
        )
        means, sems = means_arr.tolist(), sems_arr.tolist()
    return means, sems


def get_observed_pareto_frontiers(
    experiment: Experiment,
    data: Data | None = None,
    rel: bool | None = None,
    arm_names: list[str] | None = None,
) -> list[ParetoFrontierResults]:
    """
    Find all Pareto points from an experiment.

    Uses only values as observed in the data; no modeling is involved. Makes no
    assumption about the search space or types of parameters. If "data" is provided will
    use that, otherwise will use all data already attached to the experiment.

    Uses all arms present in data; does not filter according to experiment
    search space. If arm_names is specified, will filter to just those arm whose names
    are given in the list.

    Assumes experiment has a multiobjective optimization config from which the
    objectives and outcome constraints will be extracted.

    Will generate a ParetoFrontierResults for every pair of metrics in the experiment's
    multiobjective optimization config.

    Args:
        experiment: The experiment.
        data: Data to use for computing Pareto frontier. If not provided, will lookup
            data from experiment.
        rel: Relativize results wrt experiment status quo. If None, then rel will be
            taken for each objective separately from its own objective threshold.
            `rel` must be specified if there are missing objective thresholds.
        arm_names: If provided, computes Pareto frontier only from among the provided
            list of arm names, plus status quo if set on experiment.

    Returns: ParetoFrontierResults that can be used with interact_pareto_frontier.
    """
    if data is None:
        data = experiment.lookup_data()
    if experiment.optimization_config is None:
        raise ValueError("Experiment must have an optimization config")
    if arm_names is not None:
        if (
            experiment.status_quo is not None
            and experiment.status_quo.name not in arm_names
        ):
            # Make sure status quo is always included, for derelativization
            arm_names.append(experiment.status_quo.name)
        data = Data(df=data.df[data.df["arm_name"].isin(arm_names)])
    adapter = get_tensor_converter_adapter(experiment=experiment, data=data)
    pareto_observations = observed_pareto_frontier(adapter=adapter)
    # Convert to ParetoFrontierResults
    objective_metric_names = {
        metric.name
        for metric in experiment.optimization_config.objective.metrics  # pyre-ignore
    }
    obj_metr_list = sorted(objective_metric_names)
    pfr_means = {name: [] for name in obj_metr_list}
    pfr_sems = {name: [] for name in obj_metr_list}
    for obs in pareto_observations:
        obs_metric_names = []
        for signature in obs.data.metric_signatures:
            obs_metric_names.append(experiment.signature_to_metric[signature].name)
        for i, name in enumerate(obs_metric_names):
            if name in objective_metric_names:
                pfr_means[name].append(obs.data.means[i])
                pfr_sems[name].append(np.sqrt(obs.data.covariance[i, i]))

    # Get objective thresholds
    rel_objth = {}
    objective_thresholds = {}
    if experiment.optimization_config.objective_thresholds is not None:  # pyre-ignore
        for objth in experiment.optimization_config.objective_thresholds:
            rel_objth[objth.metric.signature] = objth.relative
            objective_thresholds[objth.metric.signature] = objth.bound

    # Identify which metrics should be relativized
    if rel in [True, False]:
        metric_is_rel = {name: rel for name in pfr_means}
    else:
        if len(rel_objth) != len(pfr_means):
            raise UserInputError(
                "At least one objective is missing an objective threshold. "
                "`rel` must be specified as True or False when there are missing "
                "objective thresholds."
            )
        # Default to however the threshold is specified
        metric_is_rel = rel_objth

    # Compute SQ values
    sq_means, sq_sems = _extract_sq_data(experiment, data)

    # Relativize data and thresholds as needed
    for name in pfr_means:
        if metric_is_rel[name]:
            pfr_means[name], pfr_sems[name] = _relativize_values(
                means=pfr_means[name],
                sq_mean=sq_means[name],
                sems=pfr_sems[name],
                sq_sem=sq_sems[name],
            )
            if name in objective_thresholds and not rel_objth[name]:
                # Metric is rel but obj th is not.
                # Need to relativize the objective threshold
                objective_thresholds[name] = _relativize_values(
                    means=[objective_thresholds[name]],
                    sq_mean=sq_means[name],
                    sems=[np.nan],
                    sq_sem=np.nan,
                )[0][0]
        elif name in objective_thresholds and rel_objth[name]:
            # Metric is not relative but objective threshold is, so we need to
            # derelativize the objective threshold.
            objective_thresholds[name] = derelativize_bound(
                bound=objective_thresholds[name], sq_val=sq_means[name]
            )

    absolute_metrics = [name for name, val in metric_is_rel.items() if not val]
    # Construct ParetoFrontResults for each pair
    pfr_list = []
    param_dicts = [obs.features.parameters for obs in pareto_observations]
    pfr_arm_names = [obs.arm_name for obs in pareto_observations]

    for metric_a, metric_b in combinations(obj_metr_list, 2):
        pfr_list.append(
            ParetoFrontierResults(
                param_dicts=param_dicts,
                means=pfr_means,
                sems=pfr_sems,
                primary_metric=metric_a,
                secondary_metric=metric_b,
                absolute_metrics=absolute_metrics,
                objective_thresholds=objective_thresholds,
                arm_names=pfr_arm_names,
            )
        )
    return pfr_list


def compute_posterior_pareto_frontier(
    experiment: Experiment,
    primary_objective: Metric,
    secondary_objective: Metric,
    data: Data | None = None,
    outcome_constraints: list[OutcomeConstraint] | None = None,
    absolute_metrics: list[str] | None = None,
    num_points: int = 10,
    trial_index: int | None = None,
) -> ParetoFrontierResults:
    """Compute the Pareto frontier between two objectives. For experiments
    with batch trials, a trial index or data object must be provided.

    This is done by fitting a GP and finding the pareto front according to the
    GP posterior mean.

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

    Returns:
        ParetoFrontierResults: A NamedTuple with fields listed in its definition.
    """
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
                if trial_index is not None
                else experiment.fetch_data()
            )
        except Exception as e:
            warnings.warn(
                f"Could not fetch data from experiment or trial: {e}", stacklevel=2
            )

    # The weights here are just dummy weights that we pass in to construct the
    # adapter. We set the weight to -1 if `lower_is_better` is `True` and
    # 1 otherwise. This code would benefit from a serious revamp.
    oc = _build_scalarized_optimization_config(
        weights=np.array(
            [
                -1 if primary_objective.lower_is_better else 1,
                -1 if secondary_objective.lower_is_better else 1,
            ]
        ),
        primary_objective=primary_objective,
        secondary_objective=secondary_objective,
        outcome_constraints=outcome_constraints,
    )
    model = Generators.BOTORCH_MODULAR(
        experiment=experiment,
        data=data,
        optimization_config=oc,
        botorch_acqf_class=qSimpleRegret,
    )

    status_quo = experiment.status_quo
    if status_quo:
        try:
            status_quo_prediction = model.predict(
                [
                    ObservationFeatures(
                        parameters=status_quo.parameters,
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

    param_dicts: list[TParameterization] = []

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
        oc = _build_scalarized_optimization_config(
            weights=weights,
            primary_objective=primary_objective,
            secondary_objective=secondary_objective,
            outcome_constraints=outcome_constraints,
        )
        run = model.gen(1, optimization_config=oc)
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
    param_dicts: list[TParameterization],
    means: Mu,
    variances: Cov,
    primary_metric: str,
    secondary_metric: str,
    absolute_metrics: list[str],
    outcome_constraints: list[OutcomeConstraint] | None,
    status_quo_prediction: tuple[Mu, Cov] | None,
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
                # pyre-fixme[6]: For 2nd argument expected `List[float]` but got
                #  `ndarray[typing.Any, typing.Any]`.
                means_out[metric], sems_out[metric] = relativize(
                    means_t=means_out[metric],
                    sems_t=sems_out[metric],
                    mean_c=sq_mean[metric][0],
                    sem_c=np.sqrt(sq_sem[metric][metric][0]),
                    as_percent=True,
                )

    return ParetoFrontierResults(
        param_dicts=param_dicts,
        means=means_out,
        # pyre-fixme[6]: For 3rd argument expected `Dict[str, List[float]]` but got
        #  `Dict[str, ndarray[typing.Any, dtype[typing.Any]]]`.
        sems=sems_out,
        primary_metric=primary_metric,
        secondary_metric=secondary_metric,
        absolute_metrics=absolute_metrics,
        objective_thresholds=None,
        arm_names=None,
    )


def _validate_outcome_constraints(
    outcome_constraints: list[OutcomeConstraint],
    primary_objective: Metric,
    secondary_objective: Metric,
) -> None:
    """Validate that outcome constraints don't involve objectives."""
    objective_metrics = [primary_objective.name, secondary_objective.name]
    if outcome_constraints is not None:
        for oc in outcome_constraints:
            if oc.metric.signature in objective_metrics:
                raise ValueError(
                    "Metric `{metric_name}` occurs in both outcome constraints "
                    "and objectives".format(metric_name=oc.metric.signature)
                )


def _build_scalarized_optimization_config(
    weights: npt.NDArray,
    primary_objective: Metric,
    secondary_objective: Metric,
    outcome_constraints: list[OutcomeConstraint] | None = None,
) -> MultiObjectiveOptimizationConfig:
    obj = ScalarizedObjective(
        metrics=[primary_objective, secondary_objective],
        weights=weights.tolist(),
        minimize=False,
    )
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=obj, outcome_constraints=outcome_constraints
    )
    return optimization_config
