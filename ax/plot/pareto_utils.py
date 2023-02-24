#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from copy import deepcopy
from itertools import combinations
from logging import Logger
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import ScalarizedObjective
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import MultiObjectiveOptimizationConfig
from ax.core.outcome_constraint import (
    ComparisonOp,
    ObjectiveThreshold,
    OutcomeConstraint,
)
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.core.types import TParameterization
from ax.exceptions.core import AxError, UnsupportedError
from ax.modelbridge.modelbridge_utils import (
    _get_modelbridge_training_data,
    get_pareto_frontier_and_configs,
    observed_pareto_frontier,
)
from ax.modelbridge.registry import Models
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.choice_encode import OrderedChoiceEncode
from ax.modelbridge.transforms.derelativize import Derelativize
from ax.modelbridge.transforms.int_to_float import IntToFloat
from ax.modelbridge.transforms.one_hot import OneHot
from ax.modelbridge.transforms.remove_fixed import RemoveFixed
from ax.modelbridge.transforms.search_space_to_choice import SearchSpaceToChoice
from ax.models.torch.posterior_mean import get_PosteriorMean
from ax.models.torch_base import TorchModel
from ax.utils.common.logger import get_logger
from ax.utils.stats.statstools import relativize
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.hypervolume import infer_reference_point

# type aliases
Mu = Dict[str, List[float]]
Cov = Dict[str, Dict[str, List[float]]]


logger: Logger = get_logger(__name__)


def _extract_observed_pareto_2d(
    Y: np.ndarray,
    reference_point: Optional[Tuple[float, float]],
    minimize: Union[bool, Tuple[bool, bool]] = True,
) -> np.ndarray:
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

    param_dicts: List[TParameterization]
    means: Dict[str, List[float]]
    sems: Dict[str, List[float]]
    primary_metric: str
    secondary_metric: str
    absolute_metrics: List[str]
    objective_thresholds: Optional[Dict[str, float]]
    arm_names: Optional[List[Optional[str]]]


def get_observed_pareto_frontiers(
    experiment: Experiment,
    data: Optional[Data] = None,
    rel: bool = True,
    arm_names: Optional[List[str]] = None,
    is_pex: Optional[bool] = False,
) -> List[ParetoFrontierResults]:
    """
    Find all Pareto points from an experiment.

    Uses only values as observed in the data; no modeling is involved. Makes no
    assumption about the search space or types of parameters. If "data" is provided will
    use that, otherwise will use all data attached to the experiment.

    Uses all arms present in data; does not filter according to experiment
    search space. If arm_names is specified, will filter to just those arm whose names
    are given in the list.

    Assumes experiment has a multiobjective optimization config from which the
    objectives and outcome constraints will be extracted.

    Will generate a ParetoFrontierResults for every pair of metrics in the experiment's
    multiobjective optimization config.

    Args:
        experiment: The experiment.
        data: Data to use for computing Pareto frontier. If not provided, will fetch
            data from experiment.
        rel: Relativize, if status quo on experiment.
        arm_names: If provided, computes Pareto frontier only from among the provided
            list of arm names, plus status quo if set on experiment.

    Returns: ParetoFrontierResults that can be used with interact_pareto_frontier.
    """
    if data is None:
        data = experiment.fetch_data()
    if experiment.optimization_config is None:
        raise ValueError("Experiment must have an optimization config")
    if arm_names is not None:
        if (
            experiment.status_quo is not None
            and experiment.status_quo.name not in arm_names
        ):
            # Make sure status quo is always included, for derelativization
            arm_names.append(experiment.status_quo.name)
        data = Data(data.df[data.df["arm_name"].isin(arm_names)])
    if is_pex:
        mb = pex_get_tensor_converter_model(experiment=experiment, data=data)
    else:
        mb = get_tensor_converter_model(experiment=experiment, data=data)
    pareto_observations = observed_pareto_frontier(modelbridge=mb)
    # Convert to ParetoFrontierResults
    objective_metric_names = {
        metric.name
        for metric in experiment.optimization_config.objective.metrics  # pyre-ignore
    }
    obj_metr_list = sorted(objective_metric_names)
    pfr_means = {name: [] for name in obj_metr_list}
    pfr_sems = {name: [] for name in obj_metr_list}

    for obs in pareto_observations:
        for i, name in enumerate(obs.data.metric_names):
            if name in objective_metric_names:
                pfr_means[name].append(obs.data.means[i])
                pfr_sems[name].append(np.sqrt(obs.data.covariance[i, i]))

    # Relativize as needed
    if rel and experiment.status_quo is not None:
        # Get status quo values
        # pyre-fixme[16]: `Optional` has no attribute `name`.
        sq_df = data.df[data.df["arm_name"] == experiment.status_quo.name]
        sq_df = sq_df.to_dict(orient="list")
        sq_means = {}
        sq_sems = {}
        # pyre-fixme[6]: Expected `_SupportsIndex` for 1st param but got `str`.
        for i, metric in enumerate(sq_df["metric_name"]):
            # pyre-fixme[6]: Expected `_SupportsIndex` for 1st param but got `str`.
            sq_means[metric] = sq_df["mean"][i]
            # pyre-fixme[6]: Expected `_SupportsIndex` for 1st param but got `str`.
            sq_sems[metric] = sq_df["sem"][i]
        # Relativize
        for name in pfr_means:
            if np.isnan(sq_sems[name]) or np.isnan(pfr_sems[name]).any():
                # Just relativize means
                pfr_means[name] = [
                    (mu / sq_means[name] - 1) * 100 for mu in pfr_means[name]
                ]
            else:
                # Use delta method
                pfr_means[name], pfr_sems[name] = relativize(
                    means_t=pfr_means[name],
                    sems_t=pfr_sems[name],
                    mean_c=sq_means[name],
                    sem_c=sq_sems[name],
                    as_percent=True,
                )
        absolute_metrics = []
    else:
        absolute_metrics = obj_metr_list

    objective_thresholds = {}
    if experiment.optimization_config.objective_thresholds is not None:  # pyre-ignore
        for objth in experiment.optimization_config.objective_thresholds:
            is_rel = objth.metric.name not in absolute_metrics
            if objth.relative != is_rel:
                raise ValueError(
                    f"Objective threshold for {objth.metric.name} has "
                    f"rel={objth.relative} but was specified here as rel={is_rel}"
                )
            objective_thresholds[objth.metric.name] = objth.bound

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


def to_nonrobust_search_space(search_space: SearchSpace) -> SearchSpace:
    """Reduces a RobustSearchSpace to a SearchSpace.

    This is a no-op for all other search spaces.
    """
    if isinstance(search_space, RobustSearchSpace):
        return SearchSpace(
            parameters=[p.clone() for p in search_space._parameters.values()],
            parameter_constraints=[
                pc.clone() for pc in search_space._parameter_constraints
            ],
        )
    else:
        return search_space


# TODO: delete this function after merging with get_tensor_converter_model
def pex_get_tensor_converter_model(
    experiment: Experiment, data: Data
) -> TorchModelBridge:
    """
    Copy of get_tensor_converter_model which retains the old transforms for pex usage

    Args:
        experiment: Experiment.
        data: Data for fitting the model.

    Returns: A torch modelbridge with transforms set.
    """
    # Transforms is the minimal set that will work for converting any search
    # space to tensors.
    return TorchModelBridge(
        experiment=experiment,
        search_space=to_nonrobust_search_space(experiment.search_space),
        data=data,
        model=TorchModel(),
        transforms=[Derelativize, SearchSpaceToChoice, OrderedChoiceEncode, IntToFloat],
        transform_configs={
            "Derelativize": {"use_raw_status_quo": True},
            "SearchSpaceToChoice": {"use_ordered": True},
        },
        fit_out_of_design=True,
    )


def get_tensor_converter_model(experiment: Experiment, data: Data) -> TorchModelBridge:
    """
    Constructs a minimal model for converting things to tensors.

    Model fitting will instantiate all of the transforms but will not do any
    expensive (i.e. GP) fitting beyond that. The model will raise an error if
    it is used for predicting or generating.

    Will work for any search space regardless of types of parameters.

    Args:
        experiment: Experiment.
        data: Data for fitting the model.

    Returns: A torch modelbridge with transforms set.
    """
    # Transforms is the minimal set that will work for converting any search
    # space to tensors.
    return TorchModelBridge(
        experiment=experiment,
        search_space=to_nonrobust_search_space(experiment.search_space),
        data=data,
        model=TorchModel(),
        transforms=[RemoveFixed, OrderedChoiceEncode, OneHot, IntToFloat],
        fit_out_of_design=True,
    )


def compute_posterior_pareto_frontier(
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
        chebyshev: Whether to use augmented_chebyshev_scalarization
            when computing Pareto Frontier points.

    Returns:
        ParetoFrontierResults: A NamedTuple with fields listed in its definition.
    """
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
        objective_thresholds=None,
        arm_names=None,
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
    # pyre-fixme[2]: Parameter must be annotated.
    weights,
    # pyre-fixme[2]: Parameter must be annotated.
    primary_objective,
    # pyre-fixme[2]: Parameter must be annotated.
    secondary_objective,
    # pyre-fixme[2]: Parameter must be annotated.
    outcome_constraints=None,
) -> MultiObjectiveOptimizationConfig:
    obj = ScalarizedObjective(
        metrics=[primary_objective, secondary_objective],
        weights=weights,
        minimize=False,
    )
    optimization_config = MultiObjectiveOptimizationConfig(
        objective=obj, outcome_constraints=outcome_constraints
    )
    return optimization_config


def infer_reference_point_from_experiment(
    experiment: Experiment,
) -> List[ObjectiveThreshold]:
    """This functions is a wrapper around ``infer_reference_point`` to find the nadir
    point from the pareto front of an experiment. Aside from converting experiment
    to tensors, this wrapper transforms back and forth the objectives of the experiment
    so that they are appropriately used by ``infer_reference_point``.

    Args:
        experiment: The experiment for which we want to infer the reference point.

    Returns:
        List of obejective thresholds representing the reference point.
    """
    if not experiment.is_moo_problem:
        raise ValueError(
            "This function works for MOO experiments only."
            f" Experiment {experiment.name} is single objective."
        )

    # Reading experiment data.
    mb_reference = get_tensor_converter_model(
        experiment=experiment, data=experiment.fetch_data()
    )
    obs_feats, obs_data, _ = _get_modelbridge_training_data(modelbridge=mb_reference)

    # Since objectives could have arbitrary orders in objective_thresholds and
    # further down the road `get_pareto_frontier_and_configs` arbitrarily changes the
    # orders of the objectives, we fix the objective orders here based on the
    # observation_data and maintain it throughout the flow.
    objective_orders = obs_data[0].metric_names

    # Defining a dummy reference point so that all observed points are considered
    # when calculating the Pareto front. Also, defining a multiplier to turn all
    # the objectives to be maximized. Note that the multiplier at this point
    # contains 0 for outcome_constraint metrics, but this will be dropped later.
    dummy_rp = copy.deepcopy(
        experiment.optimization_config.objective_thresholds  # pyre-ignore
    )
    multiplier = [0] * len(objective_orders)
    for ot in dummy_rp:
        # In the following, we find the index of the objective in
        # `objective_orders`. If there is an objective that does not exist
        # in `obs_data`, a ValueError is raised.
        try:
            objective_index = objective_orders.index(ot.metric.name)
        except ValueError:
            raise ValueError(f"Metric {ot.metric.name} does not exist in `obs_data`.")

        if ot.op == ComparisonOp.LEQ:
            ot.bound = np.inf
            multiplier[objective_index] = -1
        else:
            ot.bound = -np.inf
            multiplier[objective_index] = 1

    # Finding the pareto frontier
    frontier_observations, f, obj_w, _ = get_pareto_frontier_and_configs(
        modelbridge=mb_reference,
        observation_features=obs_feats,
        observation_data=obs_data,
        objective_thresholds=dummy_rp,
        use_model_predictions=False,
    )

    # Need to reshuffle columns of `f` and `obj_w` to be consistent
    # with objective_orders.
    order = [
        objective_orders.index(metric_name)
        for metric_name in frontier_observations[0].data.metric_names
    ]
    f = f[:, order]
    obj_w = obj_w[order]

    # Dropping the columns related to outcome constraints.
    f = f[:, obj_w.nonzero().view(-1)]
    multiplier_tensor = torch.tensor(multiplier, dtype=f.dtype, device=f.device)
    multiplier_nonzero = multiplier_tensor[obj_w.nonzero().view(-1)]

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

    # Constructing the objective thresholds.
    nadir_objective_thresholds = copy.deepcopy(
        experiment.optimization_config.objective_thresholds
    )

    for obj_threshold in nadir_objective_thresholds:
        obj_threshold.bound = rp[
            objective_orders_reduced.index(obj_threshold.metric.name)
        ].item()

    return nadir_objective_thresholds
