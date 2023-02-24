#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings

from collections import defaultdict
from copy import deepcopy
from functools import partial

from logging import Logger
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

import numpy as np
import torch
from ax.core.base_trial import TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
    TRefPoint,
)
from ax.core.outcome_constraint import (
    ComparisonOp,
    OutcomeConstraint,
    ScalarizedOutcomeConstraint,
)
from ax.core.parameter import ChoiceParameter, Parameter, ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.risk_measures import RiskMeasure
from ax.core.search_space import (
    RobustSearchSpace,
    RobustSearchSpaceDigest,
    SearchSpace,
    SearchSpaceDigest,
)
from ax.core.trial import Trial
from ax.core.types import TBounds, TCandidateMetadata
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.utils import (
    derelativize_optimization_config_with_raw_status_quo,
)
from ax.models.torch.botorch_moo_defaults import pareto_frontier_evaluator
from ax.models.torch.frontier_utils import (
    get_weighted_mc_objective_and_objective_thresholds,
)
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import (
    checked_cast,
    checked_cast_optional,
    checked_cast_to_tuple,
    not_none,
)
from botorch.acquisition.multi_objective.multi_output_risk_measures import (
    IndependentCVaR,
    IndependentVaR,
    MARS,
    MultiOutputExpectation,
    MVaR,
)
from botorch.acquisition.risk_measures import (
    CVaR,
    Expectation,
    RiskMeasureMCObjective,
    VaR,
    WorstCase,
)
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from torch import Tensor

logger: Logger = get_logger(__name__)


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401  # pragma: no cover


"""A mapping of risk measure names to the corresponding classes.

NOTE: This can be extended with user-defined risk measure classes by
importing the dictionary and adding the new risk measure class as
`RISK_MEASURE_NAME_TO_CLASS["my_risk_measure"] = MyRiskMeasure`.
An example of this is found in `tests/test_risk_measure`.
"""
RISK_MEASURE_NAME_TO_CLASS: Dict[str, Type[RiskMeasureMCObjective]] = {
    "Expectation": Expectation,
    "CVaR": CVaR,
    "MARS": MARS,
    "MVaR": MVaR,
    "IndependentCVaR": IndependentCVaR,
    "IndependentVaR": IndependentVaR,
    "MultiOutputExpectation": MultiOutputExpectation,
    "VaR": VaR,
    "WorstCase": WorstCase,
}


def extract_risk_measure(risk_measure: RiskMeasure) -> RiskMeasureMCObjective:
    r"""Extracts the BoTorch risk measure objective from an Ax `RiskMeasure`.

    Args:
        risk_measure: The RiskMeasure object.

    Returns:
        The corresponding `RiskMeasureMCObjective` object.
    """
    try:
        risk_measure_class = RISK_MEASURE_NAME_TO_CLASS[risk_measure.risk_measure]
        # Add dummy chebyshev weights to initialize MARS.
        additional_options = (
            {"chebyshev_weights": []} if risk_measure_class is MARS else {}
        )
        return risk_measure_class(
            # pyre-ignore Incompatible parameter type [6]
            **risk_measure.options,
            **additional_options,
        )
    except (KeyError, RuntimeError, ValueError):
        raise UserInputError(
            "Got an error while constructing the risk measure. Make sure that "
            f"{risk_measure.risk_measure} exists in  `RISK_MEASURE_NAME_TO_CLASS` "
            f"and accepts arguments {risk_measure.options}."
        )


def check_has_multi_objective_and_data(
    experiment: Experiment,
    data: Data,
    optimization_config: Optional[OptimizationConfig] = None,
) -> None:
    """Raise an error if not using a `MultiObjective` or if the data is empty."""
    optimization_config = not_none(
        optimization_config or experiment.optimization_config
    )
    if not isinstance(optimization_config.objective, MultiObjective):
        raise ValueError("Multi-objective optimization requires multiple objectives.")
    if data.df.empty:
        raise ValueError("MultiObjectiveOptimization requires non-empty data.")


def extract_parameter_constraints(
    parameter_constraints: List[ParameterConstraint], param_names: List[str]
) -> TBounds:
    """Extract parameter constraints."""
    if len(parameter_constraints) == 0:
        return None
    A = np.zeros((len(parameter_constraints), len(param_names)))
    b = np.zeros((len(parameter_constraints), 1))
    for i, c in enumerate(parameter_constraints):
        b[i, 0] = c.bound
        for name, val in c.constraint_dict.items():
            A[i, param_names.index(name)] = val
    return (A, b)


def extract_search_space_digest(
    search_space: SearchSpace, param_names: List[str]
) -> SearchSpaceDigest:
    """Extract basic parameter properties from a search space."""
    bounds: List[Tuple[Union[int, float], Union[int, float]]] = []
    ordinal_features: List[int] = []
    categorical_features: List[int] = []
    discrete_choices: Dict[int, List[Union[int, float]]] = {}
    task_features: List[int] = []
    fidelity_features: List[int] = []
    target_fidelities: Dict[int, Union[int, float]] = {}

    for i, p_name in enumerate(param_names):
        p = search_space.parameters[p_name]
        if isinstance(p, ChoiceParameter):
            if p.is_task:
                task_features.append(i)
            elif p.is_ordered:  # pragma: no cover
                ordinal_features.append(i)  # pragma: no cover
            else:  # pragma: no cover
                categorical_features.append(i)  # pragma: no cover
            # at this point we can assume that values are numeric due to transforms
            discrete_choices[i] = p.values  # pyre-ignore [6]
            bounds.append((min(p.values), max(p.values)))  # pyre-ignore [6]
        elif isinstance(p, RangeParameter):
            if p.log_scale:
                raise ValueError(f"{p} is log scale")  # pragma: no cover
            if p.parameter_type == ParameterType.INT:
                ordinal_features.append(i)  # pragma: no cover
                d_choices = list(  # pragma: no cover
                    range(int(p.lower), int(p.upper) + 1)
                )
                # pyre-ignore [6]
                discrete_choices[i] = d_choices  # pragma: no cover
            bounds.append((p.lower, p.upper))
        else:
            raise ValueError(f"Unknown parameter type {type(p)}")  # pragma: no cover
        if p.is_fidelity:
            if not isinstance(not_none(p.target_value), (int, float)):
                raise NotImplementedError(  # pragma: no cover
                    "Only numerical target values are supported."
                )
            target_fidelities[i] = checked_cast_to_tuple((int, float), p.target_value)
            fidelity_features.append(i)

    return SearchSpaceDigest(
        feature_names=param_names,
        bounds=bounds,
        ordinal_features=ordinal_features,
        categorical_features=categorical_features,
        discrete_choices=discrete_choices,
        task_features=task_features,
        fidelity_features=fidelity_features,
        target_fidelities=target_fidelities,
        robust_digest=extract_robust_digest(
            search_space=search_space, param_names=param_names
        ),
    )


def extract_robust_digest(
    search_space: SearchSpace, param_names: List[str]
) -> Optional[RobustSearchSpaceDigest]:
    """Extracts the `RobustSearchSpaceDigest`.

    Args:
        search_space: A `SearchSpace` to digest.
        param_names: A list of names of the parameters that are used in optimization.
            If environmental variables are present, these should be the last entries
            in `param_names`.

    Returns:
        If the `search_space` is not a `RobustSearchSpace`, this returns None.
        Otherwise, it returns a `RobustSearchSpaceDigest` with entries populated
        from the properties of the `search_space`. In particular, this constructs
        two optional callables, `sample_param_perturbations` and `sample_environmental`,
        that require no inputs and return a `num_samples x d`-dim array of samples
        from the corresponding parameter distributions, where `d` is the number of
        environmental variables for `environmental_sampler and the number of
        non-environmental parameters in `param_names` for `distribution_sampler`.
    """
    if not isinstance(search_space, RobustSearchSpace):
        return None
    dist_params = search_space._distributional_parameters
    env_vars: Dict[str, Parameter] = search_space._environmental_variables
    pert_params = [p for p in dist_params if p not in env_vars]
    # Make sure all distributional parameters are in param_names.
    dist_idcs: Dict[str, int] = {}
    for p_name in dist_params:
        if p_name not in param_names:
            raise RuntimeError(
                "All distributional parameters must be included in `param_names`."
            )
        dist_idcs[p_name] = param_names.index(p_name)
    num_samples: int = search_space.num_samples
    if len(env_vars) > 0:
        num_non_env_vars: int = len(param_names) - len(env_vars)
        env_idcs = {idx for p, idx in dist_idcs.items() if p in env_vars}
        if env_idcs != set(range(num_non_env_vars, len(param_names))):
            raise RuntimeError(
                "Environmental variables must be last entries in `param_names`. "
                "Otherwise, `AppendFeatures` will not work."
            )
        # NOTE: Extracting it from `param_names` in case the ordering is different.
        environmental_variables = param_names[num_non_env_vars:]

        def sample_environmental() -> np.ndarray:
            """Get samples from the environmental distributions.

            Samples have the same dimension as the number of environmental variables.
            The samples of an environmental variable appears in the same order it is
            in `param_names`.
            """
            samples = np.zeros((num_samples, len(env_vars)))
            # pyre-ignore [16]
            for dist in search_space._environmental_distributions:
                dist_samples = dist.distribution.rvs(num_samples).reshape(
                    num_samples, -1
                )
                for i, p_name in enumerate(dist.parameters):
                    target_idx = dist_idcs[p_name] - num_non_env_vars
                    samples[:, target_idx] = dist_samples[:, i]
            return samples

    else:
        sample_environmental = None
        environmental_variables = []

    if len(pert_params) > 0:
        constructor: Callable[[Tuple[int, int]], np.ndarray] = (
            np.ones if search_space.multiplicative else np.zeros
        )

        def sample_param_perturbations() -> np.ndarray:
            """Get samples of the input perturbations.

            Samples have the same dimension as the length of `param_names`
            minus the number of environmental variables. The samples of a
            parameter appears in the same order it is in `param_names`. For
            non-distributional parameters, their values are filled as 0 if
            the perturbations are additive and 1 if multiplicative.
            """
            samples = constructor((num_samples, len(param_names) - len(env_vars)))
            # pyre-ignore [16]
            for dist in search_space._perturbation_distributions:
                dist_samples = dist.distribution.rvs(num_samples).reshape(
                    num_samples, -1
                )
                for i, p_name in enumerate(dist.parameters):
                    samples[:, dist_idcs[p_name]] = dist_samples[:, i]
            return samples

    else:
        sample_param_perturbations = None

    return RobustSearchSpaceDigest(
        sample_param_perturbations=sample_param_perturbations,
        sample_environmental=sample_environmental,
        environmental_variables=environmental_variables,
        multiplicative=search_space.multiplicative,
    )


def extract_objective_thresholds(
    objective_thresholds: TRefPoint,
    objective: Objective,
    outcomes: List[str],
) -> Optional[np.ndarray]:
    """Extracts objective thresholds' values, in the order of `outcomes`.

    Will return None if no objective thresholds, otherwise the extracted tensor
    will be the same length as `outcomes`.

    Outcomes that are not part of an objective and the objectives that do no have
    a corresponding objective threshold will be given a threshold of NaN. We will
    later infer appropriate threshold values for the objectives that are given a
    threshold of NaN.

    Args:
        objective_thresholds: Objective thresholds to extract values from.
        objective: The corresponding Objective, for validation purposes.
        outcomes: n-length list of names of metrics.

    Returns:
        (n,) array of thresholds
    """
    if len(objective_thresholds) == 0:
        return None

    objective_threshold_dict = {}
    for ot in objective_thresholds:
        if ot.relative:
            raise ValueError(
                f"Objective {ot.metric.name} has a relative threshold that is not "
                f"supported here."
            )
        objective_threshold_dict[ot.metric.name] = ot.bound

    # Check that all thresholds correspond to a metric.
    if set(objective_threshold_dict.keys()).difference(set(objective.metric_names)):
        raise ValueError(
            "Some objective thresholds do not have corresponding metrics."
            f"Got {objective_thresholds=} and {objective=}."
        )

    # Initialize these to be NaN to make sure that objective thresholds for
    # non-objective metrics are never used.
    obj_t = np.full(len(outcomes), float("nan"))
    for metric, threshold in objective_threshold_dict.items():
        obj_t[outcomes.index(metric)] = threshold
    return obj_t


def extract_objective_weights(objective: Objective, outcomes: List[str]) -> np.ndarray:
    """Extract a weights for objectives.

    Weights are for a maximization problem.

    Give an objective weight to each modeled outcome. Outcomes that are modeled
    but not part of the objective get weight 0.

    In the single metric case, the objective is given either +/- 1, depending
    on the minimize flag.

    In the multiple metric case, each objective is given the input weight,
    multiplied by the minimize flag.

    Args:
        objective: Objective to extract weights from.
        outcomes: n-length list of names of metrics.

    Returns:
        n-length array of weights.

    """
    objective_weights = np.zeros(len(outcomes))
    if isinstance(objective, ScalarizedObjective):
        s = -1.0 if objective.minimize else 1.0  # pragma: no cover
        for obj_metric, obj_weight in objective.metric_weights:  # pragma: no cover
            objective_weights[outcomes.index(obj_metric.name)] = (  # pragma: no cover
                obj_weight * s
            )
    elif isinstance(objective, MultiObjective):
        for obj, obj_weight in objective.objective_weights:
            s = -1.0 if obj.minimize else 1.0
            objective_weights[outcomes.index(obj.metric.name)] = obj_weight * s
    else:
        s = -1.0 if objective.minimize else 1.0
        objective_weights[outcomes.index(objective.metric.name)] = s
    return objective_weights


def extract_outcome_constraints(
    outcome_constraints: List[OutcomeConstraint], outcomes: List[str]
) -> TBounds:
    if len(outcome_constraints) == 0:
        return None
    # Extract outcome constraints
    A = np.zeros((len(outcome_constraints), len(outcomes)))
    b = np.zeros((len(outcome_constraints), 1))
    for i, c in enumerate(outcome_constraints):
        s = 1 if c.op == ComparisonOp.LEQ else -1
        if isinstance(c, ScalarizedOutcomeConstraint):
            for c_metric, c_weight in c.metric_weights:
                j = outcomes.index(c_metric.name)
                A[i, j] = s * c_weight
        else:
            j = outcomes.index(c.metric.name)
            A[i, j] = s
        b[i, 0] = s * c.bound
    return (A, b)


def validate_and_apply_final_transform(
    objective_weights: np.ndarray,
    outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]],
    linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]],
    pending_observations: Optional[List[np.ndarray]],
    objective_thresholds: Optional[np.ndarray] = None,
    final_transform: Callable[[np.ndarray], Tensor] = torch.tensor,
) -> Tuple[
    Tensor,
    Optional[Tuple[Tensor, Tensor]],
    Optional[Tuple[Tensor, Tensor]],
    Optional[List[Tensor]],
    Optional[Tensor],
]:
    # TODO: use some container down the road (similar to
    # SearchSpaceDigest) to limit the return arguments
    # pyre-fixme[35]: Target cannot be annotated.
    objective_weights: Tensor = final_transform(objective_weights)
    if outcome_constraints is not None:  # pragma: no cover
        # pyre-fixme[35]: Target cannot be annotated.
        outcome_constraints: Tuple[Tensor, Tensor] = (
            final_transform(outcome_constraints[0]),
            final_transform(outcome_constraints[1]),
        )
    if linear_constraints is not None:  # pragma: no cover
        # pyre-fixme[35]: Target cannot be annotated.
        linear_constraints: Tuple[Tensor, Tensor] = (
            final_transform(linear_constraints[0]),
            final_transform(linear_constraints[1]),
        )
    if pending_observations is not None:  # pragma: no cover
        # pyre-fixme[35]: Target cannot be annotated.
        pending_observations: List[Tensor] = [
            final_transform(pending_obs) for pending_obs in pending_observations
        ]
    if objective_thresholds is not None:
        # pyre-fixme[35]: Target cannot be annotated.
        objective_thresholds: Tensor = final_transform(objective_thresholds)
    return (
        objective_weights,
        outcome_constraints,
        linear_constraints,
        pending_observations,
        objective_thresholds,
    )


def get_fixed_features(
    fixed_features: Optional[ObservationFeatures], param_names: List[str]
) -> Optional[Dict[int, float]]:
    """Reformat a set of fixed_features."""
    if fixed_features is None:
        return None
    fixed_features_dict = {}
    for p_name, val in fixed_features.parameters.items():
        # These all need to be floats at this point.
        # pyre-ignore[6]: All float here.
        val_ = float(val)
        fixed_features_dict[param_names.index(p_name)] = val_
    fixed_features_dict = fixed_features_dict if len(fixed_features_dict) > 0 else None
    return fixed_features_dict


def pending_observations_as_array_list(
    pending_observations: Dict[str, List[ObservationFeatures]],
    outcome_names: List[str],
    param_names: List[str],
) -> Optional[List[np.ndarray]]:
    """Re-format pending observations.

    Args:
        pending_observations: List of raw numpy pending observations.
        outcome_names: List of outcome names.
        param_names: List fitted param names.

    Returns:
        Filtered pending observations data, by outcome and param names.
    """
    if len(pending_observations) == 0:
        return None

    pending = [np.array([]) for _ in outcome_names]
    for metric_name, po_list in pending_observations.items():
        # It is possible that some metrics attached to the experiment should
        # not be included in pending features for a given model. For example,
        # if a model is fit to the initial data that is missing some of the
        # metrics on the experiment or if a model just should not be fit for
        # some of the metrics attached to the experiment, so metrics that
        # appear in pending_observations (drawn from an experiment) but not
        # in outcome_names (metrics, expected for the model) are filtered out.
        if metric_name not in outcome_names:
            continue
        pending[outcome_names.index(metric_name)] = np.array(
            [[po.parameters[p] for p in param_names] for po in po_list]
        )
    return pending


def parse_observation_features(
    X: np.ndarray,
    param_names: List[str],
    candidate_metadata: Optional[List[TCandidateMetadata]] = None,
) -> List[ObservationFeatures]:
    """Re-format raw model-generated candidates into ObservationFeatures.

    Args:
        param_names: List of param names.
        X: Raw np.ndarray of candidate values.
        candidate_metadata: Model's metadata for candidates it produced.

    Returns:
        List of candidates, represented as ObservationFeatures.
    """
    if candidate_metadata and len(candidate_metadata) != len(X):
        raise ValueError(  # pragma: no cover
            "Observations metadata list provided is not of "
            "the same size as the number of candidates."
        )
    observation_features = []
    for i, x in enumerate(X):
        observation_features.append(
            ObservationFeatures(
                parameters=dict(zip(param_names, x)),
                metadata=candidate_metadata[i] if candidate_metadata else None,
            )
        )
    return observation_features


def transform_callback(
    param_names: List[str], transforms: MutableMapping[str, Transform]
) -> Callable[[np.ndarray], np.ndarray]:
    """A closure for performing the `round trip` transformations.

    The function round points by de-transforming points back into
    the original space (done by applying transforms in reverse), and then
    re-transforming them.
    This function is specifically for points which are formatted as numpy
    arrays. This function is passed to _model_gen.

    Args:
        param_names: Names of parameters to transform.
        transforms: Ordered set of transforms which were applied to the points.

    Returns:
        a function with for performing the roundtrip transform.
    """

    def _roundtrip_transform(x: np.ndarray) -> np.ndarray:
        """Inner function for performing aforementioned functionality.

        Args:
            x: points in the transformed space (e.g. all transforms have been applied
                to them)

        Returns:
            points in the transformed space, but rounded via the original space.
        """
        # apply reverse terminal transform to turn array to ObservationFeatures
        observation_features = [
            ObservationFeatures(
                parameters={p: float(x[i]) for i, p in enumerate(param_names)}
            )
        ]
        # reverse loop through the transforms and do untransform
        for t in reversed(list(transforms.values())):
            observation_features = t.untransform_observation_features(
                observation_features
            )
        # forward loop through the transforms and do transform
        for t in transforms.values():
            observation_features = t.transform_observation_features(
                observation_features
            )
        # parameters are guaranteed to be float compatible here, but pyre doesn't know
        new_x: List[float] = [
            # pyre-fixme[6]: Expected `Union[_SupportsIndex, bytearray, bytes, str,
            #  typing.SupportsFloat]` for 1st param but got `Union[None, bool, float,
            #  int, str]`.
            float(observation_features[0].parameters[p])
            for p in param_names
        ]
        # turn it back into an array
        return np.array(new_x)

    return _roundtrip_transform


def get_pending_observation_features(
    experiment: Experiment, include_failed_as_pending: bool = False
) -> Optional[Dict[str, List[ObservationFeatures]]]:
    """Computes a list of pending observation features (corresponding to arms that
    have been generated and deployed in the course of the experiment, but have not
    been completed with data or to arms that have been abandoned or belong to
    abandoned trials).

    NOTE: Pending observation features are passed to the model to
    instruct it to not generate the same points again.

    Args:
        experiment: Experiment, pending features on which we seek to compute.
        include_failed_as_pending: Whether to include failed trials as pending
            (for example, to avoid the model suggesting them again).

    Returns:
        An optional mapping from metric names to a list of observation features,
        pending for that metric (i.e. do not have evaluation data for that metric).
        If there are no pending features for any of the metrics, return is None.
    """
    pending_features = {}
    # Note that this assumes that if a metric appears in fetched data, the trial is
    # not pending for the metric. Where only the most recent data matters, this will
    # work, but may need to add logic to check previously added data objects, too.
    for trial_index, trial in experiment.trials.items():
        dat = trial.lookup_data()
        for metric_name in experiment.metrics:
            if metric_name not in pending_features:
                pending_features[metric_name] = []
            include_since_failed = include_failed_as_pending and trial.status.is_failed
            if isinstance(trial, BatchTrial):
                if trial.status.is_abandoned or (
                    (trial.status.is_deployed or include_since_failed)
                    and metric_name not in dat.df.metric_name.values
                    and trial.arms is not None
                ):
                    for arm in trial.arms:
                        not_none(pending_features.get(metric_name)).append(
                            ObservationFeatures.from_arm(
                                arm=arm,
                                trial_index=np.int64(trial_index),
                                metadata=trial._get_candidate_metadata(
                                    arm_name=arm.name
                                ),
                            )
                        )
                abandoned_arms = trial.abandoned_arms
                for abandoned_arm in abandoned_arms:
                    not_none(pending_features.get(metric_name)).append(
                        ObservationFeatures.from_arm(
                            arm=abandoned_arm,
                            trial_index=np.int64(trial_index),
                            metadata=trial._get_candidate_metadata(
                                arm_name=abandoned_arm.name
                            ),
                        )
                    )

            if isinstance(trial, Trial):
                if trial.status.is_abandoned or (
                    (trial.status.is_deployed or include_since_failed)
                    and metric_name not in dat.df.metric_name.values
                    and trial.arm is not None
                ):
                    not_none(pending_features.get(metric_name)).append(
                        ObservationFeatures.from_arm(
                            arm=not_none(trial.arm),
                            trial_index=np.int64(trial_index),
                            metadata=trial._get_candidate_metadata(
                                arm_name=not_none(trial.arm).name
                            ),
                        )
                    )
    return pending_features if any(x for x in pending_features.values()) else None


def get_pending_observation_features_based_on_trial_status(
    experiment: Experiment,
) -> Optional[Dict[str, List[ObservationFeatures]]]:
    """A faster analogue of ``get_pending_observation_features`` that makes
    assumptions about trials in experiment in order to speed up extraction
    of pending points.

    Assumptions:

    * All arms in all trials in ``STAGED,`` ``RUNNING`` and ``ABANDONED`` statuses
      are to be considered pending for all outcomes.
    * All arms in all trials in other statuses are to be considered not pending for
      all outcomes.

    This entails:

    * No actual data-fetching for trials to determine whether arms in them are pending
      for specific outcomes.
    * Even if data is present for some outcomes in ``RUNNING`` trials, their arms will
      still be considered pending for those outcomes.

    NOTE: This function should not be used to extract pending features in field
    experiments, where arms in running trials should not be considered pending if
    there is data for those arms.

    Args:
        experiment: Experiment, pending features on which we seek to compute.

    Returns:
        An optional mapping from metric names to a list of observation features,
        pending for that metric (i.e. do not have evaluation data for that metric).
        If there are no pending features for any of the metrics, return is None.
    """
    pending_features = defaultdict(list)
    for status in [TrialStatus.STAGED, TrialStatus.RUNNING, TrialStatus.ABANDONED]:
        for trial in experiment.trials_by_status[status]:
            for metric_name in experiment.metrics:
                pending_features[metric_name].extend(
                    ObservationFeatures.from_arm(
                        arm=arm,
                        trial_index=np.int64(trial.index),
                        metadata=trial._get_candidate_metadata(arm_name=arm.name),
                    )
                    for arm in trial.arms
                )

    return dict(pending_features) if any(x for x in pending_features.values()) else None


def extend_pending_observations(
    experiment: Experiment,
    pending_observations: Dict[str, List[ObservationFeatures]],
    generator_run: GeneratorRun,
) -> None:
    """Extend given pending observations dict (from metric name to observations
    that are pending for that metric), with arms in a given generator run.
    """
    for m in experiment.metrics:
        if m not in pending_observations:
            pending_observations[m] = []
        pending_observations[m].extend(
            ObservationFeatures.from_arm(a) for a in generator_run.arms
        )


def get_pareto_frontier_and_configs(
    modelbridge: modelbridge_module.torch.TorchModelBridge,
    observation_features: List[ObservationFeatures],
    observation_data: Optional[List[ObservationData]] = None,
    objective_thresholds: Optional[TRefPoint] = None,
    optimization_config: Optional[MultiObjectiveOptimizationConfig] = None,
    arm_names: Optional[List[Optional[str]]] = None,
    use_model_predictions: bool = True,
    transform_outcomes_and_configs: Optional[bool] = None,
) -> Tuple[List[Observation], Tensor, Tensor, Optional[Tensor]]:
    """Helper that applies transforms and calls ``frontier_evaluator``.

    Returns the ``frontier_evaluator`` configs in addition to the Pareto
    observations.

    Args:
        modelbridge: ``Modelbridge`` used to predict metrics outcomes.
        observation_features: Observation features to consider for the Pareto
            frontier.
        observation_data: Data for computing the Pareto front, unless
            ``observation_features`` are provided and ``model_predictions is True``.
        objective_thresholds: Metric values bounding the region of interest in
            the objective outcome space; used to override objective thresholds
            specified in ``optimization_config``, if necessary.
        optimization_config: Multi-objective optimization config.
        arm_names: Arm names for each observation in ``observation_features``.
        use_model_predictions: If ``True``, will use model predictions at
            ``observation_features`` to compute Pareto front. If ``False``,
            will use ``observation_data`` directly to compute Pareto front, ignoring
            ``observation_features``.
        transform_outcomes_and_configs: Deprecated and must be ``False`` if provided.
            Previously, if ``True``, would transform the optimization
            config, observation features and observation data, before calling
            ``frontier_evaluator``, then will untransform all of the above before
            returning the observations.

    Returns: Four-item tuple of:
          - frontier_observations: Observations of points on the pareto frontier,
          - f: n x m tensor representation of the Pareto frontier values where n is the
            length of frontier_observations and m is the number of metrics,
          - obj_w: m tensor of objective weights,
          - obj_t: m tensor of objective thresholds corresponding to Y, or None if no
            objective thresholds used.
    """
    if transform_outcomes_and_configs is None:
        warnings.warn(
            "FYI: The default behavior of `get_pareto_frontier_and_configs` when "
            "`transform_outcomes_and_configs` is not specified has changed. Previously,"
            " the default was `transform_outcomes_and_configs=True`; now this argument "
            "is deprecated and behavior is as if "
            "`transform_outcomes_and_configs=False`. You did not specify "
            "`transform_outcomes_and_configs`, so this warning requires no action."
        )
    elif transform_outcomes_and_configs:
        raise UnsupportedError(
            "`transform_outcomes_and_configs=True` is no longer supported, and the "
            "`transform_outcomes_and_configs` argument is deprecated. Please do not "
            "specify this argument."
        )
    else:
        warnings.warn(
            "You passed `transform_outcomes_and_configs=False`. Specifying "
            "`transform_outcomes_and_configs` at all is deprecated because `False` is "
            "now the only allowed behavior. In the future, this will become an error.",
            DeprecationWarning,
        )
    # Input validation
    if use_model_predictions:
        if observation_data is not None:
            warnings.warn(
                "You provided `observation_data` when `use_model_predictions` is True; "
                "`observation_data` will not be used."
            )
    else:
        if observation_data is None:
            raise ValueError(
                "`observation_data` must not be None when `use_model_predictions` is "
                "True."
            )

    array_to_tensor = partial(_array_to_tensor, modelbridge=modelbridge)
    if use_model_predictions:
        observation_data = modelbridge._predict_observation_data(
            observation_features=observation_features
        )
    Y, Yvar = observation_data_to_array(
        outcomes=modelbridge.outcomes, observation_data=not_none(observation_data)
    )
    Y, Yvar = (array_to_tensor(Y), array_to_tensor(Yvar))
    if arm_names is None:
        arm_names = [None] * len(observation_features)

    # Extract optimization config: make sure that the problem is a MOO
    # problem and clone the optimization config with specified
    # `objective_thresholds` if those are provided. If `optimization_config`
    # is not specified, uses the one stored on `modelbridge`.
    optimization_config = _get_multiobjective_optimization_config(
        modelbridge=modelbridge,
        optimization_config=optimization_config,
        objective_thresholds=objective_thresholds,
    )

    # Transform optimization config.

    # de-relativize outcome constraints and objective thresholds
    observations = modelbridge.get_training_data()

    optimization_config = checked_cast(
        MultiObjectiveOptimizationConfig,
        derelativize_optimization_config_with_raw_status_quo(
            optimization_config=optimization_config,
            modelbridge=modelbridge,
            observations=observations,
        ),
    )
    # Extract weights, constraints, and objective_thresholds
    objective_weights = extract_objective_weights(
        objective=optimization_config.objective, outcomes=modelbridge.outcomes
    )
    outcome_constraints = extract_outcome_constraints(
        outcome_constraints=optimization_config.outcome_constraints,
        outcomes=modelbridge.outcomes,
    )
    obj_t = extract_objective_thresholds(
        objective_thresholds=optimization_config.objective_thresholds,
        objective=optimization_config.objective,
        outcomes=modelbridge.outcomes,
    )
    obj_t = array_to_tensor(obj_t)
    # Transform to tensors.
    obj_w, oc_c, _, _, _ = validate_and_apply_final_transform(
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
        linear_constraints=None,
        pending_observations=None,
        final_transform=array_to_tensor,
    )
    f, cov, indx = pareto_frontier_evaluator(
        model=None,
        X=None,
        Y=Y,
        Yvar=Yvar,
        objective_thresholds=obj_t,
        objective_weights=obj_w,
        outcome_constraints=oc_c,
    )
    f, cov = f.detach().cpu().clone(), cov.detach().cpu().clone()
    indx = indx.tolist()
    frontier_observation_data = array_to_observation_data(
        f=f.numpy(), cov=cov.numpy(), outcomes=not_none(modelbridge.outcomes)
    )
    # Construct observations
    frontier_observations = []
    for i, obsd in enumerate(frontier_observation_data):
        frontier_observations.append(
            Observation(
                features=deepcopy(observation_features[indx[i]]),
                data=deepcopy(obsd),
                arm_name=arm_names[indx[i]],
            )
        )

    return frontier_observations, f, obj_w.cpu(), obj_t.cpu()


def pareto_frontier(
    modelbridge: modelbridge_module.torch.TorchModelBridge,
    observation_features: List[ObservationFeatures],
    observation_data: Optional[List[ObservationData]] = None,
    objective_thresholds: Optional[TRefPoint] = None,
    optimization_config: Optional[MultiObjectiveOptimizationConfig] = None,
    arm_names: Optional[List[Optional[str]]] = None,
    use_model_predictions: bool = True,
) -> List[Observation]:
    """Compute the list of points on the Pareto frontier as `Observation`-s
    in the untransformed search space.

    Args:
        modelbridge: ``Modelbridge`` used to predict metrics outcomes.
        observation_features: Observation features to consider for the Pareto
            frontier.
        observation_data: Data for computing the Pareto front, unless
            ``observation_features`` are provided and ``model_predictions is True``.
        objective_thresholds: Metric values bounding the region of interest in
            the objective outcome space; used to override objective thresholds
            specified in ``optimization_config``, if necessary.
        optimization_config: Multi-objective optimization config.
        arm_names: Arm names for each observation in ``observation_features``.
        use_model_predictions: If ``True``, will use model predictions at
            ``observation_features`` to compute Pareto front. If ``False``,
            will use ``observation_data`` directly to compute Pareto front, ignoring
            ``observation_features``.

    Returns: Points on the Pareto frontier as `Observation`-s.
    """
    return get_pareto_frontier_and_configs(
        modelbridge=modelbridge,
        observation_features=observation_features,
        observation_data=observation_data,
        objective_thresholds=objective_thresholds,
        optimization_config=optimization_config,
        arm_names=arm_names,
        use_model_predictions=use_model_predictions,
    )[0]


def predicted_pareto_frontier(
    modelbridge: modelbridge_module.torch.TorchModelBridge,
    objective_thresholds: Optional[TRefPoint] = None,
    observation_features: Optional[List[ObservationFeatures]] = None,
    optimization_config: Optional[MultiObjectiveOptimizationConfig] = None,
) -> List[Observation]:
    """Generate a Pareto frontier based on the posterior means of given
    observation features. Given a model and optionally features to evaluate
    (will use model training data if not specified), use the model to predict
    which points lie on the Pareto frontier.

    Args:
        modelbridge: ``Modelbridge`` used to predict metrics outcomes.
        observation_features: Observation features to predict, if provided and
            ``use_model_predictions is True``.
        objective_thresholds: Metric values bounding the region of interest in
            the objective outcome space; used to override objective thresholds
            specified in ``optimization_config``, if necessary.
        optimization_config: Multi-objective optimization config.

    Returns:
        Observations representing points on the Pareto frontier.
    """
    if observation_features is None:
        observation_features, _, arm_names = _get_modelbridge_training_data(
            modelbridge=modelbridge
        )
    else:
        arm_names = None
    if not observation_features:
        raise ValueError(
            "Must receive observation_features as input or the model must "
            "have training data."
        )

    pareto_observations = pareto_frontier(
        modelbridge=modelbridge,
        objective_thresholds=objective_thresholds,
        observation_features=observation_features,
        optimization_config=optimization_config,
        arm_names=arm_names,
    )
    return pareto_observations


def observed_pareto_frontier(
    modelbridge: modelbridge_module.torch.TorchModelBridge,
    objective_thresholds: Optional[TRefPoint] = None,
    optimization_config: Optional[MultiObjectiveOptimizationConfig] = None,
) -> List[Observation]:
    """Generate a pareto frontier based on observed data. Given observed data
    (sourced from model training data), return points on the Pareto frontier
    as `Observation`-s.

    Args:
        modelbridge: ``Modelbridge`` that holds previous training data.
        objective_thresholds: Metric values bounding the region of interest in
            the objective outcome space; used to override objective thresholds
            in the optimization config, if needed.
        optimization_config: Multi-objective optimization config.

    Returns:
        Data representing points on the pareto frontier.
    """
    # Get observation_data from current training data
    obs_feats, obs_data, arm_names = _get_modelbridge_training_data(
        modelbridge=modelbridge
    )

    pareto_observations = pareto_frontier(
        modelbridge=modelbridge,
        objective_thresholds=objective_thresholds,
        observation_data=obs_data,
        observation_features=obs_feats,
        optimization_config=optimization_config,
        arm_names=arm_names,
        use_model_predictions=False,
    )
    return pareto_observations


def hypervolume(
    modelbridge: modelbridge_module.torch.TorchModelBridge,
    observation_features: List[ObservationFeatures],
    objective_thresholds: Optional[TRefPoint] = None,
    observation_data: Optional[List[ObservationData]] = None,
    optimization_config: Optional[MultiObjectiveOptimizationConfig] = None,
    selected_metrics: Optional[List[str]] = None,
    use_model_predictions: bool = True,
) -> float:
    """Helper function that computes (feasible) hypervolume.

    Args:
        modelbridge: The modelbridge.
        observation_features: The observation features for the in-sample arms.
        objective_thresholds: The objective thresholds to be used for computing
            the hypervolume. If None, these are extracted from the optimization
            config.
        observation_data: The observed outcomes for the in-sample arms.
        optimization_config: The optimization config specifying the objectives,
            objectives thresholds, and outcome constraints.
        selected_metrics: A list of objective metric names specifying which
            objectives to use in hypervolume computation. By default, all
            objectives are used.
        use_model_predictions: A boolean indicating whether to use model predictions
            for determining the in-sample Pareto frontier instead of the raw observed
            values.

    Returns:
        The (feasible) hypervolume.

    """
    frontier_observations, f, obj_w, obj_t = get_pareto_frontier_and_configs(
        modelbridge=modelbridge,
        observation_features=observation_features,
        observation_data=observation_data,
        objective_thresholds=objective_thresholds,
        optimization_config=optimization_config,
        use_model_predictions=use_model_predictions,
    )
    if obj_t is None:
        raise ValueError(  # pragma: no cover
            "Cannot compute hypervolume without having objective thresholds specified."
        )
    oc = _get_multiobjective_optimization_config(
        modelbridge=modelbridge,
        optimization_config=optimization_config,
        objective_thresholds=objective_thresholds,
    )
    # Set to all metrics if unspecified
    if selected_metrics is None:
        selected_metrics = oc.objective.metric_names
    # filter to only include objectives
    else:
        if any(m not in oc.objective.metric_names for m in selected_metrics):
            raise ValueError("All selected metrics must be objectives.")

    # Create a mask indicating selected metrics
    selected_metrics_mask = torch.tensor(
        [metric in selected_metrics for metric in modelbridge.outcomes],
        dtype=torch.bool,
        device=f.device,
    )
    # Apply appropriate weights and thresholds
    obj, obj_t = get_weighted_mc_objective_and_objective_thresholds(
        objective_weights=obj_w, objective_thresholds=not_none(obj_t)
    )
    f_t = obj(f)
    obj_mask = obj_w.nonzero().view(-1)
    selected_metrics_mask = selected_metrics_mask[obj_mask]
    f_t = f_t[:, selected_metrics_mask]
    obj_t = obj_t[selected_metrics_mask]
    bd = DominatedPartitioning(ref_point=obj_t, Y=f_t)
    return bd.compute_hypervolume().item()


def _get_multiobjective_optimization_config(
    modelbridge: modelbridge_module.torch.TorchModelBridge,
    optimization_config: Optional[OptimizationConfig] = None,
    objective_thresholds: Optional[TRefPoint] = None,
) -> MultiObjectiveOptimizationConfig:
    # Optimization_config
    mooc = optimization_config or checked_cast_optional(
        MultiObjectiveOptimizationConfig, modelbridge._optimization_config
    )
    if not mooc:
        raise ValueError(  # pragma: no cover
            (
                "Experiment must have an existing optimization_config "
                "of type `MultiObjectiveOptimizationConfig` "
                "or `optimization_config` must be passed as an argument."
            )
        )
    if not isinstance(mooc, MultiObjectiveOptimizationConfig):
        raise ValueError(  # pragma: no cover
            "optimization_config must be a MultiObjectiveOptimizationConfig."
        )
    if objective_thresholds:
        mooc = mooc.clone_with_args(objective_thresholds=objective_thresholds)

    return mooc


def predicted_hypervolume(
    modelbridge: modelbridge_module.torch.TorchModelBridge,
    objective_thresholds: Optional[TRefPoint] = None,
    observation_features: Optional[List[ObservationFeatures]] = None,
    optimization_config: Optional[MultiObjectiveOptimizationConfig] = None,
    selected_metrics: Optional[List[str]] = None,
) -> float:
    """Calculate hypervolume of a pareto frontier based on the posterior means of
    given observation features.

    Given a model and features to evaluate calculate the hypervolume of the pareto
    frontier formed from their predicted outcomes.

    Args:
        modelbridge: Modelbridge used to predict metrics outcomes.
        objective_thresholds: point defining the origin of hyperrectangles that
            can contribute to hypervolume.
        observation_features: observation features to predict. Model's training
            data used by default if unspecified.
        optimization_config: Optimization config
        selected_metrics: If specified, hypervolume will only be evaluated on
            the specified subset of metrics. Otherwise, all metrics will be used.

    Returns:
        calculated hypervolume.
    """
    if observation_features is None:
        (
            observation_features,
            _,
            __,
        ) = _get_modelbridge_training_data(  # pragma: no cover
            modelbridge=modelbridge
        )
    if not observation_features:
        raise ValueError(
            "Must receive observation_features as input or the model must "
            "have training data."
        )

    return hypervolume(
        modelbridge=modelbridge,
        objective_thresholds=objective_thresholds,
        observation_features=observation_features,
        optimization_config=optimization_config,
        selected_metrics=selected_metrics,
    )


def observed_hypervolume(
    modelbridge: modelbridge_module.torch.TorchModelBridge,
    objective_thresholds: Optional[TRefPoint] = None,
    optimization_config: Optional[MultiObjectiveOptimizationConfig] = None,
    selected_metrics: Optional[List[str]] = None,
) -> float:
    """Calculate hypervolume of a pareto frontier based on observed data.

    Given observed data, return the hypervolume of the pareto frontier formed from
    those outcomes.

    Args:
        modelbridge: Modelbridge that holds previous training data.
        objective_thresholds: point defining the origin of hyperrectangles that
            can contribute to hypervolume.
        observation_features: observation features to predict. Model's training
            data used by default if unspecified.
        optimization_config: Optimization config
        selected_metrics: If specified, hypervolume will only be evaluated on
            the specified subset of metrics. Otherwise, all metrics will be used.

    Returns:
        (float) calculated hypervolume.
    """
    # Get observation_data from current training data.
    obs_feats, obs_data, _ = _get_modelbridge_training_data(modelbridge=modelbridge)

    return hypervolume(
        modelbridge=modelbridge,
        objective_thresholds=objective_thresholds,
        observation_features=obs_feats,
        observation_data=obs_data,
        optimization_config=optimization_config,
        selected_metrics=selected_metrics,
        use_model_predictions=False,
    )


def array_to_observation_data(
    f: np.ndarray, cov: np.ndarray, outcomes: List[str]
) -> List[ObservationData]:
    """Convert arrays of model predictions to a list of ObservationData.

    Args:
        f: An (n x m) array
        cov: An (n x m x m) array
        outcomes: A list of d outcome names

    Returns: A list of n ObservationData
    """
    observation_data = []
    for i in range(f.shape[0]):
        observation_data.append(
            ObservationData(
                metric_names=list(outcomes),
                means=f[i, :].copy(),
                covariance=cov[i, :, :].copy(),
            )
        )
    return observation_data


def observation_data_to_array(
    outcomes: List[str],
    observation_data: List[ObservationData],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a list of Observation data to arrays.

    Args:
        observation_data: A list of n ObservationData

    Returns:
        An array of n ObservationData, each containing
            - f: An (n x m) array
            - cov: An (n x m x m) array
    """
    means = []
    cov = []
    for obsd in observation_data:
        metric_idxs = np.array([obsd.metric_names.index(m) for m in outcomes])
        means.append(obsd.means[metric_idxs])
        cov.append(obsd.covariance[metric_idxs][:, metric_idxs])
    return np.array(means), np.array(cov)


def observation_features_to_array(
    parameters: List[str], obsf: List[ObservationFeatures]
) -> np.ndarray:
    """Convert a list of Observation features to arrays."""
    return np.array([[of.parameters[p] for p in parameters] for of in obsf])


def detect_duplicates(
    X: Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Iterator[Tuple[int, int]]:
    """Returns an iterator over index pairs `(duplicate index, original index)` for all
    duplicate entries of `X`.
    """
    tols = atol
    if rtol:
        rval = X.abs().max(dim=-1, keepdim=True).values
        tols = tols + rtol * rval.max(rval.transpose(-1, -2))

    n = X.shape[-2]
    dist = torch.full((n, n), float("inf"), device=X.device, dtype=X.dtype)
    dist[torch.triu_indices(n, n, offset=1).unbind()] = torch.nn.functional.pdist(
        X, p=float("inf")
    )
    return (
        (i, int(j))
        # pyre-fixme[19]: Expected 1 positional argument.
        for diff, j, i in zip(*(dist - tols).min(dim=-2), range(n))
        if diff < 0
    )


def feasible_hypervolume(  # pragma: no cover
    optimization_config: MultiObjectiveOptimizationConfig, values: Dict[str, np.ndarray]
) -> np.ndarray:
    """Compute the feasible hypervolume each iteration.

    Args:
        optimization_config: Optimization config.
        values: Dictionary from metric name to array of value at each
            iteration (each array is `n`-dim). If optimization config contains
            outcome constraints, values for them must be present in `values`.

    Returns: Array of feasible hypervolumes.
    """
    # Get objective at each iteration
    obj_threshold_dict = {
        ot.metric.name: ot.bound for ot in optimization_config.objective_thresholds
    }
    f_vals = np.hstack(
        [values[m.name].reshape(-1, 1) for m in optimization_config.objective.metrics]
    )
    obj_thresholds = np.array(
        [obj_threshold_dict[m.name] for m in optimization_config.objective.metrics]
    )
    # Set infeasible points to be the objective threshold
    for oc in optimization_config.outcome_constraints:
        if oc.relative:
            raise ValueError(  # pragma: no cover
                "Benchmark aggregation does not support relative constraints"
            )
        g = values[oc.metric.name]
        feas = g <= oc.bound if oc.op == ComparisonOp.LEQ else g >= oc.bound
        f_vals[~feas] = obj_thresholds

    obj_weights = np.array(
        [-1 if m.lower_is_better else 1 for m in optimization_config.objective.metrics]
    )
    obj_thresholds = obj_thresholds * obj_weights
    f_vals = f_vals * obj_weights
    partitioning = DominatedPartitioning(
        ref_point=torch.from_numpy(obj_thresholds).double()
    )
    f_vals_torch = torch.from_numpy(f_vals).double()
    # compute hv at each iteration
    hvs = []
    for i in range(f_vals.shape[0]):
        # update with new point
        partitioning.update(Y=f_vals_torch[i : i + 1])
        hv = partitioning.compute_hypervolume().item()
        hvs.append(hv)
    return np.array(hvs)


def _array_to_tensor(
    array: Union[np.ndarray, List[float]],
    modelbridge: Optional[modelbridge_module.base.ModelBridge] = None,
) -> Tensor:
    if modelbridge and hasattr(modelbridge, "_array_to_tensor"):
        # pyre-ignore[16]: modelbridge does not have attribute `_array_to_tensor`
        return modelbridge._array_to_tensor(array)
    else:
        return torch.tensor(array)


def _get_modelbridge_training_data(
    modelbridge: modelbridge_module.torch.TorchModelBridge,
) -> Tuple[List[ObservationFeatures], List[ObservationData], List[Optional[str]]]:
    obs = modelbridge.get_training_data()
    return _unpack_observations(obs=obs)


def _unpack_observations(
    obs: List[Observation],
) -> Tuple[List[ObservationFeatures], List[ObservationData], List[Optional[str]]]:
    obs_feats, obs_data, arm_names = [], [], []
    for ob in obs:
        obs_feats.append(ob.features)
        obs_data.append(ob.data)
        arm_names.append(ob.arm_name)
    return obs_feats, obs_data, arm_names
