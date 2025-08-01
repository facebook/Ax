#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from copy import deepcopy
from logging import Logger
from typing import Any, SupportsFloat, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.utils import (
    derelativize_optimization_config_with_raw_status_quo,
)
from ax.core.data import Data
from ax.core.experiment import Experiment
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
from ax.core.types import TBounds, TCandidateMetadata
from ax.exceptions.core import DataRequiredError, UserInputError
from ax.generators.torch.botorch_moo_defaults import (
    get_weighted_mc_objective_and_objective_thresholds,
    pareto_frontier_evaluator,
)
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import (
    assert_is_instance_of_tuple,
    assert_is_instance_optional,
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
from botorch.models.utils.assorted import consolidate_duplicates
from botorch.utils.containers import SliceContainer
from botorch.utils.datasets import ContextualDataset, RankingDataset, SupervisedDataset
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from pyre_extensions import assert_is_instance, none_throws
from torch import Tensor

logger: Logger = get_logger(__name__)


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


"""A mapping of risk measure names to the corresponding classes.

NOTE: This can be extended with user-defined risk measure classes by
importing the dictionary and adding the new risk measure class as
`RISK_MEASURE_NAME_TO_CLASS["my_risk_measure"] = MyRiskMeasure`.
An example of this is found in `tests/test_risk_measure`.
"""
RISK_MEASURE_NAME_TO_CLASS: dict[str, type[RiskMeasureMCObjective]] = {
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
    optimization_config: OptimizationConfig | None = None,
) -> None:
    """Raise an error if not using a `MultiObjective` or if the data is empty."""
    optimization_config = none_throws(
        optimization_config or experiment.optimization_config
    )
    if not isinstance(optimization_config.objective, MultiObjective):
        raise ValueError("Multi-objective optimization requires multiple objectives.")
    if data.df.empty:
        raise ValueError("MultiObjectiveOptimization requires non-empty data.")


def extract_parameter_constraints(
    parameter_constraints: list[ParameterConstraint], param_names: list[str]
) -> TBounds:
    """Convert Ax parameter constraints into a tuple of NumPy arrays representing the
    system of linear inequality constraints.

    Args:
        parameter_constraints: A list of parameter constraint objects.
        param_names: A list of parameter names.

    Returns:
        An optional tuple of NumPy arrays (A, b) representing the system of linear
        inequality constraints A x < b.
    """
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
    search_space: SearchSpace, param_names: list[str]
) -> SearchSpaceDigest:
    """Extract basic parameter properties from a search space.

    This is typically called with the transformed search space and makes certain
    assumptions regarding the parameters being transformed.

    For ChoiceParameters:
    * The choices are assumed to be numerical. ChoiceToNumericChoice
    and OrderedChoiceToIntegerRange
    transforms handle this.
    * If is_task, its index is added to task_features.
    * If ordered, its index is added to ordinal_features.
    * Otherwise, its index is added to categorical_features.
    * In all cases, the choices are added to discrete_choices.
    * The minimum and maximum value are added to the bounds.
    * The target_value is added to target_values.

    For RangeParameters:
    * They're assumed not to be in the log_scale. The Log transform handles this.
    * If integer, its index is added to ordinal_features and the choices are added
    to discrete_choices.
    * The minimum and maximum value are added to the bounds.

    If a parameter is_fidelity:
    * Its target_value is assumed to be numerical.
    * The target_value is added to target_values.
    * Its index is added to fidelity_features.
    """
    bounds: list[tuple[int | float, int | float]] = []
    ordinal_features: list[int] = []
    categorical_features: list[int] = []
    discrete_choices: dict[int, list[int | float]] = {}
    task_features: list[int] = []
    fidelity_features: list[int] = []
    target_values: dict[int, int | float] = {}

    for i, p_name in enumerate(param_names):
        p = search_space.parameters[p_name]
        if isinstance(p, ChoiceParameter):
            if p.is_task:
                task_features.append(i)
                target_values[i] = assert_is_instance_of_tuple(
                    p.target_value, (int, float)
                )
            elif p.is_ordered:
                ordinal_features.append(i)
            else:
                categorical_features.append(i)
            # at this point we can assume that values are numeric due to transforms
            discrete_choices[i] = p.values  # pyre-ignore [6]
            bounds.append((min(p.values), max(p.values)))  # pyre-ignore [6]
        elif isinstance(p, RangeParameter):
            if p.log_scale or p.logit_scale:
                raise UserInputError(
                    "Log and Logit scale parameters must be transformed using the "
                    "corresponding transform within the `Adapter`. After applying "
                    f"the transforms, we have {p.log_scale=} and {p.logit_scale=}."
                )
            if p.parameter_type == ParameterType.INT:
                ordinal_features.append(i)
                d_choices = list(range(int(p.lower), int(p.upper) + 1))
                # pyre-ignore [6]
                discrete_choices[i] = d_choices
            bounds.append((p.lower, p.upper))
        else:
            raise ValueError(f"Unknown parameter type {type(p)}")
        if p.is_fidelity:
            fidelity_features.append(i)
            target_values[i] = assert_is_instance_of_tuple(p.target_value, (int, float))

    return SearchSpaceDigest(
        feature_names=param_names,
        bounds=bounds,
        ordinal_features=ordinal_features,
        categorical_features=categorical_features,
        discrete_choices=discrete_choices,
        task_features=task_features,
        fidelity_features=fidelity_features,
        target_values=target_values,
        robust_digest=extract_robust_digest(
            search_space=search_space, param_names=param_names
        ),
    )


def extract_robust_digest(
    search_space: SearchSpace, param_names: list[str]
) -> RobustSearchSpaceDigest | None:
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
    env_vars: dict[str, Parameter] = search_space._environmental_variables
    pert_params = [p for p in dist_params if p not in env_vars]
    # Make sure all distributional parameters are in param_names.
    dist_idcs: dict[str, int] = {}
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

        def sample_environmental() -> npt.NDArray:
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
        constructor: Callable[[tuple[int, int]], npt.NDArray] = (
            np.ones if search_space.multiplicative else np.zeros
        )

        def sample_param_perturbations() -> npt.NDArray:
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
    outcomes: list[str],
) -> npt.NDArray | None:
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


def extract_objective_weights(objective: Objective, outcomes: list[str]) -> npt.NDArray:
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
        s = -1.0 if objective.minimize else 1.0
        for obj_metric, obj_weight in objective.metric_weights:
            objective_weights[outcomes.index(obj_metric.name)] = obj_weight * s
    elif isinstance(objective, MultiObjective):
        for obj in objective.objectives:
            s = -1.0 if obj.minimize else 1.0
            objective_weights[outcomes.index(obj.metric.name)] = s
    else:
        s = -1.0 if objective.minimize else 1.0
        objective_weights[outcomes.index(objective.metric.name)] = s
    return objective_weights


def extract_outcome_constraints(
    outcome_constraints: list[OutcomeConstraint], outcomes: list[str]
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
    objective_weights: npt.NDArray,
    outcome_constraints: tuple[npt.NDArray, npt.NDArray] | None,
    linear_constraints: tuple[npt.NDArray, npt.NDArray] | None,
    pending_observations: list[npt.NDArray] | None,
    objective_thresholds: npt.NDArray | None = None,
    final_transform: Callable[[npt.NDArray], Tensor] = torch.tensor,
) -> tuple[
    Tensor,
    tuple[Tensor, Tensor] | None,
    tuple[Tensor, Tensor] | None,
    list[Tensor] | None,
    Tensor | None,
]:
    # TODO: use some container down the road (similar to
    # SearchSpaceDigest) to limit the return arguments
    # pyre-fixme[35]: Target cannot be annotated.
    objective_weights: Tensor = final_transform(objective_weights)
    if outcome_constraints is not None:
        # pyre-fixme[35]: Target cannot be annotated.
        outcome_constraints: tuple[Tensor, Tensor] = (
            final_transform(outcome_constraints[0]),
            final_transform(outcome_constraints[1]),
        )
    if linear_constraints is not None:
        # pyre-fixme[35]: Target cannot be annotated.
        linear_constraints: tuple[Tensor, Tensor] = (
            final_transform(linear_constraints[0]),
            final_transform(linear_constraints[1]),
        )
    if pending_observations is not None:
        # pyre-fixme[35]: Target cannot be annotated.
        pending_observations: list[Tensor] = [
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
    fixed_features: ObservationFeatures | None, param_names: list[str]
) -> dict[int, float] | None:
    """Reformat a set of fixed_features."""
    if fixed_features is None or not fixed_features.parameters:
        return None
    params = fixed_features.parameters
    params_set = set(params)
    param_names_set = set(param_names)
    if params_set > param_names_set:
        raise ValueError(
            "Fixed features contains parameters not in "
            f"`param_names`: {params_set - param_names_set}."
        )
    fixed_features_dict = {
        i: float(assert_is_instance(params[p_name], SupportsFloat))
        for i, p_name in enumerate(param_names)
        if p_name in params
    }
    return fixed_features_dict


def get_fixed_features_from_experiment(
    experiment: Experiment,
) -> ObservationFeatures:
    completed_indices = [t.index for t in experiment.completed_trials]
    completed_indices.append(0)  # handle case of no completed trials
    return ObservationFeatures(
        parameters={},
        trial_index=max(completed_indices),
    )


def pending_observations_as_array_list(
    pending_observations: dict[str, list[ObservationFeatures]],
    outcome_names: list[str],
    param_names: list[str],
) -> list[npt.NDArray] | None:
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
    X: npt.NDArray,
    param_names: list[str],
    candidate_metadata: Sequence[TCandidateMetadata] | None = None,
) -> list[ObservationFeatures]:
    """Re-format raw model-generated candidates into ObservationFeatures.

    Args:
        param_names: List of param names.
        X: Raw np.ndarray of candidate values.
        candidate_metadata: Model's metadata for candidates it produced.

    Returns:
        List of candidates, represented as ObservationFeatures.
    """
    if candidate_metadata and len(candidate_metadata) != len(X):
        raise ValueError(
            "Observations metadata list provided is not of "
            "the same size as the number of candidates."
        )
    observation_features = []
    for i, x in enumerate(X):
        observation_features.append(
            ObservationFeatures(
                parameters=dict(zip(param_names, x, strict=True)),
                metadata=candidate_metadata[i] if candidate_metadata else None,
            )
        )
    return observation_features


def transform_callback(
    param_names: list[str],
    transforms: MutableMapping[str, Transform],
) -> Callable[[npt.NDArray], npt.NDArray]:
    """A closure for performing the `round trip` transformations.

    The function rounds points by de-transforming points back into
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

    def _roundtrip_transform(x: npt.NDArray) -> npt.NDArray:
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
        new_x: list[float] = [
            float(observation_features[0].parameters[p]) for p in param_names
        ]
        # turn it back into an array
        return np.array(new_x)

    return _roundtrip_transform


def get_pareto_frontier_and_configs(
    adapter: adapter_module.torch.TorchAdapter,
    observation_features: list[ObservationFeatures],
    observation_data: list[ObservationData] | None = None,
    objective_thresholds: TRefPoint | None = None,
    optimization_config: MultiObjectiveOptimizationConfig | None = None,
    arm_names: list[str | None] | None = None,
    use_model_predictions: bool = True,
) -> tuple[list[Observation], Tensor, Tensor, Tensor | None]:
    """Helper that applies transforms and calls ``frontier_evaluator``.

    Returns the ``frontier_evaluator`` configs in addition to the Pareto
    observations.

    Args:
        adapter: ``Adapter`` used to predict metrics outcomes.
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

    Returns: Four-item tuple of:
          - frontier_observations: Observations of points on the pareto frontier,
          - f: n x m tensor representation of the Pareto frontier values where n is the
            length of frontier_observations and m is the number of metrics,
          - obj_w: m tensor of objective weights,
          - obj_t: m tensor of objective thresholds corresponding to Y, or None if no
            objective thresholds used.
    """
    # Input validation
    if use_model_predictions:
        if observation_data is not None:
            warnings.warn(
                "You provided `observation_data` when `use_model_predictions` is True; "
                "`observation_data` will not be used.",
                stacklevel=2,
            )
    else:
        if observation_data is None:
            raise ValueError(
                "`observation_data` must not be None when `use_model_predictions` is "
                "True."
            )

    array_to_tensor = adapter._array_to_tensor
    if use_model_predictions:
        observation_data = adapter._predict_observation_data(
            observation_features=observation_features
        )
    Y, Yvar = observation_data_to_array(
        outcomes=adapter.outcomes, observation_data=none_throws(observation_data)
    )
    Y, Yvar = (array_to_tensor(Y), array_to_tensor(Yvar))
    if arm_names is None:
        arm_names = [None] * len(observation_features)

    # Extract optimization config: make sure that the problem is a MOO
    # problem and clone the optimization config with specified
    # `objective_thresholds` if those are provided. If `optimization_config`
    # is not specified, uses the one stored on `adapter`.
    optimization_config = _get_multiobjective_optimization_config(
        adapter=adapter,
        optimization_config=optimization_config,
        objective_thresholds=objective_thresholds,
    )

    # Transform optimization config.

    # de-relativize outcome constraints and objective thresholds
    optimization_config = assert_is_instance(
        derelativize_optimization_config_with_raw_status_quo(
            optimization_config=optimization_config, adapter=adapter
        ),
        MultiObjectiveOptimizationConfig,
    )
    # Extract weights, constraints, and objective_thresholds
    objective_weights = extract_objective_weights(
        objective=optimization_config.objective, outcomes=adapter.outcomes
    )
    outcome_constraints = extract_outcome_constraints(
        outcome_constraints=optimization_config.outcome_constraints,
        outcomes=adapter.outcomes,
    )
    obj_t = extract_objective_thresholds(
        objective_thresholds=optimization_config.objective_thresholds,
        objective=optimization_config.objective,
        outcomes=adapter.outcomes,
    )
    if obj_t is not None:
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
        f=f.numpy(), cov=cov.numpy(), outcomes=none_throws(adapter.outcomes)
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

    return (
        frontier_observations,
        f,
        obj_w.cpu(),
        obj_t.cpu() if obj_t is not None else None,
    )


def pareto_frontier(
    adapter: adapter_module.torch.TorchAdapter,
    observation_features: list[ObservationFeatures],
    observation_data: list[ObservationData] | None = None,
    objective_thresholds: TRefPoint | None = None,
    optimization_config: MultiObjectiveOptimizationConfig | None = None,
    arm_names: list[str | None] | None = None,
    use_model_predictions: bool = True,
) -> list[Observation]:
    """Compute the list of points on the Pareto frontier as `Observation`-s
    in the untransformed search space.

    Args:
        adapter: ``Adapter`` used to predict metrics outcomes.
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

    Returns: Points on the Pareto frontier as `Observation`-s in order of descending
        individual hypervolume if possible.
    """
    frontier_observations, f, obj_w, obj_t = get_pareto_frontier_and_configs(
        adapter=adapter,
        observation_features=observation_features,
        observation_data=observation_data,
        objective_thresholds=objective_thresholds,
        optimization_config=optimization_config,
        arm_names=arm_names,
        use_model_predictions=use_model_predictions,
    )

    # If no objective thresholds are present we cannot compute hypervolume -- return
    # frontier observations in arbitrary order
    if obj_t is None:
        return frontier_observations

    # Apply appropriate weights and thresholds
    obj, obj_t = get_weighted_mc_objective_and_objective_thresholds(
        objective_weights=obj_w, objective_thresholds=obj_t
    )
    f_t = obj(f)

    # Compute individual hypervolumes by taking the difference between the observation
    # and the reference point and multiplying
    individual_hypervolumes = (
        (f_t.unsqueeze(dim=0) - obj_t).clamp_min(0).prod(dim=-1).squeeze().tolist()
    )

    if not isinstance(individual_hypervolumes, list):
        individual_hypervolumes = [individual_hypervolumes]

    return [
        obs
        for obs, _ in sorted(
            zip(frontier_observations, individual_hypervolumes),
            key=lambda tup: tup[1],
            reverse=True,
        )
    ]


def predicted_pareto_frontier(
    adapter: adapter_module.torch.TorchAdapter,
    objective_thresholds: TRefPoint | None = None,
    observation_features: list[ObservationFeatures] | None = None,
    optimization_config: MultiObjectiveOptimizationConfig | None = None,
) -> list[Observation]:
    """Generate a Pareto frontier based on the posterior means of given
    observation features. Given a model and optionally features to evaluate
    (will use model training data if not specified), use the model to predict
    which points lie on the Pareto frontier.

    Args:
        adapter: ``Adapter`` used to predict metrics outcomes.
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
        observation_features, _, arm_names = _get_adapter_training_data(
            adapter=adapter, in_design_only=True
        )
    else:
        arm_names = None
    if not observation_features:
        raise ValueError(
            "Must receive observation_features as input or the model must "
            "have training data."
        )

    pareto_observations = pareto_frontier(
        adapter=adapter,
        objective_thresholds=objective_thresholds,
        observation_features=observation_features,
        optimization_config=optimization_config,
        arm_names=arm_names,
    )
    return pareto_observations


def observed_pareto_frontier(
    adapter: adapter_module.torch.TorchAdapter,
    objective_thresholds: TRefPoint | None = None,
    optimization_config: MultiObjectiveOptimizationConfig | None = None,
) -> list[Observation]:
    """Generate a pareto frontier based on observed data. Given observed data
    (sourced from model training data), return points on the Pareto frontier
    as `Observation`-s.

    Args:
        adapter: ``Adapter`` that holds previous training data.
        objective_thresholds: Metric values bounding the region of interest in
            the objective outcome space; used to override objective thresholds
            in the optimization config, if needed.
        optimization_config: Multi-objective optimization config.

    Returns:
        Data representing points on the pareto frontier.
    """
    # Get observation_data from current training data
    obs_feats, obs_data, arm_names = _get_adapter_training_data(
        adapter=adapter, in_design_only=True
    )

    pareto_observations = pareto_frontier(
        adapter=adapter,
        objective_thresholds=objective_thresholds,
        observation_data=obs_data,
        observation_features=obs_feats,
        optimization_config=optimization_config,
        arm_names=arm_names,
        use_model_predictions=False,
    )
    return pareto_observations


def hypervolume(
    adapter: adapter_module.torch.TorchAdapter,
    observation_features: list[ObservationFeatures],
    objective_thresholds: TRefPoint | None = None,
    observation_data: list[ObservationData] | None = None,
    optimization_config: MultiObjectiveOptimizationConfig | None = None,
    selected_metrics: list[str] | None = None,
    use_model_predictions: bool = True,
) -> float:
    """Helper function that computes (feasible) hypervolume.

    Args:
        adapter: The adapter.
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
        adapter=adapter,
        observation_features=observation_features,
        observation_data=observation_data,
        objective_thresholds=objective_thresholds,
        optimization_config=optimization_config,
        use_model_predictions=use_model_predictions,
    )
    if obj_t is None:
        raise ValueError(
            "Cannot compute hypervolume without having objective thresholds specified."
        )
    oc = _get_multiobjective_optimization_config(
        adapter=adapter,
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
        [metric in selected_metrics for metric in adapter.outcomes],
        dtype=torch.bool,
        device=f.device,
    )
    # Apply appropriate weights and thresholds
    obj, obj_t = get_weighted_mc_objective_and_objective_thresholds(
        objective_weights=obj_w, objective_thresholds=none_throws(obj_t)
    )
    f_t = obj(f)
    obj_mask = obj_w.nonzero().view(-1)
    selected_metrics_mask = selected_metrics_mask[obj_mask]
    f_t = f_t[:, selected_metrics_mask]
    obj_t = obj_t[selected_metrics_mask]
    bd = DominatedPartitioning(ref_point=obj_t, Y=f_t)
    return bd.compute_hypervolume().item()


def _get_multiobjective_optimization_config(
    adapter: adapter_module.torch.TorchAdapter,
    optimization_config: OptimizationConfig | None = None,
    objective_thresholds: TRefPoint | None = None,
) -> MultiObjectiveOptimizationConfig:
    # Optimization_config
    mooc = optimization_config or assert_is_instance_optional(
        adapter._optimization_config, MultiObjectiveOptimizationConfig
    )
    if not mooc:
        raise ValueError(
            "Experiment must have an existing optimization_config "
            "of type `MultiObjectiveOptimizationConfig` "
            "or `optimization_config` must be passed as an argument."
        )
    if not isinstance(mooc, MultiObjectiveOptimizationConfig):
        raise ValueError(
            "optimization_config must be a MultiObjectiveOptimizationConfig."
        )
    if objective_thresholds:
        mooc = mooc.clone_with_args(objective_thresholds=objective_thresholds)

    return mooc


def predicted_hypervolume(
    adapter: adapter_module.torch.TorchAdapter,
    objective_thresholds: TRefPoint | None = None,
    observation_features: list[ObservationFeatures] | None = None,
    optimization_config: MultiObjectiveOptimizationConfig | None = None,
    selected_metrics: list[str] | None = None,
) -> float:
    """Calculate hypervolume of a pareto frontier based on the posterior means of
    given observation features.

    Given a model and features to evaluate calculate the hypervolume of the pareto
    frontier formed from their predicted outcomes.

    Args:
        adapter: Adapter used to predict metrics outcomes.
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
        ) = _get_adapter_training_data(adapter=adapter)
    if not observation_features:
        raise ValueError(
            "Must receive observation_features as input or the model must "
            "have training data."
        )

    return hypervolume(
        adapter=adapter,
        objective_thresholds=objective_thresholds,
        observation_features=observation_features,
        optimization_config=optimization_config,
        selected_metrics=selected_metrics,
    )


def observed_hypervolume(
    adapter: adapter_module.torch.TorchAdapter,
    objective_thresholds: TRefPoint | None = None,
    optimization_config: MultiObjectiveOptimizationConfig | None = None,
    selected_metrics: list[str] | None = None,
) -> float:
    """Calculate hypervolume of a pareto frontier based on observed data.

    Given observed data, return the hypervolume of the pareto frontier formed from
    those outcomes.

    Args:
        adapter: Adapter that holds previous training data.
        objective_thresholds: Point defining the origin of hyperrectangles that
            can contribute to hypervolume. Note that if this is None,
            `objective_thresholds` must be present on the
            `adapter.optimization_config`.
        observation_features: observation features to predict. Model's training
            data used by default if unspecified.
        optimization_config: Optimization config
        selected_metrics: If specified, hypervolume will only be evaluated on
            the specified subset of metrics. Otherwise, all metrics will be used.

    Returns:
        (float) calculated hypervolume.
    """
    # Get observation_data from current training data.
    obs_feats, obs_data, _ = _get_adapter_training_data(adapter=adapter)

    return hypervolume(
        adapter=adapter,
        objective_thresholds=objective_thresholds,
        observation_features=obs_feats,
        observation_data=obs_data,
        optimization_config=optimization_config,
        selected_metrics=selected_metrics,
        use_model_predictions=False,
    )


def array_to_observation_data(
    f: npt.NDArray,
    cov: npt.NDArray,
    outcomes: list[str],
) -> list[ObservationData]:
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
    outcomes: list[str],
    observation_data: list[ObservationData],
) -> tuple[npt.NDArray, npt.NDArray]:
    """Convert a list of Observation data to arrays.

    Any missing mean or covariance values will be returned as NaNs.

    Args:
        outcomes: A list of `m` outcomes to extract observation data for.
        observation_data: A list of `n` ``ObservationData`` objects.

    Returns:
        - means: An (n x m) array of mean observations.
        - cov: An (n x m x m) array of covariance observations.
    """
    means = []
    cov = []
    # Initialize arrays with all NaN values.
    means = np.full((len(observation_data), len(outcomes)), np.nan)
    cov = np.full((len(observation_data), len(outcomes), len(outcomes)), np.nan)
    # Iterate over observations and extract the relevant data.
    for i, obsd in enumerate(observation_data):
        # Indices of outcomes that are observed.
        outcome_idx = [j for j, o in enumerate(outcomes) if o in obsd.metric_names]
        # Corresponding indices in the observation data.
        observation_idx = [obsd.metric_names.index(outcomes[j]) for j in outcome_idx]
        means[i, outcome_idx] = obsd.means[observation_idx]
        # We can't use advanced indexing over two dimensions jointly for assignment,
        # so this has to be done in two steps.
        cov_pick = np.full((len(outcome_idx), len(outcomes)), np.nan)
        cov_pick[:, outcome_idx] = obsd.covariance[observation_idx][:, observation_idx]
        cov[i, outcome_idx] = cov_pick
    return means, cov


def observation_features_to_array(
    parameters: list[str],
    obsf: list[ObservationFeatures],
) -> npt.NDArray:
    """Convert a list of Observation features to arrays."""
    return np.array([[of.parameters[p] for p in parameters] for of in obsf])


def feasible_hypervolume(
    optimization_config: MultiObjectiveOptimizationConfig,
    values: dict[str, npt.NDArray],
) -> npt.NDArray:
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
            raise ValueError(
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


def _get_adapter_training_data(
    adapter: adapter_module.torch.TorchAdapter, in_design_only: bool = False
) -> tuple[list[ObservationFeatures], list[ObservationData], list[str | None]]:
    """
    Get training data for adapter, optionally filtering out out-of-design points.
    """
    obs = adapter.get_training_data()

    if in_design_only:
        obs = [obs[i] for i in range(len(obs)) if adapter.training_in_design[i]]

    return _unpack_observations(obs=obs)


def _unpack_observations(
    obs: list[Observation],
) -> tuple[list[ObservationFeatures], list[ObservationData], list[str | None]]:
    obs_feats, obs_data, arm_names = [], [], []
    for ob in obs:
        obs_feats.append(ob.features)
        obs_data.append(ob.data)
        arm_names.append(ob.arm_name)
    return obs_feats, obs_data, arm_names


def transform_search_space(
    search_space: SearchSpace,
    transforms: Iterable[type[Transform]],
    transform_configs: Mapping[str, Any],
) -> SearchSpace:
    """
    Apply all given transforms to a copy of the SearchSpace iteratively.
    """
    search_space = search_space.clone()

    for t in transforms:
        try:
            t_instance = t(
                search_space=search_space,
                observations=[],
                adapter=None,
                config=transform_configs.get(t.__name__),
            )

            search_space = t_instance.transform_search_space(search_space=search_space)
        except DataRequiredError:
            # Skip this transform if data is required. Data is only required for
            # transforms that operate on Observations.
            pass

    return search_space


def process_contextual_datasets(
    datasets: list[SupervisedDataset],
    outcomes: list[str],
    parameter_decomposition: dict[str, list[str]],
    metric_decomposition: dict[str, list[str]] | None = None,
) -> list[ContextualDataset]:
    """Contruct a list of `ContextualDataset`.

    Args:
        datasets: A list of `Dataset` objects.
        outcomes: The names of the outcomes to extract observations for.
        parameter_decomposition: Keys are context names. Values are the lists
            of parameter names belonging to the context, e.g.
            {'context1': ['p1_c1', 'p2_c1'],'context2': ['p1_c2', 'p2_c2']}.
        metric_decomposition: Context breakdown metrics. Keys are context names.
            Values are the lists of metric names belonging to the context:
            {
                'context1': ['m1_c1', 'm2_c1', 'm3_c1'],
                'context2': ['m1_c2', 'm2_c2', 'm3_c2'],
            }

    Returns: A list of `ContextualDataset` objects. Order generally will not be that of
        `outcomes`.
    """
    context_buckets = list(parameter_decomposition.keys())
    remaining_metrics = deepcopy(outcomes)
    contextual_datasets = []
    if metric_decomposition is not None:
        M = len(metric_decomposition[context_buckets[0]])
        for j in range(M):
            metric_list = [metric_decomposition[c][j] for c in context_buckets]
            contextual_datasets.append(
                ContextualDataset(
                    datasets=[
                        datasets[outcomes.index(metric_i)] for metric_i in metric_list
                    ],
                    parameter_decomposition=parameter_decomposition,
                    metric_decomposition=metric_decomposition,
                )
            )
            remaining_metrics = list(set(remaining_metrics) - set(metric_list))
    else:
        logger.warning(
            "No metric decomposition found in experiment properties. Using "
            "LCEA model to fit each outcome independently."
        )
    if len(remaining_metrics) > 0:
        for metric_i in remaining_metrics:
            contextual_datasets.append(
                ContextualDataset(
                    datasets=[datasets[outcomes.index(metric_i)]],
                    parameter_decomposition=parameter_decomposition,
                )
            )
    return contextual_datasets


def prep_pairwise_data(
    X: Tensor,
    Y: Tensor,
    group_indices: Tensor,
    outcome: str,
    parameters: list[str],
) -> RankingDataset:
    """Prep data for pairwise modeling
    Args:
        X: Tensor of shape `(n, d)` where `n` is the number of datapoints and `d` is
            the number of features.
        Y: Tensor of shape `(n, 1)` with binary 0 or 1 outcomes with 1 indicating
            that is the preferred arm in the trial.
        group_indices: Indices of groups of each observation. We have exactly two
            arms per group with exactly one 0 and one 1 in Y.
        outcome: Name of the outcome.
        parameters: Names of the features.

    Returns:
        A `RankingDataset` for pairwise preference modeling.
    """
    sorted_indices = torch.argsort(group_indices)
    X = X[sorted_indices]
    Y = Y[sorted_indices]

    # Update Xs and Ys shapes for PairwiseGP
    Y = _binary_pref_to_comp_pair(Y=Y)
    X, Y = _consolidate_comparisons(X=X, Y=Y)

    datapoints, comparisons = X, Y.long()
    event_shape = torch.Size([2 * datapoints.shape[-1]])
    # pyre-ignore[6]: For 2nd param expected `LongTensor` but
    dataset_X = SliceContainer(datapoints, comparisons, event_shape=event_shape)
    dataset_Y = torch.tensor([[0, 1]]).expand(comparisons.shape)
    dataset = RankingDataset(
        X=dataset_X,
        Y=dataset_Y,
        feature_names=parameters,
        outcome_names=[outcome],
    )
    return dataset


def _binary_pref_to_comp_pair(Y: Tensor) -> Tensor:
    """Convert Y from binary indicator pair to index pair comparisons

    Convert Y from binary indicator pair such as [[0, 1], [1, 0], ...]
    to index comparisons like [[1, 0], [2, 3], ...]
    """
    Y_shape = Y.shape[:-2] + (-1, 2)
    Y = Y.reshape(Y_shape)

    # ==== Check if Ys have valid values ====
    # Y must have even number of elements
    if Y.shape[-1] != 2:
        raise ValueError(
            f"Trailing dimension of `Y` should be size 2 but is {Y.shape[-1]}"
        )
    # all adjacent pairs must have exactly a 0 and a 1
    if not (Y.min(dim=-1).values.eq(0).all() and Y.max(dim=-1).values.eq(1).all()):
        raise ValueError("`Y` values must be `{0, 1}.`")

    idx_shift = (torch.arange(0, Y.shape[-2]) * 2).unsqueeze(-1).expand_as(Y)
    comparison_pairs = idx_shift + (1 - Y)
    return comparison_pairs


def _consolidate_comparisons(X: Tensor, Y: Tensor) -> tuple[Tensor, Tensor]:
    """Drop duplicated Xs and update the indices in Ys accordingly"""
    if Y.shape[-1] != 2:
        raise ValueError(
            "The last dimension of Y must contain 2 elements "
            "representing the pairwise comparison."
        )

    if len(Y.shape) != 2:
        raise ValueError("Y must have 2 dimensions.")

    X, Y, _ = consolidate_duplicates(X, Y)
    return X, Y
