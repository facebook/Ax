#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Dict, List, MutableMapping, Optional, Tuple

import numpy as np
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.core.types import TBounds
from ax.modelbridge.transforms.base import Transform


def extract_parameter_constraints(
    parameter_constraints: List[ParameterConstraint], param_names: List[str]
) -> Optional[TBounds]:
    """Extract parameter constraints."""
    if len(parameter_constraints) > 0:
        A = np.zeros((len(parameter_constraints), len(param_names)))
        b = np.zeros((len(parameter_constraints), 1))
        for i, c in enumerate(parameter_constraints):
            b[i, 0] = c.bound
            for name, val in c.constraint_dict.items():
                A[i, param_names.index(name)] = val
        linear_constraints: TBounds = (A, b)
    else:
        linear_constraints = None
    return linear_constraints


def get_bounds_and_task(
    search_space: SearchSpace, param_names: List[str]
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """Extract box bounds from a search space in the usual Scipy format.
    Identify integer parameters as task features.
    """
    bounds: List[Tuple[float, float]] = []
    task_features: List[int] = []
    for i, p_name in enumerate(param_names):
        p = search_space.parameters[p_name]
        # Validation
        if not isinstance(p, RangeParameter):
            raise ValueError(f"{p} not RangeParameter")
        elif p.log_scale:
            raise ValueError(f"{p} is log scale")
        # Set value
        bounds.append((p.lower, p.upper))
        if p.parameter_type == ParameterType.INT:
            task_features.append(i)
    return bounds, task_features


def get_fixed_features(
    fixed_features: ObservationFeatures, param_names: List[str]
) -> Optional[Dict[int, float]]:
    """Reformat a set of fixed_features."""
    fixed_features_dict = {}
    for p_name, val in fixed_features.parameters.items():
        # These all need to be floats at this point.
        # pyre-ignore[6]: All float here.
        val_ = float(val)
        fixed_features_dict[param_names.index(p_name)] = val_
    fixed_features_dict = fixed_features_dict if len(fixed_features_dict) > 0 else None
    return fixed_features_dict


def get_pending_observations(
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
        pending_array: Optional[List[np.ndarray]] = None
    else:
        pending_array = [np.array([]) for _ in outcome_names]
        for metric_name, po_list in pending_observations.items():
            pending_array[outcome_names.index(metric_name)] = np.array(
                [[po.parameters[p] for p in param_names] for po in po_list]
            )
    return pending_array


def parse_observation_features(
    X: np.ndarray, param_names: List[str]
) -> List[ObservationFeatures]:
    """Re-format raw model-generated candidates into ObservationFeatures.

    Args:
        param_names: List of param names.
        X: Raw np.ndarray of candidate values.

    Returns:
        List of candidates, represented as ObservationFeatures.
    """
    observation_features = []
    for x in X:
        observation_features.append(
            ObservationFeatures(parameters={p: x[i] for i, p in enumerate(param_names)})
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
        # pyre-fixme[6]: Expected `Sequence[_T]` for 1st param but got `ValuesView[Tr...
        for t in reversed(transforms.values()):
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
            float(observation_features[0].parameters[p])  # pyre-ignore
            for p in param_names
        ]
        # turn it back into an array
        return np.array(new_x)

    return _roundtrip_transform
