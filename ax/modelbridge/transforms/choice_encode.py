#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from ax.core.observation import Observation, ObservationFeatures
from ax.core.parameter import ChoiceParameter, Parameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TParamValue
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.utils import (
    ClosestLookupDict,
    construct_new_search_space,
)
from ax.models.types import TConfig

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401


class ChoiceEncode(Transform):
    """Convert general ChoiceParameters to integer or float ChoiceParameters.

    If the parameter type is numeric (int, float) and the parameter is ordered,
    then the values are normalized to the unit interval while retaining relative
    spacing. If the parameter type is unordered (categorical) or ordered but
    non-numeric, this transform uses an integer encoding to `0, 1, ..., n_choices - 1`.
    The resulting choice parameter will be considered ordered iff the original
    parameter is.

    In the inverse transform, parameters will be mapped back onto the original domain.

    This transform does not transform task parameters (use TaskEncode for this).

    Note that this behavior is different from that of OrderedChoiceEncode, which
    transforms (ordered) ChoiceParameters to integer RangeParameters (rather than
    ChoiceParameters).

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        observations: Optional[List[Observation]] = None,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        assert search_space is not None, "ChoiceEncode requires search space"
        # Identify parameters that should be transformed
        self.encoded_parameters: Dict[str, Dict[TParamValue, TParamValue]] = {}
        self.encoded_parameters_inverse: Dict[str, ClosestLookupDict] = {}
        for p in search_space.parameters.values():
            if isinstance(p, ChoiceParameter) and not p.is_task:
                transformed_values, _ = transform_choice_values(p)
                self.encoded_parameters[p.name] = dict(
                    zip(p.values, transformed_values)
                )
                self.encoded_parameters_inverse[p.name] = ClosestLookupDict(
                    zip(transformed_values, p.values)
                )

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in self.encoded_parameters:
                if p_name in obsf.parameters:
                    obsf.parameters[p_name] = self.encoded_parameters[p_name][
                        obsf.parameters[p_name]
                    ]
        return observation_features

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        transformed_parameters: Dict[str, Parameter] = {}
        for p_name, p in search_space.parameters.items():
            if p_name in self.encoded_parameters and isinstance(p, ChoiceParameter):
                if p.is_fidelity:
                    raise ValueError(
                        f"Cannot choice-encode fidelity parameter {p_name}"
                    )
                tvals, ptype = transform_choice_values(p)
                transformed_parameters[p_name] = ChoiceParameter(
                    name=p_name,
                    parameter_type=ptype,
                    values=tvals.tolist(),
                    is_ordered=p.is_ordered,
                    sort_values=p.sort_values,
                )
            else:
                transformed_parameters[p.name] = p
        return construct_new_search_space(
            search_space=search_space,
            parameters=list(transformed_parameters.values()),
            parameter_constraints=[
                pc.clone_with_transformed_parameters(
                    transformed_parameters=transformed_parameters
                )
                for pc in search_space.parameter_constraints
            ],
        )

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, reverse_transform in self.encoded_parameters_inverse.items():
                if p_name in obsf.parameters:
                    # pyre: pval is declared to have type `int` but is used as
                    # pyre-fixme[9]: type `Union[bool, float, str]`.
                    pval: int = obsf.parameters[p_name]
                    obsf.parameters[p_name] = reverse_transform[pval]
        return observation_features


class OrderedChoiceEncode(ChoiceEncode):
    """Convert ordered ChoiceParameters to integer RangeParameters.

    Parameters will be transformed to an integer RangeParameters, mapped from the
    original choice domain to a contiguous range `0, 1, ..., n_choices - 1`
    of integers. Does not transform task parameters.

    In the inverse transform, parameters will be mapped back onto the original domain.

    In order to encode all ChoiceParameters (not just ordered ChoiceParameters),
    use ChoiceEncode instead.

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        observations: List[Observation],
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        # Identify parameters that should be transformed
        self.encoded_parameters: Dict[str, Dict[TParamValue, int]] = {}
        for p in search_space.parameters.values():
            if isinstance(p, ChoiceParameter) and p.is_ordered and not p.is_task:
                self.encoded_parameters[p.name] = {
                    original_value: transformed_value
                    for transformed_value, original_value in enumerate(p.values)
                }
        self.encoded_parameters_inverse: Dict[str, Dict[int, TParamValue]] = {
            p_name: {
                transformed_value: original_value
                for original_value, transformed_value in transforms.items()
            }
            for p_name, transforms in self.encoded_parameters.items()
        }

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        transformed_parameters: Dict[str, Parameter] = {}
        for p_name, p in search_space.parameters.items():
            if p_name in self.encoded_parameters and isinstance(p, ChoiceParameter):
                if p.is_fidelity:
                    raise ValueError(
                        f"Cannot choice-encode fidelity parameter {p_name}"
                    )
                # Choice(|K|) => Range(0, K-1)
                transformed_parameters[p_name] = RangeParameter(
                    name=p_name,
                    parameter_type=ParameterType.INT,
                    lower=0,
                    upper=len(p.values) - 1,
                )
            else:
                transformed_parameters[p.name] = p
        return construct_new_search_space(
            search_space=search_space,
            parameters=list(transformed_parameters.values()),
            parameter_constraints=[
                pc.clone_with_transformed_parameters(
                    transformed_parameters=transformed_parameters
                )
                for pc in search_space.parameter_constraints
            ],
        )


def transform_choice_values(p: ChoiceParameter) -> Tuple[np.ndarray, ParameterType]:
    """Transforms the choice values and returns the new parameter type.

    If the choices were numeric (int or float) and ordered, then they're cast
    to float and rescaled to [0, 1]. Otherwise, they're cast to integers
    `0, 1, ..., n_choices - 1`.
    """
    if p.is_numeric and p.is_ordered:
        # If values are ordered numeric, retain relative distances.
        values = np.array(p.values, dtype=float)
        vmin, vmax = values.min(), values.max()
        if len(values) > 1:
            values = (values - vmin) / (vmax - vmin)
        ptype = ParameterType.FLOAT
    else:
        # If values are unordered or not numeric, use integer encoding.
        # The reason for using integers rather than floats is somewhat arcane - it has
        # to do with slightly different representation of floats in pure python and in
        # PyTorch, which require some careful handling when untransform the choices that
        # a model may generate on the botorch end. Ints do not have this issue, so we
        # are using them here.
        values = np.arange(len(p.values))
        ptype = ParameterType.INT
    return values, ptype
