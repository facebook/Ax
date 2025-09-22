#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional, TYPE_CHECKING

from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.utils import ClosestLookupDict, construct_new_search_space
from ax.core.observation import Observation, ObservationFeatures
from ax.core.parameter import ChoiceParameter, Parameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TParamValue
from ax.generators.types import TConfig

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


class ChoiceToNumericChoice(Transform):
    """Convert non-numeric ChoiceParameters to integer ChoiceParameters.

    If the parameter type is numeric (int, float), the parameter is not modified.
    If the parameter is non-numeric, this transform uses an integer encoding to
    `0, 1, ..., n_choices - 1`. The resulting choice parameter will be considered
    ordered iff the original parameter is.

    In the inverse transform, parameters will be mapped back onto the original domain.

    This transform does not transform task parameters
    (use TaskChoiceToIntTaskChoice for this).

    Note that this behavior is different from that of OrderedChoiceToIntegerRange, which
    transforms (ordered) ChoiceParameters to integer RangeParameters (rather than
    ChoiceParameters).

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: Optional["adapter_module.base.Adapter"] = None,
        config: TConfig | None = None,
    ) -> None:
        assert search_space is not None, "ChoiceToNumericChoice requires search space"
        super().__init__(
            search_space=search_space,
            observations=observations,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        # Identify parameters that should be transformed
        self.encoded_parameters: dict[str, dict[TParamValue, TParamValue]] = {}
        self.encoded_parameters_inverse: dict[str, ClosestLookupDict] = {}
        for p in search_space.parameters.values():
            if isinstance(p, ChoiceParameter) and not p.is_numeric and not p.is_task:
                transformed_values = list(range(len(p.values)))
                self.encoded_parameters[p.name] = dict(
                    zip(p.values, transformed_values)
                )
                self.encoded_parameters_inverse[p.name] = ClosestLookupDict(
                    zip(transformed_values, p.values)
                )

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in self.encoded_parameters:
                if p_name in obsf.parameters:
                    obsf.parameters[p_name] = self.encoded_parameters[p_name][
                        obsf.parameters[p_name]
                    ]
        return observation_features

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        transformed_parameters: dict[str, Parameter] = {}
        for p_name, p in search_space.parameters.items():
            if p_name in self.encoded_parameters and isinstance(p, ChoiceParameter):
                encoding = self.encoded_parameters[p_name]
                dependents = None
                if p.is_hierarchical:
                    # The dependents of hierarchical parameters need to be updated to
                    # reflect the changes by encoding.
                    dependents = {
                        encoding[val]: deps for val, deps in p.dependents.items()
                    }

                transformed_parameters[p_name] = ChoiceParameter(
                    name=p_name,
                    parameter_type=ParameterType.INT,
                    # Explicitly extracting values rather than passing
                    # `list(encoding.values())` should make it possible to correctly
                    # transform a parameter that has different (a subset of) values
                    # than the parameter used to initialize the transform.
                    values=[encoding[val] for val in p.values],
                    is_ordered=p.is_ordered,
                    is_task=p.is_task,
                    is_fidelity=p.is_fidelity,
                    target_value=encoding[p.target_value]
                    if p.target_value is not None
                    else None,
                    sort_values=p.sort_values,
                    dependents=dependents,
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
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, reverse_transform in self.encoded_parameters_inverse.items():
                if p_name in obsf.parameters:
                    # pyre: pval is declared to have type `int` but is used as
                    # pyre-fixme[9]: type `Union[bool, float, str]`.
                    pval: int = obsf.parameters[p_name]
                    if pval in reverse_transform:
                        obsf.parameters[p_name] = reverse_transform[pval]
        return observation_features

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        arm_data = experiment_data.arm_data.copy()
        for parameter_name, new_values in self.encoded_parameters.items():
            arm_data[parameter_name] = arm_data[parameter_name].map(new_values)
        return ExperimentData(
            arm_data=arm_data,
            observation_data=experiment_data.observation_data,
        )


class OrderedChoiceToIntegerRange(ChoiceToNumericChoice):
    """Convert ordered ChoiceParameters to integer RangeParameters.

    Parameters will be transformed to an integer RangeParameters, mapped from the
    original choice domain to a contiguous range `0, 1, ..., n_choices - 1`
    of integers. Does not transform task parameters.

    In the inverse transform, parameters will be mapped back onto the original domain.

    In order to encode all ChoiceParameters (not just ordered ChoiceParameters),
    use ChoiceToNumericChoice instead.

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        observations: list[Observation] | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: Optional["adapter_module.base.Adapter"] = None,
        config: TConfig | None = None,
    ) -> None:
        super().__init__(
            search_space=search_space,
            observations=observations,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        # Identify parameters that should be transformed
        self.encoded_parameters: dict[str, dict[TParamValue, int]] = {}
        for p in search_space.parameters.values():
            if isinstance(p, ChoiceParameter) and p.is_ordered and not p.is_task:
                self.encoded_parameters[p.name] = {
                    original_value: transformed_value
                    for transformed_value, original_value in enumerate(p.values)
                }
        self.encoded_parameters_inverse: dict[str, dict[int, TParamValue]] = {
            p_name: {
                transformed_value: original_value
                for original_value, transformed_value in transforms.items()
            }
            for p_name, transforms in self.encoded_parameters.items()
        }

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        transformed_parameters: dict[str, Parameter] = {}
        for p_name, p in search_space.parameters.items():
            if p_name in self.encoded_parameters and isinstance(p, ChoiceParameter):
                if p.is_fidelity:
                    raise ValueError(
                        f"Cannot choice-encode fidelity parameter {p_name}"
                    )
                # Make sure that the search space is compatible with the encoding.
                encoding = self.encoded_parameters[p_name]
                try:
                    t_values = [encoding[pv] for pv in p.values]
                except KeyError:
                    raise ValueError(
                        f"The parameter {p} contains values that are not present in "
                        "the search space used to initialize the transform. The "
                        f"supported encoding for the parameter {p_name} is {encoding}."
                    )
                min_val = min(t_values)
                len_val = len(t_values)
                # Ensure that the values span a contiguous range.
                if set(t_values) != set(range(min_val, min_val + len_val)):
                    raise ValueError(
                        f"The {self.__class__.__name__} transform requires the "
                        "parameter to be encoded with a contiguous range of integers. "
                        f"The parameter {p} maps to {t_values}, which does not span "
                        f"a contiguous range of integers. For parameter {p_name}, "
                        f"the transform uses {encoding=}."
                    )
                transformed_parameters[p_name] = RangeParameter(
                    name=p_name,
                    parameter_type=ParameterType.INT,
                    lower=min_val,
                    upper=min_val + len_val - 1,
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
