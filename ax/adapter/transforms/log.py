#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from typing import Optional, TYPE_CHECKING

import numpy as np
from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.core.observation import Observation, ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TNumeric, TParamValue
from ax.generators.types import TConfig
from pyre_extensions import assert_is_instance

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


class Log(Transform):
    """Apply log base 10 to a RangeParameter.

    Transform is done in-place.
    Integer log-scale RangeParameter are converted to ChoiceParameter.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: Optional["adapter_module.base.Adapter"] = None,
        config: TConfig | None = None,
    ) -> None:
        assert search_space is not None, "Log requires search space"
        super().__init__(
            search_space=search_space,
            observations=observations,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        # Identify parameters that should be transformed
        self.transform_parameters: dict[str, ParameterType] = {
            p_name: p.parameter_type
            for p_name, p in search_space.parameters.items()
            if isinstance(p, RangeParameter) and p.is_numeric and p.log_scale
        }

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in self.transform_parameters:
                if p_name in obsf.parameters:
                    value = assert_is_instance(obsf.parameters[p_name], TNumeric)
                    obsf.parameters[p_name] = math.log10(value)
        return observation_features

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        for p_name, p in search_space.parameters.items():
            if p_name in self.transform_parameters:
                if (
                    isinstance(p, RangeParameter)
                    and p.parameter_type == ParameterType.FLOAT
                ):
                    # Don't round in log space
                    if p.digits is not None:
                        p.set_digits(digits=None)
                    p.set_log_scale(False).update_range(
                        lower=math.log10(p.lower), upper=math.log10(p.upper)
                    )
                    if p.target_value is not None:
                        target_value = assert_is_instance(p.target_value, float)
                        p._target_value = math.log10(target_value)
                elif (
                    isinstance(p, RangeParameter)
                    and p.parameter_type == ParameterType.INT
                ):
                    # Convert integer valued RangeParameter to ChoiceParameter
                    lower = assert_is_instance(p.lower, int)
                    upper = assert_is_instance(p.upper, int)
                    values = list(range(lower, upper + 1))
                    transformed_values = [
                        assert_is_instance(math.log10(v), TParamValue) for v in values
                    ]

                    target_value = p.target_value
                    if target_value is not None:
                        target_value = assert_is_instance(target_value, int)
                        target_value = math.log10(target_value)

                    # Create new ChoiceParameter to replace the RangeParameter
                    choice_param = ChoiceParameter(
                        name=p.name,
                        parameter_type=ParameterType.FLOAT,
                        values=transformed_values,
                        is_ordered=True,
                        is_fidelity=p.is_fidelity,
                        target_value=target_value,
                        sort_values=True,
                        bypass_cardinality_check=True,
                    )

                    # Replace the parameter in the search space
                    search_space.parameters[p_name] = choice_param
        return search_space

    def untransform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, p_type in self.transform_parameters.items():
                if p_name in obsf.parameters:
                    param: float = assert_is_instance(obsf.parameters[p_name], float)
                    val = math.pow(10, param)

                    # Round to nearest integer if original parameter type is int
                    if p_type == ParameterType.INT:
                        val = round(val)

                    obsf.parameters[p_name] = val
        return observation_features

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        arm_data = experiment_data.arm_data
        for p_name in self.transform_parameters:
            arm_data[p_name] = np.log10(arm_data[p_name])
        return ExperimentData(
            arm_data=arm_data, observation_data=experiment_data.observation_data
        )
