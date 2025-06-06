#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional, TYPE_CHECKING

from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.utils import construct_new_search_space
from ax.core.observation import Observation, ObservationFeatures
from ax.core.parameter import ChoiceParameter, FixedParameter, RangeParameter
from ax.core.search_space import SearchSpace
from ax.generators.types import TConfig

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


class RemoveFixed(Transform):
    """Remove fixed parameters.

    Fixed parameters should not be included in the SearchSpace.
    This transform removes these parameters, leaving only tunable parameters.

    Transform is done in-place for observation features.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: Optional["adapter_module.base.Adapter"] = None,
        config: TConfig | None = None,
    ) -> None:
        assert search_space is not None, "RemoveFixed requires search space"
        super().__init__(
            search_space=search_space,
            observations=observations,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        # Identify parameters that should be transformed
        self.fixed_parameters: dict[str, FixedParameter] = {
            p_name: p
            for p_name, p in search_space.parameters.items()
            if isinstance(p, FixedParameter)
        }

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in self.fixed_parameters:
                obsf.parameters.pop(p_name, None)
        return observation_features

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        tunable_parameters: list[ChoiceParameter | RangeParameter] = []
        for p in search_space.parameters.values():
            if p.name not in self.fixed_parameters:
                # If it's not in fixed_parameters, it must be a tunable param.
                # pyre: p_ is declared to have type `Union[ChoiceParameter,
                # pyre: RangeParameter]` but is used as type `ax.core.
                # pyre-fixme[9]: parameter.Parameter`.
                p_: ChoiceParameter | RangeParameter = p
                tunable_parameters.append(p_)
        return construct_new_search_space(
            search_space=search_space,
            # pyre-ignore Incompatible parameter type [6]
            parameters=tunable_parameters,
            parameter_constraints=[
                pc.clone() for pc in search_space.parameter_constraints
            ],
        )

    def untransform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, p in self.fixed_parameters.items():
                obsf.parameters[p_name] = p.value
        return observation_features

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        return ExperimentData(
            arm_data=experiment_data.arm_data.drop(columns=list(self.fixed_parameters)),
            observation_data=experiment_data.observation_data,
        )
