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
from ax.core.parameter import (
    ChoiceParameter,
    DerivedParameter,
    FixedParameter,
    RangeParameter,
)
from ax.core.search_space import HierarchicalSearchSpace, SearchSpace
from ax.generators.types import TConfig

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


class RemoveFixed(Transform):
    """Remove fixed and derived parameters.

    Fixed and derived parameters should not be included in the SearchSpace.
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
        self.fixed_or_derived_parameters: dict[
            str, FixedParameter | DerivedParameter
        ] = {
            p_name: p
            for p_name, p in search_space.parameters.items()
            if isinstance(p, (DerivedParameter, FixedParameter))
        }

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in self.fixed_or_derived_parameters:
                obsf.parameters.pop(p_name, None)
        return observation_features

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        # For hierarchical search spaces, the only hierarchical fixed (or derived)
        # parameter has to be the root. This is a tricky case that requires a BFS
        # traversal. We don't support it for now, since we haven't seen any use cases.
        if isinstance(search_space, HierarchicalSearchSpace):
            for p_name, param in search_space.parameters.items():
                if isinstance(param, (DerivedParameter, FixedParameter)):
                    if param.is_hierarchical and param is not search_space.root:
                        raise NotImplementedError(
                            f"{p_name} is a hierarchical fixed or derived parameter."
                            "But it is not the root."
                        )

        tunable_parameters: list[ChoiceParameter | RangeParameter] = []
        for p in search_space.parameters.values():
            if p.name not in self.fixed_or_derived_parameters:
                # If it's not in fixed_or_derived_parameters, it must be a
                # tunable param.
                # pyre: p_ is declared to have type `Union[ChoiceParameter,
                # pyre: RangeParameter]` but is used as type `ax.core.
                # pyre-fixme[9]: parameter.Parameter`.
                p_: ChoiceParameter | RangeParameter = p
                tunable_parameters.append(p_)

        # Also need to remove fixed parameters in `dependents`.
        for p in tunable_parameters:
            # NOTE: Type checking `ChoiceParameter` and `FixedParameter` is entirely
            # unnecessary, because `is_hierarchical` returns false unless it's either a
            # choice or fixed parameter. We do this solely to avoid a type check error
            # from buck tests.
            if isinstance(p, (ChoiceParameter, FixedParameter)) and p.is_hierarchical:
                dependents = {
                    p_value: [
                        child
                        for child in children
                        if child not in self.fixed_or_derived_parameters
                    ]
                    for p_value, children in p.dependents.items()
                }
                if any(len(children) > 0 for children in dependents.values()):
                    p.dependents = dependents
                else:
                    # Wipe out the dependents if all children are removed.
                    p.dependents = None

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
            for p_name, p in self.fixed_or_derived_parameters.items():
                if isinstance(p, DerivedParameter):
                    obsf.parameters[p_name] = p.compute(parameters=obsf.parameters)
                else:
                    obsf.parameters[p_name] = p.value
        return observation_features

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        return ExperimentData(
            arm_data=experiment_data.arm_data.drop(
                columns=list(self.fixed_or_derived_parameters)
            ),
            observation_data=experiment_data.observation_data,
        )
