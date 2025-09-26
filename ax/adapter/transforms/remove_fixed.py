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
from ax.core.observation import ObservationFeatures
from ax.core.parameter import (
    ChoiceParameter,
    DerivedParameter,
    FixedParameter,
    Parameter,
)
from ax.core.search_space import HierarchicalSearchSpace, SearchSpace
from ax.generators.types import TConfig

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


def find_adoptable_descendants(
    param: FixedParameter,
    search_space: HierarchicalSearchSpace,
) -> list[str]:
    """
    Find all descendants (if any) of a fixed parameter that needs to be adopted by the
    parent of the fixed parameter.

    Note that the fixed parameter will be removed in the `RemoveFixed` transform. Its
    dependents (if any) need to be adopted by its parent, i.e., the grandparent adopts
    the grandchildren.

    If the dependents of the fixed parameter also contain fixed parameters, then we need
    to recursively to find all descendants to be adopted, e.g., the great-grandparent
    adopts the great-grandchildren.

    Args:
        param: The fixed parameter to be removed.
        search_space: The hierarchical search space that `param` comes from.

    Returns:
        A list of names of adoptable descendants of `param`.
    """
    lst_adoptable_descendants = []

    if param.is_hierarchical:
        for child in next(iter(param.dependents.values())):
            if isinstance(search_space.parameters[child], DerivedParameter):
                continue
            if isinstance(search_space.parameters[child], FixedParameter):
                lst_adoptable_descendants += find_adoptable_descendants(
                    # pyre-ignore[6]: It's a fixed parameter for sure.
                    search_space.parameters[child],
                    search_space=search_space,
                )
            else:
                lst_adoptable_descendants.append(child)

    return lst_adoptable_descendants


class RemoveFixed(Transform):
    """Remove fixed and derived parameters.

    Fixed and derived parameters (i.e., nontunable parameters) should not be included
    in the search space. This transform removes these parameters, leaving only tunable
    parameters.

    Transform is done in-place for observation features.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: Optional["adapter_module.base.Adapter"] = None,
        config: TConfig | None = None,
    ) -> None:
        assert search_space is not None, "RemoveFixed requires search space"
        super().__init__(
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )

        self.nontunable_parameters: dict[str, DerivedParameter | FixedParameter] = {
            name: p.clone() for name, p in search_space.nontunable_parameters.items()
        }

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in self.nontunable_parameters:
                obsf.parameters.pop(p_name, None)
        return observation_features

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        tunable_parameters: list[Parameter] = list(
            search_space.tunable_parameters.values()
        )

        # Also need to update `dependents` if the search space is hierarchical.
        if isinstance(search_space, HierarchicalSearchSpace):
            for p in tunable_parameters:
                # NOTE: Type checking `ChoiceParameter` and `FixedParameter` is entirely
                # unnecessary, because `is_hierarchical` returns false unless it's
                # either a choice or fixed parameter. We do this solely to avoid a type
                # check error from buck tests.
                if (
                    isinstance(p, (ChoiceParameter, FixedParameter))
                    and p.is_hierarchical
                ):
                    dependents = {}

                    # 1. Remove fixed and derived parameters in `dependents`.
                    # 2. If a fixed parameter has dependents, then the parent of the
                    # fixed parameter needs to adopt the dependents.
                    # 3. Set `dependents=None` if all children are removed.
                    for p_value, children in p.dependents.items():
                        updated_children = []

                        for child in children:
                            if isinstance(
                                search_space.parameters[child], DerivedParameter
                            ):
                                # Do nothing. This derived parameter will be removed.
                                continue
                            elif isinstance(
                                search_space.parameters[child], FixedParameter
                            ):
                                updated_children += find_adoptable_descendants(
                                    # pyre-ignore[6]: It's a fixed parameter for sure.
                                    param=search_space.parameters[child],
                                    search_space=search_space,
                                )
                            else:
                                updated_children.append(child)

                        dependents[p_value] = updated_children

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
            # Only untransform observations with specified parameters
            # where at least one of them is not a fixed or derived parameter.
            # This would be empty when status quo param values are not specified
            if obsf.parameters:
                for p_name, p in self.nontunable_parameters.items():
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
                columns=list(self.nontunable_parameters)
            ),
            observation_data=experiment_data.observation_data,
        )
