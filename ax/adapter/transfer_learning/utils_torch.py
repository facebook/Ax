# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import MutableMapping
from typing import cast

from ax.adapter.transforms.base import Transform
from ax.core.auxiliary_source import AuxiliarySource
from ax.core.parameter import ChoiceParameter, FixedParameter
from ax.core.search_space import SearchSpace
from ax.utils.common.typeutils import assert_is_instance_of_tuple


def get_mapped_parameter_names(
    aux_src: AuxiliarySource,
    target_search_space: SearchSpace,
    transforms: MutableMapping[str, Transform] | None = None,
) -> list[str]:
    """
    Get a list of parameter names that corresponds to the observations that
    are produced by `aux_src.map_observations`.

    Applies the parameter name map specified in `aux_src.transfer_param_config`. If
    `aux_src.update_fixed_params`, then replaces all fixed params with those from
    the target search space.

    Args:
        aux_src: The auxiliary source from which to extract the parameter names.
        target_search_space: The target search space.
        transforms: An optional mapping from transform name to the transform
            instance from the adapter that will consume the mapped parameter
            names. This will ensure that the mapped parameter names are compatible
            with the transformed observations.

    Returns:
        A list of parameter names corresponding to mapped observations.
    """
    # Get the parameters from the auxiliary experiment.
    parameters = {
        p_name: p.clone()
        for p_name, p in aux_src.experiment.search_space.parameters.items()
        # We don't want to include fixed params if we are updating them.
        if not (aux_src.update_fixed_params and isinstance(p, FixedParameter))
    }

    for p_name, p in target_search_space.parameters.items():
        if isinstance(p, FixedParameter) and aux_src.update_fixed_params:
            # Add new fixed params if we're updating them.
            parameters[p_name] = p.clone()
        elif p_name in aux_src.transfer_param_config:
            # Rename any parameters that have different names.
            old_name = aux_src.transfer_param_config[p_name]
            old_param = parameters.pop(old_name)
            old_param._name = p_name
            parameters[p_name] = old_param

    # Rename the dependents of any hierarchical parameters.
    reverse_param_config = {v: k for k, v in aux_src.transfer_param_config.items()}
    for p in parameters.values():
        if p.is_hierarchical:
            assert_is_instance_of_tuple(p, (FixedParameter, ChoiceParameter))
            cast(FixedParameter | ChoiceParameter, p)._dependents = {
                k: [reverse_param_config.get(v_, v_) for v_ in v]
                for k, v in p.dependents.items()
            }

    # Transform the mapped search space to get the final parameter names.
    search_space = aux_src.experiment.search_space.__class__(
        parameters=list(parameters.values())
    )
    if transforms is not None:
        for t in transforms.values():
            search_space = t.transform_search_space(search_space)
    return list(search_space.parameters.keys())
