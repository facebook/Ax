# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# This should not include any torch dependencies!

from typing import Union

from ax.core.auxiliary_source import AuxiliarySource
from ax.core.parameter import (
    ChoiceParameter,
    DerivedParameter,
    FixedParameter,
    Parameter,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.core.types import TParamValue


TFixedOrChoice = Union[FixedParameter, ChoiceParameter]


def merge_dependents(
    p1: TFixedOrChoice,
    p2: TFixedOrChoice,
    reverse_param_config: dict[str, str],
) -> dict[TParamValue, list[str]] | None:
    """Merge the dependents of two fixed or choice parameters.
    The dependent parameters from p2 are renamed according to the
    reverse_param_config.
    """
    dependents1 = p1._dependents
    dependents2 = p2._dependents
    if dependents2 is None:
        return dependents1
    elif dependents1 is None:
        if dependents2 is None:
            return dependents2
        else:
            return {
                k: [reverse_param_config.get(v_, v_) for v_ in v]
                for k, v in dependents2.items()
            }
    else:
        new_dependents = {}
        all_keys = set(dependents1.keys()).union(set(dependents2.keys()))
        for key in all_keys:
            new_dependents[key] = list(
                set(dependents1.get(key, set())).union(
                    {
                        reverse_param_config.get(v_, v_)
                        for v_ in dependents2.get(key, set())
                    }
                )
            )
        return new_dependents


def merge_parameters(
    p1: Parameter,
    p2: Parameter,
    reverse_param_config: dict[str, str],
    update_fixed_params: bool = False,
) -> Parameter:
    """Merge two parameters into a single parameter. The parameters should either
    have the same name or p2 should be mapped to p1 using `reverse_param_config`.
    If any attribute not mentioned below is different between the two parameters,
    we will use the attribute from p1.

    If the two parameters are of different types, this function will raise an error.

    If both are range parameters, they will be merged into a range parameter
    that encapsulates the bounds of both parameters.

    If both are fixed parameters, the value from p1 will be used.

    If both are choice parameters, they will be merged into a choice parameter that
    includes the union of the values of the two parameters.

    If one is a fixed parameter and the other a choice parameter, they will be
    merged into a choice parameter whose values include the fixed value.

    If the parameters have dependents (for hierarchical search spaces), then the
    dependents will be merged together.
    """
    if p1.name != reverse_param_config.get(p2.name, p2.name):
        raise ValueError(
            f"Cannot merge parameters with different names: {p1=}, {p2=}. "
            "Use `AuxiliarySource.transfer_param_config` to map p2 to p1."
        )
    p1_type = type(p1)
    p2_type = type(p2)
    allowed_mixed_pairs = (
        {FixedParameter, RangeParameter},
        {FixedParameter, ChoiceParameter},
    )
    if (
        p1_type is not p2_type and ({p1_type, p2_type} not in allowed_mixed_pairs)
    ) or p1.parameter_type != p2.parameter_type:
        raise ValueError(f"Cannot merge parameters of different types: {p1}, {p2}.")
    if isinstance(p1, RangeParameter) and isinstance(p2, RangeParameter):
        return RangeParameter(
            name=p1.name,
            parameter_type=p1.parameter_type,
            lower=min(p1.lower, p2.lower),
            upper=max(p1.upper, p2.upper),
            log_scale=p1.log_scale,
            logit_scale=p1.logit_scale,
            digits=p1.digits,
            is_fidelity=p1.is_fidelity,
            target_value=p1.target_value,
        )
    elif isinstance(p1, FixedParameter) and isinstance(p2, FixedParameter):
        return FixedParameter(
            name=p1.name,
            parameter_type=p1.parameter_type,
            value=p1.value,
            is_fidelity=p1.is_fidelity,
            target_value=p1.target_value,
            dependents=merge_dependents(
                p1=p1, p2=p2, reverse_param_config=reverse_param_config
            ),
        )
    elif (
        isinstance(fixed_param := p1, FixedParameter)
        and isinstance(range_param := p2, RangeParameter)
    ) or (
        isinstance(fixed_param := p2, FixedParameter)
        and isinstance(range_param := p1, RangeParameter)
    ):
        # Copy the range parameter. FixedToTunable will
        # then convert the fixed parameter in the target
        # search space into a tunable parameter so that it
        # is included in the model. fixed_features are set
        # accordingly in TransferLearningAdapter.gen to
        # make sure we don't adjust the value.
        # Expand the range here, since the TL Adapter will not use
        # information from auxiliary sources in determining the model_space
        new_range_param = range_param.clone()
        new_range_param.update_range(
            lower=min(range_param.lower, range_param.cast(fixed_param.value)),
            upper=max(range_param.upper, range_param.cast(fixed_param.value)),
        )
        return new_range_param
    elif (
        isinstance(fixed_param := p1, FixedParameter)
        and isinstance(choice_param := p2, ChoiceParameter)
    ) or (
        isinstance(fixed_param := p2, FixedParameter)
        and isinstance(choice_param := p1, ChoiceParameter)
    ):
        # Merge FixedParameter into ChoiceParameter by including the fixed
        # value in the set of choice values.
        values = list(set(choice_param.values) | {fixed_param.value})
        return ChoiceParameter(
            name=p1.name,
            parameter_type=p1.parameter_type,
            values=values,
            is_ordered=choice_param.is_ordered,
            is_task=choice_param.is_task,
            is_fidelity=choice_param.is_fidelity,
            target_value=choice_param.target_value,
            sort_values=choice_param.sort_values,
            dependents=merge_dependents(
                # pyre-ignore[6]: p1/p2 are FixedParameter | ChoiceParameter here.
                p1=p1,
                # pyre-ignore[6]: p1/p2 are FixedParameter | ChoiceParameter here.
                p2=p2,
                reverse_param_config=reverse_param_config,
            ),
        )
    elif isinstance(p1, ChoiceParameter) and isinstance(p2, ChoiceParameter):
        return ChoiceParameter(
            name=p1.name,
            parameter_type=p1.parameter_type,
            values=list(set(p1.values).union(set(p2.values))),
            is_ordered=p1.is_ordered,
            is_task=p1.is_task,
            is_fidelity=p1.is_fidelity,
            target_value=p1.target_value,
            sort_values=p1.sort_values,
            dependents=merge_dependents(
                p1=p1, p2=p2, reverse_param_config=reverse_param_config
            ),
        )
    elif isinstance(p1, DerivedParameter) and isinstance(p2, DerivedParameter):
        if p1.expression_str != p2.expression_str:
            raise ValueError(
                f"Cannot merge DerivedParameters with different expressions: "
                f"{p1.name} has '{p1.expression_str}' vs '{p2.expression_str}'."
            )
        return p1.clone()
    else:  # pragma: no cover
        raise NotImplementedError(f"Unknown parameter type: {p1}.")


def get_joint_search_space(
    search_space: SearchSpace,
    auxiliary_sources: list[AuxiliarySource],
) -> SearchSpace:
    """Get the joint search space consisting of all the parameters from
    the experiment and auxiliary sources, after renaming the parameters
    from the auxiliary sources according to the config.

    Args:
        search_space: The target search space.
        auxiliary_sources: The list of auxiliary sources.

    Returns:
        The joint search space. This will not contain any parameter constraints.
    """
    all_parameters = search_space.parameters.copy()
    for aux_src in auxiliary_sources:
        aux_params = aux_src.experiment.search_space.parameters.copy()
        transfer_param_values = aux_src.transfer_param_config.values()
        reverse_param_config = {v: k for k, v in aux_src.transfer_param_config.items()}
        for param_name, param in aux_params.items():
            if param_name in transfer_param_values:
                # The parameter is mapped to one of the target parameters.
                # We need to merge the two parameters.
                new_param_name = reverse_param_config[param_name]
                all_parameters[new_param_name] = merge_parameters(
                    p1=all_parameters[new_param_name],
                    p2=param,
                    reverse_param_config=reverse_param_config,
                    update_fixed_params=aux_src.update_fixed_params,
                )
            elif param_name in all_parameters:
                # A parameter with the same name exists. Merge the two parameters.
                all_parameters[param_name] = merge_parameters(
                    p1=all_parameters[param_name],
                    p2=param,
                    reverse_param_config=reverse_param_config,
                    update_fixed_params=aux_src.update_fixed_params,
                )
            else:
                # This is a new parameter.
                all_parameters[param_name] = param
    # Clone the parameters to be safe against any in-place modification.
    new_search_space = search_space.__class__(
        parameters=[p.clone() for p in all_parameters.values()]
    )
    return new_search_space
