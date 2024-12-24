#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Optional, TYPE_CHECKING

from ax.core.observation import Observation
from ax.core.parameter import ChoiceParameter, Parameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.core.types import TParamValue
from ax.modelbridge.transforms.choice_encode import OrderedChoiceToIntegerRange

from ax.modelbridge.transforms.deprecated_transform_mixin import (
    DeprecatedTransformMixin,
)
from ax.modelbridge.transforms.utils import construct_new_search_space
from ax.models.types import TConfig

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401


class TaskChoiceToIntTaskChoice(OrderedChoiceToIntegerRange):
    """Convert task ChoiceParameters to integer-valued ChoiceParameters.

    Parameters will be transformed to an integer ChoiceParameter with
    property `is_task=True`, mapping values from the original choice domain to a
    contiguous range integers `0, 1, ..., n_choices-1`.

    In the inverse transform, parameters will be mapped back onto the original domain.

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: TConfig | None = None,
    ) -> None:
        assert (
            search_space is not None
        ), "TaskChoiceToIntTaskChoice requires search space"
        # Identify parameters that should be transformed
        self.encoded_parameters: dict[str, dict[TParamValue, int]] = {}
        self.target_values: dict[str, int] = {}
        for p in search_space.parameters.values():
            if isinstance(p, ChoiceParameter) and p.is_task:
                if p.is_fidelity:
                    raise ValueError(
                        f"Task parameter {p.name} cannot simultaneously be "
                        "a fidelity parameter."
                    )
                self.encoded_parameters[p.name] = {
                    original_value: transformed_value
                    for transformed_value, original_value in enumerate(p.values)
                }
                self.target_values[p.name] = self.encoded_parameters[p.name][
                    p.target_value
                ]
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
                        f"Cannot choice-encode fidelity parameter {p_name}."
                    )
                # Choice(|K|) => Choice(0, K-1, is_task=True)
                transformed_parameters[p_name] = ChoiceParameter(
                    name=p_name,
                    parameter_type=ParameterType.INT,
                    values=list(range(len(p.values))),
                    is_ordered=p.is_ordered,
                    is_task=True,
                    sort_values=True,
                    target_value=self.target_values[p_name],
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


class TaskEncode(DeprecatedTransformMixin, TaskChoiceToIntTaskChoice):
    """Deprecated alias for TaskChoiceToIntTaskChoice."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
