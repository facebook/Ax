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
from ax.core.observation import Observation
from ax.core.parameter import FixedParameter, Parameter, RangeParameter
from ax.core.search_space import SearchSpace
from ax.generators.types import TConfig

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


class FixedToTunable(Transform):
    """Convert any FixedParameter to a RangeParameter if the SearchSpace that
    is passed on initialization has a RangeParameter of the same name.

    TODO: Add support for ChoiceParameters.
    """

    no_op_for_experiment_data = True

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: Optional["adapter_module.base.Adapter"] = None,
        config: TConfig | None = None,
    ) -> None:
        assert search_space is not None, "FixedToTunable requires search space"
        super().__init__(
            search_space=search_space,
            observations=observations,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        # Store search space. In transfer learning, this will be the joint search
        # space that contains range parameters when one experiment has a fixed
        # parameter and another experiment has a range parameter of the same name.
        self.search_space: SearchSpace = search_space

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        transformed_parameters: dict[str, Parameter] = {}
        for p_name, p in search_space.parameters.items():
            if isinstance(p, FixedParameter) and isinstance(
                target_p := self.search_space.parameters[p_name], RangeParameter
            ):
                transformed_parameters[p_name] = target_p.clone()
            else:
                transformed_parameters[p.name] = p

        return construct_new_search_space(
            search_space=search_space,
            parameters=list(transformed_parameters.values()),
            # the target search space cannot have constraints on a fixed
            # so the parameter constraints will not have changed here.
            parameter_constraints=search_space.parameter_constraints,
        )
