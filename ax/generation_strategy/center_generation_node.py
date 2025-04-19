#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from dataclasses import dataclass

from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.parameter import ChoiceParameter, FixedParameter, RangeParameter
from ax.core.search_space import HierarchicalSearchSpace, SearchSpace
from ax.core.types import TParameterization
from ax.generation_strategy.external_generation_node import ExternalGenerationNode
from ax.generation_strategy.transition_criterion import AutoTransitionAfterGen
from pyre_extensions import none_throws


@dataclass(init=False)
class CenterGenerationNode(ExternalGenerationNode):
    next_node_name: str

    def __init__(self, next_node_name: str) -> None:
        """A generation node that samples the center of the search space.
        This generation node is only used to generate the first point of the experiment.
        After one point is generated, it will transition to `next_node_name`.

        If the generated point is a duplicate of an arm already attached to the
        experiment, this will fallback to Sobol through the use of ``GenerationNode``
        deduplication logic.
        """
        super().__init__(
            node_name="CenterOfSearchSpace",
            transition_criteria=[
                AutoTransitionAfterGen(
                    transition_to=next_node_name,
                    continue_trial_generation=False,
                )
            ],
            should_deduplicate=True,
        )
        self.search_space: SearchSpace | None = None
        self.next_node_name = next_node_name

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        self.search_space = experiment.search_space

    def get_next_candidate(
        self, pending_parameters: list[TParameterization]
    ) -> TParameterization:
        """Sample the center of the search space.

        For range parameters, the center is the midpoint of the range. If the
        parameter is log-scale, then the center point will correspond to the
        mid-point in log-scale.
        For choice parameters, the center point is determined as the value
        that is at the middle of the values list.
        For both choice and integer range parameters, the ties are broken in
        favor of the larger value / index. For example, a binary parameter with
        values [0, 1] will be sampled as 1.
        Fixed parameters are returned at their only allowed value.
        """
        parameters = {}
        for name, p in none_throws(self.search_space).parameters.items():
            if isinstance(p, RangeParameter):
                if p.logit_scale:
                    raise NotImplementedError(f"`logit_scale` is not supported. {p=}")
                if p.log_scale:
                    center = 10 ** ((math.log10(p.lower) + math.log10(p.upper)) / 2.0)
                else:
                    center = (float(p.lower) + float(p.upper)) / 2.0
                parameters[name] = p.cast(center)
            elif isinstance(p, ChoiceParameter):
                parameters[name] = p.values[int(len(p.values) / 2)]
            elif isinstance(p, FixedParameter):
                parameters[name] = p.value
            else:
                raise NotImplementedError(f"Parameter type {type(p)} is not supported.")
        if isinstance(self.search_space, HierarchicalSearchSpace):
            parameters = self.search_space._cast_parameterization(parameters=parameters)
        return parameters
