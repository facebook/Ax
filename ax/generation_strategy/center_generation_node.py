#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from dataclasses import dataclass
from typing import Any

from ax.adapter.registry import Generators
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.core.parameter import (
    ChoiceParameter,
    DerivedParameter,
    FixedParameter,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.core.types import TParameterization
from ax.exceptions.generation_strategy import AxGenerationException
from ax.generation_strategy.external_generation_node import ExternalGenerationNode
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import (
    AutoTransitionAfterGenOrExhaustion,
)
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
            name="CenterOfSearchSpace",
            transition_criteria=[
                AutoTransitionAfterGenOrExhaustion(
                    transition_to=next_node_name,
                    continue_trial_generation=False,
                )
            ],
            should_deduplicate=True,
        )
        self.search_space: SearchSpace | None = None
        self.next_node_name = next_node_name
        self.fallback_specs: dict[type[Exception], GeneratorSpec] = {
            AxGenerationException: GeneratorSpec(
                generator_enum=Generators.SOBOL, generator_key_override="Fallback_Sobol"
            ),
            **self.fallback_specs,  # This includes the default fallbacks.
        }

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        self.search_space = experiment.search_space

    def gen(
        self,
        experiment: Experiment,
        data: Data | None = None,
        pending_observations: dict[str, list[ObservationFeatures]] | None = None,
        skip_fit: bool = False,
        **gs_gen_kwargs: Any,
    ) -> GeneratorRun | None:
        """Generate candidates or skip if search space is exhausted.

        This method checks if the center point already exists or is infeasible
        before attempting generation. If so, it sets _should_skip to True and
        returns None, allowing the generation strategy to transition to the next node.
        """
        # Check if center already exists or is infeasible
        self.search_space = experiment.search_space
        center_params = self._compute_center_params()
        search_space = none_throws(self.search_space)

        # Check if center already exists in experiment
        center_arm = Arm(parameters=center_params)
        if center_arm.signature in experiment.arms_by_signature:
            self._should_skip = True
            return None

        # Check if center violates parameter constraints
        if not search_space.check_membership(parameterization=center_params):
            self._should_skip = True
            return None

        # Otherwise, proceed with normal generation
        return super().gen(
            experiment=experiment,
            data=data,
            pending_observations=pending_observations,
            skip_fit=skip_fit,
            **gs_gen_kwargs,
        )

    def _compute_center_params(self) -> TParameterization:
        """Compute the center of the search space."""
        search_space = none_throws(self.search_space)
        parameters = {}
        derived_params = []
        for name, p in search_space.parameters.items():
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
            elif isinstance(p, DerivedParameter):
                derived_params.append(p)
            else:
                raise NotImplementedError(f"Parameter type {type(p)} is not supported.")
        for p in derived_params:
            parameters[p.name] = p.compute(parameters=parameters)
        if search_space.is_hierarchical:
            parameters = search_space._cast_parameterization(parameters=parameters)
        return parameters

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
        search_space = none_throws(self.search_space)
        parameters = self._compute_center_params()

        # Check for search space membership, which will check if the generated
        # point satisfies the parameter constraints.
        if not search_space.check_membership(parameterization=parameters):
            # TODO: Improve this handling by instead choosing the point
            # in the center of the feasible set (e.g. by finding the)
            # Chebyshev center of the constraint polytope.
            raise AxGenerationException(
                "Center of the search space does not satisfy parameter constraints. "
                "The generation strategy will fallback to Sobol. "
            )
        return parameters
