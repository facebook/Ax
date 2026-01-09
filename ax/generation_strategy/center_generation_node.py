#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import Any

from ax.adapter.registry import Generators
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationFeatures
from ax.core.parameter import DerivedParameter
from ax.core.search_space import HierarchicalSearchSpace, SearchSpace
from ax.core.types import TParameterization
from ax.exceptions.generation_strategy import AxGenerationException
from ax.generation_strategy.external_generation_node import ExternalGenerationNode
from ax.generation_strategy.generator_spec import GeneratorSpec
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
            name="CenterOfSearchSpace",
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
        self.fallback_specs: dict[type[Exception], GeneratorSpec] = {
            AxGenerationException: GeneratorSpec(
                generator_enum=Generators.SOBOL, generator_key_override="Fallback_Sobol"
            ),
            **self.fallback_specs,  # This includes the default fallbacks.
        }
        # custom property to enable single center point computation
        self._center_params: TParameterization | None = None

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        # State is already set in gen() and will persist during generation
        pass

    def gen(
        self,
        *,
        experiment: Experiment,
        pending_observations: dict[str, list[ObservationFeatures]] | None,
        skip_fit: bool = False,
        data: Data | None = None,
        n: int | None = None,
        arms_per_node: dict[str, int] | None = None,
        **gs_gen_kwargs: Any,
    ) -> GeneratorRun | None:
        """Generate candidates or skip if search space is exhausted.

        This method checks if the center point already exists or is infeasible
        before attempting generation. If so, it sets _should_skip to True and
        returns None, allowing the generation strategy to transition to the next node.
        """
        self.search_space = experiment.search_space
        self._center_params = self.compute_center_params()

        # Check if unable to find a suitable center
        if self._center_params is None:
            self._should_skip = True
            return None

        # Check if center already exists in experiment
        center_arm = Arm(parameters=self._center_params)
        if center_arm.signature in experiment.arms_by_signature:
            self._should_skip = True
            return None

        return super().gen(
            experiment=experiment,
            pending_observations=pending_observations,
            skip_fit=skip_fit,
            data=data,
            n=n,
            arms_per_node=arms_per_node,
            **gs_gen_kwargs,
        )

    def compute_center_params(self) -> TParameterization | None:
        """Compute the center of the search space.

        Returns:
            The center parameters, or None if the center cannot be computed
            (e.g., due to infeasible constraints).
        """
        search_space = none_throws(self.search_space)
        parameters = search_space.compute_naive_center()

        # Check for search space membership, which will check if the generated
        # point satisfies the parameter constraints. Fallback to Chebyshev center
        if not search_space.check_membership(parameterization=parameters):
            chebyshev_center = search_space.compute_chebyshev_center()
            if chebyshev_center is not None:
                for name, value in chebyshev_center.items():
                    if name in parameters:
                        parameters[name] = search_space[name].cast(value)

            # recompute derived parameters using the updated parameter values
            derived_params = [
                p
                for p in search_space.parameters.values()
                if isinstance(p, DerivedParameter)
            ]
            for p in derived_params:
                parameters[p.name] = p.compute(parameters=parameters)

            if isinstance(search_space, HierarchicalSearchSpace):
                parameters = search_space._cast_parameterization(parameters=parameters)

            # Return None if something goes wrong, or some non-range parameter
            # remains out of search space
            if chebyshev_center is None or not search_space.check_membership(
                parameterization=parameters
            ):
                return None
        return parameters

    def get_next_candidate(
        self, pending_parameters: list[TParameterization]
    ) -> TParameterization:
        """Sample the center of the search space.

        For range parameters, the center is the midpoint of the range. If the
        parameter is log-scale, then the center point will correspond to the
        mid-point in log-scale. If the parameter is logit-scale, then the center
        point will correspond to the mid-point in logit-scale.
        For choice parameters, the center point is determined as the value
        that is at the middle of the values list.
        For both choice and integer range parameters, the ties are broken in
        favor of the larger value / index. For example, a binary parameter with
        values [0, 1] will be sampled as 1.
        Fixed parameters are returned at their only allowed value.

        Note: If range naive midpoint fails to remain within parameter constraints, we
        attempt to compute the Chebyshev center of the constraint polytope defined by
        parameter bounds and parameter constraints w.r.t non-log range parameters.
        This finds the center of the largest inscribed ball in the feasible region.
        """
        return none_throws(self._center_params)
