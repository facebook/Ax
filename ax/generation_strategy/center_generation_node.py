#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from dataclasses import dataclass

import numpy as np
from ax.adapter.registry import Generators
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.parameter import (
    ChoiceParameter,
    DerivedParameter,
    FixedParameter,
    RangeParameter,
)
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import HierarchicalSearchSpace, SearchSpace
from ax.core.types import TParameterization
from ax.exceptions.generation_strategy import AxGenerationException
from ax.generation_strategy.external_generation_node import ExternalGenerationNode
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import AutoTransitionAfterGen
from pyre_extensions import none_throws
from scipy.optimize import linprog


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
        self.fallback_specs: dict[type[Exception], GeneratorSpec] = {
            AxGenerationException: GeneratorSpec(
                generator_enum=Generators.SOBOL, model_key_override="Fallback_Sobol"
            ),
            **self.fallback_specs,  # This includes the default fallbacks.
        }

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        self.search_space = experiment.search_space

    def _compute_chebyshev_center(
        self,
        parameters: dict[str, RangeParameter],
        parameter_constraints: list[ParameterConstraint],
    ) -> dict[str, float] | None:
        """Compute the Chebyshev center of the constraint polytope.

        The Chebyshev center is the center of the largest inscribed ball in the
        feasible region defined by the parameter constraints. This is computed
        by solving a linear program. It is most limited by the tightest constraint.

        For a polytope defined by a @ x <= b, the Chebyshev center (x_c, r) is
        the solution to:
            maximize r, where r is the radius of the inscribed ball
            subject to: a_i^T x + r ||a_i||_2 <= b_i for all i

        Args:
            parameters: all parameters in a given search space to be used in chebyshev
                calculation
            parameter_constraints: all parameter constraints in a given search space

        Note: this only considers natural range parameters, other parameter types are
            handled naively.

        Returns:
            A dictionary mapping parameter names to values at the Chebyshev center,
            or none if the problem is infeasible.
        """
        # construct constraint matrix, each row represents one constraint and number
        # columns == number of params, with each column representing a parameter
        # constraints are of the form: sum(A_i,n*param_1 + A_i,n+1*param_1, ..) < b_i
        constraint_matrix = []
        bound_vector = []
        param_names = list(parameters.keys())
        num_params = len(parameters)
        param_name_to_idx = {name: idx for idx, name in enumerate(param_names)}

        # add parameter constraints
        for constraint in parameter_constraints:
            row = np.zeros(num_params)
            for param_name, weight in constraint.constraint_dict.items():
                if param_name in param_name_to_idx:
                    row[param_name_to_idx[param_name]] = weight
            constraint_matrix.append(row)
            bound_vector.append(constraint.bound)

        # add parameter bounds
        for name, idx in param_name_to_idx.items():
            param = parameters[name]
            # lower bound: -x_i <= -lower_i
            row_lower = np.zeros(num_params)
            row_lower[idx] = -1.0
            constraint_matrix.append(row_lower)
            bound_vector.append(-float(param.lower))

            # upper bound: x_i <= upper_i
            row_upper = np.zeros(num_params)
            row_upper[idx] = 1.0
            constraint_matrix.append(row_upper)
            bound_vector.append(float(param.upper))

        constraint_matrix = np.array(constraint_matrix)
        bound_vector = np.array(bound_vector)

        # compute norm for each vector in constraint matrix and add this to the
        # constraint matrix as a new column representing variable r, which will
        # allow us to solve the linear program to maximize r. The row norms
        # represent how much the radius contributes to violating the constraint
        # and are ||a_i||_2]
        row_norms = np.linalg.norm(constraint_matrix, axis=1)
        augmented_constraint_matrix = np.column_stack([constraint_matrix, row_norms])

        # set objective vector which maximizes r (minimize -r == maximize r)
        radius_objective_vector = np.zeros(num_params + 1)
        radius_objective_vector[-1] = -1.0
        result = linprog(
            c=radius_objective_vector,
            A_ub=augmented_constraint_matrix,
            b_ub=bound_vector,
            bounds=[(None, None)] * num_params + [(0, None)],  # no bounds except r >= 0
        )

        if not result.success or result.x is None:
            return None

        center_values = result.x[:num_params]  # remove r
        center_dict = {
            name: float(center_values[param_name_to_idx[name]]) for name in param_names
        }
        return center_dict

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

        Note: If range naive midpoint fails to remain within parameter constraints, we
        attempt to compute the Chebyshev center of the constraint polytope defined by
        parameter bounds and parameter constraints w.r.t non-log range parameters.
        This finds the center of the largest inscribed ball in the feasible region.
        """
        search_space = none_throws(self.search_space)
        parameters = {}
        derived_params = []

        # Compute naive mid-point
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

        # compute derived midpoint using parameter midpoint values
        for p in derived_params:
            parameters[p.name] = p.compute(parameters=parameters)

        if isinstance(search_space, HierarchicalSearchSpace):
            parameters = search_space._cast_parameterization(parameters=parameters)

        # Check for search space membership, which will check if the generated
        # point satisfies the parameter constraints. Fallback to Chebyshev center
        if not search_space.check_membership(parameterization=parameters):
            # Note: only consider non-logscale range parameters, since logscale params
            # are not able to have parameter constraints currently. If we modify this
            # expectation, we'll need to extend the chebyshev center calculation to
            # work in mixed log/natural space
            natural_range_params = {
                name: param
                for name, param in search_space.range_parameters.items()
                if not param.log_scale and not param.logit_scale
            }
            chebyshev_center = self._compute_chebyshev_center(
                parameters=natural_range_params,
                parameter_constraints=search_space.parameter_constraints,
            )
            if chebyshev_center is not None:
                for name, value in chebyshev_center.items():
                    if name in parameters:
                        parameters[name] = search_space[name].cast(value)

            # compute derived midpoint using parameter midpoint values
            for p in derived_params:
                parameters[p.name] = p.compute(parameters=parameters)

            if isinstance(search_space, HierarchicalSearchSpace):
                parameters = search_space._cast_parameterization(parameters=parameters)

            # fallback in case something goes wrong, or some non-range parameter
            # remains out of search space
            if chebyshev_center is None or not search_space.check_membership(
                parameterization=parameters
            ):
                raise AxGenerationException(
                    "Center of the search space does not satisfy parameter "
                    "constraints. The generation strategy will fallback to Sobol. "
                )
        return parameters
