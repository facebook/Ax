#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import math
from typing import List, TYPE_CHECKING

from ax.core.parameter import FixedParameter, ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.base import Transform
from ax.utils.common.typeutils import checked_cast

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401


class SimplifyParameterConstraints(Transform):
    """Convert parameter constraints on one parameter to an updated bound.

    This transform converts parameter constraints on only one parameter into an updated
    upper or lower bound. Note that this transform will convert parameters that can only
    take on one value into a `FixedParameter`. Make sure this transform is applied
    before `RemoveFixed` if you want to remove all fixed parameters.
    """

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        # keeps track of the constraints that cannot be converted to bounds
        nontrivial_constraints: List[ParameterConstraint] = []
        for pc in search_space.parameter_constraints:
            if len(pc.constraint_dict) == 1:
                # This can be turned into an updated bound since only one variable is
                # involved in the constraint.
                [(p_name, weight)] = pc.constraint_dict.items()
                # NOTE: We only allow parameter constraints on range parameters
                p = checked_cast(RangeParameter, search_space.parameters[p_name])
                lb, ub = p.lower, p.upper
                if weight == 0 and pc.bound < 0:  # Cannot be satisfied
                    raise ValueError(
                        "Parameter constraint cannot be satisfied since the weight "
                        "is zero and the bound is negative."
                    )
                elif weight == 0:  # Constraint is always satisfied
                    continue
                elif weight > 0:  # New upper bound
                    ub = float(pc.bound) / float(weight)
                    if p.parameter_type == ParameterType.INT:
                        ub = math.floor(ub)  # Round down
                else:  # New lower bound
                    lb = float(pc.bound) / float(weight)
                    if p.parameter_type == ParameterType.INT:
                        lb = math.ceil(lb)  # Round up

                if lb == ub:  # Need to turn this into a fixed parameter
                    search_space.parameters[p_name] = FixedParameter(
                        name=p_name, parameter_type=p.parameter_type, value=lb
                    )
                elif weight > 0:
                    p._upper = ub
                else:
                    p._lower = lb
            else:
                nontrivial_constraints.append(pc)
        search_space.set_parameter_constraints(nontrivial_constraints)
        return search_space
