# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.api.utils.instantiation.from_config import parameter_from_config
from ax.api.utils.instantiation.from_string import parse_parameter_constraint
from ax.api.utils.structs import ExperimentStruct
from ax.core.experiment import Experiment
from ax.core.formatting_utils import DataType
from ax.core.parameter_constraint import validate_constraint_parameters
from ax.core.search_space import HierarchicalSearchSpace, SearchSpace


def experiment_from_struct(struct: ExperimentStruct) -> Experiment:
    """Create an Experiment from an ExperimentStruct."""
    parameters = [
        parameter_from_config(config=parameter_config)
        for parameter_config in struct.parameters
    ]

    constraints = [
        parse_parameter_constraint(constraint_str=constraint_str)
        for constraint_str in struct.parameter_constraints
    ]

    # Ensure that all ParameterConstraints are valid and acting on existing parameters
    for constraint in constraints:
        validate_constraint_parameters(
            parameters=[
                parameter
                for parameter in parameters
                if parameter.name in constraint.constraint_dict.keys()
            ]
        )

    if any(p.is_hierarchical for p in parameters):
        search_space = HierarchicalSearchSpace(
            parameters=parameters, parameter_constraints=constraints
        )
    else:
        search_space = SearchSpace(
            parameters=parameters, parameter_constraints=constraints
        )

    return Experiment(
        search_space=search_space,
        name=struct.name,
        description=struct.description,
        experiment_type=struct.experiment_type,
        properties={"owners": [struct.owner]},
        default_data_type=DataType.MAP_DATA,
    )
