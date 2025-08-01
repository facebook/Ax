# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from contextlib import ExitStack

from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig
from ax.api.utils.instantiation.from_config import (
    _parameter_type_converter,
    parameter_from_config,
)
from ax.api.utils.instantiation.from_struct import experiment_from_struct
from ax.api.utils.structs import ExperimentStruct
from ax.core.experiment import Experiment
from ax.core.formatting_utils import DataType
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType as CoreParameterType,
    RangeParameter,
)
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import HierarchicalSearchSpace, SearchSpace
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase


class TestFromConfig(TestCase):
    def test_create_range_parameter(self) -> None:
        float_config = RangeParameterConfig(
            name="float_param",
            parameter_type="float",
            bounds=(0, 1),
        )

        self.assertEqual(
            parameter_from_config(config=float_config),
            RangeParameter(
                name="float_param",
                parameter_type=CoreParameterType.FLOAT,
                lower=0,
                upper=1,
            ),
        )

        float_config_with_log_scaling = RangeParameterConfig(
            name="float_param_with_log_scaling",
            parameter_type="float",
            bounds=(1e-10, 1),
            scaling="log",
        )

        self.assertEqual(
            parameter_from_config(config=float_config_with_log_scaling),
            RangeParameter(
                name="float_param_with_log_scaling",
                parameter_type=CoreParameterType.FLOAT,
                lower=1e-10,
                upper=1,
                log_scale=True,
            ),
        )

        int_config = RangeParameterConfig(
            name="int_param",
            parameter_type="int",
            bounds=(0, 1),
        )

        self.assertEqual(
            parameter_from_config(config=int_config),
            RangeParameter(
                name="int_param",
                parameter_type=CoreParameterType.INT,
                lower=0,
                upper=1,
            ),
        )

        with warnings.catch_warnings(record=True) as ws:
            step_size_config = RangeParameterConfig(
                name="step_size_param",
                parameter_type="float",
                bounds=(0, 100),
                step_size=10,
            )
        self.assertFalse(any("sort_values" in str(w.message) for w in ws))

        self.assertEqual(
            parameter_from_config(config=step_size_config),
            ChoiceParameter(
                name="step_size_param",
                parameter_type=CoreParameterType.FLOAT,
                values=[
                    0.0,
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                    60.0,
                    70.0,
                    80.0,
                    90.0,
                    100.0,
                ],
                is_ordered=True,
                sort_values=False,
            ),
        )

        with self.assertRaisesRegex(
            UserInputError,
            "Non-linear parameter scaling is not supported when using step_size",
        ):
            parameter_from_config(
                config=RangeParameterConfig(
                    name="step_size_param_with_scaling",
                    parameter_type="float",
                    bounds=(0, 100),
                    step_size=10,
                    scaling="log",
                )
            )

    def test_create_choice_parameter(self) -> None:
        es = ExitStack()
        ws = es.enter_context(warnings.catch_warnings(record=True))
        choice_config = ChoiceParameterConfig(
            name="choice_param",
            parameter_type="str",
            values=["a", "b", "c"],
        )

        self.assertEqual(
            parameter_from_config(config=choice_config),
            ChoiceParameter(
                name="choice_param",
                parameter_type=CoreParameterType.STRING,
                values=["a", "b", "c"],
            ),
        )

        choice_config_with_order = ChoiceParameterConfig(
            name="choice_param_with_order",
            parameter_type="str",
            values=["a", "b", "c"],
            is_ordered=True,
        )
        self.assertEqual(
            parameter_from_config(config=choice_config_with_order),
            ChoiceParameter(
                name="choice_param_with_order",
                parameter_type=CoreParameterType.STRING,
                values=["a", "b", "c"],
                is_ordered=True,
            ),
        )

        choice_config_with_dependents = ChoiceParameterConfig(
            name="choice_param_with_dependents",
            parameter_type="str",
            values=["a", "b", "c"],
            dependent_parameters={
                "a": ["a1", "a2"],
                "b": ["b1", "b2", "b3"],
            },
        )
        self.assertEqual(
            parameter_from_config(config=choice_config_with_dependents),
            ChoiceParameter(
                name="choice_param_with_dependents",
                parameter_type=CoreParameterType.STRING,
                values=["a", "b", "c"],
                dependents={
                    "a": ["a1", "a2"],
                    "b": ["b1", "b2", "b3"],
                },
            ),
        )

        single_element_choice_config = ChoiceParameterConfig(
            name="single_element_choice_param",
            parameter_type="str",
            values=["a"],
        )
        self.assertEqual(
            parameter_from_config(config=single_element_choice_config),
            FixedParameter(
                name="single_element_choice_param",
                parameter_type=CoreParameterType.STRING,
                value="a",
            ),
        )
        self.assertFalse(any("sort_values" in str(w.message) for w in ws))

    def test_experiment_from_config(self) -> None:
        float_parameter = RangeParameterConfig(
            name="float_param",
            parameter_type="float",
            bounds=(0, 1),
        )
        int_parameter = RangeParameterConfig(
            name="int_param",
            parameter_type="int",
            bounds=(0, 1),
        )
        choice_parameter = ChoiceParameterConfig(
            name="choice_param",
            parameter_type="str",
            values=["a", "b", "c"],
        )

        experiment_struct = ExperimentStruct(
            name="test_experiment",
            parameters=[float_parameter, int_parameter, choice_parameter],
            parameter_constraints=["int_param <= float_param"],
            description="test description",
            experiment_type="TEST",
            owner="miles",
        )

        self.assertEqual(
            experiment_from_struct(struct=experiment_struct),
            Experiment(
                search_space=SearchSpace(
                    parameters=[
                        RangeParameter(
                            name="float_param",
                            parameter_type=CoreParameterType.FLOAT,
                            lower=0,
                            upper=1,
                        ),
                        RangeParameter(
                            name="int_param",
                            parameter_type=CoreParameterType.INT,
                            lower=0,
                            upper=1,
                        ),
                        ChoiceParameter(
                            name="choice_param",
                            parameter_type=CoreParameterType.STRING,
                            values=["a", "b", "c"],
                            is_ordered=False,
                            sort_values=False,
                        ),
                    ],
                    parameter_constraints=[
                        ParameterConstraint(
                            constraint_dict={"int_param": 1, "float_param": -1}, bound=0
                        )
                    ],
                ),
                name="test_experiment",
                description="test description",
                experiment_type="TEST",
                properties={"owners": ["miles"]},
                default_data_type=DataType.MAP_DATA,
            ),
        )

        root_parameter = ChoiceParameterConfig(
            name="root_param",
            parameter_type="str",
            values=["left", "right"],
            dependent_parameters={
                "left": ["float_param"],
                "right": ["int_param"],
            },
        )

        hss_struct = ExperimentStruct(
            parameters=[float_parameter, int_parameter, root_parameter],
            parameter_constraints=["int_param <= float_param"],
            name="test_experiment",
            description="test description",
            experiment_type=None,
            owner="miles",
        )

        self.assertEqual(
            experiment_from_struct(struct=hss_struct),
            Experiment(
                search_space=HierarchicalSearchSpace(
                    parameters=[
                        RangeParameter(
                            name="float_param",
                            parameter_type=CoreParameterType.FLOAT,
                            lower=0,
                            upper=1,
                        ),
                        RangeParameter(
                            name="int_param",
                            parameter_type=CoreParameterType.INT,
                            lower=0,
                            upper=1,
                        ),
                        ChoiceParameter(
                            name="root_param",
                            parameter_type=CoreParameterType.STRING,
                            values=["left", "right"],
                            is_ordered=False,
                            sort_values=False,
                            dependents={
                                "left": ["float_param"],
                                "right": ["int_param"],
                            },
                        ),
                    ],
                    parameter_constraints=[
                        ParameterConstraint(
                            constraint_dict={"int_param": 1, "float_param": -1}, bound=0
                        )
                    ],
                ),
                name="test_experiment",
                description="test description",
                properties={"owners": ["miles"]},
                default_data_type=DataType.MAP_DATA,
            ),
        )

    def test_parameter_type_converter(self) -> None:
        self.assertEqual(
            _parameter_type_converter(parameter_type="bool"),
            CoreParameterType.BOOL,
        )
        self.assertEqual(
            _parameter_type_converter(parameter_type="int"),
            CoreParameterType.INT,
        )
        self.assertEqual(
            _parameter_type_converter(parameter_type="float"),
            CoreParameterType.FLOAT,
        )
        self.assertEqual(
            _parameter_type_converter(parameter_type="str"),
            CoreParameterType.STRING,
        )
        with self.assertRaisesRegex(UserInputError, "Unsupported parameter type"):
            _parameter_type_converter(parameter_type="bad")
