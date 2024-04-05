#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from copy import deepcopy
from typing import List

from ax.core.observation import ObservationFeatures
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    Parameter,
    ParameterType,
    RangeParameter,
)
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.simplify_parameter_constraints import (
    SimplifyParameterConstraints,
)
from ax.utils.common.testutils import TestCase


class SimplifyParameterConstraintsTest(TestCase):
    def setUp(self) -> None:
        self.parameters: List[Parameter] = [
            RangeParameter("x", lower=1, upper=3, parameter_type=ParameterType.FLOAT),
            RangeParameter("y", lower=2, upper=5, parameter_type=ParameterType.INT),
            ChoiceParameter(
                "z", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
            ),
        ]
        self.observation_features = [
            ObservationFeatures(parameters={"x": 2, "y": 2, "z": "b"})
        ]

    def test_transform_no_constraints(self) -> None:
        t = SimplifyParameterConstraints()
        ss = SearchSpace(parameters=self.parameters)
        ss_transformed = t.transform_search_space(search_space=ss)
        self.assertEqual(ss, ss_transformed)
        self.assertEqual(
            self.observation_features,
            t.transform_observation_features(self.observation_features),
        )

    def test_transform_weight_zero(self) -> None:
        t = SimplifyParameterConstraints()
        ss = SearchSpace(
            parameters=self.parameters,
            parameter_constraints=[
                ParameterConstraint(constraint_dict={"x": 0}, bound=1)
            ],
        )
        ss_transformed = t.transform_search_space(search_space=deepcopy(ss))
        self.assertEqual(ss_transformed.parameter_constraints, [])
        self.assertEqual(ss.parameters, ss_transformed.parameters)
        ss_raises = SearchSpace(
            parameters=self.parameters,
            parameter_constraints=[
                ParameterConstraint(constraint_dict={"x": 0}, bound=-1)
            ],
        )
        with self.assertRaisesRegex(
            ValueError, "Parameter constraint cannot be satisfied since the weight"
        ):
            ss_transformed = t.transform_search_space(search_space=deepcopy(ss_raises))

    def test_transform_search_space(self) -> None:
        t = SimplifyParameterConstraints()
        ss = SearchSpace(
            parameters=self.parameters,
            parameter_constraints=[
                ParameterConstraint(constraint_dict={"x": 1}, bound=2),  # x <= 2
                ParameterConstraint(constraint_dict={"y": -1}, bound=-4),  # y => 4
            ],
        )
        ss_transformed = t.transform_search_space(search_space=deepcopy(ss))
        self.assertEqual(
            {
                **ss.parameters,
                "x": RangeParameter(
                    "x", parameter_type=ParameterType.FLOAT, lower=1, upper=2
                ),
                "y": RangeParameter(
                    "y", parameter_type=ParameterType.INT, lower=4, upper=5
                ),
            },
            ss_transformed.parameters,
        )
        self.assertEqual(ss_transformed.parameter_constraints, [])
        self.assertEqual(  # No-op
            self.observation_features,
            t.transform_observation_features(self.observation_features),
        )

    def test_transform_to_fixed(self) -> None:
        t = SimplifyParameterConstraints()
        ss = SearchSpace(
            parameters=self.parameters,
            parameter_constraints=[
                ParameterConstraint(constraint_dict={"x": 1}, bound=1),  # x == 1
                ParameterConstraint(constraint_dict={"y": -1}, bound=-5),  # y == 5
            ],
        )
        ss_transformed = t.transform_search_space(search_space=deepcopy(ss))
        self.assertEqual(
            {
                **ss.parameters,
                "x": FixedParameter("x", parameter_type=ParameterType.FLOAT, value=1),
                "y": FixedParameter("y", parameter_type=ParameterType.INT, value=5),
            },
            ss_transformed.parameters,
        )
        self.assertEqual(ss_transformed.parameter_constraints, [])
        self.assertEqual(  # No-op
            self.observation_features,
            t.transform_observation_features(self.observation_features),
        )
