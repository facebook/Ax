#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.cap_parameter import CapParameter
from ax.utils.common.testutils import TestCase


class CapParameterTest(TestCase):
    def setUp(self):
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "a", lower=1, upper=3, parameter_type=ParameterType.FLOAT
                ),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
            ]
        )

    def test_transform_search_space(self):
        t = CapParameter(
            search_space=self.search_space,
            observation_features=[],
            observation_data=[],
            config={"a": "2"},
        )
        t.transform_search_space(self.search_space)
        self.assertEqual(self.search_space.parameters.get("a").upper, 2)
        t2 = CapParameter(
            search_space=self.search_space,
            observation_features=[],
            observation_data=[],
            config={"b": "2"},
        )
        with self.assertRaises(NotImplementedError):
            t2.transform_search_space(self.search_space)
