#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List

from ax.core import SearchSpace
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.types import TParameterization
from ax.health_check.search_space import search_space_update_recommendation
from ax.utils.common.testutils import TestCase


class SearchSpaceTest(TestCase):
    def test_search_space_update(self) -> None:
        parameter_a = RangeParameter(
            name="a", lower=0.0, upper=1.0, parameter_type=ParameterType.FLOAT
        )
        parameter_b = RangeParameter(
            name="b", lower=-1.0, upper=2.0, parameter_type=ParameterType.FLOAT
        )
        search_space = SearchSpace(parameters=[parameter_a, parameter_b])
        parametrizations: List[TParameterization] = [
            {"a": 0.5, "b": -1.0},
            {"a": 0.0, "b": 2.0},
            {"a": 0.0, "b": 2.0},
            {"a": 1.0, "b": -1.0},
        ]
        param_boundary_prop, msg = search_space_update_recommendation(
            search_space, parametrizations=parametrizations
        )
        self.assertEqual(param_boundary_prop, {"a": (0.5, 0.25), "b": (0.5, 0.5)})
