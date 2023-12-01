#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.transforms.cap_parameter import CapParameter
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_robust_search_space


class CapParameterTest(TestCase):
    def setUp(self) -> None:
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

    def test_transform_search_space(self) -> None:
        t = CapParameter(
            search_space=self.search_space,
            observations=[],
            config={"a": "2"},
        )
        t.transform_search_space(self.search_space)
        # pyre-fixme[16]: Optional type has no attribute `upper`.
        self.assertEqual(self.search_space.parameters.get("a").upper, 2)
        t2 = CapParameter(
            search_space=self.search_space,
            observations=[],
            config={"b": "2"},
        )
        with self.assertRaises(NotImplementedError):
            t2.transform_search_space(self.search_space)

    def test_w_parameter_distributions(self) -> None:
        rss = get_robust_search_space()
        # Transform a non-distributional parameter.
        t = CapParameter(
            search_space=rss,
            observations=[],
            config={"z": "2"},
        )
        t.transform_search_space(rss)
        # pyre-fixme[16]: Optional type has no attribute `upper`.
        self.assertEqual(rss.parameters.get("z").upper, 2)
        # Error with distributional parameter.
        t = CapParameter(
            search_space=rss,
            observations=[],
            config={"x": "2"},
        )
        with self.assertRaisesRegex(UnsupportedError, "transform is not supported"):
            t.transform_search_space(rss)
