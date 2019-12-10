#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple

from ax.utils.common.serialization import named_tuple_to_dict
from ax.utils.common.testutils import TestCase


class TestSerializationUtils(TestCase):
    def test_named_tuple_to_dict(self):
        class Foo(NamedTuple):
            x: int
            y: str

        foo = Foo(x=5, y="g")
        self.assertEqual(named_tuple_to_dict(foo), {"x": 5, "y": "g"})

        bar = {"x": 5, "foo": foo, "y": [(1, True), foo]}
        self.assertEqual(
            named_tuple_to_dict(bar),
            {"x": 5, "foo": {"x": 5, "y": "g"}, "y": [(1, True), {"x": 5, "y": "g"}]},
        )
