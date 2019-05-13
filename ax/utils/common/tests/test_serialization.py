#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

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
