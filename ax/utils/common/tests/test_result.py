# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.utils.common.result import Err, Ok, Result, UnwrapError
from ax.utils.common.testutils import TestCase


class ResultTest(TestCase):
    def setUp(self) -> None:
        def safeDivide(a: float, b: float) -> Result[float, str]:
            if b == 0:
                return Err("yikes")

            return Ok(a / b)

        self.ok = safeDivide(0, 2)
        self.err = safeDivide(0, 0)

    def test_eq(self) -> None:
        self.assertEqual(self.ok, Ok(0))
        self.assertEqual(self.err, Err("yikes"))

    def test_repr(self) -> None:
        self.assertEqual(str(self.ok), "Ok(0.0)")
        self.assertEqual(str(self.err), "Err(yikes)")

    def test_is(self) -> None:
        self.assertTrue(self.ok.is_ok())
        self.assertFalse(self.ok.is_err())

        self.assertFalse(self.err.is_ok())
        self.assertTrue(self.err.is_err())

    def test_map(self) -> None:
        def f(n: int) -> int:
            return n + 1

        def g(val: str) -> int:
            return len(val)

        def h() -> int:
            return -1

        self.assertEqual(self.ok.map(op=f), Ok(1))
        self.assertEqual(self.ok.map_err(op=g), Ok(0))
        self.assertEqual(self.ok.map_or(default="foo", op=f), 1)
        self.assertEqual(self.ok.map_or_else(default_op=h, op=f), 1)

        self.assertEqual(self.err.map(op=f), Err("yikes"))
        self.assertEqual(self.err.map_err(op=g), Err(5))
        self.assertEqual(self.err.map_or(default="foo", op=f), "foo")
        self.assertEqual(self.err.map_or_else(default_op=h, op=f), -1)

    def test_unwrap(self) -> None:
        self.assertEqual(self.ok.unwrap(), 0)
        with self.assertRaises(UnwrapError):
            self.ok.unwrap_err()
        self.assertEqual(self.ok.unwrap_or(1), 0)
        self.assertEqual(self.ok.unwrap_or_else(1), 0)

        with self.assertRaises(UnwrapError):
            self.err.unwrap()
        self.assertEqual(self.err.unwrap_err(), "yikes")
        self.assertEqual(self.err.unwrap_or(1), 1)
        self.assertEqual(self.err.unwrap_or_else(lambda s: 2), 2)
