#!/usr/bin/env python3

from ae.lazarus.ae.utils.common.testutils import TestCase
from ae.lazarus.ae.utils.common.typeutils import (
    checked_cast,
    checked_cast_list,
    checked_cast_optional,
    not_none,
)


class TestTypeUtils(TestCase):
    def test_not_none(self):
        self.assertEqual(not_none("not_none"), "not_none")
        with self.assertRaises(ValueError):
            not_none(None)

    def test_checked_cast(self):
        self.assertEqual(checked_cast(float, 2.0), 2.0)
        with self.assertRaises(ValueError):
            checked_cast(float, 2)

    def test_checked_cast_list(self):
        self.assertEqual(checked_cast_list(float, [1.0, 2.0]), [1.0, 2.0])
        with self.assertRaises(ValueError):
            checked_cast_list(float, [1.0, 2])

    def test_checked_cast_optional(self):
        self.assertEqual(checked_cast_optional(float, None), None)
        with self.assertRaises(ValueError):
            checked_cast_optional(float, 2)
