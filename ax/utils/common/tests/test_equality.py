#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from datetime import datetime

from ax.utils.common.equality import (
    datetime_equals,
    equality_typechecker,
    same_elements,
)
from ax.utils.common.testutils import TestCase


class EqualityTest(TestCase):
    def testEqualityTypechecker(self):
        @equality_typechecker
        def eq(x, y):
            return x == y

        self.assertFalse(eq(5, 5.0))
        self.assertTrue(eq(5, 5))

    def testListsEquals(self):
        self.assertFalse(same_elements([0], [0, 1]))
        self.assertFalse(same_elements([1, 0], [0, 2]))
        self.assertTrue(same_elements([1, 0], [0, 1]))

    def testDatetimeEquals(self):
        now = datetime.now()
        self.assertTrue(datetime_equals(None, None))
        self.assertFalse(datetime_equals(None, now))
        self.assertTrue(datetime_equals(now, now))
