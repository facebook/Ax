#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import sys

from ax.utils.common.base import Base
from ax.utils.common.testutils import TestCase


# pyre-fixme[3]: Return type must be annotated.
def _f():
    e = RuntimeError("Test")
    raise e


F_FAILURE_LINENO = 17  # Line # for the error in `_f`.


def _g() -> None:
    _f()  # Lines along the path are matched too


class MyBase(Base):
    def __init__(self, val: str) -> None:
        self.field = val


class TestTestUtils(TestCase):
    def test_equal(self) -> None:
        try:  # Check case where values aren't `Base` subclasses.
            self.assertEqual(1, 2)
        except AssertionError as err:
            self.assertEqual(str(err), "1 != 2")

        try:  # Check case where values are `Base` subclasses.
            self.assertEqual(MyBase("red"), MyBase("panda"))
        except AssertionError as err:
            expected_suffix = (
                "\n\nFields with different values:\n\n1) field: red "
                "(type <class 'str'>) != panda (type <class 'str'>)"
            )
            self.assertIn(expected_suffix, str(err))

    def test_raises_on(self) -> None:
        with self.assertRaisesOn(RuntimeError, "raise e"):
            _f()

        # Check that we fail if the source line is not what we expect
        with self.assertRaisesRegex(Exception, "was not found in the traceback"):
            with self.assertRaisesOn(RuntimeError, 'raise Exception("Test")'):
                _f()

        # Check that the exception passes through if it's not the one we meant to catch
        with self.assertRaisesRegex(RuntimeError, "Test"):
            with self.assertRaisesOn(AssertionError, "raise e"):
                _f()

        with self.assertRaisesOn(
            RuntimeError, "_f()  # Lines along the path are matched too"
        ):
            _g()

        # Use this as a context manager to get the position of an error
        with self.assertRaisesOn(RuntimeError) as cm:
            _f()
        # pyre-fixme[16]: `None` has no attribute `filename`.
        self.assertEqual(cm.filename, __file__)
        # pyre-fixme[16]: `None` has no attribute `lineno`.
        self.assertEqual(cm.lineno, F_FAILURE_LINENO)

    def test_silence_warning_normal(self) -> None:
        new_stderr = io.StringIO()
        old_err = sys.stderr
        try:
            sys.stderr = new_stderr
            with self.silence_stderr():
                print("A message", file=sys.stderr)
        finally:
            sys.stderr = old_err
        self.assertEqual(new_stderr.getvalue(), "")

    def test_silence_warning(self) -> None:
        new_stderr = io.StringIO()
        old_err = sys.stderr
        with self.assertRaises(AssertionError):
            try:
                sys.stderr = new_stderr
                with self.silence_stderr():
                    print("A message", file=sys.stderr)
                    raise AssertionError()
            finally:
                sys.stderr = old_err
        self.assertTrue(new_stderr.getvalue().startswith("A message\n"))

    def test_fail_deprecated(self) -> None:
        self.assertEqual(1, 1)
        with self.assertRaises(RuntimeError):
            self.assertEquals(1, 1)
