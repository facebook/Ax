#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import sys

from ax.utils.common.testutils import TestCase


def _f():
    e = RuntimeError("Test")
    raise e


def _g():
    _f()  # Lines along the path are matched too


class TestTestUtils(TestCase):
    def test_raises_on(self):
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
        self.assertEqual(cm.filename, __file__)
        self.assertEqual(cm.lineno, 15)

    def test_silence_warning_normal(self):
        new_stderr = io.StringIO()
        old_err = sys.stderr
        try:
            sys.stderr = new_stderr
            with self.silence_stderr():
                print("A message", file=sys.stderr)
        finally:
            sys.stderr = old_err
        self.assertEqual(new_stderr.getvalue(), "")

    def test_silence_warning(self):
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

    def test_fail_deprecated(self):
        self.assertEqual(1, 1)
        with self.assertRaises(RuntimeError):
            self.assertEquals(1, 1)
