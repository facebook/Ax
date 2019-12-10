#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Support functions for tests
"""

import contextlib
import io
import linecache
import sys
import types
import unittest
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
)


def _get_tb_lines(tb: types.TracebackType) -> List[Tuple[str, int, str]]:
    """Get the filename and line number and line contents of all the lines in the
    traceback with the root at the top.
    """
    res = []
    opt_tb = tb
    while opt_tb is not None:
        lineno = opt_tb.tb_frame.f_lineno
        filename = opt_tb.tb_frame.f_code.co_filename
        line = linecache.getline(filename, lineno).strip()
        res.append((filename, lineno, line))
        opt_tb = opt_tb.tb_next
    res.reverse()
    return res


# pyre-fixme[24]: Generic type `unittest.case._AssertRaisesContext` expects 1 type
#  parameter.
class _AssertRaisesContextOn(unittest.case._AssertRaisesContext):
    """
    Attributes:
       lineno: the line number on which the error occured
       filename: the file in which the error occured
    """

    _expected_line: Optional[str]
    lineno: Optional[int]
    filename: Optional[str]

    def __init__(
        self,
        expected: Type[Exception],
        test_case: unittest.TestCase,
        expected_line: Optional[str] = None,
        expected_regex: Optional[str] = None,
    ) -> None:
        self._expected_line = (
            expected_line.strip() if expected_line is not None else None
        )
        self.lineno = None
        self.filename = None
        # pyre-fixme[28]: Unexpected keyword argument `expected`.
        super().__init__(
            expected=expected, test_case=test_case, expected_regex=expected_regex
        )

    def __exit__(
        self,
        exc_type: Optional[Type[Exception]],
        exc_value: Optional[Exception],
        tb: Optional[types.TracebackType],
    ) -> bool:
        """This is called when the context closes. If an exception was raised
        `exc_type`, `exc_value` and `tb` will be set.
        """
        if not super().__exit__(exc_type, exc_value, tb):
            return False  # reraise
        # super().__exit__ will throw if exc_type is None
        assert exc_type is not None
        assert exc_value is not None
        assert tb is not None
        frames = _get_tb_lines(tb)
        self.filename, self.lineno, _ = frames[0]
        lines = [line for _, _, line in frames]
        if self._expected_line is not None and self._expected_line not in lines:
            # pyre-ignore [16]: ... has no attribute `_raiseFailure`.
            self._raiseFailure(
                f"{self._expected_line!r} was not found in the traceback: {lines!r}"
            )

        return True


# Instead of showing a warning (like in the standard library) we throw an error when
# deprecated functions are called.
# pyre-fixme[3]: Return annotation cannot contain `Any`.
# pyre-fixme[2]: Parameter annotation cannot contain `Any`.
def _deprecate(original_func: Callable) -> Callable:
    def _deprecated_func(*args: List[Any], **kwargs: Dict[str, Any]) -> None:
        raise RuntimeError(
            f"This function is deprecated please use {original_func.__name__} "
            "instead."
        )

    return _deprecated_func


class TestCase(unittest.TestCase):
    """The base test case for Ax, contains various helper functions to write unittest.
    """

    def assertRaisesOn(
        self,
        exc: Type[Exception],
        line: Optional[str] = None,
        regex: Optional[str] = None,
    ) -> ContextManager[None]:
        """Assert that an exception is raised on a specific line.
        """
        context = _AssertRaisesContextOn(exc, self, line, regex)
        # pyre-ignore [16]: ... has no attribute `handle`.
        return context.handle("assertRaisesOn", [], {})

    @staticmethod
    @contextlib.contextmanager
    def silence_stderr() -> Generator[None, None, None]:
        """A context manager that silences stderr for part of a test.

        If any exception passes through this context manager the stderr will be printed,
        otherwise it will be discarded.
        """
        new_err = io.StringIO()
        old_err = sys.stderr
        try:
            sys.stderr = new_err
            yield
        except Exception:
            print(new_err.getvalue(), file=old_err, flush=True)
            raise
        finally:
            sys.stderr = old_err

    # This list is taken from the python standard library
    # pyre-fixme[4]: Attribute must be annotated.
    # pyre-fixme[4]: Attribute must be annotated.
    failUnlessEqual = assertEquals = _deprecate(unittest.TestCase.assertEqual)
    # pyre-fixme[4]: Attribute must be annotated.
    # pyre-fixme[4]: Attribute must be annotated.
    failIfEqual = assertNotEquals = _deprecate(unittest.TestCase.assertNotEqual)
    # pyre-fixme[4]: Attribute must be annotated.
    # pyre-fixme[4]: Attribute must be annotated.
    failUnlessAlmostEqual = assertAlmostEquals = _deprecate(
        unittest.TestCase.assertAlmostEqual
    )
    # pyre-fixme[4]: Attribute must be annotated.
    # pyre-fixme[4]: Attribute must be annotated.
    failIfAlmostEqual = assertNotAlmostEquals = _deprecate(
        unittest.TestCase.assertNotAlmostEqual
    )
    # pyre-fixme[4]: Attribute must be annotated.
    # pyre-fixme[4]: Attribute must be annotated.
    failUnless = assert_ = _deprecate(unittest.TestCase.assertTrue)
    # pyre-fixme[4]: Attribute must be annotated.
    failUnlessRaises = _deprecate(unittest.TestCase.assertRaises)
    # pyre-fixme[4]: Attribute must be annotated.
    failIf = _deprecate(unittest.TestCase.assertFalse)
    # pyre-fixme[4]: Attribute must be annotated.
    assertRaisesRegexp = _deprecate(unittest.TestCase.assertRaisesRegex)
    # pyre-fixme[4]: Attribute must be annotated.
    assertRegexpMatches = _deprecate(unittest.TestCase.assertRegex)
    # pyre-fixme[4]: Attribute must be annotated.
    assertNotRegexpMatches = _deprecate(unittest.TestCase.assertNotRegex)
