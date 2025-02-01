#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Support functions for tests"""

import builtins
import contextlib
import cProfile
import io
import linecache
import logging
import os
import signal
import sys
import types
import unittest
import warnings
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager
from logging import Logger
from pstats import Stats
from types import FrameType, ModuleType
from typing import Any, TypeVar, Union
from unittest.mock import MagicMock

import numpy as np
from ax.exceptions.core import AxParameterWarning
from ax.utils.common.base import Base
from ax.utils.common.constants import TESTENV_ENV_KEY, TESTENV_ENV_VAL
from ax.utils.common.equality import object_attribute_dicts_find_unequal_fields
from ax.utils.common.logger import get_logger
from botorch.exceptions.warnings import InputDataWarning
from pyfakefs import fake_filesystem_unittest


T_AX_BASE_OR_ATTR_DICT = Union[Base, dict[str, Any]]
COMPARISON_STR_MAX_LEVEL = 8
T = TypeVar("T")

logger: Logger = get_logger(__name__)


def _get_tb_lines(tb: types.TracebackType) -> list[tuple[str, int, str]]:
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

    _expected_line: str | None
    lineno: int | None
    filename: str | None

    def __init__(
        self,
        expected: type[Exception],
        test_case: unittest.TestCase,
        expected_line: str | None = None,
        expected_regex: str | None = None,
    ) -> None:
        self._expected_line = (
            expected_line.strip() if expected_line is not None else None
        )
        self.lineno = None
        self.filename = None
        super().__init__(
            expected=expected, test_case=test_case, expected_regex=expected_regex
        )

    # pyre-fixme[14]: `__exit__` overrides method defined in `_AssertRaisesContext`
    #  inconsistently.
    def __exit__(
        self,
        exc_type: type[Exception] | None,
        exc_value: Exception | None,
        tb: types.TracebackType | None,
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
def _deprecate(original_func: Callable) -> Callable:
    def _deprecated_func(*args: list[Any], **kwargs: dict[str, Any]) -> None:
        raise RuntimeError(
            f"This function is deprecated please use {original_func.__name__} "
            "instead."
        )

    return _deprecated_func


def _build_comparison_str(
    first: T_AX_BASE_OR_ATTR_DICT,
    second: T_AX_BASE_OR_ATTR_DICT,
    level: int = 0,
    values_in_suffix: str = "",
    skip_db_id_check: bool = False,
) -> str:
    """Recursively build a comparison string for classes that extend Ax `Base`
    or two dictionaries (dictionaries are passed in in the recursive case).
    Prints out an 'inequality report' that includes nested fields.

    NOTE: Allows recursion only up to level 4, with markers like '1)', then 'a)',
    then 'i)', then '*' for the nested lists.

    For example, for two experiments, the 'report' might look like this, if their
    search spaces are unequal because of difference of parameters in parameter
    constraints:

    Experiment(test_1) (type `Experiment`) != Experiment(test_2) (type `Experiment`).

    Fields with different values:

    1) _search_space: ... != ...

        Fields with different values in 1):

            a) _parameter_constraints: ... != ...

                Fields with different values in a):

                    i) _parameter: ... != ...

                        Fields with different values in i):

                            * db_id: ... != ...

    NOTE: If ``skip_db_id_check`` is ``True``, will exclude the ``db_id`` attributes
    from the equality check. Useful for ensuring that all attributes of an object are
    equal except the ids, with which one or both of them are saved to the database
    (e.g. if confirming an object before it was saved, to the version reloaded
    from the DB).
    """

    def _unequal_str(first: Any, second: Any) -> str:  # pyre-ignore[2]
        return f"{first} (type {type(first)}) != {second} (type {type(second)})."

    if first == second:
        return ""

    if level > COMPARISON_STR_MAX_LEVEL:
        # Don't go deeper than 4 levels as the inequality report will not be legible.
        return (
            f"\n... also there were unequal fields at levels {level}+; "
            "to see full comparison past this level, adjust `ax.utils.common.testutils."
            "COMPARISON_STR_MAX_LEVEL`"
        )

    msg = ""
    indent = " " * level * 4
    unequal_types, unequal_val = object_attribute_dicts_find_unequal_fields(
        one_dict=first.__dict__ if isinstance(first, Base) else first,
        other_dict=second.__dict__ if isinstance(second, Base) else second,
        fast_return=False,
        skip_db_id_check=skip_db_id_check,
    )
    unequal_types_suffixed = {
        f"{k} (field had values of unequal type)": v for k, v in unequal_types.items()
    }
    if level == 0:
        msg += f"{_unequal_str(first=first, second=second)}\n"

    msg += f"\n{indent}Fields with different values{values_in_suffix}:\n"
    joint_unequal_field_dict = {**unequal_val, **unequal_types_suffixed}
    for idx, (field, (first, second)) in enumerate(joint_unequal_field_dict.items()):
        # For level 0, use numbers as bullets. For 1, use letters. For 2, use "i".
        # For 3+, use "*".
        bul = "*"
        if level == 0:
            bul = f"{idx + 1})"
        elif level == 1:
            bul = f"{chr(ord('a') + idx)})"
        elif level == 2:
            bul = f"{'i' * (idx + 1)})"
        elif level <= COMPARISON_STR_MAX_LEVEL:
            # Add default for when setting `COMPARISON_STR_MAX_LEVEL` to higher value
            # during debugging.
            bul = "*"
        else:
            raise RuntimeError(
                "Reached level > `COMPARISON_STR_MAX_LEVEL`, which should've been "
                "unreachable."
            )
        msg += f"\n{indent}{bul} {field}: {_unequal_str(first=first, second=second)}\n"
        if isinstance(first, (dict, Base)) and isinstance(second, (dict, Base)):
            msg += _build_comparison_str(
                first=first,
                second=second,
                level=level + 1,
                values_in_suffix=f" in {bul}",
                skip_db_id_check=skip_db_id_check,
            )
        elif isinstance(first, list) and isinstance(second, list):
            # To compare lists recursively via same function, making them into dicts
            # with index keys.
            msg += _build_comparison_str(
                first=dict(zip([str(x) for x in range(len(first))], first)),
                second=dict(zip([str(x) for x in range(len(second))], second)),
                level=level + 1,
                values_in_suffix=f" in {bul}",
                skip_db_id_check=skip_db_id_check,
            )
    return msg


def setup_import_mocks(
    mocked_import_paths: list[str], mock_config_dict: dict[str, Any] | None = None
) -> None:
    """This function mocks expensive modules used in tests. It must be called before
    those modules are imported or it will not work.  Stubbing out these modules
    will obviously affect the behavior of all tests that use it, so be sure modules
    being mocked are not important to your test.  It will also mock all child modules.

    Args:
        mocked_import_paths: List of module paths to mock.
        mock_config_dict: Dictionary of attributes to mock on the modules being mocked.
            This is useful if the import is expensive, but there is still some
            functionality it has the test relies on.  These attributes will be
            set on all modules being mocked.
    """

    def custom_import(name: str, *args: Any, **kwargs: Any) -> ModuleType:
        for import_path in mocked_import_paths:
            if name == import_path or name.startswith(f"{import_path}."):
                mymock = MagicMock()
                if mock_config_dict is not None:
                    mymock.configure_mock(**mock_config_dict)
                return mymock
        return original_import(name, *args, **kwargs)

    for import_path in mocked_import_paths:
        if import_path in sys.modules and not isinstance(
            sys.modules[import_path], MagicMock
        ):
            raise Exception(f"{import_path} has already been imported!")

    # Replace the original import with the custom one
    # pyre-fixme[61][53]
    original_import: Callable[..., ModuleType] = builtins.__import__
    # pyre-fixme[9]: __import__ has type `(name: str, globals: Optional[Mapping[str,
    #  object]] = ..., locals: Optional[Mapping[str, object]] = ..., fromlist:
    #  Sequence[str] = ..., level: int = ...) -> ModuleType`; used as `(name: str,
    #  *(Any), **(Any)) -> Any`.
    builtins.__import__ = custom_import


class TestCase(fake_filesystem_unittest.TestCase):
    """The base Ax test case, contains various helper functions to write unittests."""

    MAX_TEST_SECONDS = 60
    NUMBER_OF_PROFILER_LINES_TO_OUTPUT = 20
    PROFILE_TESTS = False
    _long_test_active_reason: str | None = None

    def __init__(self, methodName: str = "runTest") -> None:
        def signal_handler(signum: int, frame: FrameType | None) -> None:
            message = f"Test took longer than {self.MAX_TEST_SECONDS} seconds."
            if self.PROFILE_TESTS:
                self._print_profiler_output()
            else:
                message += (
                    " To see a profiler output, set `TestCase.PROFILE_TESTS` to `True`."
                )
            if hasattr(sys, "gettrace") and sys.gettrace() is not None:
                # If we're in a debugger session, let the test continue running.
                return
            elif self._long_test_active_reason is None:
                message += (
                    " To specify a reason for a long running test,"
                    + " utilize the @ax_long_test decorator. If your test "
                    + "is long because it's doing modeling, please use the "
                    + "@mock_botorch_optimize decorator and see if that helps."
                )
                raise TimeoutError(message)
            else:
                message += (
                    " Reason for long running test: " + self._long_test_active_reason
                )
                logger.warning(message)

        super().__init__(methodName=methodName)
        signal.signal(signal.SIGALRM, signal_handler)
        # This is set to indicate we are running in a test environment.  Code can check
        # this to:
        # * more strictly enforce SQL encoding
        #  (https://github.com/facebook/Ax/blob/main/ax/storage/sqa_store/save.py#L598)
        # * avoid actions that will affect product environments
        os.environ[TESTENV_ENV_KEY] = TESTENV_ENV_VAL

    def setUp(self) -> None:
        """
        Only show log messages of WARNING or higher while testing.

        Ax prints a lot of INFO logs that are not relevant for unit tests.

        Also silences a number of common warnings originating from Ax & BoTorch.
        """
        super().setUp()
        self.profiler = cProfile.Profile()
        if self.PROFILE_TESTS:
            self.profiler.enable()
            self.addCleanup(self.profiler.disable)
        logger = get_logger(__name__, level=logging.WARNING)
        # Parent handlers are shared, so setting the level this
        # way applies it to all Ax loggers.
        if logger.parent is not None and hasattr(logger.parent, "handlers"):
            logger.parent.handlers[0].setLevel(logging.WARNING)

        # Choice parameter default parameter type / is_ordered warnings.
        warnings.filterwarnings(
            "ignore",
            message=".*is not specified for .ChoiceParameter.*",
            category=AxParameterWarning,
        )
        # BoTorch float32 warning.
        warnings.filterwarnings(
            "ignore",
            message="The model inputs are of type",
            category=InputDataWarning,
        )
        # BoTorch input standardization warnings.
        warnings.filterwarnings(
            "ignore",
            message=r"Data \(outcome observations\) is not standardized ",
            category=InputDataWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Data \(input features\) is not",
            category=InputDataWarning,
        )

    def run(
        self, result: unittest.result.TestResult | None = ...
    ) -> unittest.result.TestResult | None:
        # Arrange for a SIGALRM signal to be delivered to the calling process
        # in specified number of seconds.
        signal.alarm(self.MAX_TEST_SECONDS)
        try:
            result = super().run(result)
        finally:
            signal.alarm(0)
        return result

    def assertEqual(
        self,
        first: Any,  # pyre-ignore[2]
        second: Any,  # pyre-ignore[2]
        msg: str | None = None,
    ) -> None:
        if isinstance(first, Base) and isinstance(second, Base):
            self.assertAxBaseEqual(first=first, second=second, msg=msg)
        else:
            super().assertEqual(first=first, second=second, msg=msg)

    def assertAxBaseEqual(
        self,
        first: Base,
        second: Base,
        msg: str | None = None,
        skip_db_id_check: bool = False,
    ) -> None:
        """Check that two Ax objects that subclass ``Base`` are equal or raise
        assertion error otherwise.

        Args:
            first: ``Base``-subclassing object to compare to ``second``.
            second: ``Base``-subclassing object to compare to ``first``.
            msg: Message to put into the assertion error raised on inequality; if not
                specified, a default message is used.
            skip_db_id_check: If ``True``, will exclude the ``db_id`` attributes from
                the equality check. Useful for ensuring that all attributes of an object
                are equal except the ids, with which one or both of them are saved to
                the database (e.g. if confirming an object before it was saved, to the
                 version reloaded from the DB).
        """
        self.assertIsInstance(
            first, Base, "First argument is not a subclass of Ax `Base`."
        )
        self.assertIsInstance(
            second, Base, "Second argument is not a subclass of Ax `Base`."
        )
        if (
            not first._eq_skip_db_id_check(other=second)
            if skip_db_id_check
            else first != second
        ):
            raise self.failureException(
                "Encountered unequal objects. This Ax utility will now attempt an "
                "in-depth comparison of the objects to print out the actually "
                "unequal fields within them. Note that the resulting printout is "
                "a nested comparison, and you'll find the actual unequal fields at "
                "the very bottom. Don't be scared of the long printout : )\n\n"
                + _build_comparison_str(
                    first=first, second=second, skip_db_id_check=skip_db_id_check
                ),
            )

    def assertRaisesOn(
        self,
        exc: type[Exception],
        line: str | None = None,
        regex: str | None = None,
        # pyre-ignore[24]: Generic type `AbstractContextManager`
        # expects 2 type parameters, received 1.
    ) -> AbstractContextManager[None]:
        """Assert that an exception is raised on a specific line."""
        context = _AssertRaisesContextOn(exc, self, line, regex)
        return context.handle("assertRaisesOn", [], {})

    def assertDictsAlmostEqual(
        self, a: dict[str, Any], b: dict[str, Any], consider_nans_equal: bool = False
    ) -> None:
        """Testing utility that checks that
        1) the keys of `a` and `b` are identical, and that
        2) the values of `a` and `b` are almost equal if they have a floating point
        type, considering NaNs as equal, and otherwise just equal.

        Args:
            test: The test case object.
            a: A dictionary.
            b: Another dictionary.
            consider_nans_equal: Whether to consider NaNs equal when comparing floating
                point numbers.
        """
        set_a = set(a.keys())
        set_b = set(b.keys())
        key_msg = (
            "Dict keys differ."
            f"Keys that are in a but not b: {set_a - set_b}."
            f"Keys that are in b but not a: {set_b - set_a}."
        )
        self.assertEqual(set_a, set_b, msg=key_msg)
        for field in b:
            a_field = a[field]
            b_field = b[field]
            msg = f"Dict values differ for key {field}: {a[field]=}, {b[field]=}."
            # for floating point values, compare approximately and consider NaNs equal
            if isinstance(a_field, float):
                if consider_nans_equal and np.isnan(a_field):
                    self.assertTrue(np.isnan(b_field), msg=msg)
                else:
                    self.assertAlmostEqual(a_field, b_field, msg=msg)
            else:
                self.assertEqual(a_field, b_field, msg=msg)

    def assertIsSubDict(
        self,
        subdict: dict[str, Any],
        superdict: dict[str, Any],
        almost_equal: bool = False,
        consider_nans_equal: bool = False,
    ) -> None:
        """Testing utility that checks that all keys and values of `subdict` are
        contained in `dict`.

        Args:
            subdict: A smaller dictionary.
            superdict: A larger dictionary which should contain all keys of subdict
                and the same values as subdict for the corresponding keys.
        """
        intersection_dict = {k: superdict[k] for k in subdict if k in superdict}
        if consider_nans_equal and not almost_equal:
            raise ValueError(
                "`consider_nans_equal` can only be used with `almost_equal`"
            )
        if almost_equal:
            self.assertDictsAlmostEqual(
                subdict, intersection_dict, consider_nans_equal=consider_nans_equal
            )
        else:
            self.assertEqual(subdict, intersection_dict)

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

    def _print_profiler_output(self) -> None:
        """Print profiler output to stdout."""
        s = io.StringIO()
        ps = Stats(self.profiler, stream=s).sort_stats("cumulative").reverse_order()
        ps.print_stats()
        output = s.getvalue().splitlines()
        headers = output[:5]
        # Print the headers
        for line in headers:
            print(line)
        # Print the longest running functions
        for line in output[-self.NUMBER_OF_PROFILER_LINES_TO_OUTPUT :]:
            print(line)

    @classmethod
    @contextlib.contextmanager
    def ax_long_test(cls, reason: str | None) -> Generator[None, None, None]:
        cls._long_test_active_reason = reason
        yield
        cls._long_test_active_reason = None

    # This list is taken from the python standard library
    # pyre-fixme[4]: Attribute must be annotated.
    failUnlessEqual = assertEquals = _deprecate(unittest.TestCase.assertEqual)
    # pyre-fixme[4]: Attribute must be annotated.
    failIfEqual = assertNotEquals = _deprecate(unittest.TestCase.assertNotEqual)
    # pyre-fixme[4]: Attribute must be annotated.
    failUnlessAlmostEqual = assertAlmostEquals = _deprecate(
        unittest.TestCase.assertAlmostEqual
    )
    # pyre-fixme[4]: Attribute must be annotated.
    failIfAlmostEqual = assertNotAlmostEquals = _deprecate(
        unittest.TestCase.assertNotAlmostEqual
    )
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
