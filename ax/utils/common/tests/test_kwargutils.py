#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, Dict
from unittest.mock import patch

from ax.utils.common.kwargs import validate_kwarg_typing, warn_on_kwargs
from ax.utils.common.logger import get_logger
from ax.utils.common.testutils import TestCase

logger = get_logger("ax.utils.common.kwargs")


class TestKwargUtils(TestCase):
    def test_validate_kwarg_typing(self):
        def typed_callable(arg1: int, arg2: str = None) -> None:
            pass

        def typed_callable_with_dict(arg3: int, arg4: Dict[str, int]) -> None:
            pass

        def typed_callable_valid(arg3: int, arg4: str = None) -> None:
            pass

        def typed_callable_dup_keyword(arg2: int, arg4: str = None) -> None:
            pass

        def typed_callable_with_callable(
            arg1: int, arg2: Callable[[int], Dict[str, int]]
        ) -> None:
            pass

        def typed_callable_extra_arg(arg1: int, arg2: str, arg3: bool) -> None:
            pass

        # pass
        try:
            kwargs = {"arg1": 1, "arg2": "test", "arg3": 2}
            validate_kwarg_typing([typed_callable, typed_callable_valid], **kwargs)
        except Exception:
            self.assertTrue(False, "Exception raised on valid kwargs")

        # pass with complex data structure
        try:
            kwargs = {"arg1": 1, "arg2": "test", "arg3": 2, "arg4": {"k1": 1}}
            validate_kwarg_typing([typed_callable, typed_callable_with_dict], **kwargs)
        except Exception:
            self.assertTrue(False, "Exception raised on valid kwargs")

        # callable as arg (same arg count but diff type)
        try:
            kwargs = {"arg1": 1, "arg2": typed_callable}
            validate_kwarg_typing([typed_callable_with_callable], **kwargs)
        except Exception:
            self.assertTrue(False, "Exception raised on valid kwargs")

        # callable as arg (diff arg count)
        try:
            kwargs = {"arg1": 1, "arg2": typed_callable_extra_arg}
            validate_kwarg_typing([typed_callable_with_callable], **kwargs)
        except Exception:
            self.assertTrue(False, "Exception raised on valid kwargs")

        # kwargs contains extra keywords
        with self.assertRaises(ValueError):
            kwargs = {"arg1": 1, "arg2": "test", "arg3": 3, "arg5": 4}
            typed_callables = [typed_callable, typed_callable_valid]
            validate_kwarg_typing(typed_callables, **kwargs)

        # callables have duplicate keywords
        with patch.object(logger, "debug") as mock_debug:
            kwargs = {"arg1": 1, "arg2": "test", "arg4": "test_again"}
            typed_callables = [typed_callable, typed_callable_dup_keyword]
            validate_kwarg_typing(typed_callables, **kwargs)
            mock_debug.assert_called_once_with(
                f"`{typed_callables}` have duplicate keyword argument: arg2."
            )

        # mismatch types
        with patch.object(logger, "warning") as mock_warning:
            kwargs = {"arg1": 1, "arg2": "test", "arg3": "test_again"}
            typed_callables = [typed_callable, typed_callable_valid]
            validate_kwarg_typing(typed_callables, **kwargs)
            expected_message = (
                f"`{typed_callable_valid}` expected argument `arg3` to be of type"
                f" {type(1)}. Got test_again (type: {type('test_again')})."
            )
            mock_warning.assert_called_once_with(expected_message)

        # mismatch types with Dict
        with patch.object(logger, "warning") as mock_warning:
            str_dic = {"k1": "test"}
            kwargs = {"arg1": 1, "arg2": "test", "arg3": 2, "arg4": str_dic}
            typed_callables = [typed_callable, typed_callable_with_dict]
            validate_kwarg_typing(typed_callables, **kwargs)
            expected_message = (
                f"`{typed_callable_with_dict}` expected argument `arg4` to be of type"
                f" typing.Dict[str, int]. Got {str_dic} (type: {type(str_dic)})."
            )
            mock_warning.assert_called_once_with(expected_message)

        # mismatch types with callable as arg
        with patch.object(logger, "warning") as mock_warning:
            kwargs = {"arg1": 1, "arg2": "test_again"}
            typed_callables = [typed_callable_with_callable]
            validate_kwarg_typing(typed_callables, **kwargs)
            expected_message = (
                f"`{typed_callable_with_callable}` expected argument `arg2` to be of"
                f" type typing.Callable[[int], typing.Dict[str, int]]. "
                f"Got test_again (type: {type('test_again')})."
            )
            mock_warning.assert_called_once_with(expected_message)


class TestWarnOnKwargs(TestCase):
    def test_it_warns_if_kwargs_are_passed(self):
        with patch.object(logger, "warning") as mock_warning:

            def callable_arg():
                return

            warn_on_kwargs(callable_with_kwargs=callable_arg, foo="")
            mock_warning.assert_called_once_with(
                "Found unexpected kwargs: %s while calling %s "
                "from JSON. These kwargs will be ignored.",
                {"foo": ""},
                callable_arg,
            )

    def test_it_does_not_warn_if_no_kwargs_are_passed(self):
        with patch.object(logger, "warning") as mock_warning:
            warn_on_kwargs(callable_with_kwargs=lambda: None)
            mock_warning.assert_not_called()
