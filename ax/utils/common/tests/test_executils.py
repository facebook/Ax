#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from asyncio import iscoroutinefunction
from unittest.mock import Mock

from ax.utils.common.executils import retry_on_exception
from ax.utils.common.testutils import TestCase


class TestRetryDecorator(TestCase):
    def test_default_return(self) -> None:
        """
        Tests if the decorator correctly returns the default value.
        """

        class DecoratorTester:
            @retry_on_exception(
                suppress_all_errors=True, default_return_on_suppression="SUCCESS"
            )
            def error_throwing_function(self) -> None:
                raise RuntimeError("ERROR THROWN FOR TESTING. SHOULD HAVE BEEN CAUGHT")

        decorator_tester = DecoratorTester()
        self.assertEqual("SUCCESS", decorator_tester.error_throwing_function())

    def test_kwarg_passage(self) -> None:
        """
        Tests if the decorator correctly takes the suppress all errors
        flag from the kwargs of the decorated function.
        """

        class DecoratorTester:
            @retry_on_exception(default_return_on_suppression="SUCCESS")
            def error_throwing_function(
                self, suppress_all_errors=False, extra_kwarg="1234"
            ):
                # Testing that kwargs get passed down correctly
                self.assertEqual(extra_kwarg, "abcd")
                raise RuntimeError("ERROR THROWN FOR TESTING. SHOULD HAVE BEEN CAUGHT")

        decorator_tester = DecoratorTester()
        self.assertEqual(
            "SUCCESS",
            decorator_tester.error_throwing_function(
                suppress_all_errors=True, extra_kwarg="abcd"
            ),
        )

    def test_message_checking(self) -> None:
        """
        Tests if the decorator correctly checks the list of messages
        provided for which you may suppress the error.
        """

        # Also pass along the logger to ensure coverage
        logger = logging.getLogger("test_message_checking")

        class DecoratorTester:
            @retry_on_exception(
                default_return_on_suppression="SUCCESS",
                check_message_contains=["Hello", "World"],
                exception_types=(RuntimeError,),
                logger=logger,
                suppress_all_errors=True,
            )
            def error_throwing_function(self) -> None:
                # The exception thrown below should be caught and handled since it
                # has the keywords we want
                raise RuntimeError("Hello World")

        decorator_tester = DecoratorTester()
        self.assertEqual("SUCCESS", decorator_tester.error_throwing_function())

    def test_empty_exception_type_tuple(self) -> None:
        """
        Tests if the decorator correctly handles an empty list
        of exception types to suppress.
        """

        # Also pass along the logger to ensure coverage
        logger = logging.getLogger("test_message_checking")

        class DecoratorTester:
            @retry_on_exception(
                default_return_on_suppression="SUCCESS",
                exception_types=(),
                logger=logger,
                suppress_all_errors=False,
            )
            def error_throwing_function(self) -> None:
                # The exception thrown below should not be caught
                # because we specified an empty list.
                raise RuntimeError("Hello World")

        decorator_tester = DecoratorTester()
        with self.assertRaises(RuntimeError):
            decorator_tester.error_throwing_function()

    def test_message_checking_fail(self) -> None:
        """
        Tests if the decorator correctly checks the list of messages
        provided. In this case, we check if it correctly fails.
        """

        class DecoratorTester:
            @retry_on_exception(
                default_return_on_suppression="SUCCESS",
                check_message_contains=["Hello", "World"],
                exception_types=(RuntimeError,),
            )
            def error_throwing_function(self) -> None:
                # The execption thrown below should NOT be caught as it does not
                # contain the keywords we want
                raise RuntimeError

        decorator_tester = DecoratorTester()
        with self.assertRaises(RuntimeError):
            decorator_tester.error_throwing_function()

    def test_retry_mechanism(self) -> None:
        """
        Tests if the decorator retries sufficient number of times
        """

        class DecoratorTester:
            def __init__(self) -> None:
                self.retries_done = 0

            @retry_on_exception(retries=4)
            def error_throwing_function(self) -> None:
                # The call below will succeed only on the 3rd try
                return self.succeed_on_3rd_try()

            def succeed_on_3rd_try(self) -> None:
                if self.retries_done < 2:
                    self.retries_done += 1
                    raise RuntimeError(
                        "This error surfacing means enough retries were not done"
                    )
                else:
                    return "SUCCESS"

        decorator_tester = DecoratorTester()
        self.assertEqual("SUCCESS", decorator_tester.error_throwing_function())

    def test_retry_with_wait(self) -> None:
        """
        Tests if the decorator retries sufficient number of times
        """

        class DecoratorTester:
            def __init__(self) -> None:
                self.start_time = time.time()

            @retry_on_exception(retries=4, initial_wait_seconds=1)
            def error_throwing_function(self) -> None:
                # The call below will succeed only on the 3rd try
                return self.succeed_after_five_seconds()

            @retry_on_exception(retries=4)
            def no_wait_error_throwing_function(self) -> None:
                # The call below will succeed only on the 3rd try
                return self.succeed_after_five_seconds()

            def succeed_after_five_seconds(self) -> None:
                if time.time() - self.start_time < 5:
                    raise RuntimeError(
                        "This error surfacing means enough retries were not done"
                    )
                else:
                    return "SUCCESS"

        decorator_tester = DecoratorTester()
        self.assertEqual("SUCCESS", decorator_tester.error_throwing_function())

        decorator_tester = DecoratorTester()
        with self.assertRaises(RuntimeError):
            decorator_tester.no_wait_error_throwing_function()

    def test_retry_mechanism_fail(self) -> None:
        """
        Tests that the decorator does not retry too many times
        """

        # Also pass along the logger to ensure coverage
        logger = logging.getLogger("test_retry_mechanism_fail")

        class DecoratorTester:
            def __init__(self) -> None:
                self.xyz = 0

            @retry_on_exception(retries=2, logger=logger)
            def error_throwing_function(self) -> None:
                # The call below will succeed only on the 3rd try
                return self.succeed_on_3rd_try()

            def succeed_on_3rd_try(self) -> None:
                if self.xyz < 2:
                    self.xyz += 1
                    raise KeyError
                else:
                    return "SUCCESS"

        decorator_tester = DecoratorTester()
        with self.assertRaises(KeyError):
            decorator_tester.error_throwing_function()

    def test_no_retry_on_exception_types(self) -> None:
        class MyRuntimeError(RuntimeError):
            pass

        class DecoratorTester:
            error_throwing_function_call_count = 0

            @retry_on_exception(no_retry_on_exception_types=(MyRuntimeError,))
            def error_throwing_function(self) -> None:
                self.error_throwing_function_call_count += 1
                # The exception thrown below should NOT be caught as it does mathes
                # an exception type in `no_retry_on_exception_type`
                raise MyRuntimeError

        decorator_tester = DecoratorTester()
        with self.assertRaises(MyRuntimeError):
            decorator_tester.error_throwing_function()

        self.assertEqual(decorator_tester.error_throwing_function_call_count, 1)

        # Check that `MyRuntimeError` isn't retriable even if `RuntimeError` is.
        class DecoratorTester:
            error_throwing_function_call_count = 0

            @retry_on_exception(
                exception_types=(RuntimeError,),
                no_retry_on_exception_types=(MyRuntimeError,),
            )
            def error_throwing_function(self) -> None:
                self.error_throwing_function_call_count += 1
                # The exception thrown below should NOT be caught as it does mathes
                # an exception type in `no_retry_on_exception_type`
                raise MyRuntimeError

        decorator_tester = DecoratorTester()
        with self.assertRaises(MyRuntimeError):
            decorator_tester.error_throwing_function()

    def test_on_function_with_wrapper_message(self) -> None:
        """Tests that the decorator works on standalone functions as well as on
        instance methods.
        """

        mock = Mock()

        @retry_on_exception(wrap_error_message_in="Wrapper error message")
        # pyre-fixme[53]: Captured variable `mock` is not annotated.
        # pyre-fixme[3]: Return type must be annotated.
        def error_throwing_function():
            mock()
            raise RuntimeError("I failed")

        # Check that the function remains non-async.
        self.assertFalse(iscoroutinefunction(error_throwing_function))

        with self.assertRaisesRegex(
            RuntimeError, "Wrapper error message: RuntimeError: I failed"
        ):
            error_throwing_function()

        self.assertEqual(mock.call_count, 3)
