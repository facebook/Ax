#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from logging import Logger
from unittest.mock import patch

from ax.utils.common.kwargs import warn_on_kwargs
from ax.utils.common.logger import get_logger
from ax.utils.common.testutils import TestCase

logger: Logger = get_logger("ax.utils.common.kwargs")


class TestWarnOnKwargs(TestCase):
    def test_it_warns_if_kwargs_are_passed(self) -> None:
        with patch.object(logger, "warning") as mock_warning:

            def callable_arg() -> None:
                return

            warn_on_kwargs(callable_with_kwargs=callable_arg, foo="")
            mock_warning.assert_called_once_with(
                "Found unexpected kwargs: %s while calling %s "
                "from JSON. These kwargs will be ignored.",
                {"foo": ""},
                callable_arg,
            )

    def test_it_does_not_warn_if_no_kwargs_are_passed(self) -> None:
        with patch.object(logger, "warning") as mock_warning:
            warn_on_kwargs(callable_with_kwargs=lambda: None)
            mock_warning.assert_not_called()
