#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from contextlib import contextmanager
from typing import Any, Callable, TypeVar

from unittest.mock import MagicMock, patch


T = TypeVar("T")
C = TypeVar("C")


@contextmanager
def mock_patch_method_original(
    mock_path: str,
    original_method: Callable[..., T],
) -> MagicMock:
    """Context manager for patching a method returning type T on class C,
    to track calls to it while still executing the original method. There
    is not a native way to do this with `mock.patch`.
    """

    def side_effect(self: C, *args: Any, **kwargs: Any) -> T:
        # pyre-ignore[16]: Anonymous callable has no attribute `self`
        # (We can ignore because we expect C to be a class).
        side_effect.self = self
        return original_method(self, *args, **kwargs)

    patcher = patch(mock_path, autospec=True, side_effect=side_effect)
    yield patcher.start()
    patcher.stop()
