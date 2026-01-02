# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Any

# backward compatibility
from ax.core.data import MAP_KEY  # noqa F401

from ax.exceptions.core import AxError


class MapData:
    """MapData no longer exists. Use Data instead."""

    def __init__(self, *_: Any, **__: Any) -> None:
        raise AxError("MapData no longer exists. Use Data instead.")
