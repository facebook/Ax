#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from numbers import Real

# type aliases
TRGB = tuple[Real, ...]


def rgba(rgb_tuple: TRGB, alpha: float = 1) -> str:
    """Convert RGB tuple to an RGBA string."""
    return "rgba({},{},{},{alpha})".format(*rgb_tuple, alpha=alpha)
