#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


class Unset:
    """Sentinel type for distinguishing "not provided" from an explicit ``None``.

    Use the module-level ``UNSET`` instance as the default value for
    optional fields where ``None`` is a valid, meaningful value and a
    separate "not set" state is needed.
    """

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Unset = Unset()
