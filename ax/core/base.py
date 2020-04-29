#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from ax.utils.common.equality import equality_typechecker, object_attribute_dicts_equal


class Base(object):
    """Metaclass for core Ax classes."""

    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        return object_attribute_dicts_equal(
            one_dict=self.__dict__, other_dict=other.__dict__
        )
