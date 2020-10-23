#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional

from ax.utils.common.equality import equality_typechecker, object_attribute_dicts_equal


class Base:
    """Metaclass for core Ax classes. Provides an equality check and `db_id`
    property for SQA storage.
    """

    _db_id: Optional[int] = None

    @property
    def db_id(self) -> Optional[int]:
        return self._db_id

    @db_id.setter
    def db_id(self, db_id: int) -> None:
        self._db_id = db_id

    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        return object_attribute_dicts_equal(
            one_dict=self.__dict__, other_dict=other.__dict__
        )
