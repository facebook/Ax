#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import abc

from ax.utils.common.equality import equality_typechecker, object_attribute_dicts_equal


class Base:
    """Metaclass for core Ax classes. Provides an equality check and `db_id`
    property for SQA storage.
    """

    _db_id: int | None = None

    @property
    def db_id(self) -> int | None:
        return self._db_id

    @db_id.setter
    def db_id(self, db_id: int) -> None:
        self._db_id = db_id

    @equality_typechecker
    def __eq__(self, other: Base) -> bool:
        return object_attribute_dicts_equal(
            one_dict=self.__dict__, other_dict=other.__dict__
        )

    @equality_typechecker
    def _eq_skip_db_id_check(self, other: Base) -> bool:
        return object_attribute_dicts_equal(
            one_dict=self.__dict__, other_dict=other.__dict__, skip_db_id_check=True
        )


class SortableBase(Base, metaclass=abc.ABCMeta):
    """Extension to the base class that also provides an inequality check."""

    @property
    @abc.abstractmethod
    def _unique_id(self) -> str:
        """Returns an identification string that can be used to uniquely
        identify this instance from others attached to the same parent
        object. For example, for ``Trials`` this can be their index,

        since that is unique w.r.t. to parent ``Experiment`` object.
        For ``GenerationNode``-s attached to a ``GenerationStrategy``,
        this can be their name since we ensure uniqueness of it upon
        ``GenerationStrategy`` instantiation.

        This method is needed to correctly update SQLAlchemy objects
        that appear as children of other objects, in lists or other
        sortable collections or containers.
        """
        pass

    def __lt__(self, other: SortableBase) -> bool:
        return self._unique_id < other._unique_id
