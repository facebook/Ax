#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
from typing import Any, Dict, List

from ax.storage.sqa_store.db import NAME_OR_TYPE_FIELD_LENGTH
from sqlalchemy import types


class BaseNullableEnum(types.TypeDecorator):
    cache_ok = True

    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def __init__(self, enum: Any, *arg: List[Any], **kw: Dict[Any, Any]) -> None:
        types.TypeDecorator.__init__(self, *arg, **kw)
        # pyre-fixme[4]: Attribute must be annotated.
        self._member_map = enum._member_map_
        # pyre-fixme[4]: Attribute must be annotated.
        self._value2member_map = enum._value2member_map_

    # pyre-fixme[3]: Return annotation cannot be `Any`.
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return value  # pragma: no cover
        if not isinstance(value, enum.Enum):
            raise TypeError("Value is not an instance of Enum.")  # pragma: no cover
        val = self._member_map.get(value.name)
        if val is None:
            raise ValueError(  # pragma: no cover
                "Member '{value}' is not a supported enum: {members}".format(
                    value=value, members=list(self._member_map.keys())
                )
            )
        return val._value_

    # pyre-fixme[3]: Return annotation cannot be `Any`.
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return value  # pragma: no cover
        member = self._value2member_map.get(value)
        if member is None:
            raise ValueError(  # pragma: no cover
                f"Value '{value}' is not one of the supported "
                + "enum values: {supported_values}".format(
                    supported_values=list(self._value2member_map.keys())
                )
            )
        return member


class IntEnum(BaseNullableEnum):
    # pyre-fixme[8]: Attribute has type `SmallInteger`; used as
    #  `Type[sqlalchemy.sql.sqltypes.SmallInteger]`.
    impl: types.SmallInteger = types.SmallInteger


class StringEnum(BaseNullableEnum):
    impl = types.VARCHAR(NAME_OR_TYPE_FIELD_LENGTH)
