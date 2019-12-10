#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import enum
from typing import Any, Dict, List

from ax.storage.sqa_store.db import NAME_OR_TYPE_FIELD_LENGTH
from sqlalchemy import types


class BaseNullableEnum(types.TypeDecorator):
    def __init__(self, enum: Any, *arg: List[Any], **kw: Dict[Any, Any]) -> None:
        types.TypeDecorator.__init__(self, *arg, **kw)
        self._member_map = enum._member_map_
        self._value2member_map = enum._value2member_map_

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
    impl: types.SmallInteger = types.SmallInteger


class StringEnum(BaseNullableEnum):
    impl = types.VARCHAR(NAME_OR_TYPE_FIELD_LENGTH)
