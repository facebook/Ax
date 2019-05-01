#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import datetime
from typing import Optional

from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.types import Integer, TypeDecorator


class IntTimestamp(TypeDecorator):
    impl = Integer

    def process_bind_param(
        self, value: Optional[datetime.datetime], dialect: Dialect
    ) -> Optional[int]:
        if value is None:
            return None  # pragma: no cover
        else:
            return int(value.timestamp())

    def process_result_value(
        self, value: Optional[int], dialect: Dialect
    ) -> Optional[datetime.datetime]:
        return None if value is None else datetime.datetime.fromtimestamp(value)
