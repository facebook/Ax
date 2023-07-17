#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from typing import Optional

from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.types import Integer, TypeDecorator


class IntTimestamp(TypeDecorator):
    impl = Integer
    cache_ok = True

    # pyre-fixme[15]: `process_bind_param` overrides method defined in
    #  `TypeDecorator` inconsistently.
    def process_bind_param(
        self, value: Optional[datetime.datetime], dialect: Dialect
    ) -> Optional[int]:
        if value is None:
            return None
        else:
            return int(value.timestamp())

    def process_result_value(
        self, value: Optional[int], dialect: Dialect
    ) -> Optional[datetime.datetime]:
        return None if value is None else datetime.datetime.fromtimestamp(value)
