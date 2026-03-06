#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
from json import JSONDecodeError
from typing import Any

from ax.storage.sqa_store.db import JSON_FIELD_LENGTH, LONGTEXT_BYTES, MEDIUMTEXT_BYTES
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.types import Text, TypeDecorator, VARCHAR


class JSONEncodedObject(TypeDecorator):
    """Class for JSON-encoding objects in SQLAlchemy.

    Represents an object that is automatically marshalled and unmarshalled
    to/from the corresponding JSON string. By itself, this data structure does
    not track any changes.

    """

    impl: VARCHAR = VARCHAR(JSON_FIELD_LENGTH)

    cache_ok = True

    def __init__(
        self,
        object_pairs_hook: type[Any] | None = None,
        *args: list[Any],
        **kwargs: dict[Any, Any],
    ) -> None:
        self.object_pairs_hook: type[Any] | None = object_pairs_hook
        super().__init__(*args, **kwargs)

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is not None:
            return json.dumps(value)
        else:
            return None

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is not None:
            try:  # TODO T61331534: revert this; just a hotfix for AutoML
                # pyre-fixme[6]: `object_pairs_hook` expects a callable but
                #  `type[Any] | None` is stored; compatible at runtime.
                return json.loads(value, object_pairs_hook=self.object_pairs_hook)
            except JSONDecodeError:
                return None
        else:
            return None


class JSONEncodedText(JSONEncodedObject):
    """Class for JSON-encoding objects in SQLAlchemy, backed by TEXT (MySQL).

    See description in JSONEncodedObject.

    """

    # pyre-fixme[15]: `impl` overrides attribute in `JSONEncodedObject` with
    #  incompatible type; SQLAlchemy allows broader `impl` types at runtime.
    impl = Text


class JSONEncodedMediumText(JSONEncodedObject):
    """Class for JSON-encoding objects in SQLAlchemy, backed by MEDIUMTEXT
    (MySQL).

    See description in JSONEncodedObject.

    """

    # pyre-fixme[15]: `impl` overrides attribute in `JSONEncodedObject` with
    #  incompatible type; SQLAlchemy allows broader `impl` types at runtime.
    impl = Text(MEDIUMTEXT_BYTES)


class JSONEncodedLongText(JSONEncodedObject):
    """Class for JSON-encoding objects in SQLAlchemy, backed by LONGTEXT
    (MySQL).

    See description in JSONEncodedObject.

    """

    # pyre-fixme[15]: `impl` overrides attribute in `JSONEncodedObject` with
    #  incompatible type; SQLAlchemy allows broader `impl` types at runtime.
    impl = Text(LONGTEXT_BYTES)


JSONEncodedList: TypeDecorator = MutableList.as_mutable(JSONEncodedObject)
JSONEncodedDict: TypeDecorator = MutableDict.as_mutable(JSONEncodedObject)
JSONEncodedTextDict: TypeDecorator = MutableDict.as_mutable(JSONEncodedText)
JSONEncodedLongTextDict: TypeDecorator = MutableDict.as_mutable(JSONEncodedLongText)
