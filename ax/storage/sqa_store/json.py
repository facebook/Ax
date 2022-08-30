#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from json import JSONDecodeError
from typing import Any, Dict, List, Optional

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
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        object_pairs_hook: Any = None,
        *args: List[Any],
        **kwargs: Dict[Any, Any],
    ) -> None:
        # pyre-fixme[4]: Attribute annotation cannot be `Any`.
        self.object_pairs_hook: Any = object_pairs_hook
        super().__init__(*args, **kwargs)

    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def process_bind_param(self, value: Any, dialect: Any) -> Optional[str]:
        if value is not None:
            return json.dumps(value)
        else:
            return None

    # pyre-fixme[3]: Return annotation cannot be `Any`.
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is not None:
            try:  # TODO T61331534: revert this; just a hotfix for AutoML
                return json.loads(value, object_pairs_hook=self.object_pairs_hook)
            except JSONDecodeError:
                return None
        else:
            return None


class JSONEncodedText(JSONEncodedObject):
    """Class for JSON-encoding objects in SQLAlchemy, backed by TEXT (MySQL).

    See description in JSONEncodedObject.

    """

    # pyre-fixme[15]: `impl` overrides attribute defined in `JSONEncodedObject`
    #  inconsistently.
    impl = Text


class JSONEncodedMediumText(JSONEncodedObject):
    """Class for JSON-encoding objects in SQLAlchemy, backed by MEDIUMTEXT
    (MySQL).

    See description in JSONEncodedObject.

    """

    # pyre-fixme[15]: `impl` overrides attribute defined in `JSONEncodedObject`
    #  inconsistently.
    impl = Text(MEDIUMTEXT_BYTES)


class JSONEncodedLongText(JSONEncodedObject):
    """Class for JSON-encoding objects in SQLAlchemy, backed by MEDIUMTEXT
    (MySQL).

    See description in JSONEncodedObject.

    """

    # pyre-fixme[15]: `impl` overrides attribute defined in `JSONEncodedObject`
    #  inconsistently.
    impl = Text(LONGTEXT_BYTES)


# pyre-fixme[5]: Global expression must be annotated.
JSONEncodedList = MutableList.as_mutable(JSONEncodedObject)
# pyre-fixme[5]: Global expression must be annotated.
JSONEncodedDict = MutableDict.as_mutable(JSONEncodedObject)
# pyre-fixme[5]: Global expression must be annotated.
JSONEncodedTextDict = MutableDict.as_mutable(JSONEncodedText)
# pyre-fixme[5]: Global expression must be annotated.
JSONEncodedLongTextDict = MutableDict.as_mutable(JSONEncodedLongText)
