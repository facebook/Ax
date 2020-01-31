#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from json import JSONDecodeError
from typing import Any, Dict, List, Optional

from ax.storage.sqa_store.db import JSON_FIELD_LENGTH, MEDIUMTEXT_BYTES
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.types import VARCHAR, Text, TypeDecorator


class JSONEncodedObject(TypeDecorator):
    """Class for JSON-encoding objects in SQLAlchemy.

    Represents an object that is automatically marshalled and unmarshalled
    to/from the corresponding JSON string. By itself, this data structure does
    not track any changes.

    """

    impl: VARCHAR = VARCHAR(JSON_FIELD_LENGTH)

    def __init__(
        self, object_pairs_hook: Any = None, *args: List[Any], **kwargs: Dict[Any, Any]
    ) -> None:
        self.object_pairs_hook: Any = object_pairs_hook
        super().__init__(*args, **kwargs)

    def process_bind_param(self, value: Any, dialect: Any) -> Optional[str]:
        if value is not None:
            return json.dumps(value)
        else:
            return None

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


JSONEncodedList = MutableList.as_mutable(JSONEncodedObject)
JSONEncodedDict = MutableDict.as_mutable(JSONEncodedObject)
JSONEncodedTextDict = MutableDict.as_mutable(JSONEncodedText)
