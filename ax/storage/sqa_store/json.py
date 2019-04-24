#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
from typing import Any, Dict, List, Optional

from ax.storage.sqa_store.db import JSON_FIELD_LENGTH
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
            return json.loads(value, object_pairs_hook=self.object_pairs_hook)
        else:
            return None


class JSONEncodedText(JSONEncodedObject):
    """Class for JSON-encoding objects in SQLAlchemy, backed by TEXT (MySQL).

    See description in JSONEncodedObject.

    """

    impl = Text


JSONEncodedList = MutableList.as_mutable(JSONEncodedObject)
JSONEncodedDict = MutableDict.as_mutable(JSONEncodedObject)
JSONEncodedTextDict = MutableDict.as_mutable(JSONEncodedText)
