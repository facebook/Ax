#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import datetime
import enum
from collections import OrderedDict
from typing import Any

import pandas as pd
from ax.exceptions.storage import JSONEncodeError
from ax.storage.json_store.registry import ENCODER_REGISTRY
from ax.utils.common.serialization import _is_named_tuple
from ax.utils.common.typeutils import numpy_type_to_python_type


def object_to_json(object: Any) -> Any:
    """Convert an Ax object to a JSON-serializable dictionary.

    The root node passed to this function should always be an instance of a
    core Ax class. The sub-fields of this object will then be recursively
    passed to this function.

    e.g. if we pass an instance of Experiment, we will first fall through
    to the line `object_dict = ENCODER_REGISTRY[_type](object)`, which
    will convert the Experiment to a (shallow) dictionary, where search
    subfield remains "unconverted", i.e.:
    {"name": <name: string>, "search_space": <search space: SearchSpace>}.
    We then pass each item of the dictionary back into this function to
    recursively convert the entire object.
    """
    object = numpy_type_to_python_type(object)
    _type = type(object)
    if _type in (str, int, float, bool, type(None)):
        return object
    elif _type is list:
        return [object_to_json(x) for x in object]
    elif _type is tuple:
        return tuple(object_to_json(x) for x in object)
    elif _type is dict:
        return {k: object_to_json(v) for k, v in object.items()}
    elif _type is OrderedDict:
        return {
            "__type": _type.__name__,
            "value": [(k, object_to_json(v)) for k, v in object.items()],
        }
    elif _type is datetime.datetime:
        return {
            "__type": _type.__name__,
            "value": datetime.datetime.strftime(object, "%Y-%m-%d %H:%M:%S.%f"),
        }
    elif _type is pd.DataFrame:
        return {"__type": _type.__name__, "value": object.to_json()}
    elif _is_named_tuple(object):
        return {
            "__type": _type.__name__,
            **{k: object_to_json(v) for k, v in object._asdict().items()},
        }
    elif issubclass(_type, enum.Enum):
        return {"__type": _type.__name__, "name": object.name}

    if _type not in ENCODER_REGISTRY:
        err = (
            f"Object passed to `object_to_json` (of type {_type}) is not "
            f"registered with a corresponding encoder in ENCODER_REGISTRY."
        )
        raise JSONEncodeError(err)
    object_dict = ENCODER_REGISTRY[_type](object)
    return {k: object_to_json(v) for k, v in object_dict.items()}
