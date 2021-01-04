#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import enum
from collections import OrderedDict
from inspect import isclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from ax.exceptions.storage import JSONEncodeError
from ax.storage.json_store.registry import CLASS_ENCODER_REGISTRY, ENCODER_REGISTRY
from ax.utils.common.serialization import _is_named_tuple
from ax.utils.common.typeutils import numpy_type_to_python_type, torch_type_to_str


def object_to_json(obj: Any) -> Any:
    """Convert an Ax object to a JSON-serializable dictionary.

    The root node passed to this function should always be an instance of a
    core Ax class or a JSON-compatible python builtin. The sub-fields of the
    input will then be recursively passed to this function.

    e.g. if we pass an instance of Experiment, we will first fall through
    to the line `object_dict = ENCODER_REGISTRY[_type](object)`, which
    will convert the Experiment to a (shallow) dictionary, where search
    subfield remains "unconverted", i.e.:
    {"name": <name: string>, "search_space": <search space: SearchSpace>}.
    We then pass each item of the dictionary back into this function to
    recursively convert the entire object.
    """
    obj = numpy_type_to_python_type(obj)
    _type = type(obj)

    # Type[MyClass] encoding (encoding of classes, not instances)
    if isclass(obj):
        for class_type in CLASS_ENCODER_REGISTRY:
            if issubclass(obj, class_type):
                obj_dict = CLASS_ENCODER_REGISTRY[class_type](obj)
                return {k: object_to_json(v) for k, v in obj_dict.items()}
        raise ValueError(
            f"{obj} is a class. Add it to the CLASS_ENCODER_REGISTRY "
            "(and remove it from the ENCODER_REGISTRY if needed)."
        )

    if _type in ENCODER_REGISTRY:
        obj_dict = ENCODER_REGISTRY[_type](obj)
        return {k: object_to_json(v) for k, v in obj_dict.items()}

    # Python built-in types + `typing` module types
    if _type in (str, int, float, bool, type(None)):
        return obj
    elif _type is list:
        return [object_to_json(x) for x in obj]
    elif _type is tuple:
        return tuple(object_to_json(x) for x in obj)
    elif _type is dict:
        return {k: object_to_json(v) for k, v in obj.items()}
    elif _is_named_tuple(obj):
        return {
            "__type": _type.__name__,
            **{k: object_to_json(v) for k, v in obj._asdict().items()},
        }

    # Types from libraries, commonly used in Ax (e.g., numpy, pandas, torch)
    elif _type is OrderedDict:
        return {
            "__type": _type.__name__,
            "value": [(k, object_to_json(v)) for k, v in obj.items()],
        }
    elif _type is datetime.datetime:
        return {
            "__type": _type.__name__,
            "value": datetime.datetime.strftime(obj, "%Y-%m-%d %H:%M:%S.%f"),
        }
    elif _type is pd.DataFrame:
        return {"__type": _type.__name__, "value": obj.to_json()}
    elif issubclass(_type, enum.Enum):
        return {"__type": _type.__name__, "name": obj.name}
    elif _type is np.ndarray or issubclass(_type, np.ndarray):
        return {"__type": _type.__name__, "value": obj.tolist()}
    elif _type is torch.Tensor:
        return {
            "__type": _type.__name__,
            # TODO: check size and add warning for large tensors: T69137799
            "value": obj.tolist(),
            "dtype": object_to_json(obj.dtype),
            "device": object_to_json(obj.device),
        }
    elif _type.__module__ == "torch":
        # Torch does not support saving to string, so save to buffer first
        return {"__type": f"torch_{_type.__name__}", "value": torch_type_to_str(obj)}

    err = (
        f"Object {obj} passed to `object_to_json` (of type {_type}) is "
        f"not registered with a corresponding encoder in ENCODER_REGISTRY."
    )
    raise JSONEncodeError(err)
