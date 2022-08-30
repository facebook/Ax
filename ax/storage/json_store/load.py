#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Callable, Dict, Type

from ax.core.experiment import Experiment
from ax.storage.json_store.decoder import object_from_json
from ax.storage.json_store.registry import (
    CORE_CLASS_DECODER_REGISTRY,
    CORE_DECODER_REGISTRY,
)


def load_experiment(
    filepath: str,
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
    #  `typing.Type` to avoid runtime subscripting errors.
    decoder_registry: Dict[str, Type] = CORE_DECODER_REGISTRY,
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    class_decoder_registry: Dict[
        str, Callable[[Dict[str, Any]], Any]
    ] = CORE_CLASS_DECODER_REGISTRY,
) -> Experiment:
    """Load experiment from file.

    1) Read file.
    2) Convert dictionary to Ax experiment instance.
    """
    with open(filepath, "r") as file:
        json_experiment = json.loads(file.read())
        return object_from_json(
            json_experiment, decoder_registry, class_decoder_registry
        )
