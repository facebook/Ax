#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Callable, Dict, Type

from ax.core.experiment import Experiment
from ax.storage.json_store.encoder import object_to_json
from ax.storage.json_store.registry import (
    CORE_CLASS_ENCODER_REGISTRY,
    CORE_ENCODER_REGISTRY,
)


def save_experiment(
    experiment: Experiment,
    filepath: str,
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
    #  `typing.Type` to avoid runtime subscripting errors.
    encoder_registry: Dict[
        Type, Callable[[Any], Dict[str, Any]]
    ] = CORE_ENCODER_REGISTRY,
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
    #  `typing.Type` to avoid runtime subscripting errors.
    class_encoder_registry: Dict[
        Type, Callable[[Any], Dict[str, Any]]
    ] = CORE_CLASS_ENCODER_REGISTRY,
) -> None:
    """Save experiment to file.

    1) Convert Ax experiment to JSON-serializable dictionary.
    2) Write to file.
    """
    if not isinstance(experiment, Experiment):
        raise ValueError("Can only save instances of Experiment")

    if not filepath.endswith(".json"):
        raise ValueError("Filepath must end in .json")

    json_experiment = object_to_json(
        experiment,
        encoder_registry=encoder_registry,
        class_encoder_registry=class_encoder_registry,
    )
    with open(filepath, "w+") as file:
        file.write(json.dumps(json_experiment))
