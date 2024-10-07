#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
from collections.abc import Callable
from typing import Any

from ax.core.experiment import Experiment
from ax.storage.json_store.decoder import object_from_json
from ax.storage.json_store.registry import (
    CORE_CLASS_DECODER_REGISTRY,
    CORE_DECODER_REGISTRY,
)
from ax.utils.common.serialization import TDecoderRegistry


def load_experiment(
    filepath: str,
    decoder_registry: TDecoderRegistry = CORE_DECODER_REGISTRY,
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    class_decoder_registry: dict[
        str, Callable[[dict[str, Any]], Any]
    ] = CORE_CLASS_DECODER_REGISTRY,
) -> Experiment:
    """Load experiment from file.

    1) Read file.
    2) Convert dictionary to Ax experiment instance.
    """
    with open(filepath) as file:
        json_experiment = json.loads(file.read())
        return object_from_json(
            json_experiment, decoder_registry, class_decoder_registry
        )
