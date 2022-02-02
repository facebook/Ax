#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Callable, Type, Dict

from ax.core.experiment import Experiment
from ax.storage.json_store.decoder import object_from_json


def load_experiment(
    filepath: str,
    decoder_registry: Dict[str, Type],
    class_decoder_registry: Dict[str, Callable[[Dict[str, Any]], Any]],
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
