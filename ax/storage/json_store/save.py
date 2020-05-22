#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

from ax.core.experiment import Experiment
from ax.storage.json_store.encoder import object_to_json


def save_experiment(experiment: Experiment, filepath: str) -> None:
    """Save experiment to file.

    1) Convert Ax experiment to JSON-serializable dictionary.
    2) Write to file.
    """
    if not isinstance(experiment, Experiment):
        raise ValueError("Can only save instances of Experiment")

    if not filepath.endswith(".json"):
        raise ValueError("Filepath must end in .json")

    json_experiment = object_to_json(experiment)
    with open(filepath, "w+") as file:
        file.write(json.dumps(json_experiment))
