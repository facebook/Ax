#!/usr/bin/env python3

import json

from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.storage.json_store.encoder import object_to_json


def save_experiment(experiment: Experiment, filepath: str) -> None:
    """Save experiment to file.

    1) Convert AE experiment to JSON-serializable dictionary.
    2) Write to file.
    """
    if not filepath.endswith(".json"):
        raise ValueError("Filepath must end in .json")

    json_experiment = object_to_json(experiment)
    with open(filepath, "w+") as file:
        file.write(json.dumps(json_experiment))
