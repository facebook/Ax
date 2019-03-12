#!/usr/bin/env python3
import json

from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.storage.json_store.decoder import object_from_json


def load_experiment(filepath: str) -> Experiment:
    """Load experiment from file.

    1) Read file.
    2) Convert dictionary to AE experiment instance.
    """
    with open(filepath, "r") as file:
        json_experiment = json.loads(file.read())
        return object_from_json(json_experiment)
