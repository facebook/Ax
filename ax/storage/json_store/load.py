#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import json

from ax.core.experiment import Experiment
from ax.storage.json_store.decoder import object_from_json


def load_experiment(filepath: str) -> Experiment:
    """Load experiment from file.

    1) Read file.
    2) Convert dictionary to Ax experiment instance.
    """
    with open(filepath, "r") as file:
        json_experiment = json.loads(file.read())
        return object_from_json(json_experiment)
