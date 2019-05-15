#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from ax.core.experiment import Experiment
from ax.storage.sqa_store.db import init_engine_and_session_factory
from ax.storage.sqa_store.load import _load_experiment
from ax.storage.sqa_store.save import _save_experiment
from ax.storage.sqa_store.structs import DBSettings


"""Utilities for storing experiment to the database for AxClient."""


def load_experiment(name: str, db_settings: DBSettings) -> Experiment:
    """
    Load experiment from the db. Service API only supports `Experiment`.

    Args:
        name: Experiment name.
        db_settings: Defines behavior for loading/saving experiment to/from db.

    Returns:
        ax.core.Experiment: Loaded experiment.
    """
    init_engine_and_session_factory(creator=db_settings.creator, url=db_settings.url)
    experiment = _load_experiment(name, decoder=db_settings.decoder)
    if not isinstance(experiment, Experiment) or experiment.is_simple_experiment:
        raise ValueError("Service API only supports Experiment")
    return experiment


def save_experiment(experiment: Experiment, db_settings: DBSettings) -> None:
    """
    Save experiment to db.

    Args:
        experiment: `Experiment` object.
        db_settings: Defines behavior for loading/saving experiment to/from db.
    """
    init_engine_and_session_factory(creator=db_settings.creator, url=db_settings.url)
    _save_experiment(experiment, encoder=db_settings.encoder)
