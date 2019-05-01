#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from ax.core.experiment import Experiment
from ax.storage.sqa_store.db import init_engine_and_session_factory
from ax.storage.sqa_store.load import _load_experiment
from ax.storage.sqa_store.save import _save_experiment
from ax.storage.sqa_store.structs import DBSettings


"""Utilities for storing experiment to the database for AxClient."""


def initialize_db(db_settings: DBSettings) -> None:
    """
    Initialize db connections given settings.

    Args:
        db_settings: Optional[DBSettings] = None
    """
    if db_settings.creator is not None:
        init_engine_and_session_factory(creator=db_settings.creator)


def load_experiment(name: str, db_settings: DBSettings) -> Experiment:
    """
    Load experiment from the db. Service API only supports `Experiment`.

    Args:
        name: Experiment name.
        db_settings: Specifies decoder and xdb tier where experiment is stored.

    Returns:
        ax.core.Experiment: Loaded experiment.
    """
    initialize_db(db_settings)
    experiment = _load_experiment(name, db_settings.decoder)
    if not isinstance(experiment, Experiment):
        raise ValueError("Service API only supports Experiment")
    return experiment


def save_experiment(experiment: Experiment, db_settings: DBSettings) -> None:
    """
    Save experiment to db.

    Args:
        experiment: `Experiment` object.
        db_settings: Specifies decoder and xdb tier where experiment is stored.
    """
    initialize_db(db_settings)
    _save_experiment(experiment, db_settings.encoder)
