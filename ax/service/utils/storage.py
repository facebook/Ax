#!/usr/bin/env python3
from ax.core.simple_experiment import SimpleExperiment
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


def load_experiment(name: str, db_settings: DBSettings) -> SimpleExperiment:
    """
    Load experiment from the db. Service API only supports `SimpleExperiment`.

    Args:
        name: Experiment name.
        db_settings: Specifies decoder and xdb tier where experiment is stored.

    Returns:
        SimpleExperiment: created `SimpleExperiment` object.
    """
    initialize_db(db_settings)
    experiment = _load_experiment(name, db_settings.decoder)
    if not isinstance(experiment, SimpleExperiment):
        raise ValueError("Service API only supports SimpleExperiment")
    return experiment


def save_experiment(experiment: SimpleExperiment, db_settings: DBSettings) -> None:
    """
    Save experiment to db.

    Args:
        experiment: `SimpleExperiment` object.
        db_settings: Specifies decoder and xdb tier where experiment is stored.
    """
    initialize_db(db_settings)
    _save_experiment(experiment, db_settings.encoder)
