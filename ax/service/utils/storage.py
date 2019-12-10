#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

from ax.core.experiment import Experiment
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.storage.sqa_store.db import init_engine_and_session_factory
from ax.storage.sqa_store.load import (
    _load_experiment,
    _load_generation_strategy_by_experiment_name,
)
from ax.storage.sqa_store.save import _save_experiment, _save_generation_strategy
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


def save_experiment(
    experiment: Experiment,
    db_settings: DBSettings,
    overwrite_existing_experiment: bool = False,
) -> None:
    """
    Save experiment to db.

    Args:
        experiment: `Experiment` object.
        db_settings: Defines behavior for loading/saving experiment to/from db.
        overwrite_existing_experiment: If the experiment being created
            has the same name as some experiment already stored, this flag
            determines whether to overwrite the existing experiment.
            Defaults to False.
    """
    init_engine_and_session_factory(creator=db_settings.creator, url=db_settings.url)
    _save_experiment(
        experiment, encoder=db_settings.encoder, overwrite=overwrite_existing_experiment
    )


def load_experiment_and_generation_strategy(
    experiment_name: str, db_settings: DBSettings
) -> Tuple[Experiment, Optional[GenerationStrategy]]:
    """Load experiment and the corresponding generation strategy from the DB.

    Args:
        name: Experiment name.
        db_settings: Defines behavior for loading/saving experiment to/from db.

    Returns:
        A tuple of the loaded experiment and generation strategy.
    """
    experiment = load_experiment(name=experiment_name, db_settings=db_settings)
    try:
        generation_strategy = _load_generation_strategy_by_experiment_name(
            experiment_name=experiment_name, decoder=db_settings.decoder
        )
    except ValueError:  # Generation strategy has not yet been attached to this
        return experiment, None  # experiment.
    return experiment, generation_strategy


def save_experiment_and_generation_strategy(
    experiment: Experiment,
    generation_strategy: GenerationStrategy,
    db_settings: DBSettings,
    overwrite_existing_experiment: bool = False,
) -> None:
    """Save experiment and generation strategy to DB.

    Args:
        experiment: `Experiment` object.
        generation_strategy: Corresponding generation strategy.
        db_settings: Defines behavior for loading/saving experiment to/from db.
        overwrite_existing_experiment: If the experiment being created
            has the same name as some experiment already stored, this flag
            determines whether to overwrite the existing experiment.
            Defaults to False.
    """
    save_experiment(
        experiment=experiment,
        db_settings=db_settings,
        overwrite_existing_experiment=overwrite_existing_experiment,
    )
    _save_generation_strategy(
        generation_strategy=generation_strategy, encoder=db_settings.encoder
    )
