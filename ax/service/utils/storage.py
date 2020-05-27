#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Optional, Tuple

from ax.core.base_trial import BaseTrial
from ax.core.experiment import Experiment
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.storage.sqa_store.db import init_engine_and_session_factory
from ax.storage.sqa_store.load import (
    _load_experiment,
    _load_generation_strategy_by_experiment_name,
)
from ax.storage.sqa_store.save import (
    _save_experiment,
    _save_generation_strategy,
    _save_new_trial,
    _update_trial,
)
from ax.storage.sqa_store.structs import DBSettings
from ax.utils.common.logger import _round_floats_for_logging, get_logger


logger = get_logger(__name__)


"""Utilities for storing experiment to the database via `DBSettings`."""


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
    start_time = time.time()
    experiment = _load_experiment(name, decoder=db_settings.decoder)
    if not isinstance(experiment, Experiment) or experiment.is_simple_experiment:
        raise ValueError("Service API only supports Experiment")
    logger.info(
        f"Loaded experiment {name} in "
        f"{_round_floats_for_logging(time.time() - start_time)} seconds"
    )
    return experiment


def save_experiment(experiment: Experiment, db_settings: DBSettings) -> None:
    """
    Save experiment to db.

    Args:
        experiment: `Experiment` object.
        db_settings: Defines behavior for loading/saving experiment to/from db.
    """
    init_engine_and_session_factory(creator=db_settings.creator, url=db_settings.url)
    start_time = time.time()
    _save_experiment(experiment, encoder=db_settings.encoder)
    logger.info(
        f"Saved experiment {experiment.name} in "
        f"{_round_floats_for_logging(time.time() - start_time)} seconds"
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


def save_generation_strategy(
    generation_strategy: GenerationStrategy, db_settings: DBSettings
) -> None:
    """Save generation strategy to DB.

    Args:
        generation_strategy: Corresponding generation strategy.
        db_settings: Defines behavior for loading/saving experiment to/from db.
    """
    _save_generation_strategy(
        generation_strategy=generation_strategy, encoder=db_settings.encoder
    )


def save_new_trial(
    experiment: Experiment, trial: BaseTrial, db_settings: DBSettings
) -> None:
    """
    Save experiment to db.

    NOTE: This function also saves data attached to experiment
    for this trial.

    Args:
        experiment: `Experiment` object.
        trial: `BaseTrial` object.
        db_settings: Defines behavior for loading/saving experiment to/from db.
    """
    init_engine_and_session_factory(creator=db_settings.creator, url=db_settings.url)
    start_time = time.time()
    _save_new_trial(experiment=experiment, trial=trial, encoder=db_settings.encoder)
    logger.info(
        f"Saved trial {trial.index} in "
        f"{_round_floats_for_logging(time.time() - start_time)} seconds"
    )


def save_updated_trial(
    experiment: Experiment, trial: BaseTrial, db_settings: DBSettings
) -> None:
    """
    Save experiment to db.

    NOTE: This function also saves data attached to experiment
    for this trial.

    Args:
        experiment: `Experiment` object.
        trial: `BaseTrial` object.
        db_settings: Defines behavior for loading/saving experiment to/from db.
    """
    init_engine_and_session_factory(creator=db_settings.creator, url=db_settings.url)
    start_time = time.time()
    _update_trial(experiment=experiment, trial=trial, encoder=db_settings.encoder)
    logger.info(
        f"Saved trial {trial.index} in "
        f"{_round_floats_for_logging(time.time() - start_time)} seconds"
    )
