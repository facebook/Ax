#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from ax.core.experiment import Experiment
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.storage.sqa_store.db import session_scope
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.utils.common.equality import datetime_equals


def save_experiment(
    experiment: Experiment, config: Optional[SQAConfig] = None, overwrite: bool = False
) -> None:
    """Save experiment (using default SQAConfig)."""
    if not isinstance(experiment, Experiment):
        raise ValueError("Can only save instances of Experiment")
    if not experiment.has_name:
        raise ValueError("Experiment name must be set prior to saving.")

    config = config or SQAConfig()
    encoder = Encoder(config=config)
    _save_experiment(experiment=experiment, encoder=encoder, overwrite=overwrite)


def _save_experiment(
    experiment: Experiment, encoder: Encoder, overwrite: bool = False
) -> None:
    """Save experiment, using given Encoder instance.

    1) Convert Ax object to SQLAlchemy object.
    2) Determine if there is an existing experiment with that name in the DB.
    3) If not, create a new one.
    4) If so, update the old one.
        The update works by merging the new SQLAlchemy object into the
        existing SQLAlchemy object, and then letting SQLAlchemy handle the
        actual DB updates.
    """
    # Convert user-facing class to SQA outside of session scope to avoid timeouts
    new_sqa_experiment = encoder.experiment_to_sqa(experiment)
    exp_sqa_class = encoder.config.class_to_sqa_class[Experiment]
    with session_scope() as session:
        existing_sqa_experiment = (
            session.query(exp_sqa_class).filter_by(name=experiment.name).one_or_none()
        )

    if existing_sqa_experiment is not None:
        if (
            not datetime_equals(
                existing_sqa_experiment.time_created, new_sqa_experiment.time_created
            )
            and overwrite is False
        ):
            raise Exception(
                "An experiment already exists with the name "
                f"{new_sqa_experiment.name}. To overwrite, specify "
                "`overwrite = True` when calling `save_experiment`."
            )

        # Update the SQA object outside of session scope to avoid timeouts.
        # This object is detached from the session, but contains a database
        # identity marker, so when we do `session.add` below, SQA knows to
        # perform an update rather than an insert.
        existing_sqa_experiment.update(new_sqa_experiment)
        new_sqa_experiment = existing_sqa_experiment

    with session_scope() as session:
        session.add(new_sqa_experiment)


def save_generation_strategy(
    generation_strategy: GenerationStrategy, config: Optional[SQAConfig] = None
) -> int:
    """Save generation strategy (using default SQAConfig if no config is
    specified). If the generation strategy has an experiment set, the experiment
    will be saved first.

    Returns:
        The ID of the saved generation strategy.
    """
    # Start up SQA encoder.
    config = config or SQAConfig()
    encoder = Encoder(config=config)

    return _save_generation_strategy(
        generation_strategy=generation_strategy, encoder=encoder
    )


def _save_generation_strategy(
    generation_strategy: GenerationStrategy, encoder: Encoder
) -> int:
    # If the generation strategy has not yet generated anything, there will be no
    # experiment set on it.
    if generation_strategy._experiment is None:
        experiment_id = None
    else:
        # Experiment was set on the generation strategy, so need to check whether
        # if has been saved and create a relationship b/w GS and experiment if so.
        experiment_id = _get_experiment_id(
            # pyre-fixme[6]: Expected `Experiment` for 1st param but got
            #  `Optional[Experiment]`.
            experiment=generation_strategy._experiment,
            encoder=encoder,
        )

    gs_sqa = encoder.generation_strategy_to_sqa(
        generation_strategy=generation_strategy, experiment_id=experiment_id
    )

    with session_scope() as session:
        if generation_strategy._db_id is None:
            session.add(gs_sqa)
            session.flush()  # Ensures generation strategy id is set.
            generation_strategy._db_id = gs_sqa.id
        else:
            gs_sqa_class = encoder.config.class_to_sqa_class[GenerationStrategy]
            existing_gs_sqa = session.query(gs_sqa_class).get(
                generation_strategy._db_id
            )
            existing_gs_sqa.update(gs_sqa)
            # our update logic ignores foreign keys, i.e. fields ending in _id,
            # because we want SQLAlchemy to handle those relationships for us
            # however, generation_strategy.experiment_id is an exception, so we
            # need to update that manually
            existing_gs_sqa.experiment_id = gs_sqa.experiment_id

    # pyre-fixme[7]: Expected `int` but got `Optional[int]`.
    return generation_strategy._db_id


def _get_experiment_id(experiment: Experiment, encoder: Encoder) -> int:
    exp_sqa_class = encoder.config.class_to_sqa_class[Experiment]
    with session_scope() as session:
        sqa_experiment = (
            session.query(exp_sqa_class).filter_by(name=experiment.name).one_or_none()
        )
    if sqa_experiment is None:  # pragma: no cover (this is technically unreachable)
        raise ValueError(
            "The undelying experiment must be saved before the generation strategy."
        )
    return sqa_experiment.id
