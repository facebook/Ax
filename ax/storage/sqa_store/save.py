#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Optional

from ax.core.experiment import Experiment
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.storage.sqa_store.db import session_scope
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.sqa_config import SQAConfig


def save_experiment(experiment: Experiment, config: Optional[SQAConfig] = None) -> None:
    """Save experiment (using default SQAConfig)."""
    if not isinstance(experiment, Experiment):
        raise ValueError("Can only save instances of Experiment")
    if not experiment.has_name:
        raise ValueError("Experiment name must be set prior to saving.")

    config = config or SQAConfig()
    encoder = Encoder(config=config)
    _save_experiment(experiment=experiment, encoder=encoder)


def _save_experiment(experiment: Experiment, encoder: Encoder) -> None:
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

    # If the generation strategy has not yet generated anything, there will be no
    # experiment set on it.
    if generation_strategy._experiment is None:
        with session_scope() as session:
            gs_sqa = encoder.generation_strategy_to_sqa(
                generation_strategy=generation_strategy, experiment_id=None
            )
            session.add(gs_sqa)
            session.flush()  # Ensures generation strategy id is set.
            return gs_sqa.id

    # Experiment was set on the generation strategy, so we need to save it first.
    save_experiment(experiment=generation_strategy._experiment, config=config)
    exp_sqa_class = encoder.config.class_to_sqa_class[Experiment]
    with session_scope() as session:
        sqa_experiment = (
            session.query(exp_sqa_class)
            .filter_by(name=generation_strategy._experiment.name)
            .one_or_none()
        )
    if sqa_experiment is None:  # pragma: no cover (this is technically unreachable)
        raise ValueError(
            "The undelying experiment must be saved before the generation strategy."
        )

    # Having saved the experiment, we can save the generation strategy with the
    # experiment ID attached to it.
    with session_scope() as session:
        gs_sqa = encoder.generation_strategy_to_sqa(
            generation_strategy=generation_strategy, experiment_id=sqa_experiment.id
        )
        session.add(gs_sqa)
        session.flush()  # Ensures generation strategy id is set.
        return gs_sqa.id
