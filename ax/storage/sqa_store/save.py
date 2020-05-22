#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from ax.core.base_trial import BaseTrial
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
    exp_sqa_class = encoder.config.class_to_sqa_class[Experiment]
    with session_scope() as session:
        existing_sqa_experiment = (
            session.query(exp_sqa_class).filter_by(name=experiment.name).one_or_none()
        )
    encoder.validate_experiment_metadata(
        experiment, existing_sqa_experiment=existing_sqa_experiment
    )
    new_sqa_experiment = encoder.experiment_to_sqa(experiment)

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

    if generation_strategy._db_id is not None:
        gs_sqa_class = encoder.config.class_to_sqa_class[GenerationStrategy]
        with session_scope() as session:
            existing_gs_sqa = session.query(gs_sqa_class).get(
                generation_strategy._db_id
            )

        existing_gs_sqa.update(gs_sqa)
        # our update logic ignores foreign keys, i.e. fields ending in _id,
        # because we want SQLAlchemy to handle those relationships for us
        # however, generation_strategy.experiment_id is an exception, so we
        # need to update that manually
        existing_gs_sqa.experiment_id = gs_sqa.experiment_id
        gs_sqa = existing_gs_sqa

    with session_scope() as session:
        session.add(gs_sqa)
        session.flush()  # Ensures generation strategy id is set.

    generation_strategy._db_id = gs_sqa.id
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


def save_new_trial(
    experiment: Experiment, trial: BaseTrial, config: Optional[SQAConfig] = None
) -> None:
    """Add new trial to the experiment (using default SQAConfig)."""
    config = config or SQAConfig()
    encoder = Encoder(config=config)
    _save_new_trial(experiment=experiment, trial=trial, encoder=encoder)


def _save_new_trial(experiment: Experiment, trial: BaseTrial, encoder: Encoder) -> None:
    """Add new trial to the experiment, using given Encoder instance."""
    exp_sqa_class = encoder.config.class_to_sqa_class[Experiment]
    with session_scope() as session:
        existing_sqa_experiment = (
            session.query(exp_sqa_class).filter_by(name=experiment.name).one_or_none()
        )

    if existing_sqa_experiment is None:
        raise ValueError("Must save experiment before adding a new trial.")

    existing_trial_indices = {trial.index for trial in existing_sqa_experiment.trials}
    if trial.index in existing_trial_indices:
        raise ValueError(f"Trial {trial.index} already attached to experiment.")

    new_sqa_trial = encoder.trial_to_sqa(trial)

    with session_scope() as session:
        existing_sqa_experiment.trials.append(new_sqa_trial)
        session.add(existing_sqa_experiment)


def update_trial(
    experiment: Experiment, trial: BaseTrial, config: Optional[SQAConfig] = None
) -> None:
    """Update trial and attach data (using default SQAConfig)."""
    config = config or SQAConfig()
    encoder = Encoder(config=config)
    _update_trial(experiment=experiment, trial=trial, encoder=encoder)


def _update_trial(experiment: Experiment, trial: BaseTrial, encoder: Encoder) -> None:
    """Update trial and attach data, using given Encoder instance."""
    exp_sqa_class = encoder.config.class_to_sqa_class[Experiment]
    with session_scope() as session:
        existing_sqa_experiment = (
            session.query(exp_sqa_class).filter_by(name=experiment.name).one_or_none()
        )

    if existing_sqa_experiment is None:
        raise ValueError("Must save experiment before updating a trial.")

    existing_sqa_trial = None
    for sqa_trial in existing_sqa_experiment.trials:
        if sqa_trial.index == trial.index:
            # There should only be one existing trial with the same index
            assert existing_sqa_trial is None
            existing_sqa_trial = sqa_trial

    if existing_sqa_trial is None:
        raise ValueError(f"Trial {trial.index} is not attached to the experiment.")

    new_sqa_trial = encoder.trial_to_sqa(trial)
    existing_sqa_trial.update(new_sqa_trial)

    with session_scope() as session:
        session.add(existing_sqa_trial)

    data, ts = experiment.lookup_data_for_trial(trial_index=trial.index)
    if ts != -1:
        sqa_data = encoder.data_to_sqa(data=data, trial_index=trial.index, timestamp=ts)
        with session_scope() as session:
            existing_sqa_experiment.data.append(sqa_data)
            session.add(existing_sqa_experiment)
