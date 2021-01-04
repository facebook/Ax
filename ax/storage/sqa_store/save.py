#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

from ax.core.base_trial import BaseTrial
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.trial import Trial
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.storage.sqa_store.db import SQABase, optional_session_scope, session_scope
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.utils.common.base import Base
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none
from sqlalchemy.orm import Session


logger = get_logger(__name__)


def _set_db_ids(obj_to_sqa: List[Tuple[Base, SQABase]]) -> None:
    for obj, sqa_obj in obj_to_sqa:
        if sqa_obj.id is not None:  # pyre-ignore[16]
            obj.db_id = not_none(sqa_obj.id)
        elif obj.db_id is None:
            is_sq_gr = (
                isinstance(obj, GeneratorRun)
                and obj._generator_run_type == "STATUS_QUO"
            )
            # TODO: Remove this warning when storage & perf project is complete.
            if not is_sq_gr:
                logger.warning(
                    f"User-facing object {obj} does not already have a db_id, "
                    f"and the corresponding SQA object: {sqa_obj} does not either."
                )


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
        experiment,
        # pyre-fixme[6]: Expected
        #  `Optional[ax.storage.sqa_store.sqa_classes.SQAExperiment]` for 2nd param but
        #  got `Optional[ax.storage.sqa_store.db.SQABase]`.
        existing_sqa_experiment=existing_sqa_experiment,
    )
    new_sqa_experiment, obj_to_sqa = encoder.experiment_to_sqa(experiment)

    if existing_sqa_experiment is not None:
        # Update the SQA object outside of session scope to avoid timeouts.
        # This object is detached from the session, but contains a database
        # identity marker, so when we do `session.add` below, SQA knows to
        # perform an update rather than an insert.
        # pyre-fixme[6]: Expected `SQABase` for 1st param but got `SQAExperiment`.
        existing_sqa_experiment.update(new_sqa_experiment)
        new_sqa_experiment = existing_sqa_experiment

    with session_scope() as session:
        session.add(new_sqa_experiment)
        session.flush()

    _set_db_ids(obj_to_sqa=obj_to_sqa)


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

    gs_sqa, obj_to_sqa = encoder.generation_strategy_to_sqa(
        generation_strategy=generation_strategy, experiment_id=experiment_id
    )

    if generation_strategy._db_id is not None:
        gs_sqa_class = encoder.config.class_to_sqa_class[GenerationStrategy]
        with session_scope() as session:
            existing_gs_sqa = session.query(gs_sqa_class).get(
                generation_strategy._db_id
            )

        # pyre-fixme[16]: `Optional` has no attribute `update`.
        existing_gs_sqa.update(gs_sqa)
        # our update logic ignores foreign keys, i.e. fields ending in _id,
        # because we want SQLAlchemy to handle those relationships for us
        # however, generation_strategy.experiment_id is an exception, so we
        # need to update that manually
        # pyre-fixme[16]: `Optional` has no attribute `experiment_id`.
        existing_gs_sqa.experiment_id = gs_sqa.experiment_id
        gs_sqa = existing_gs_sqa

    with session_scope() as session:
        session.add(gs_sqa)
        session.flush()  # Ensures generation strategy id is set.

    _set_db_ids(obj_to_sqa=obj_to_sqa)

    return not_none(generation_strategy.db_id)


def _get_experiment_id(
    experiment: Experiment, encoder: Encoder, session: Optional[Session] = None
) -> int:
    exp_sqa_class = encoder.config.class_to_sqa_class[Experiment]
    with optional_session_scope(session=session) as _session:
        sqa_experiment_id = (
            _session.query(exp_sqa_class.id)  # pyre-ignore
            .filter_by(name=experiment.name)
            .one_or_none()
        )

    if sqa_experiment_id is None:  # pragma: no cover (this is technically unreachable)
        raise ValueError("No corresponding experiment found.")
    return sqa_experiment_id[0]


def save_new_trial(
    experiment: Experiment, trial: BaseTrial, config: Optional[SQAConfig] = None
) -> None:
    """Add new trial to the experiment (using default SQAConfig)."""
    config = config or SQAConfig()
    encoder = Encoder(config=config)
    _save_new_trial(experiment=experiment, trial=trial, encoder=encoder)


def _save_new_trial(experiment: Experiment, trial: BaseTrial, encoder: Encoder) -> None:
    """Add new trial to the experiment."""
    _save_new_trials(experiment=experiment, trials=[trial], encoder=encoder)


def _save_new_trials(
    experiment: Experiment, trials: List[BaseTrial], encoder: Encoder
) -> None:
    """Add new trials to the experiment."""
    trial_sqa_class = encoder.config.class_to_sqa_class[Trial]
    obj_to_sqa = []
    with session_scope() as session:
        experiment_id = _get_experiment_id(
            experiment=experiment, encoder=encoder, session=session
        )
        existing_trial_indices = (
            session.query(trial_sqa_class.index)  # pyre-ignore
            .filter_by(experiment_id=experiment_id)
            .all()
        )

    existing_trial_indices = {x[0] for x in existing_trial_indices}
    new_trial_idcs = set()

    trials_sqa = []
    for trial in trials:
        if trial.index in existing_trial_indices:
            raise ValueError(f"Trial {trial.index} already attached to experiment.")

        if trial.index in new_trial_idcs:
            raise ValueError(f"Trial {trial.index} appears in `trials` more than once.")

        new_sqa_trial, _obj_to_sqa = encoder.trial_to_sqa(trial)
        obj_to_sqa.extend(_obj_to_sqa)
        new_sqa_trial.experiment_id = experiment_id
        trials_sqa.append(new_sqa_trial)
        new_trial_idcs.add(trial.index)

    with session_scope() as session:
        session.add_all(trials_sqa)
        session.flush()

    _set_db_ids(obj_to_sqa=obj_to_sqa)


def update_trial(
    experiment: Experiment, trial: BaseTrial, config: Optional[SQAConfig] = None
) -> None:
    """Update trial and attach data (using default SQAConfig)."""
    config = config or SQAConfig()
    encoder = Encoder(config=config)
    _update_trial(experiment=experiment, trial=trial, encoder=encoder)


def _update_trial(experiment: Experiment, trial: BaseTrial, encoder: Encoder) -> None:
    """Update trial and attach data."""
    _update_trials(experiment=experiment, trials=[trial], encoder=encoder)


def _update_trials(
    experiment: Experiment, trials: List[BaseTrial], encoder: Encoder
) -> None:
    """Update trials and attach data."""
    trial_sqa_class = encoder.config.class_to_sqa_class[Trial]
    trial_indices = [trial.index for trial in trials]
    obj_to_sqa = []
    with session_scope() as session:
        experiment_id = _get_experiment_id(
            experiment=experiment, encoder=encoder, session=session
        )
        existing_trials = (
            session.query(trial_sqa_class)
            .filter_by(experiment_id=experiment_id)
            .filter(trial_sqa_class.index.in_(trial_indices))  # pyre-ignore
            .all()
        )

    trial_index_to_existing_trial = {trial.index: trial for trial in existing_trials}

    updated_sqa_trials, new_sqa_data = [], []
    for trial in trials:
        existing_trial = trial_index_to_existing_trial.get(trial.index)
        if existing_trial is None:
            raise ValueError(f"Trial {trial.index} is not attached to the experiment.")

        new_sqa_trial, _obj_to_sqa = encoder.trial_to_sqa(trial)
        obj_to_sqa.extend(_obj_to_sqa)
        existing_trial.update(new_sqa_trial)
        updated_sqa_trials.append(existing_trial)

        data, ts = experiment.lookup_data_for_trial(trial_index=trial.index)
        if ts != -1:
            sqa_data = encoder.data_to_sqa(
                data=data, trial_index=trial.index, timestamp=ts
            )
            obj_to_sqa.append((data, sqa_data))
            sqa_data.experiment_id = experiment_id
            new_sqa_data.append(sqa_data)

    with session_scope() as session:
        session.add_all(updated_sqa_trials)
        session.add_all(new_sqa_data)
        session.flush()

    _set_db_ids(obj_to_sqa=obj_to_sqa)


def update_generation_strategy(
    generation_strategy: GenerationStrategy,
    generator_runs: List[GeneratorRun],
    config: Optional[SQAConfig] = None,
) -> None:
    """Update generation strategy's current step and attach generator runs
    (using default SQAConfig)."""
    config = config or SQAConfig()
    encoder = Encoder(config=config)
    _update_generation_strategy(
        generation_strategy=generation_strategy,
        generator_runs=generator_runs,
        encoder=encoder,
    )


def _update_generation_strategy(
    generation_strategy: GenerationStrategy,
    generator_runs: List[GeneratorRun],
    encoder: Encoder,
) -> None:
    """Update generation strategy's current step and attach generator runs."""
    gs_sqa_class = encoder.config.class_to_sqa_class[GenerationStrategy]

    gs_id = generation_strategy.db_id
    if gs_id is None:
        raise ValueError("GenerationStrategy must be saved before being updated.")

    obj_to_sqa = []
    with session_scope() as session:
        experiment_id = _get_experiment_id(
            experiment=generation_strategy.experiment, encoder=encoder, session=session
        )
        session.query(gs_sqa_class).filter_by(id=gs_id).update(
            {
                "curr_index": generation_strategy._curr.index,
                "experiment_id": experiment_id,
            }
        )

    generator_runs_sqa = []
    for generator_run in generator_runs:
        gr_sqa, _obj_to_sqa = encoder.generator_run_to_sqa(generator_run=generator_run)
        obj_to_sqa.extend(_obj_to_sqa)
        gr_sqa.generation_strategy_id = gs_id
        generator_runs_sqa.append(gr_sqa)

    with session_scope() as session:
        session.add_all(generator_runs_sqa)

    _set_db_ids(obj_to_sqa=obj_to_sqa)
