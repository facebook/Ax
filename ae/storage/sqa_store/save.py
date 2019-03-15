#!/usr/bin/env python3

from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.storage.sqa_store.db import session_scope
from ae.lazarus.ae.storage.sqa_store.encoder import Encoder


def save_experiment(experiment: Experiment) -> None:
    """Save experiment (using default Encoder)."""
    return _save_experiment(experiment=experiment, encoder=Encoder())


def _save_experiment(experiment: Experiment, encoder: Encoder) -> None:
    """Save experiment, using given Encoder instance.

    1) Convert AE object to SQLAlchemy object.
    2) Determine if there is an existing experiment with that name in the DB.
    3) If not, create a new one.
    4) If so, update the old one.
        The update works by merging the new SQLAlchemy object into the
        existing SQLAlchemy object, and then letting SQLAlchemy handle the
        actual DB updates.
    """
    with session_scope() as session:
        new_sqa_experiment = encoder.experiment_to_sqa(experiment)
        exp_sqa_class = encoder.class_to_sqa_class[Experiment]
        existing_sqa_experiment = (
            session.query(exp_sqa_class).filter_by(name=experiment.name).one_or_none()
        )
        if existing_sqa_experiment is None:
            session.add(new_sqa_experiment)
        else:
            existing_sqa_experiment.update(new_sqa_experiment)
