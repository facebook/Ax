#!/usr/bin/env python3

from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.storage.sqa_store.base_decoder import Decoder
from ae.lazarus.ae.storage.sqa_store.db import get_session
from ae.lazarus.ae.storage.sqa_store.sqa_classes import SQAExperiment


def load_experiment(experiment_name: str) -> Experiment:
    """Load experiment by name (uses default Decoder)."""
    return _load_experiment(experiment_name=experiment_name, decoder=Decoder())


def _load_experiment(experiment_name: str, decoder: Decoder) -> Experiment:
    """Load experiment by name, using given Decoder instance.

    1) Get SQLAlchemy object from DB.
    2) Convert to corresponding AE object.
    """
    sqa_experiment = (
        get_session()  # noqa P203
        .query(SQAExperiment)
        .filter_by(name=experiment_name)
        .one_or_none()
    )
    if sqa_experiment is None:
        raise ValueError(f"Experiment `{experiment_name}` not found.")
    return decoder.experiment_from_sqa(sqa_experiment)
