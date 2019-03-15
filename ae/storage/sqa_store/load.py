#!/usr/bin/env python3

from typing import Optional

from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.storage.sqa_store.db import session_scope
from ae.lazarus.ae.storage.sqa_store.decoder import Decoder
from ae.lazarus.ae.storage.sqa_store.sqa_classes import SQAExperiment
from ae.lazarus.ae.storage.sqa_store.sqa_config import SQAConfig


def load_experiment(
    experiment_name: str, config: Optional[SQAConfig] = None
) -> Experiment:
    """Load experiment by name (uses default SQAConfig)."""
    config = config or SQAConfig()
    decoder = Decoder(config=config)
    return _load_experiment(experiment_name=experiment_name, decoder=decoder)


def _load_experiment(experiment_name: str, decoder: Decoder) -> Experiment:
    """Load experiment by name, using given Decoder instance.

    1) Get SQLAlchemy object from DB.
    2) Convert to corresponding AE object.
    """
    with session_scope() as session:
        sqa_experiment = (
            session.query(SQAExperiment).filter_by(name=experiment_name).one_or_none()
        )
        if sqa_experiment is None:
            raise ValueError(f"Experiment `{experiment_name}` not found.")
        return decoder.experiment_from_sqa(sqa_experiment)
