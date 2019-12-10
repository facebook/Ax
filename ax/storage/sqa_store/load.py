#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from ax.core.experiment import Experiment
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.storage.sqa_store.db import session_scope
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.sqa_classes import SQAExperiment, SQAGenerationStrategy
from ax.storage.sqa_store.sqa_config import SQAConfig


# ---------------------------- Loading `Experiment`. ---------------------------


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
    2) Convert to corresponding Ax object.
    """
    # Convert SQA to user-facing class outside of session scope to avoid timeouts
    return decoder.experiment_from_sqa(
        experiment_sqa=_get_experiment_sqa(experiment_name=experiment_name)
    )


def _get_experiment_sqa(experiment_name: str) -> SQAExperiment:
    """Obtains SQLAlchemy experiment object from DB."""
    with session_scope() as session:
        sqa_experiment = (
            session.query(SQAExperiment).filter_by(name=experiment_name).one_or_none()
        )
        if sqa_experiment is None:
            raise ValueError(f"Experiment `{experiment_name}` not found.")
    return sqa_experiment


# ------------------------ Loading `GenerationStrategy`. -----------------------


def load_generation_strategy_by_experiment_name(
    experiment_name: str, config: Optional[SQAConfig] = None
) -> GenerationStrategy:
    """Finds a generation strategy attached to an experiment specified by a name
    and restores it from its corresponding SQA object.
    """
    config = config or SQAConfig()
    decoder = Decoder(config=config)
    return _load_generation_strategy_by_experiment_name(
        experiment_name=experiment_name, decoder=decoder
    )


def load_generation_strategy_by_id(
    gs_id: int, config: Optional[SQAConfig] = None
) -> GenerationStrategy:
    """Finds a generation strategy stored by a given ID and restores it."""
    config = config or SQAConfig()
    decoder = Decoder(config=config)
    return _load_generation_strategy_by_id(gs_id=gs_id, decoder=decoder)


def _load_generation_strategy_by_id(gs_id: int, decoder: Decoder) -> GenerationStrategy:
    """Finds a generation strategy stored by a given ID and restores it."""
    with session_scope() as session:
        gs_sqa = session.query(SQAGenerationStrategy).filter_by(id=gs_id).one_or_none()
        if gs_sqa is None:
            raise ValueError(f"Generation strategy with ID #{gs_id} not found.")
    return decoder.generation_strategy_from_sqa(gs_sqa=gs_sqa)


def _load_generation_strategy_by_experiment_name(
    experiment_name: str, decoder: Decoder
) -> GenerationStrategy:
    """Load a generation strategy attached to an experiment specified by a name,
    using given Decoder instance.

    1) Get SQLAlchemy object from DB.
    2) Convert to corresponding Ax object.
    """
    experiment_sqa = _get_experiment_sqa(experiment_name=experiment_name)
    if experiment_sqa.generation_strategy is None:
        raise ValueError(
            f"Experiment {experiment_name} does not have a generation strategy "
            "attached to it."
        )
    gs_sqa = experiment_sqa.generation_strategy
    # pyre-fixme[6]: Expected `SQAGenerationStrategy` for 1st param but got
    #  `Optional[SQAGenerationStrategy]`.
    return decoder.generation_strategy_from_sqa(gs_sqa=gs_sqa)
