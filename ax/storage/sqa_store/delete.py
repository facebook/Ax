# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger
from typing import Optional

from ax.core.experiment import Experiment
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.storage.sqa_store.db import session_scope
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.sqa_classes import SQAExperiment
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.utils.common.logger import get_logger

logger: Logger = get_logger(__name__)


def delete_experiment(exp_name: str) -> None:
    """Delete experiment by name.

    Args:
        experiment_name: Name of the experiment to delete.
    """
    with session_scope() as session:
        exp = session.query(SQAExperiment).filter_by(name=exp_name).one_or_none()
        session.delete(exp)
        session.flush()

    logger.info(
        f"You are deleting {exp_name} and all its associated data from the database."
    )


def delete_generation_strategy(
    exp_name: str, config: Optional[SQAConfig] = None
) -> None:
    """Delete the generation strategy associated with an experiment

    Args:
        exp_name: Name of the experiment for which the generation strategy
            should be deleted.
        config: The SQAConfig.
    """
    config = config or SQAConfig()
    decoder = Decoder(config=config)
    exp_sqa_class = decoder.config.class_to_sqa_class[Experiment]
    gs_sqa_class = decoder.config.class_to_sqa_class[GenerationStrategy]
    # get the generation strategy's db_id
    with session_scope() as session:
        sqa_gs_id = (
            session.query(gs_sqa_class.id)  # pyre-ignore[16]
            .join(exp_sqa_class.generation_strategy)  # pyre-ignore[16]
            # pyre-fixme[16]: `SQABase` has no attribute `name`.
            .filter(exp_sqa_class.name == exp_name)
            .one_or_none()
        )

    if sqa_gs_id is None:
        return None

    gs_id = sqa_gs_id[0]
    # delete generation strategy
    with session_scope() as session:
        gs = session.query(gs_sqa_class).filter_by(id=gs_id).one_or_none()
        session.delete(gs)
        session.flush()
