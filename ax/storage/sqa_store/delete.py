# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger

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
    exp_name: str, config: SQAConfig | None = None, max_gs_to_delete: int = 1
) -> None:
    """Delete the generation strategy associated with an experiment

    Args:
        exp_name: Name of the experiment for which the generation strategy
            should be deleted.
        config: The SQAConfig.
        max_gs_to_delete: There is never supposed to be more than one generation
            strategy associated with an experiment. However, we've seen cases where
            there are, and we don't know why. This parameter allows us to delete
            multiple generation strategies, but we raise an error if there are more
            than `max_gs_to_delete` generation strategies associated with the
            experiment.
            This is a safeguard in case we have a bug in this code that deletes
            all generation strategies.

    """
    config = config or SQAConfig()
    decoder = Decoder(config=config)
    exp_sqa_class = decoder.config.class_to_sqa_class[Experiment]
    gs_sqa_class = decoder.config.class_to_sqa_class[GenerationStrategy]
    # get the generation strategy's db_id
    with session_scope() as session:
        sqa_gs_ids = (
            session.query(gs_sqa_class.id)  # pyre-ignore[16]
            .join(exp_sqa_class.generation_strategy)  # pyre-ignore[16]
            # pyre-fixme[16]: `SQABase` has no attribute `name`.
            .filter(exp_sqa_class.name == exp_name)
            .all()
        )

    if sqa_gs_ids is None:
        logger.info(f"No generation strategy found for experiment {exp_name}.")
        return None

    if len(sqa_gs_ids) > max_gs_to_delete:
        raise ValueError(
            f"Found {len(sqa_gs_ids)} generation strategies for experiment {exp_name}. "
            "If you are sure you want to delete all of them, please set "
            f"`max_gs_to_delete` (currently {max_gs_to_delete}) to a higher value."
        )

    # delete generation strategy
    with session_scope() as session:
        gs_list = (
            session.query(gs_sqa_class)
            .filter(gs_sqa_class.id.in_([id[0] for id in sqa_gs_ids]))
            .all()
        )
        for gs in gs_list:
            session.delete(gs)
        session.flush()
