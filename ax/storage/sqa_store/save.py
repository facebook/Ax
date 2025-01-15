#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
from collections.abc import Callable, Sequence
from datetime import datetime

from logging import Logger
from typing import Any, cast

from ax.analysis.analysis import AnalysisCard

from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.outcome_constraint import ObjectiveThreshold, OutcomeConstraint
from ax.core.runner import Runner
from ax.core.trial import Trial
from ax.exceptions.core import UserInputError
from ax.exceptions.storage import SQADecodeError
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.storage.sqa_store.db import session_scope, SQABase
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.sqa_classes import (
    SQAData,
    SQAGeneratorRun,
    SQAMetric,
    SQARunner,
    SQATrial,
)
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.storage.sqa_store.utils import copy_db_ids
from ax.utils.common.base import Base
from ax.utils.common.logger import get_logger
from pyre_extensions import assert_is_instance, none_throws

logger: Logger = get_logger(__name__)


def save_experiment(experiment: Experiment, config: SQAConfig | None = None) -> None:
    """Save experiment (using default SQAConfig)."""
    if not isinstance(experiment, Experiment):
        raise ValueError("Can only save instances of Experiment")
    if not experiment.has_name:
        raise ValueError("Experiment name must be set prior to saving.")
    config = SQAConfig() if config is None else config
    encoder = Encoder(config=config)
    decoder = Decoder(config=config)
    _save_experiment(experiment=experiment, encoder=encoder, decoder=decoder)


def _save_experiment(
    experiment: Experiment,
    encoder: Encoder,
    decoder: Decoder,
    return_sqa: bool = False,
    validation_kwargs: dict[str, Any] | None = None,
) -> SQABase | None:
    """Save experiment, using given Encoder instance.

    1) Convert Ax object to SQLAlchemy object.
    2) Determine if there is an existing experiment with that name in the DB.
    3) If not, create a new one.
    4) If so, update the old one.
        The update works by merging the new SQLAlchemy object into the
        existing SQLAlchemy object, and then letting SQLAlchemy handle the
        actual DB updates.
    """
    exp_sqa_class = encoder.config.class_to_sqa_class[Experiment]
    with session_scope() as session:
        existing_sqa_experiment_id = (
            # pyre-ignore Undefined attribute [16]: `SQABase` has no attribute `id`
            session.query(exp_sqa_class.id)
            .filter_by(name=experiment.name)
            .one_or_none()
        )
    if existing_sqa_experiment_id:
        existing_sqa_experiment_id = existing_sqa_experiment_id[0]

    encoder.validate_experiment_metadata(
        experiment,
        existing_sqa_experiment_id=existing_sqa_experiment_id,
        **(validation_kwargs or {}),
    )

    experiment_sqa = _merge_into_session(
        obj=experiment,
        encode_func=encoder.experiment_to_sqa,
        decode_func=decoder.experiment_from_sqa,
    )

    return assert_is_instance(experiment_sqa, SQABase) if return_sqa else None


def save_generation_strategy(
    generation_strategy: GenerationStrategy, config: SQAConfig | None = None
) -> int:
    """Save generation strategy (using default SQAConfig if no config is
    specified). If the generation strategy has an experiment set, the experiment
    will be saved first.

    Returns:
        The ID of the saved generation strategy.
    """
    # Start up SQA encoder.
    config = SQAConfig() if config is None else config
    encoder = Encoder(config=config)
    decoder = Decoder(config=config)

    return _save_generation_strategy(
        generation_strategy=generation_strategy, encoder=encoder, decoder=decoder
    )


def _save_generation_strategy(
    generation_strategy: GenerationStrategy, encoder: Encoder, decoder: Decoder
) -> int:
    # If the generation strategy has not yet generated anything, there will be no
    # experiment set on it.
    experiment = generation_strategy._experiment
    if experiment is None:
        experiment_id = None
    else:
        # Experiment was set on the generation strategy, so need to check whether
        # if has been saved and create a relationship b/w GS and experiment if so.
        experiment_id = experiment.db_id
        if experiment_id is None:
            raise ValueError(
                f"Experiment {experiment.name} should be saved before "
                "generation strategy."
            )

    _merge_into_session(
        obj=generation_strategy,
        encode_func=encoder.generation_strategy_to_sqa,
        decode_func=decoder.generation_strategy_from_sqa,
        encode_args={"experiment_id": experiment_id},
        decode_args={"experiment": experiment},
    )

    return none_throws(generation_strategy.db_id)


def save_or_update_trial(
    experiment: Experiment, trial: BaseTrial, config: SQAConfig | None = None
) -> None:
    """Add new trial to the experiment, or update if already exists
    (using default SQAConfig)."""
    config = SQAConfig() if config is None else config
    encoder = Encoder(config=config)
    decoder = Decoder(config=config)
    _save_or_update_trial(
        experiment=experiment, trial=trial, encoder=encoder, decoder=decoder
    )


def _save_or_update_trial(
    experiment: Experiment,
    trial: BaseTrial,
    encoder: Encoder,
    decoder: Decoder,
    reduce_state_generator_runs: bool = False,
) -> None:
    """Add new trial to the experiment, or update if already exists."""
    _save_or_update_trials(
        experiment=experiment,
        trials=[trial],
        encoder=encoder,
        decoder=decoder,
        reduce_state_generator_runs=reduce_state_generator_runs,
    )


def save_or_update_trials(
    experiment: Experiment,
    trials: list[BaseTrial],
    config: SQAConfig | None = None,
    batch_size: int | None = None,
    reduce_state_generator_runs: bool = False,
) -> None:
    """Add new trials to the experiment, or update if already exists
    (using default SQAConfig).

    Note that new data objects (whether attached to existing or new trials)
    will also be added to the experiment, but existing data objects in the
    database will *not* be updated or removed.
    """
    config = SQAConfig() if config is None else config
    encoder = Encoder(config=config)
    decoder = Decoder(config=config)
    _save_or_update_trials(
        experiment=experiment,
        trials=trials,
        encoder=encoder,
        decoder=decoder,
        batch_size=batch_size,
        reduce_state_generator_runs=reduce_state_generator_runs,
    )


def _save_or_update_trials(
    experiment: Experiment,
    trials: list[BaseTrial],
    encoder: Encoder,
    decoder: Decoder,
    batch_size: int | None = None,
    reduce_state_generator_runs: bool = False,
) -> None:
    """Add new trials to the experiment, or update if they already exist.

    Note that new data objects (whether attached to existing or new trials)
    will also be added to the experiment, but existing data objects in the
    database will *not* be updated or removed.
    """
    if experiment._db_id is None:
        raise ValueError("Must save experiment before saving/updating its trials.")

    experiment_id: int = experiment._db_id

    def add_experiment_id(sqa: SQATrial) -> None:
        sqa.experiment_id = experiment_id

    if reduce_state_generator_runs:
        latest_trial = trials[-1]
        trials_to_reduce_state = trials[0:-1]

        def trial_to_reduced_state_sqa_encoder(t: BaseTrial) -> SQATrial:
            return encoder.trial_to_sqa(t, generator_run_reduced_state=True)

        _bulk_merge_into_session(
            objs=trials_to_reduce_state,
            encode_func=trial_to_reduced_state_sqa_encoder,
            decode_func=decoder.trial_from_sqa,
            decode_args_list=[{"experiment": experiment} for _ in range(len(trials))],
            modify_sqa=add_experiment_id,
            batch_size=batch_size,
        )

        _bulk_merge_into_session(
            objs=[latest_trial],
            encode_func=encoder.trial_to_sqa,
            decode_func=decoder.trial_from_sqa,
            decode_args_list=[{"experiment": experiment} for _ in range(len(trials))],
            modify_sqa=add_experiment_id,
            batch_size=batch_size,
        )
    else:
        _bulk_merge_into_session(
            objs=trials,
            encode_func=encoder.trial_to_sqa,
            decode_func=decoder.trial_from_sqa,
            decode_args_list=[{"experiment": experiment} for _ in range(len(trials))],
            modify_sqa=add_experiment_id,
            batch_size=batch_size,
        )

    save_or_update_data_for_trials(
        experiment=experiment,
        trials=trials,
        encoder=encoder,
        decoder=decoder,
        batch_size=batch_size,
    )


def save_or_update_data_for_trials(
    experiment: Experiment,
    trials: list[BaseTrial],
    encoder: Encoder,
    decoder: Decoder,
    batch_size: int | None = None,
    update_trial_statuses: bool = False,
) -> None:
    if experiment.db_id is None:
        raise ValueError("Must save experiment before saving/updating its data.")

    def add_experiment_id(sqa: SQAData) -> None:
        sqa.experiment_id = experiment.db_id

    datas, data_encode_args, datas_to_keep, trial_idcs = [], [], [], []
    data_sqa_class: type[SQAData] = cast(
        type[SQAData], encoder.config.class_to_sqa_class[Data]
    )
    for trial in trials:
        trial_idcs.append(trial.index)
        trial_datas = experiment.data_by_trial.get(trial.index, {})
        for ts, data in trial_datas.items():
            if data.db_id is None:
                # This is data we have not saved before; we should add it to the
                # database. Previously saved data for this experiment can be removed.
                datas.append(data)
                data_encode_args.append({"trial_index": trial.index, "timestamp": ts})
            else:
                datas_to_keep.append(data.db_id)

        # For trials, for which we saved new data, we can first remove previously
        # saved data if it's no longer on the experiment.
        with session_scope() as session:
            session.query(data_sqa_class).filter_by(
                experiment_id=experiment.db_id
            ).filter(data_sqa_class.trial_index.isnot(None)).filter(
                data_sqa_class.trial_index.in_(trial_idcs)
            ).filter(data_sqa_class.id not in datas_to_keep).delete()

    _bulk_merge_into_session(
        objs=datas,
        encode_func=encoder.data_to_sqa,
        decode_func=decoder.data_from_sqa,
        encode_args_list=data_encode_args,
        decode_args_list=[
            {"data_constructor": experiment.default_data_constructor}
            for _ in range(len(datas))
        ],
        modify_sqa=add_experiment_id,
        batch_size=batch_size,
    )

    if update_trial_statuses:
        for trial in trials:
            update_trial_status(trial_with_updated_status=trial, config=encoder.config)


def update_generation_strategy(
    generation_strategy: GenerationStrategy,
    generator_runs: list[GeneratorRun],
    config: SQAConfig | None = None,
    batch_size: int | None = None,
    reduce_state_generator_runs: bool = False,
) -> None:
    """Update generation strategy's current step and attach generator runs
    (using default SQAConfig)."""
    config = SQAConfig() if config is None else config
    encoder = Encoder(config=config)
    decoder = Decoder(config=config)
    _update_generation_strategy(
        generation_strategy=generation_strategy,
        generator_runs=generator_runs,
        encoder=encoder,
        decoder=decoder,
        batch_size=batch_size,
        reduce_state_generator_runs=reduce_state_generator_runs,
    )


def _update_generation_strategy(
    generation_strategy: GenerationStrategy,
    generator_runs: list[GeneratorRun],
    encoder: Encoder,
    decoder: Decoder,
    batch_size: int | None = None,
    reduce_state_generator_runs: bool = False,
) -> None:
    """Update generation strategy's current step and attach generator runs."""
    gs_sqa_class = encoder.config.class_to_sqa_class[GenerationStrategy]

    gs_id = generation_strategy.db_id
    if gs_id is None:
        raise ValueError("GenerationStrategy must be saved before being updated.")

    experiment_id = generation_strategy.experiment.db_id
    if experiment_id is None:
        raise ValueError(
            f"Experiment {generation_strategy.experiment.name} "
            "should be saved before generation strategy."
        )

    curr_index = (
        None
        if generation_strategy.is_node_based
        else generation_strategy.current_step_index
    )
    # there is always a node name
    curr_node_name = generation_strategy.current_node_name
    with session_scope() as session:
        session.query(gs_sqa_class).filter_by(id=gs_id).update(
            {
                "curr_index": curr_index,
                "experiment_id": experiment_id,
                "curr_node_name": curr_node_name,
            }
        )

    # pyre-fixme[53]: Captured variable `gs_id` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    def add_generation_strategy_id(sqa: SQAGeneratorRun):
        sqa.generation_strategy_id = gs_id

    # pyre-fixme[3]: Return type must be annotated.
    def generator_run_to_sqa_encoder(gr: GeneratorRun, weight: float | None = None):
        return encoder.generator_run_to_sqa(
            gr,
            weight=weight,
            reduced_state=reduce_state_generator_runs,
        )

    _bulk_merge_into_session(
        objs=generator_runs,
        encode_func=generator_run_to_sqa_encoder,
        decode_func=decoder.generator_run_from_sqa,
        decode_args_list=[
            {
                "reduced_state": False,
                "immutable_search_space_and_opt_config": False,
            }
            for _ in range(len(generator_runs))
        ],
        modify_sqa=add_generation_strategy_id,
        batch_size=batch_size,
    )


def update_runner_on_experiment(
    experiment: Experiment, runner: Runner, encoder: Encoder, decoder: Decoder
) -> None:
    runner_sqa_class = encoder.config.class_to_sqa_class[Runner]

    exp_id = experiment.db_id
    if exp_id is None:
        raise ValueError("Experiment must be saved before being updated.")

    with session_scope() as session:
        session.query(runner_sqa_class).filter_by(experiment_id=exp_id).delete()

    # pyre-fixme[53]: Captured variable `exp_id` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    def add_experiment_id(sqa: SQARunner):
        sqa.experiment_id = exp_id

    _merge_into_session(
        obj=runner,
        encode_func=encoder.runner_to_sqa,
        decode_func=decoder.runner_from_sqa,
        modify_sqa=add_experiment_id,
    )


def update_outcome_constraint_on_experiment(
    experiment: Experiment,
    outcome_constraint: OutcomeConstraint,
    encoder: Encoder,
    decoder: Decoder,
) -> None:
    oc_sqa_class = encoder.config.class_to_sqa_class[Metric]

    exp_id: int | None = experiment.db_id
    if exp_id is None:
        raise UserInputError("Experiment must be saved before being updated.")
    oc_id = outcome_constraint.db_id
    if oc_id is not None:
        with session_scope() as session:
            session.query(oc_sqa_class).filter_by(experiment_id=exp_id).filter_by(
                id=oc_id
            ).delete()

    def add_experiment_id(sqa: SQAMetric) -> None:
        sqa.experiment_id = exp_id

    encode_func = (
        encoder.objective_threshold_to_sqa
        if isinstance(outcome_constraint, ObjectiveThreshold)
        else encoder.outcome_constraint_to_sqa
    )
    _merge_into_session(
        obj=outcome_constraint,
        encode_func=encode_func,
        decode_func=decoder.metric_from_sqa,
        modify_sqa=add_experiment_id,
    )


def update_properties_on_experiment(
    experiment_with_updated_properties: Experiment,
    config: SQAConfig | None = None,
) -> None:
    config = SQAConfig() if config is None else config
    exp_sqa_class = config.class_to_sqa_class[Experiment]

    exp_id = experiment_with_updated_properties.db_id
    if exp_id is None:
        raise ValueError("Experiment must be saved before being updated.")

    with session_scope() as session:
        session.query(exp_sqa_class).filter_by(id=exp_id).update(
            {
                "properties": experiment_with_updated_properties._properties,
            }
        )


def update_properties_on_trial(
    trial_with_updated_properties: BaseTrial,
    config: SQAConfig | None = None,
) -> None:
    config = SQAConfig() if config is None else config
    trial_sqa_class = config.class_to_sqa_class[Trial]

    trial_id = trial_with_updated_properties.db_id
    if trial_id is None:
        raise ValueError("Trial must be saved before being updated.")

    with session_scope() as session:
        session.query(trial_sqa_class).filter_by(id=trial_id).update(
            {
                "properties": trial_with_updated_properties._properties,
            }
        )


def update_trial_status(
    trial_with_updated_status: BaseTrial,
    config: SQAConfig | None = None,
) -> None:
    config = SQAConfig() if config is None else config
    trial_sqa_class = config.class_to_sqa_class[Trial]

    trial_id = trial_with_updated_status.db_id
    if trial_id is None:
        raise ValueError("Trial must be saved before being updated.")

    with session_scope() as session:
        session.query(trial_sqa_class).filter_by(id=trial_id).update(
            {
                "status": trial_with_updated_status.status,
            }
        )


def save_analysis_cards(
    analysis_cards: list[AnalysisCard],
    experiment: Experiment,
    config: SQAConfig | None = None,
) -> None:
    # Start up SQA encoder.
    config = SQAConfig() if config is None else config
    encoder = Encoder(config=config)
    decoder = Decoder(config=config)
    timestamp = datetime.utcnow()
    _save_analysis_cards(
        analysis_cards=analysis_cards,
        experiment=experiment,
        timestamp=timestamp,
        encoder=encoder,
        decoder=decoder,
    )


def _save_analysis_cards(
    analysis_cards: list[AnalysisCard],
    experiment: Experiment,
    timestamp: datetime,
    encoder: Encoder,
    decoder: Decoder,
) -> None:
    if any(analysis_card.db_id is not None for analysis_card in analysis_cards):
        raise ValueError("Analysis cards cannot be updated.")
    if experiment.db_id is None:
        raise ValueError(
            f"Experiment {experiment.name} should be saved before analysis cards."
        )
    _bulk_merge_into_session(
        objs=analysis_cards,
        encode_func=encoder.analysis_card_to_sqa,
        decode_func=decoder.analysis_card_from_sqa,
        encode_args_list=[
            {
                "experiment_id": experiment.db_id,
                "timestamp": timestamp,
            }
            for _analysis_card in analysis_cards
        ],
    )


def _merge_into_session(
    obj: Base,
    encode_func: Callable,
    decode_func: Callable,
    encode_args: dict[str, Any] | None = None,
    decode_args: dict[str, Any] | None = None,
    modify_sqa: Callable | None = None,
) -> SQABase:
    """Given a user-facing object (that may or may not correspond to an
    existing DB object), perform the following steps to either create or
    update the necessary DB objects, and ensure the user-facing object
    is annotated with the appropriate db_ids:

    1.  Encode the user-facing object `obj` to a sqa object `sqa`
    2.  If the `modify_sqa` argument is passed in, apply this to `sqa`
        before continuing
    3.  Merge `sqa` into the session
        Note: if `sqa` and its children contain ids, they will be merged into
        those corresponding DB objects. If not, new DB objects will be created.
    4. `session.merge` returns `new_sqa`, which is the same as `sqa` but
        but annotated ids.
    5. Decode `new_sqa` into a new user-facing object `new_obj`
    6. Copy db_ids from `new_obj` to the originally passed-in `obj`
    """
    sqa = encode_func(obj, **(encode_args or {}))

    if modify_sqa is not None:
        modify_sqa(sqa=sqa)

    with session_scope() as session:
        new_sqa = session.merge(sqa)
        session.flush()

    new_obj = decode_func(new_sqa, **(decode_args or {}))
    _copy_db_ids_if_possible(obj=obj, new_obj=new_obj)

    return new_sqa


def _bulk_merge_into_session(
    objs: Sequence[Base],
    encode_func: Callable,
    decode_func: Callable,
    encode_args_list: list[None] | list[dict[str, Any]] | None = None,
    decode_args_list: list[None] | list[dict[str, Any]] | None = None,
    modify_sqa: Callable | None = None,
    batch_size: int | None = None,
) -> list[SQABase]:
    """Bulk version of _merge_into_session.

    Takes in a list of objects to merge into the session together
    (i.e. within one session scope), along with corresponding (but optional)
    lists of encode and decode arguments.

    If batch_size is specified, the list of objects will be chunked
    accordingly, and multiple session scopes will be used to merge
    the objects in, one batch at a time.
    """
    if len(objs) == 0:
        return []

    encode_args_list = encode_args_list or [None for _ in range(len(objs))]
    decode_args_list = decode_args_list or [None for _ in range(len(objs))]

    sqas = []
    for obj, encode_args in zip(objs, encode_args_list):
        sqa = encode_func(obj, **(encode_args or {}))
        if modify_sqa is not None:
            modify_sqa(sqa=sqa)
        sqas.append(sqa)

    # https://stackoverflow.com/a/312464
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def split_into_batches(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    new_sqas = []
    batch_size = batch_size or len(sqas)
    for batch in split_into_batches(lst=sqas, n=batch_size):
        with session_scope() as session:
            for sqa in batch:
                new_sqa = session.merge(sqa)
                new_sqas.append(new_sqa)
            session.flush()

    new_objs = []
    for new_sqa, decode_args in zip(new_sqas, decode_args_list):
        new_obj = decode_func(new_sqa, **(decode_args or {}))
        new_objs.append(new_obj)

    for obj, new_obj in zip(objs, new_objs):
        _copy_db_ids_if_possible(obj=obj, new_obj=new_obj)

    return new_sqas


# pyre-fixme[2]: Parameter annotation cannot be `Any`.
def _copy_db_ids_if_possible(new_obj: Any, obj: Any) -> None:
    """Wraps _copy_db_ids in a try/except, and logs warnings on error."""
    try:
        copy_db_ids(new_obj, obj, [])
    except SQADecodeError as e:
        # Raise these warnings in unittests only
        if os.environ.get("TESTENV"):
            raise e
        logger.warning(
            f"Error encountered when copying db_ids from {new_obj} "
            f"back to user-facing object {obj}. "
            "This might cause issues if you re-save this experiment. "
            f"Exception: {e}"
        )
