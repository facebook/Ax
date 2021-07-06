#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from ax.core.base_trial import BaseTrial
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.runner import Runner
from ax.exceptions.storage import SQADecodeError
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.storage.sqa_store.db import SQABase, session_scope
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.sqa_classes import (
    SQATrial,
    SQAData,
    SQAGeneratorRun,
    SQARunner,
)
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.storage.sqa_store.utils import copy_db_ids
from ax.utils.common.base import Base
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none


logger = get_logger(__name__)


def save_experiment(experiment: Experiment, config: Optional[SQAConfig] = None) -> None:
    """Save experiment (using default SQAConfig)."""
    if not isinstance(experiment, Experiment):
        raise ValueError("Can only save instances of Experiment")
    if not experiment.has_name:
        raise ValueError("Experiment name must be set prior to saving.")

    config = config or SQAConfig()
    encoder = Encoder(config=config)
    decoder = Decoder(config=config)
    _save_experiment(experiment=experiment, encoder=encoder, decoder=decoder)


def _save_experiment(
    experiment: Experiment,
    encoder: Encoder,
    decoder: Decoder,
    return_sqa: bool = False,
    validation_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[SQABase]:
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

    return checked_cast(SQABase, experiment_sqa) if return_sqa else None


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
            raise ValueError(  # pragma: no cover
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

    return not_none(generation_strategy.db_id)


def save_or_update_trial(
    experiment: Experiment, trial: BaseTrial, config: Optional[SQAConfig] = None
) -> None:
    """Add new trial to the experiment, or update if already exists
    (using default SQAConfig)."""
    config = config or SQAConfig()
    encoder = Encoder(config=config)
    decoder = Decoder(config=config)
    _save_or_update_trial(
        experiment=experiment, trial=trial, encoder=encoder, decoder=decoder
    )


def _save_or_update_trial(
    experiment: Experiment, trial: BaseTrial, encoder: Encoder, decoder: Decoder
) -> None:
    """Add new trial to the experiment, or update if already exists."""
    _save_or_update_trials(
        experiment=experiment, trials=[trial], encoder=encoder, decoder=decoder
    )


def save_or_update_trials(
    experiment: Experiment,
    trials: List[BaseTrial],
    config: Optional[SQAConfig] = None,
    batch_size: Optional[int] = None,
) -> None:
    """Add new trials to the experiment, or update if already exists
    (using default SQAConfig).

    Note that new data objects (whether attached to existing or new trials)
    will also be added to the experiment, but existing data objects in the
    database will *not* be updated or removed.
    """
    config = config or SQAConfig()
    encoder = Encoder(config=config)
    decoder = Decoder(config=config)
    _save_or_update_trials(
        experiment=experiment,
        trials=trials,
        encoder=encoder,
        decoder=decoder,
        batch_size=batch_size,
    )


def _save_or_update_trials(
    experiment: Experiment,
    trials: List[BaseTrial],
    encoder: Encoder,
    decoder: Decoder,
    batch_size: Optional[int] = None,
) -> None:
    """Add new trials to the experiment, or update if they already exist.

    Note that new data objects (whether attached to existing or new trials)
    will also be added to the experiment, but existing data objects in the
    database will *not* be updated or removed.
    """
    experiment_id = experiment._db_id
    if experiment_id is None:
        raise ValueError("Must save experiment first.")

    def add_experiment_id(sqa: Union[SQATrial, SQAData]):
        sqa.experiment_id = experiment_id

    _bulk_merge_into_session(
        objs=trials,
        encode_func=encoder.trial_to_sqa,
        decode_func=decoder.trial_from_sqa,
        decode_args_list=[{"experiment": experiment} for _ in range(len(trials))],
        modify_sqa=add_experiment_id,
        batch_size=batch_size,
    )

    datas = []
    data_encode_args = []
    for trial in trials:
        trial_datas = experiment.data_by_trial.get(trial.index, {})
        for ts, data in trial_datas.items():
            if data.db_id is None:
                # Only need to worry about new data, since it's not really possible
                # or supported to modify or remove existing data.
                datas.append(data)
                data_encode_args.append({"trial_index": trial.index, "timestamp": ts})

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


def update_generation_strategy(
    generation_strategy: GenerationStrategy,
    generator_runs: List[GeneratorRun],
    config: Optional[SQAConfig] = None,
    batch_size: Optional[int] = None,
) -> None:
    """Update generation strategy's current step and attach generator runs
    (using default SQAConfig)."""
    config = config or SQAConfig()
    encoder = Encoder(config=config)
    decoder = Decoder(config=config)
    _update_generation_strategy(
        generation_strategy=generation_strategy,
        generator_runs=generator_runs,
        encoder=encoder,
        decoder=decoder,
        batch_size=batch_size,
    )


def _update_generation_strategy(
    generation_strategy: GenerationStrategy,
    generator_runs: List[GeneratorRun],
    encoder: Encoder,
    decoder: Decoder,
    batch_size: Optional[int] = None,
) -> None:
    """Update generation strategy's current step and attach generator runs."""
    gs_sqa_class = encoder.config.class_to_sqa_class[GenerationStrategy]

    gs_id = generation_strategy.db_id
    if gs_id is None:
        raise ValueError("GenerationStrategy must be saved before being updated.")

    experiment_id = generation_strategy.experiment.db_id
    if experiment_id is None:
        raise ValueError(  # pragma: no cover
            f"Experiment {generation_strategy.experiment.name} "
            "should be saved before generation strategy."
        )

    with session_scope() as session:
        session.query(gs_sqa_class).filter_by(id=gs_id).update(
            {
                "curr_index": generation_strategy._curr.index,
                "experiment_id": experiment_id,
            }
        )

    def add_generation_strategy_id(sqa: SQAGeneratorRun):
        sqa.generation_strategy_id = gs_id

    _bulk_merge_into_session(
        objs=generator_runs,
        encode_func=encoder.generator_run_to_sqa,
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

    def add_experiment_id(sqa: SQARunner):
        sqa.experiment_id = exp_id

    _merge_into_session(
        obj=runner,
        encode_func=encoder.runner_to_sqa,
        decode_func=decoder.runner_from_sqa,
        modify_sqa=add_experiment_id,
    )


def update_properties_on_experiment(
    experiment_with_updated_properties: Experiment,
    config: Optional[SQAConfig] = None,
) -> None:
    config = config or SQAConfig()
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


def _merge_into_session(
    obj: Base,
    encode_func: Callable,
    decode_func: Callable,
    encode_args: Optional[Dict[str, Any]] = None,
    decode_args: Optional[Dict[str, Any]] = None,
    modify_sqa: Optional[Callable] = None,
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
    encode_args_list: Optional[Union[List[None], List[Dict[str, Any]]]] = None,
    decode_args_list: Optional[Union[List[None], List[Dict[str, Any]]]] = None,
    modify_sqa: Optional[Callable] = None,
    batch_size: Optional[int] = None,
) -> List[SQABase]:
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
