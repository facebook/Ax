#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Type, cast

from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.storage.sqa_store.db import session_scope
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.sqa_classes import (
    SQAExperiment,
    SQAGenerationStrategy,
    SQAGeneratorRun,
)
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.utils.common.constants import Keys
from ax.utils.common.typeutils import not_none
from sqlalchemy.orm import defaultload, lazyload


# ---------------------------- Loading `Experiment`. ---------------------------


def load_experiment(
    experiment_name: str,
    config: Optional[SQAConfig] = None,
    reduced_state: bool = False,
) -> Experiment:
    """Load experiment by name.

    Args:
        experiment_name: Name of the expeirment to load.
        config: `SQAConfig`, from which to retrieve the decoder. Optional,
            defaults to base `SQAConfig`.
        reduced_state: Whether to load experiment with a slightly reduced state
            (without abandoned arms on experiment and withoug model state,
            search space, and optimization config on generator runs).
    """
    config = config or SQAConfig()
    decoder = Decoder(config=config)
    return _load_experiment(
        experiment_name=experiment_name, decoder=decoder, reduced_state=reduced_state
    )


def _load_experiment(
    experiment_name: str, decoder: Decoder, reduced_state: bool = False
) -> Experiment:
    """Load experiment by name, using given Decoder instance.

    1) Get SQLAlchemy object from DB.
    2) Convert to corresponding Ax object.
    """
    # pyre-ignore Incompatible variable type [9]: exp_sqa_class is declared to have type
    # `Type[SQAExperiment]` but is used as type `Type[ax.storage.sqa_store.db.SQABase]`
    exp_sqa_class: Type[SQAExperiment] = decoder.config.class_to_sqa_class[Experiment]

    if reduced_state:
        return decoder.experiment_from_sqa(
            experiment_sqa=_get_experiment_sqa_reduced_state(
                experiment_name=experiment_name, exp_sqa_class=exp_sqa_class
            ),
            reduced_state=reduced_state,
        )

    immutable_opt_config_and_search_space = (
        _get_experiment_immutable_opt_config_and_search_space(
            experiment_name=experiment_name, exp_sqa_class=exp_sqa_class
        )
    )
    if immutable_opt_config_and_search_space:
        experiment_sqa = _get_experiment_sqa_immutable_opt_config_and_search_space(
            experiment_name=experiment_name, exp_sqa_class=exp_sqa_class
        )
    else:
        experiment_sqa = _get_experiment_sqa(
            experiment_name=experiment_name, exp_sqa_class=exp_sqa_class
        )
    return decoder.experiment_from_sqa(experiment_sqa=experiment_sqa)


def _get_experiment_sqa(
    experiment_name: str,
    exp_sqa_class: Type[SQAExperiment],
    query_options: Optional[List[Any]] = None,
) -> SQAExperiment:
    """Obtains SQLAlchemy experiment object from DB."""
    with session_scope() as session:
        query = session.query(exp_sqa_class).filter_by(name=experiment_name)
        if query_options is not None:
            query = query.options(*query_options)
        sqa_experiment = query.one_or_none()
        if sqa_experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found.")
    return sqa_experiment


def _get_experiment_sqa_reduced_state(
    experiment_name: str, exp_sqa_class: Type[SQAExperiment]
) -> SQAExperiment:
    """Obtains most of the SQLAlchemy experiment object from DB, with some attributes
    (model state on generator runs, abandoned arms) omitted. Used for loading
    large experiments, in cases where model state history is not required.
    """
    return _get_experiment_sqa(
        experiment_name=experiment_name,
        exp_sqa_class=exp_sqa_class,
        query_options=[
            lazyload("trials.generator_runs.parameters"),
            lazyload("trials.generator_runs.parameter_constraints"),
            lazyload("trials.generator_runs.metrics"),
            lazyload("trials.abandoned_arms"),
            defaultload(exp_sqa_class.trials)
            .defaultload("generator_runs")
            .defer("model_kwargs"),
            defaultload(exp_sqa_class.trials)
            .defaultload("generator_runs")
            .defer("bridge_kwargs"),
            defaultload(exp_sqa_class.trials)
            .defaultload("generator_runs")
            .defer("model_state_after_gen"),
            defaultload(exp_sqa_class.trials)
            .defaultload("generator_runs")
            .defer("gen_metadata"),
        ],
    )


def _get_experiment_sqa_immutable_opt_config_and_search_space(
    experiment_name: str, exp_sqa_class: Type[SQAExperiment]
) -> SQAExperiment:
    """For experiments where the search space and opt config are
    immutable, we don't store copies of search space and opt config
    on each generator run. Therefore, there's no need to try to
    load these copies from the DB -- these queries will always return
    an empty list, and are therefore unnecessary and wasteful.
    """
    return _get_experiment_sqa(
        experiment_name=experiment_name,
        exp_sqa_class=exp_sqa_class,
        query_options=[
            lazyload("trials.generator_runs.parameters"),
            lazyload("trials.generator_runs.parameter_constraints"),
            lazyload("trials.generator_runs.metrics"),
        ],
    )


def _get_experiment_immutable_opt_config_and_search_space(
    experiment_name: str, exp_sqa_class: Type[SQAExperiment]
) -> bool:
    """Return true iff the experiment is designated as having an
    immutable optimization config and search space.
    """
    with session_scope() as session:
        sqa_experiment_properties = (
            session.query(exp_sqa_class.properties)
            .filter_by(name=experiment_name)
            .one_or_none()
        )
        if sqa_experiment_properties is None:
            raise ValueError(f"Experiment '{experiment_name}' not found.")

    sqa_experiment_properties = sqa_experiment_properties[0] or {}
    return sqa_experiment_properties.get(
        Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF, False
    )


def _get_experiment_id(experiment_name: str, config: SQAConfig) -> Optional[int]:
    """Get DB ID of the experiment by the given name if its in DB,
    return None otherwise.
    """
    exp_sqa_class = config.class_to_sqa_class[Experiment]
    with session_scope() as session:
        sqa_experiment_id = (
            session.query(exp_sqa_class.id)  # pyre-ignore
            .filter_by(name=experiment_name)
            .one_or_none()
        )

    if sqa_experiment_id is None:
        return None
    return sqa_experiment_id[0]


# ------------------------ Loading `GenerationStrategy`. -----------------------


def load_generation_strategy_by_experiment_name(
    experiment_name: str,
    config: Optional[SQAConfig] = None,
    experiment: Optional[Experiment] = None,
    reduced_state: bool = False,
) -> GenerationStrategy:
    """Finds a generation strategy attached to an experiment specified by a name
    and restores it from its corresponding SQA object.
    """
    config = config or SQAConfig()
    decoder = Decoder(config=config)
    return _load_generation_strategy_by_experiment_name(
        experiment_name=experiment_name,
        decoder=decoder,
        experiment=experiment,
        reduced_state=reduced_state,
    )


def load_generation_strategy_by_id(
    gs_id: int,
    config: Optional[SQAConfig] = None,
    experiment: Optional[Experiment] = None,
    reduced_state: bool = False,
) -> GenerationStrategy:
    """Finds a generation strategy stored by a given ID and restores it."""
    config = config or SQAConfig()
    decoder = Decoder(config=config)
    return _load_generation_strategy_by_id(
        gs_id=gs_id, decoder=decoder, experiment=experiment, reduced_state=reduced_state
    )


def _load_generation_strategy_by_experiment_name(
    experiment_name: str,
    decoder: Decoder,
    experiment: Optional[Experiment] = None,
    reduced_state: bool = False,
) -> GenerationStrategy:
    """Load a generation strategy attached to an experiment specified by a name,
    using given Decoder instance.

    1) Get SQLAlchemy object from DB.
    2) Convert to corresponding Ax object.
    """
    gs_id = _get_generation_strategy_id(
        experiment_name=experiment_name, decoder=decoder
    )
    if gs_id is None:
        raise ValueError(
            f"Experiment {experiment_name} does not have a generation strategy "
            "attached to it."
        )
    if not experiment:
        experiment = _load_experiment(
            experiment_name=experiment_name,
            decoder=decoder,
            reduced_state=reduced_state,
        )
    return _load_generation_strategy_by_id(
        gs_id=gs_id, decoder=decoder, experiment=experiment, reduced_state=reduced_state
    )


def _load_generation_strategy_by_id(
    gs_id: int,
    decoder: Decoder,
    experiment: Optional[Experiment] = None,
    reduced_state: bool = False,
) -> GenerationStrategy:
    """Finds a generation strategy stored by a given ID and restores it."""
    if reduced_state:
        return decoder.generation_strategy_from_sqa(
            gs_sqa=_get_generation_strategy_sqa_reduced_state(
                gs_id=gs_id, decoder=decoder
            ),
            experiment=experiment,
            reduced_state=reduced_state,
        )

    exp_sqa_class = decoder.config.class_to_sqa_class[Experiment]
    immutable_opt_config_and_search_space = (
        (
            _get_experiment_immutable_opt_config_and_search_space(
                experiment_name=experiment.name,
                # pyre-ignore Incompatible parameter type [6]: Expected
                # `Type[SQAExperiment]` for 2nd parameter `exp_sqa_class`
                # to call `_get_experiment_immutable_opt_config_and_search_space`
                # but got `Type[ax.storage.sqa_store.db.SQABase]`.
                exp_sqa_class=exp_sqa_class,
            )
        )
        if experiment is not None
        else False
    )
    if immutable_opt_config_and_search_space:
        gs_sqa = _get_generation_strategy_sqa_immutable_opt_config_and_search_space(
            gs_id=gs_id, decoder=decoder
        )
    else:
        gs_sqa = _get_generation_strategy_sqa(gs_id=gs_id, decoder=decoder)

    return decoder.generation_strategy_from_sqa(
        gs_sqa=gs_sqa, experiment=experiment, reduced_state=reduced_state
    )


def _get_generation_strategy_id(
    experiment_name: str, decoder: Decoder
) -> Optional[int]:
    """Get DB ID of the generation strategy, associated with the experiment
    with the given name if its in DB, return None otherwise.
    """
    exp_sqa_class = decoder.config.class_to_sqa_class[Experiment]
    gs_sqa_class = decoder.config.class_to_sqa_class[GenerationStrategy]
    with session_scope() as session:
        sqa_gs_id = (
            session.query(gs_sqa_class.id)  # pyre-ignore[16]
            .join(exp_sqa_class.generation_strategy)  # pyre-ignore[16]
            # pyre-fixme[16]: `SQABase` has no attribute `name`.
            .filter(exp_sqa_class.name == experiment_name)
            .one_or_none()
        )

    if sqa_gs_id is None:
        return None
    return sqa_gs_id[0]


def _get_generation_strategy_sqa(
    gs_id: int,
    decoder: Decoder,
    query_options: Optional[List[Any]] = None,
) -> SQAGenerationStrategy:
    """Obtains the SQLAlchemy generation strategy object from DB."""
    gs_sqa_class = cast(
        Type[SQAGenerationStrategy],
        decoder.config.class_to_sqa_class[GenerationStrategy],
    )
    with session_scope() as session:
        query = session.query(gs_sqa_class).filter_by(id=gs_id)
        if query_options:
            query = query.options(*query_options)
        gs_sqa = query.one_or_none()
    if gs_sqa is None:
        raise ValueError(f"Generation strategy with ID #{gs_id} not found.")
    return gs_sqa


def _get_generation_strategy_sqa_reduced_state(
    gs_id: int, decoder: Decoder
) -> SQAGenerationStrategy:
    """Obtains most of the SQLAlchemy generation strategy object from DB."""
    gs_sqa_class = cast(
        Type[SQAGenerationStrategy],
        decoder.config.class_to_sqa_class[GenerationStrategy],
    )
    gr_sqa_class = cast(
        Type[SQAGeneratorRun],
        decoder.config.class_to_sqa_class[GeneratorRun],
    )

    gs_sqa = _get_generation_strategy_sqa(
        gs_id=gs_id,
        decoder=decoder,
        query_options=[
            lazyload("generator_runs.parameters"),
            lazyload("generator_runs.parameter_constraints"),
            lazyload("generator_runs.metrics"),
            defaultload(gs_sqa_class.generator_runs).defer("model_kwargs"),
            defaultload(gs_sqa_class.generator_runs).defer("bridge_kwargs"),
            defaultload(gs_sqa_class.generator_runs).defer("model_state_after_gen"),
            defaultload(gs_sqa_class.generator_runs).defer("gen_metadata"),
        ],
    )

    # Load full last generator run (including model state), for generation
    # strategy restoration
    if gs_sqa.generator_runs:
        last_generator_run_id = gs_sqa.generator_runs[-1].id
        with session_scope() as session:
            last_gr_sqa = (
                session.query(gr_sqa_class)
                .filter_by(id=last_generator_run_id)
                .one_or_none()
            )

        # Swap last generator run with no state for a generator run with
        # state.
        gs_sqa.generator_runs[len(gs_sqa.generator_runs) - 1] = not_none(last_gr_sqa)

    return gs_sqa


def _get_generation_strategy_sqa_immutable_opt_config_and_search_space(
    gs_id: int, decoder: Decoder
) -> SQAGenerationStrategy:
    """Obtains most of the SQLAlchemy generation strategy object from DB."""
    return _get_generation_strategy_sqa(
        gs_id=gs_id,
        decoder=decoder,
        query_options=[
            lazyload("generator_runs.parameters"),
            lazyload("generator_runs.parameter_constraints"),
            lazyload("generator_runs.metrics"),
        ],
    )
