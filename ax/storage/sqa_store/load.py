#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from math import ceil
from typing import Any, cast

from ax.analysis.analysis import AnalysisCard

from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.trial import Trial
from ax.exceptions.core import ExperimentNotFoundError, ObjectNotFoundError
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.storage.sqa_store.db import session_scope
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.reduced_state import (
    get_query_options_to_defer_immutable_duplicates,
    get_query_options_to_defer_large_model_cols,
)
from ax.storage.sqa_store.sqa_classes import (
    SQAAnalysisCard,
    SQAExperiment,
    SQAGenerationStrategy,
    SQAGeneratorRun,
    SQAMetric,
    SQATrial,
)
from ax.storage.sqa_store.sqa_config import SQAConfig

from ax.storage.utils import MetricIntent
from ax.utils.common.constants import Keys
from pyre_extensions import assert_is_instance, none_throws
from sqlalchemy.orm import defaultload, lazyload, noload
from sqlalchemy.orm.exc import DetachedInstanceError


# ---------------------------- Loading `Experiment`. ---------------------------


def load_experiment(
    experiment_name: str,
    config: SQAConfig | None = None,
    reduced_state: bool = False,
    load_trials_in_batches_of_size: int | None = None,
    skip_runners_and_metrics: bool = False,
    load_auxiliary_experiments: bool = True,
) -> Experiment:
    """Load experiment by name.

    Args:
        experiment_name: Name of the expeirment to load.
        config: `SQAConfig`, from which to retrieve the decoder. Optional,
            defaults to base `SQAConfig`.
        reduced_state: Whether to load experiment with a slightly reduced state
            (without abandoned arms on experiment and without model state,
            search space, and optimization config on generator runs).
        skip_runners_and_metrics: If True skip loading runners, and do only a
            minimal load of metrics. This option is intended to enable loading of
            experiments that require custom runners or metrics, without depending
            on a registry. Note that even though the intention is to skip loading
            of metrics, this option converts the loaded metrics into a base
            metric avoiding conversion related to custom properties of the metric.
        load_auxiliary_experiments: whether to load auxiliary experiments.
    """
    config = SQAConfig() if config is None else config
    decoder = Decoder(config=config)
    return _load_experiment(
        experiment_name=experiment_name,
        decoder=decoder,
        reduced_state=reduced_state,
        load_trials_in_batches_of_size=load_trials_in_batches_of_size,
        skip_runners_and_metrics=skip_runners_and_metrics,
        load_auxiliary_experiments=load_auxiliary_experiments,
    )


def _load_experiment(
    experiment_name: str,
    decoder: Decoder,
    reduced_state: bool = False,
    load_trials_in_batches_of_size: int | None = None,
    skip_runners_and_metrics: bool = False,
    load_auxiliary_experiments: bool = True,
) -> Experiment:
    """Load experiment by name, using given Decoder instance.

    1) Get SQLAlchemy object from DB.
    2) Convert to corresponding Ax object.

    Args:
        experiment_name: Name of the experiment
        decoder: Decoder used to convert SQAlchemy objects into Ax objects
        reduced_state: Whether to load experiment and generation strategy
        load_trials_in_batches_of_size: Number of trials to be fetched from database
            per batch
        load_auxiliary_experiments: whether to load auxiliary experiments.
    """

    # pyre-ignore Incompatible variable type [9]: exp_sqa_class is declared to have type
    # `Type[SQAExperiment]` but is used as type `Type[ax.storage.sqa_store.db.SQABase]`
    exp_sqa_class: type[SQAExperiment] = decoder.config.class_to_sqa_class[Experiment]
    # pyre-ignore Incompatible variable type [9]: trial_sqa_class is decl. to have type
    # `Type[SQATrial]` but is used as type `Type[ax.storage.sqa_store.db.SQABase]`
    trial_sqa_class: type[SQATrial] = decoder.config.class_to_sqa_class[Trial]
    imm_OC_and_SS = _get_experiment_immutable_opt_config_and_search_space(
        experiment_name=experiment_name, exp_sqa_class=exp_sqa_class
    )

    if reduced_state:
        _get_experiment_sqa_func = _get_experiment_sqa_reduced_state

    else:
        _get_experiment_sqa_func = (
            _get_experiment_sqa_immutable_opt_config_and_search_space
            if imm_OC_and_SS
            else _get_experiment_sqa
        )

    experiment_sqa = _get_experiment_sqa_func(
        experiment_name=experiment_name,
        exp_sqa_class=exp_sqa_class,
        trial_sqa_class=trial_sqa_class,
        load_trials_in_batches_of_size=load_trials_in_batches_of_size,
        skip_runners_and_metrics=skip_runners_and_metrics,
    )

    # If skip_runners_and_metrics is True, override metric types to
    # base metric type of 0 (to load them with minimum base properties).
    # This includes metrics attached to the experiment, and ones
    # attached to the trials.
    #
    # NOTE: Currently, we load metrics and then convert mostly go get
    # "lower_is_better". Alternatively, we can "nolod" all attributes except
    # "lower_is_better" or any other attribute we want to include. This can be
    # implemented in the future if we need to.
    if skip_runners_and_metrics:
        base_metric_type_int = decoder.config.metric_registry[Metric]
        for sqa_metric in experiment_sqa.metrics:
            _set_sqa_metric_to_base_type(
                sqa_metric, base_metric_type_int=base_metric_type_int
            )

        assign_metric_on_gr = not reduced_state and not imm_OC_and_SS
        if assign_metric_on_gr:
            try:
                for sqa_trial in experiment_sqa.trials:
                    for sqa_generator_run in sqa_trial.generator_runs:
                        for sqa_metric in sqa_generator_run.metrics:
                            _set_sqa_metric_to_base_type(
                                sqa_metric, base_metric_type_int=base_metric_type_int
                            )
            except DetachedInstanceError as e:
                raise DetachedInstanceError(
                    "Unable to retrieve metric from SQA generator run, possibly due "
                    "to parts of the experiment being lazy-loaded. This is not "
                    f"expected state, please contact Ax support. Original error: {e}"
                )

    return decoder.experiment_from_sqa(
        experiment_sqa=experiment_sqa,
        reduced_state=reduced_state,
        load_auxiliary_experiments=load_auxiliary_experiments,
    )


def _get_experiment_sqa(
    experiment_name: str,
    exp_sqa_class: type[SQAExperiment],
    trial_sqa_class: type[SQATrial],
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    trials_query_options: list[Any] | None = None,
    load_trials_in_batches_of_size: int | None = None,
    skip_runners_and_metrics: bool = False,
) -> SQAExperiment:
    """Obtains SQLAlchemy experiment object from DB."""
    with session_scope() as session:
        query = (
            session.query(exp_sqa_class)
            .filter_by(name=experiment_name)
            # Delay loading trials to a separate call to `_get_trials_sqa` below
            .options(noload("trials"))
        )

        if skip_runners_and_metrics:
            query = query.options(noload("runners")).options(noload("trials.runner"))

        sqa_experiment = query.one_or_none()

    if sqa_experiment is None:
        raise ExperimentNotFoundError(f"Experiment '{experiment_name}' not found.")

    sqa_trials = _get_trials_sqa(
        experiment_id=sqa_experiment.id,
        trial_sqa_class=trial_sqa_class,
        trials_query_options=trials_query_options,
        load_trials_in_batches_of_size=load_trials_in_batches_of_size,
        skip_runners_and_metrics=skip_runners_and_metrics,
    )

    sqa_experiment.trials = sqa_trials

    return sqa_experiment


def _get_trials_sqa(
    experiment_id: int,
    trial_sqa_class: type[SQATrial],
    load_trials_in_batches_of_size: int | None = None,
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    trials_query_options: list[Any] | None = None,
    skip_runners_and_metrics: bool = False,
) -> list[SQATrial]:
    """Obtains SQLAlchemy trial objects for given experiment ID from DB,
    optionally in mini-batches and with specified query options.
    """
    with session_scope() as session:
        query = session.query(trial_sqa_class.id).filter_by(experiment_id=experiment_id)
        trial_db_ids = query.all()
        trial_db_ids = [db_id_tuple[0] for db_id_tuple in trial_db_ids]

    if len(trial_db_ids) == 0:
        return []

    batch_size = (
        len(trial_db_ids)
        if load_trials_in_batches_of_size is None
        else load_trials_in_batches_of_size
    )

    sqa_trials = []
    for i in range(ceil(len(trial_db_ids) / batch_size)):
        mini_batch_db_ids = trial_db_ids[batch_size * i : batch_size * (i + 1)]
        with session_scope() as session:
            query = session.query(trial_sqa_class).filter(
                trial_sqa_class.id.in_(mini_batch_db_ids)
            )

            if trials_query_options is not None:
                query = query.options(*trials_query_options)

            if skip_runners_and_metrics:
                query = query.options(noload("runner"))

            sqa_trials.extend(query.all())

    return sqa_trials


def _set_sqa_metric_to_base_type(
    sqa_metric: SQAMetric, base_metric_type_int: int
) -> None:
    """Sets metric type to base type, since we don't want to load
    the metric class from the DB.
    """
    sqa_metric.metric_type = base_metric_type_int
    # Handle multi-objective metrics that are not directly attached to
    # the experiment
    if sqa_metric.intent == MetricIntent.MULTI_OBJECTIVE:
        if sqa_metric.properties is None:
            sqa_metric.properties = {}
        sqa_metric.properties["skip_runners_and_metrics"] = True


def _get_experiment_sqa_reduced_state(
    experiment_name: str,
    exp_sqa_class: type[SQAExperiment],
    trial_sqa_class: type[SQATrial],
    load_trials_in_batches_of_size: int | None = None,
    skip_runners_and_metrics: bool = False,
) -> SQAExperiment:
    """Obtains most of the SQLAlchemy experiment object from DB, with some attributes
    (model state on generator runs, abandoned arms) omitted. Used for loading
    large experiments, in cases where model state history is not required.
    """
    options = get_query_options_to_defer_immutable_duplicates()
    options.extend(get_query_options_to_defer_large_model_cols())

    return _get_experiment_sqa(
        experiment_name=experiment_name,
        exp_sqa_class=exp_sqa_class,
        trial_sqa_class=trial_sqa_class,
        trials_query_options=options,
        load_trials_in_batches_of_size=load_trials_in_batches_of_size,
        skip_runners_and_metrics=skip_runners_and_metrics,
    )


def _get_experiment_sqa_immutable_opt_config_and_search_space(
    experiment_name: str,
    exp_sqa_class: type[SQAExperiment],
    trial_sqa_class: type[SQATrial],
    load_trials_in_batches_of_size: int | None = None,
    skip_runners_and_metrics: bool = False,
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
        trial_sqa_class=trial_sqa_class,
        trials_query_options=get_query_options_to_defer_immutable_duplicates(),
        load_trials_in_batches_of_size=load_trials_in_batches_of_size,
        skip_runners_and_metrics=skip_runners_and_metrics,
    )


def _get_experiment_immutable_opt_config_and_search_space(
    experiment_name: str, exp_sqa_class: type[SQAExperiment]
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
            raise ExperimentNotFoundError(f"Experiment '{experiment_name}' not found.")

    sqa_experiment_properties = sqa_experiment_properties[0] or {}
    return sqa_experiment_properties.get(
        Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF, False
    )


def _get_experiment_id(experiment_name: str, config: SQAConfig) -> int | None:
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
    config: SQAConfig | None = None,
    experiment: Experiment | None = None,
    reduced_state: bool = False,
    skip_runners_and_metrics: bool = False,
) -> GenerationStrategy:
    """Finds a generation strategy attached to an experiment specified by a name
    and restores it from its corresponding SQA object.
    """
    config = SQAConfig() if config is None else config
    decoder = Decoder(config=config)
    return _load_generation_strategy_by_experiment_name(
        experiment_name=experiment_name,
        decoder=decoder,
        experiment=experiment,
        reduced_state=reduced_state,
        skip_runners_and_metrics=skip_runners_and_metrics,
    )


def load_generation_strategy_by_id(
    gs_id: int,
    config: SQAConfig | None = None,
    experiment: Experiment | None = None,
    reduced_state: bool = False,
) -> GenerationStrategy:
    """Finds a generation strategy stored by a given ID and restores it."""
    config = SQAConfig() if config is None else config
    decoder = Decoder(config=config)
    return _load_generation_strategy_by_id(
        gs_id=gs_id, decoder=decoder, experiment=experiment, reduced_state=reduced_state
    )


def _load_generation_strategy_by_experiment_name(
    experiment_name: str,
    decoder: Decoder,
    experiment: Experiment | None = None,
    reduced_state: bool = False,
    skip_runners_and_metrics: bool = False,
) -> GenerationStrategy:
    """Load a generation strategy attached to an experiment specified by a name,
    using given Decoder instance.

    1) Get SQLAlchemy object from DB.
    2) Convert to corresponding Ax object.
    """
    gs_id = get_generation_strategy_id(experiment_name=experiment_name, decoder=decoder)
    if gs_id is None:
        raise ObjectNotFoundError(
            f"Experiment {experiment_name} does not have a generation strategy "
            "attached to it."
        )
    if not experiment:
        experiment = _load_experiment(
            experiment_name=experiment_name,
            decoder=decoder,
            reduced_state=reduced_state,
            skip_runners_and_metrics=skip_runners_and_metrics,
        )
    return _load_generation_strategy_by_id(
        gs_id=gs_id, decoder=decoder, experiment=experiment, reduced_state=reduced_state
    )


def _load_generation_strategy_by_id(
    gs_id: int,
    decoder: Decoder,
    experiment: Experiment | None = None,
    reduced_state: bool = False,
) -> GenerationStrategy:
    """Finds a generation strategy stored by a given ID and restores it."""
    if reduced_state:
        return decoder.generation_strategy_from_sqa(
            gs_sqa=get_generation_strategy_sqa_reduced_state(
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
        gs_sqa = get_generation_strategy_sqa(gs_id=gs_id, decoder=decoder)

    return decoder.generation_strategy_from_sqa(
        gs_sqa=gs_sqa, experiment=experiment, reduced_state=reduced_state
    )


def get_generation_strategy_id(experiment_name: str, decoder: Decoder) -> int | None:
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


def get_generation_strategy_sqa(
    gs_id: int,
    decoder: Decoder,
    # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
    query_options: list[Any] | None = None,
) -> SQAGenerationStrategy:
    """Obtains the SQLAlchemy generation strategy object from DB."""
    gs_sqa_class = cast(
        type[SQAGenerationStrategy],
        decoder.config.class_to_sqa_class[GenerationStrategy],
    )
    with session_scope() as session:
        query = session.query(gs_sqa_class).filter_by(id=gs_id)
        if query_options:
            query = query.options(*query_options)
        gs_sqa = query.one_or_none()
    if gs_sqa is None:
        raise ObjectNotFoundError(f"Generation strategy with ID #{gs_id} not found.")
    return gs_sqa


def get_generation_strategy_sqa_reduced_state(
    gs_id: int, decoder: Decoder
) -> SQAGenerationStrategy:
    """Obtains most of the SQLAlchemy generation strategy object from DB."""
    gs_sqa_class = cast(
        type[SQAGenerationStrategy],
        decoder.config.class_to_sqa_class[GenerationStrategy],
    )
    gr_sqa_class = cast(
        type[SQAGeneratorRun],
        decoder.config.class_to_sqa_class[GeneratorRun],
    )

    gs_sqa = get_generation_strategy_sqa(
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
        gs_sqa.generator_runs[len(gs_sqa.generator_runs) - 1] = none_throws(last_gr_sqa)

    return gs_sqa


def get_generator_runs_by_id(
    generator_run_ids: list[int],
    decoder: Decoder,
    reduced_state: bool = False,
    immutable_search_space_and_opt_config: bool = False,
) -> list[GeneratorRun]:
    """Bulk fetches generator runs by id."""
    generator_run_sqa_class = decoder.config.class_to_sqa_class[GeneratorRun]
    with session_scope() as session:
        query = session.query(generator_run_sqa_class).filter(
            generator_run_sqa_class.id.in_(generator_run_ids)  # pyre-ignore[16]
        )
        sqa_grs = query.all()
    return [
        decoder.generator_run_from_sqa(
            generator_run_sqa=assert_is_instance(
                sqa_gr,
                SQAGeneratorRun,
            ),
            reduced_state=reduced_state,
            immutable_search_space_and_opt_config=immutable_search_space_and_opt_config,
        )
        for sqa_gr in sqa_grs
    ]


def _get_generation_strategy_sqa_immutable_opt_config_and_search_space(
    gs_id: int, decoder: Decoder
) -> SQAGenerationStrategy:
    """Obtains most of the SQLAlchemy generation strategy object from DB."""
    return get_generation_strategy_sqa(
        gs_id=gs_id,
        decoder=decoder,
        query_options=[
            lazyload("generator_runs.parameters"),
            lazyload("generator_runs.parameter_constraints"),
            lazyload("generator_runs.metrics"),
        ],
    )


def load_analysis_cards_by_experiment_name(
    experiment_name: str,
    config: SQAConfig | None = None,
) -> list[AnalysisCard]:
    """Loads analysis cards for an experiment."""
    config = SQAConfig() if config is None else config
    decoder = Decoder(config=config)
    analysis_card_sqa_class: SQAAnalysisCard = cast(
        SQAAnalysisCard, decoder.config.class_to_sqa_class[AnalysisCard]
    )
    exp_sqa_class: SQAExperiment = cast(
        SQAExperiment, decoder.config.class_to_sqa_class[Experiment]
    )
    with session_scope() as session:
        analysis_cards_sqa = (
            session.query(analysis_card_sqa_class)
            .join(exp_sqa_class.analysis_cards)
            .filter(exp_sqa_class.name == experiment_name)
        )
    return [
        decoder.analysis_card_from_sqa(analysis_card_sqa=analysis_card_sqa)
        for analysis_card_sqa in analysis_cards_sqa
    ]
