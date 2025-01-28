#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import re
import time
from collections.abc import Iterable

from logging import INFO, Logger
from typing import Optional, Sequence

from ax.analysis.analysis import AnalysisCard

from ax.core.base_trial import BaseTrial
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.core.generator_run import GeneratorRun
from ax.core.runner import Runner
from ax.exceptions.core import (
    IncompatibleDependencyVersion,
    ObjectNotFoundError,
    UnsupportedError,
)
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.utils.common.executils import retry_on_exception
from ax.utils.common.logger import _round_floats_for_logging, get_logger
from pyre_extensions import none_throws

RETRY_EXCEPTION_TYPES: tuple[type[Exception], ...] = ()


logger: Logger = get_logger(__name__)

try:  # We don't require SQLAlchemy by default.
    # pyre-fixme[21]: Could not find a name `__version__` defined in module
    # `sqlalchemy`.
    from sqlalchemy import __version__ as sqa_version

    # pyre-fixme[16]: Module `sqlalchemy` has no attribute `__version__`.
    sqa_major_version = int(none_throws(re.match(r"^\d*", sqa_version))[0])
    if sqa_major_version > 1:
        msg = (
            "Ax currently requires a sqlalchemy version below 2.0. This will be "
            "addressed in a future release. Disabling SQL storage in Ax for now, if "
            "you would like to use SQL storage please install Ax with mysql extras "
            "via `pip install ax-platform[mysql]`."
        )

        logger.warning(msg)

        raise IncompatibleDependencyVersion(msg)

    from ax.storage.sqa_store.db import init_engine_and_session_factory
    from ax.storage.sqa_store.decoder import Decoder
    from ax.storage.sqa_store.encoder import Encoder
    from ax.storage.sqa_store.load import (
        _get_experiment_id,
        _load_experiment,
        _load_generation_strategy_by_experiment_name,
        get_generation_strategy_id,
    )
    from ax.storage.sqa_store.save import (
        _save_experiment,
        _save_generation_strategy,
        _save_or_update_trials,
        _update_generation_strategy,
        save_analysis_cards,
        update_properties_on_experiment,
        update_runner_on_experiment,
    )
    from ax.storage.sqa_store.sqa_config import SQAConfig
    from ax.storage.sqa_store.structs import DBSettings
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.orm.exc import StaleDataError

    # We retry on `OperationalError` if saving to DB.
    RETRY_EXCEPTION_TYPES = (OperationalError, StaleDataError)
except (ModuleNotFoundError, IncompatibleDependencyVersion, TypeError):
    DBSettings = None
    Decoder = None
    Encoder = None
    SQAConfig = None


STORAGE_MINI_BATCH_SIZE = 50
LOADING_MINI_BATCH_SIZE = 10000


class WithDBSettingsBase:
    """Helper class providing methods for saving changes made to an experiment
    if `db_settings` property is set to a non-None value on the instance.
    """

    _db_settings: Optional[DBSettings] = None

    def __init__(
        self,
        db_settings: Optional[DBSettings] = None,
        logging_level: int = INFO,
        suppress_all_errors: bool = False,
    ) -> None:
        if db_settings and (not DBSettings or not isinstance(db_settings, DBSettings)):
            raise ValueError(
                "`db_settings` argument should be of type ax.storage.sqa_store."
                f"(Got: {db_settings} of type {type(db_settings)}. "
                "structs.DBSettings. To use `DBSettings`, you will need SQLAlchemy "
                "installed in your environment (can be installed through pip)."
            )
        self._db_settings = db_settings or self._get_default_db_settings()
        self._suppress_all_errors = suppress_all_errors
        if self.db_settings_set:
            init_engine_and_session_factory(
                creator=self.db_settings.creator, url=self.db_settings.url
            )
        logger.setLevel(logging_level)

    @staticmethod
    def _get_default_db_settings() -> Optional[DBSettings]:
        """Overridable method to get default db_settings
        if none are passed in __init__
        """
        return None

    @property
    def db_settings_set(self) -> bool:
        """Whether non-None DB settings are set on this instance."""
        return self._db_settings is not None

    @property
    def db_settings(self) -> DBSettings:
        """DB settings set on this instance; guaranteed to be non-None."""
        if self._db_settings is None:
            raise ValueError("No DB settings are set on this instance.")
        return none_throws(self._db_settings)

    def _get_experiment_and_generation_strategy_db_id(
        self, experiment_name: str
    ) -> tuple[int | None, int | None]:
        """Retrieve DB ids of experiment by the given name and the associated
        generation strategy. Each ID is None if corresponding object is not
        found.
        """
        if not self.db_settings_set:
            return None, None

        exp_id = _get_experiment_id(
            experiment_name=experiment_name, config=self.db_settings.decoder.config
        )
        if not exp_id:
            return None, None
        gs_id = get_generation_strategy_id(
            experiment_name=experiment_name, decoder=self.db_settings.decoder
        )
        return exp_id, gs_id

    def _maybe_save_experiment_and_generation_strategy(
        self, experiment: Experiment, generation_strategy: GenerationStrategyInterface
    ) -> tuple[bool, bool]:
        """If DB settings are set on this `WithDBSettingsBase` instance, checks
        whether given experiment and generation strategy are already saved and
        saves them, if not.

        Returns:
            Tuple of two booleans: whether experiment was saved in the course of
                this function's execution and whether generation strategy was
                saved.
        """
        saved_exp, saved_gs = False, False
        if self.db_settings_set:
            if experiment._name is None:
                raise ValueError(
                    "Experiment must specify a name to use storage functionality."
                )
            exp_name = none_throws(experiment.name)
            exp_id, gs_id = self._get_experiment_and_generation_strategy_db_id(
                experiment_name=exp_name
            )
            if exp_id:  # Experiment in DB.
                logger.info(f"Experiment {exp_name} is in DB, updating it.")
                self._save_experiment_to_db_if_possible(experiment=experiment)
                saved_exp = True
            else:  # Experiment not yet in DB.
                logger.info(f"Experiment {exp_name} is not yet in DB, storing it.")
                self._save_experiment_to_db_if_possible(experiment=experiment)
                saved_exp = True
            if gs_id and generation_strategy._db_id != gs_id:
                raise UnsupportedError(
                    "Experiment was associated with generation strategy in DB, "
                    f"but a new generation strategy {generation_strategy.name} "
                    "was provided. To use the generation strategy currently in DB,"
                    " instantiate scheduler via: `Scheduler.from_stored_experiment`."
                )
            if not gs_id or generation_strategy._db_id is None:
                # There is no GS associated with experiment or the generation
                # strategy passed in is different from the one associated with
                # experiment currently.
                logger.info(
                    f"Generation strategy {generation_strategy.name} is not yet in DB, "
                    "storing it."
                )
                # If generation strategy does not yet have an experiment attached,
                # attach the current experiment to it, as otherwise it will not be
                # possible to retrieve by experiment name.
                if generation_strategy._experiment is None:
                    generation_strategy.experiment = experiment
                self._save_generation_strategy_to_db_if_possible(
                    generation_strategy=generation_strategy
                )
                saved_gs = True
        return saved_exp, saved_gs

    def _load_experiment_and_generation_strategy(
        self,
        experiment_name: str,
        reduced_state: bool = False,
        skip_runners_and_metrics: bool = False,
    ) -> tuple[Experiment | None, GenerationStrategy | None]:
        """Loads experiment and its corresponding generation strategy from database
        if DB settings are set on this `WithDBSettingsBase` instance.

        Args:
            experiment_name: Name of the experiment to load, used as unique
                identifier by which to find the experiment.
            reduced_state: Whether to load experiment and generation strategy
                with a slightly reduced state (without abandoned arms on experiment
                and model state on each generator run in experiment and generation
                strategy; last generator run on generation strategy will still
                have its model state).

        Returns:
            - Tuple of `None` and `None` if `DBSettings` are set and no experiment
              exists by the given name.
            - Tuple of `Experiment` and `None` if experiment exists but does not
              have a generation strategy attached to it.
            - Tuple of `Experiment` and `GenerationStrategy` if experiment exists
              and has a generation strategy attached to it.
        """
        if not self.db_settings_set:
            raise ValueError("Cannot load from DB in absence of DB settings.")

        logger.info(
            "Loading experiment and generation strategy (with reduced state: "
            f"{reduced_state})..."
        )
        start_time = time.time()
        experiment = _load_experiment(
            experiment_name,
            decoder=self.db_settings.decoder,
            reduced_state=reduced_state,
            load_trials_in_batches_of_size=LOADING_MINI_BATCH_SIZE,
            skip_runners_and_metrics=skip_runners_and_metrics,
        )
        if not isinstance(experiment, Experiment):
            raise ValueError("Service API only supports `Experiment`.")
        num_trials = len(experiment.trials)
        logger.info(
            f"Loaded experiment {experiment_name} & {num_trials} trials in "
            f"{_round_floats_for_logging(time.time() - start_time)} seconds."
        )
        generation_strategy = try_load_generation_strategy(
            experiment_name=experiment_name,
            decoder=self.db_settings.decoder,
            experiment=experiment,
            reduced_state=reduced_state,
        )

        return experiment, generation_strategy

    def _save_experiment_to_db_if_possible(self, experiment: Experiment) -> bool:
        """Saves attached experiment and generation strategy if DB settings are
        set on this `WithDBSettingsBase` instance.

        Args:
            experiment: Experiment to save new trials in DB.

        Returns:
            bool: Whether the experiment was saved.
        """
        if self.db_settings_set:
            _save_experiment_to_db_if_possible(
                experiment=experiment,
                encoder=self.db_settings.encoder,
                decoder=self.db_settings.decoder,
                suppress_all_errors=self._suppress_all_errors,
            )
            return True
        return False

    def _save_or_update_trials_and_generation_strategy_if_possible(
        self,
        experiment: Experiment,
        trials: list[BaseTrial],
        generation_strategy: GenerationStrategyInterface,
        new_generator_runs: list[GeneratorRun],
        reduce_state_generator_runs: bool = False,
    ) -> None:
        """Saves new trials (and updates existing ones) on given experiment
        and updates the given generation strategy, if DB settings are set on
        this `WithDBSettingsBase` instance.

        Args:
            experiment: Experiment, on which to save new trials in DB.
            trials: Newly added or updated trials to save or update in DB.
            generation_strategy: Generation strategy to update in DB.
            new_generator_runs: Generator runs to add to generation strategy.
        """
        logger.debug(f"Saving or updating {len(trials)} trials in DB.")
        self._save_or_update_trials_in_db_if_possible(
            experiment=experiment,
            trials=trials,
            reduce_state_generator_runs=reduce_state_generator_runs,
        )
        logger.debug(
            "Updating generation strategy in DB with "
            f"{len(new_generator_runs)} generator runs."
        )
        self._update_generation_strategy_in_db_if_possible(
            generation_strategy=generation_strategy,
            new_generator_runs=new_generator_runs,
            reduce_state_generator_runs=reduce_state_generator_runs,
        )
        return

    # No retries needed, covered in `self._save_or_update_trials_in_db_if_possible`
    def _save_or_update_trial_in_db_if_possible(
        self,
        experiment: Experiment,
        trial: BaseTrial,
    ) -> bool:
        """Saves new trial on given experiment if DB settings are set on this
        `WithDBSettingsBase` instance.

        Args:
            experiment: Experiment, on which to save new trial in DB.
            trials: Newly added trial to save.

        Returns:
            bool: Whether the trial was saved.
        """
        return self._save_or_update_trials_in_db_if_possible(
            experiment=experiment,
            trials=[trial],
        )

    def _save_or_update_trials_in_db_if_possible(
        self,
        experiment: Experiment,
        trials: Sequence[BaseTrial],
        reduce_state_generator_runs: bool = False,
    ) -> bool:
        """Saves new trials or update existing trials on given experiment if DB
        settings are set on this `WithDBSettingsBase` instance.

        Args:
            experiment: Experiment, on which to save new trials in DB.
            trials: Newly added trials to save.

        Returns:
            bool: Whether the trials were saved.
        """
        if self.db_settings_set:
            _save_or_update_trials_in_db_if_possible(
                experiment=experiment,
                trials=trials,
                encoder=self.db_settings.encoder,
                decoder=self.db_settings.decoder,
                suppress_all_errors=self._suppress_all_errors,
                reduce_state_generator_runs=reduce_state_generator_runs,
            )
            return True
        return False

    def _save_generation_strategy_to_db_if_possible(
        self,
        generation_strategy: GenerationStrategyInterface | None = None,
    ) -> bool:
        """Saves given generation strategy if DB settings are set on this
        `WithDBSettingsBase` instance and the generation strategy is an
        instance of `GenerationStrategy`.

        Args:
            generation_strategy: GenerationStrategyInterface to update in DB.
                For now, only instances of  GenerationStrategy will be updated.
                Otherwise, this function is a no-op.

        Returns:
            bool: Whether the generation strategy was saved.
        """
        if self.db_settings_set and generation_strategy is not None:
            # only local GenerationStrategies should need to be saved to
            # the database because only they make changes locally
            if isinstance(generation_strategy, GenerationStrategy):
                _save_generation_strategy_to_db_if_possible(
                    generation_strategy=generation_strategy,
                    encoder=self.db_settings.encoder,
                    decoder=self.db_settings.decoder,
                    suppress_all_errors=self._suppress_all_errors,
                )
                return True
        return False

    def _update_generation_strategy_in_db_if_possible(
        self,
        generation_strategy: GenerationStrategyInterface,
        new_generator_runs: list[GeneratorRun],
        reduce_state_generator_runs: bool = False,
    ) -> bool:
        """Updates the given generation strategy with new generator runs (and with
        new current generation step if applicable) if DB settings are set
        on this `WithDBSettingsBase` instance and the generation strategy is an
        instance of `GenerationStrategy`.

        Args:
            generation_strategy: GenerationStrategyInterface to update in DB.
                For now, only instances of  GenerationStrategy will be updated.
                Otherwise, this function is a no-op.
            new_generator_runs: New generator runs of this generation strategy
                since its last save.

        Returns:
            bool: Whether the experiment was saved.
        """
        if self.db_settings_set:
            # only local GenerationStrategies should need to be saved to
            # the database because only they make changes locally
            if isinstance(generation_strategy, GenerationStrategy):
                _update_generation_strategy_in_db_if_possible(
                    generation_strategy=generation_strategy,
                    new_generator_runs=new_generator_runs,
                    encoder=self.db_settings.encoder,
                    decoder=self.db_settings.decoder,
                    suppress_all_errors=self._suppress_all_errors,
                    reduce_state_generator_runs=reduce_state_generator_runs,
                )
                return True
        return False

    def _update_runner_on_experiment_in_db_if_possible(
        self, experiment: Experiment, runner: Runner
    ) -> bool:
        if self.db_settings_set:
            _update_runner_on_experiment_in_db_if_possible(
                experiment=experiment,
                runner=runner,
                encoder=self.db_settings.encoder,
                decoder=self.db_settings.decoder,
                suppress_all_errors=self._suppress_all_errors,
            )
            return True

        return False

    def _update_experiment_properties_in_db(
        self,
        experiment_with_updated_properties: Experiment,
    ) -> bool:
        exp = experiment_with_updated_properties
        if self.db_settings_set:
            _update_experiment_properties_in_db(
                experiment_with_updated_properties=exp,
                sqa_config=self.db_settings.encoder.config,
                suppress_all_errors=self._suppress_all_errors,
            )
            return True
        return False

    def _save_analysis_cards_to_db_if_possible(
        self,
        experiment: Experiment,
        analysis_cards: Iterable[AnalysisCard],
    ) -> bool:
        if self.db_settings_set:
            _save_analysis_cards_to_db_if_possible(
                experiment=experiment,
                analysis_cards=analysis_cards,
                sqa_config=self.db_settings.encoder.config,
                suppress_all_errors=self._suppress_all_errors,
            )
            return True

        return False


# ------------- Utils for storage that assume `DBSettings` are provided --------


@retry_on_exception(
    retries=3,
    default_return_on_suppression=False,
    exception_types=RETRY_EXCEPTION_TYPES,
)
def _save_experiment_to_db_if_possible(
    experiment: Experiment,
    encoder: Encoder,
    decoder: Decoder,
    suppress_all_errors: bool,  # Used by the decorator.
) -> None:
    start_time = time.time()
    _save_experiment(
        experiment,
        encoder=encoder,
        decoder=decoder,
    )
    logger.debug(
        f"Saved experiment {experiment.name} in "
        f"{_round_floats_for_logging(time.time() - start_time)} seconds."
    )


@retry_on_exception(
    retries=3,
    default_return_on_suppression=False,
    exception_types=RETRY_EXCEPTION_TYPES,
)
def _save_or_update_trials_in_db_if_possible(
    experiment: Experiment,
    trials: list[BaseTrial],
    encoder: Encoder,
    decoder: Decoder,
    suppress_all_errors: bool,  # Used by the decorator.
    reduce_state_generator_runs: bool = False,
) -> None:
    start_time = time.time()
    _save_or_update_trials(
        experiment=experiment,
        trials=trials,
        encoder=encoder,
        decoder=decoder,
        batch_size=STORAGE_MINI_BATCH_SIZE,
        reduce_state_generator_runs=reduce_state_generator_runs,
    )
    logger.debug(
        f"Saved or updated trials {[trial.index for trial in trials]} in "
        f"{_round_floats_for_logging(time.time() - start_time)} seconds "
        f"in mini-batches of {STORAGE_MINI_BATCH_SIZE}."
    )


@retry_on_exception(
    retries=3,
    default_return_on_suppression=False,
    exception_types=RETRY_EXCEPTION_TYPES,
)
def _save_generation_strategy_to_db_if_possible(
    generation_strategy: GenerationStrategy,
    encoder: Encoder,
    decoder: Decoder,
    suppress_all_errors: bool,  # Used by the decorator.
) -> None:
    start_time = time.time()
    _save_generation_strategy(
        generation_strategy=generation_strategy,
        encoder=encoder,
        decoder=decoder,
    )
    logger.debug(
        f"Saved generation strategy {generation_strategy.name} in "
        f"{_round_floats_for_logging(time.time() - start_time)} seconds."
    )


@retry_on_exception(
    retries=3,
    default_return_on_suppression=False,
    exception_types=RETRY_EXCEPTION_TYPES,
)
def _update_generation_strategy_in_db_if_possible(
    generation_strategy: GenerationStrategy,
    new_generator_runs: list[GeneratorRun],
    encoder: Encoder,
    decoder: Decoder,
    suppress_all_errors: bool,  # Used by the decorator.
    reduce_state_generator_runs: bool = False,
) -> None:
    start_time = time.time()
    _update_generation_strategy(
        generation_strategy=generation_strategy,
        generator_runs=new_generator_runs,
        encoder=encoder,
        decoder=decoder,
        batch_size=STORAGE_MINI_BATCH_SIZE,
        reduce_state_generator_runs=reduce_state_generator_runs,
    )
    logger.debug(
        f"Updated generation strategy {generation_strategy.name} in "
        f"{_round_floats_for_logging(time.time() - start_time)} seconds in "
        f"mini-batches of {STORAGE_MINI_BATCH_SIZE} generator runs."
    )


@retry_on_exception(
    retries=3,
    default_return_on_suppression=False,
    exception_types=RETRY_EXCEPTION_TYPES,
)
def _update_runner_on_experiment_in_db_if_possible(
    experiment: Experiment,
    runner: Runner,
    encoder: Encoder,
    decoder: Decoder,
    suppress_all_errors: bool,  # Used by the decorator.
) -> None:
    update_runner_on_experiment(
        experiment=experiment, runner=runner, encoder=encoder, decoder=decoder
    )


@retry_on_exception(
    retries=3,
    default_return_on_suppression=False,
    exception_types=RETRY_EXCEPTION_TYPES,
)
def _update_experiment_properties_in_db(
    experiment_with_updated_properties: Experiment,
    sqa_config: SQAConfig,
    suppress_all_errors: bool,  # Used by the decorator.
) -> None:
    update_properties_on_experiment(
        experiment_with_updated_properties=experiment_with_updated_properties,
        config=sqa_config,
    )


@retry_on_exception(
    retries=3,
    default_return_on_suppression=False,
    exception_types=RETRY_EXCEPTION_TYPES,
)
def _save_analysis_cards_to_db_if_possible(
    experiment: Experiment,
    analysis_cards: Iterable[AnalysisCard],
    sqa_config: SQAConfig,
    suppress_all_errors: bool,  # Used by the decorator.
) -> None:
    save_analysis_cards(
        experiment=experiment,
        analysis_cards=[*analysis_cards],
        config=sqa_config,
    )


def try_load_generation_strategy(
    experiment_name: str,
    decoder: Decoder,
    experiment: Experiment | None = None,
    reduced_state: bool = False,
) -> GenerationStrategy | None:
    """Load generation strategy by experiment name, if it exists."""
    try:
        start_time = time.time()
        generation_strategy = _load_generation_strategy_by_experiment_name(
            experiment_name=experiment_name,
            decoder=decoder,
            experiment=experiment,
            reduced_state=reduced_state,
        )
        logger.info(
            f"Loaded generation strategy for experiment {experiment_name} in "
            f"{_round_floats_for_logging(time.time() - start_time)} seconds."
        )
    except ObjectNotFoundError:
        logger.info(
            "There is no generation strategy associated with experiment "
            f"{experiment_name}."
        )
        return None
    return generation_strategy
