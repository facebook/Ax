#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from logging import INFO
from typing import List, Optional, Tuple, Type

from ax.core.base_trial import BaseTrial
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.utils.common.executils import retry_on_exception
from ax.utils.common.logger import _round_floats_for_logging, get_logger
from ax.utils.common.typeutils import not_none


RETRY_EXCEPTION_TYPES: Tuple[Type[Exception], ...] = ()

try:  # We don't require SQLAlchemy by default.
    from ax.storage.sqa_store.db import init_engine_and_session_factory
    from ax.storage.sqa_store.load import (
        _get_experiment_id,
        _get_generation_strategy_id,
        _load_experiment,
        _load_generation_strategy_by_experiment_name,
    )
    from ax.storage.sqa_store.save import (
        _save_or_update_trials,
        _save_experiment,
        _save_generation_strategy,
        _update_generation_strategy,
        update_properties_on_experiment,
    )
    from ax.storage.sqa_store.structs import DBSettings
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.orm.exc import StaleDataError

    # We retry on `OperationalError` if saving to DB.
    RETRY_EXCEPTION_TYPES = (OperationalError, StaleDataError)
except ModuleNotFoundError:  # pragma: no cover
    DBSettings = None


STORAGE_MINI_BATCH_SIZE = 50

logger = get_logger(__name__)


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
                "structs.DBSettings. To use `DBSettings`, you will need SQLAlchemy "
                "installed in your environment (can be installed through pip)."
            )
        self._db_settings = db_settings
        self._suppress_all_errors = suppress_all_errors
        if self.db_settings_set:
            init_engine_and_session_factory(
                creator=self.db_settings.creator, url=self.db_settings.url
            )
        logger.setLevel(logging_level)

    @property
    def db_settings_set(self) -> bool:
        """Whether non-None DB settings are set on this instance."""
        return self._db_settings is not None

    @property
    def db_settings(self) -> DBSettings:
        """DB settings set on this instance; guaranteed to be non-None."""
        if self._db_settings is None:
            raise ValueError("No DB settings are set on this instance.")
        return not_none(self._db_settings)

    def _get_experiment_and_generation_strategy_db_id(
        self, experiment_name: str
    ) -> Tuple[Optional[int], Optional[int]]:
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
        gs_id = _get_generation_strategy_id(
            experiment_name=experiment_name, decoder=self.db_settings.decoder
        )
        return exp_id, gs_id

    def _maybe_save_experiment_and_generation_strategy(
        self, experiment: Experiment, generation_strategy: GenerationStrategy
    ) -> Tuple[bool, bool]:
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
            exp_name = not_none(experiment.name)
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
    ) -> Tuple[Optional[Experiment], Optional[GenerationStrategy]]:
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
        )
        if not isinstance(experiment, Experiment) or experiment.is_simple_experiment:
            raise ValueError("Service API only supports `Experiment`.")
        logger.info(
            f"Loaded experiment {experiment_name} in "
            f"{_round_floats_for_logging(time.time() - start_time)} seconds."
        )

        try:
            start_time = time.time()
            generation_strategy = _load_generation_strategy_by_experiment_name(
                experiment_name=experiment_name,
                decoder=self.db_settings.decoder,
                experiment=experiment,
                reduced_state=reduced_state,
            )
            logger.info(
                f"Loaded generation strategy for experiment {experiment_name} in "
                f"{_round_floats_for_logging(time.time() - start_time)} seconds."
            )
        except ValueError as err:
            if "does not have a generation strategy" in str(err):
                return experiment, None
            raise  # `ValueError` here could signify more than just absence of GS.
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
        trials: List[BaseTrial],
        generation_strategy: GenerationStrategy,
        new_generator_runs: List[GeneratorRun],
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
            experiment=experiment, trials=trials
        )
        logger.debug(
            "Updating generation strategy in DB with "
            f"{len(new_generator_runs)} generator runs."
        )
        self._update_generation_strategy_in_db_if_possible(
            generation_strategy=generation_strategy,
            new_generator_runs=new_generator_runs,
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
        trials: List[BaseTrial],
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
            )
            return True
        return False

    def _save_generation_strategy_to_db_if_possible(
        self, generation_strategy: GenerationStrategy, suppress_all_errors: bool = False
    ) -> bool:
        """Saves given generation strategy if DB settings are set on this
        `WithDBSettingsBase` instance.

        Args:
            generation_strategy: Generation strategy to save in DB.

        Returns:
            bool: Whether the generation strategy was saved.
        """
        if self.db_settings_set:
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
        generation_strategy: GenerationStrategy,
        new_generator_runs: List[GeneratorRun],
    ) -> bool:
        """Updates the given generation strategy with new generator runs (and with
        new current generation step if applicable) if DB settings are set
        on this `WithDBSettingsBase` instance.

        Args:
            generation_strategy: Generation strategy to update in DB.
            new_generator_runs: New generator runs of this generation strategy
                since its last save.

        Returns:
            bool: Whether the experiment was saved.
        """
        if self.db_settings_set:
            _update_generation_strategy_in_db_if_possible(
                generation_strategy=generation_strategy,
                new_generator_runs=new_generator_runs,
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
    suppress_all_errors: bool,
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
    trials: List[BaseTrial],
    encoder: Encoder,
    decoder: Decoder,
    suppress_all_errors: bool,
) -> None:
    start_time = time.time()
    _save_or_update_trials(
        experiment=experiment,
        trials=trials,
        encoder=encoder,
        decoder=decoder,
        batch_size=STORAGE_MINI_BATCH_SIZE,
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
    suppress_all_errors: bool,
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
    new_generator_runs: List[GeneratorRun],
    encoder: Encoder,
    decoder: Decoder,
    suppress_all_errors: bool,
) -> None:
    start_time = time.time()
    _update_generation_strategy(
        generation_strategy=generation_strategy,
        generator_runs=new_generator_runs,
        encoder=encoder,
        decoder=decoder,
        batch_size=STORAGE_MINI_BATCH_SIZE,
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
def _update_experiment_properties_in_db(
    experiment_with_updated_properties: Experiment,
    sqa_config: SQAConfig,
    suppress_all_errors: bool,
) -> None:
    update_properties_on_experiment(
        experiment_with_updated_properties=experiment_with_updated_properties,
        config=sqa_config,
    )
