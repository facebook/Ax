#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

from ax.core.base_trial import BaseTrial
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.utils.common.executils import retry_on_exception
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none


RETRY_EXCEPTION_TYPES: Tuple[Type[Exception], ...] = ()
try:  # We don't require SQLAlchemy by default.
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.orm.exc import StaleDataError
    from ax.storage.sqa_store.structs import DBSettings
    from ax.service.utils.storage import (  # noqa F401
        load_experiment_and_generation_strategy,
        save_experiment,
        save_generation_strategy,
        save_new_trial,
        save_new_trials,
        save_updated_trial,
        save_updated_trials,
        update_generation_strategy,
    )

    # We retry on `OperationalError` if saving to DB.
    RETRY_EXCEPTION_TYPES = (OperationalError, StaleDataError)
except ModuleNotFoundError:  # pragma: no cover
    DBSettings = None


logger = get_logger(__name__)


class WithDBSettingsBase:
    """Helper class providing methods for saving changes made to an experiment
    if `db_settings` property is set to a non-None value on the instance.
    """

    _db_settings: Optional[DBSettings] = None

    def __init__(self, db_settings: Optional[DBSettings] = None) -> None:
        if db_settings and (not DBSettings or not isinstance(db_settings, DBSettings)):
            raise ValueError(
                "`db_settings` argument should be of type ax.storage.sqa_store."
                "structs.DBSettings. To use `DBSettings`, you will need SQLAlchemy "
                "installed in your environment (can be installed through pip)."
            )
        self._db_settings = db_settings

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

    def maybe_save_experiment_and_generation_strategy(
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
            # TODO: Check existance without full load.
            existing_exp, existing_gs = self._load_experiment_and_generation_strategy(
                experiment_name=exp_name
            )
            if not existing_exp:
                logger.info(f"Experiment {exp_name} is not yet in DB, storing it.")
                self._save_experiment_to_db_if_possible(experiment=experiment)
                saved_exp = True
            if not existing_gs or generation_strategy._db_id is None:
                # There is no GS associated with experiment or the generation
                # strategy passed in is different from the one associated with
                # experiment currently.
                logger.info(
                    f"Generation strategy {generation_strategy.name} is not yet in DB, "
                    "storing it."
                )
                self._save_generation_strategy_to_db_if_possible(
                    generation_strategy=generation_strategy
                )
                saved_gs = True
        # TODO: Update experiment and GS if they already exist.
        return saved_exp, saved_gs

    def _load_experiment_and_generation_strategy(
        self, experiment_name: str
    ) -> Tuple[Optional[Experiment], Optional[GenerationStrategy]]:
        """Loads experiment and its corresponding generation strategy from database
        if DB settings are set on this `WithDBSettingsBase` instance.

        Args:
            experiment_name: Name of the experiment to load, used as unique
                identifier by which to find the experiment.

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
        try:
            return load_experiment_and_generation_strategy(
                experiment_name=experiment_name, db_settings=self.db_settings
            )
        except ValueError:
            return None, None

    @retry_on_exception(
        retries=3,
        default_return_on_suppression=False,
        exception_types=RETRY_EXCEPTION_TYPES,
    )
    def _save_experiment_to_db_if_possible(
        self, experiment: Experiment, suppress_all_errors: bool = False
    ) -> bool:
        """Saves attached experiment and generation strategy if DB settings are
        set on this `WithDBSettingsBase` instance.

        Args:
            experiment: Experiment to save new trials in DB.
            suppress_all_errors: Flag for `retry_on_exception` that makes
                the decorator suppress the thrown exception even if it
                occurred in all the retries (exception is still logged).

        Returns:
            bool: Whether the experiment was saved.
        """
        if self.db_settings_set:
            save_experiment(experiment=experiment, db_settings=self.db_settings)
            return True
        return False

    @retry_on_exception(
        retries=3,
        default_return_on_suppression=False,
        exception_types=RETRY_EXCEPTION_TYPES,
    )
    def _save_new_trial_to_db_if_possible(
        self,
        experiment: Experiment,
        trial: BaseTrial,
        suppress_all_errors: bool = False,
    ) -> bool:
        """Saves new trial on given experiment if DB settings are set on this
        `WithDBSettingsBase` instance.

        Args:
            experiment: Experiment, on which to save new trial in DB.
            trials: Newly added trial to save.
            suppress_all_errors: Flag for `retry_on_exception` that makes
                the decorator suppress the thrown exception even if it
                occurred in all the retries (exception is still logged).

        Returns:
            bool: Whether the trial was saved.
        """
        if self.db_settings_set:
            save_new_trial(
                experiment=experiment, trial=trial, db_settings=self.db_settings
            )
            return True
        return False

    @retry_on_exception(
        retries=3,
        default_return_on_suppression=False,
        exception_types=RETRY_EXCEPTION_TYPES,
    )
    def _save_new_trials_to_db_if_possible(
        self,
        experiment: Experiment,
        trials: List[BaseTrial],
        suppress_all_errors: bool = False,
    ) -> bool:
        """Saves new trials on given experiment if DB settings are set on this
        `WithDBSettingsBase` instance.

        Args:
            experiment: Experiment, on which to save new trials in DB.
            trials: Newly added trials to save.
            suppress_all_errors: Flag for `retry_on_exception` that makes
                the decorator suppress the thrown exception even if it
                occurred in all the retries (exception is still logged).

        Returns:
            bool: Whether the trials were saved.
        """
        if self.db_settings_set:
            save_new_trials(
                experiment=experiment, trials=trials, db_settings=self.db_settings
            )
            return True
        return False

    @retry_on_exception(
        retries=3,
        default_return_on_suppression=False,
        exception_types=RETRY_EXCEPTION_TYPES,
    )
    def _save_updated_trial_to_db_if_possible(
        self,
        experiment: Experiment,
        trial: BaseTrial,
        suppress_all_errors: bool = False,
    ) -> bool:
        """Saves updated trials on given experiment if DB settings are set on this
        `WithDBSettingsBase` instance.

        Args:
            experiment: Experiment, on which to save updated trials in DB.
            trial: Newly updated trial to save.
            suppress_all_errors: Flag for `retry_on_exception` that makes
                the decorator suppress the thrown exception even if it
                occurred in all the retries (exception is still logged).

        Returns:
            bool: Whether the trial was saved.
        """
        if self.db_settings_set:
            save_updated_trial(
                experiment=experiment, trial=trial, db_settings=self.db_settings
            )
            return True
        return False

    @retry_on_exception(
        retries=3,
        default_return_on_suppression=False,
        exception_types=RETRY_EXCEPTION_TYPES,
    )
    def _save_updated_trials_to_db_if_possible(
        self,
        experiment: Experiment,
        trials: List[BaseTrial],
        suppress_all_errors: bool = False,
    ) -> bool:
        """Saves updated trials on given experiment if DB settings are set on this
        `WithDBSettingsBase` instance.

        Args:
            experiment: Experiment, on which to save updated trials in DB.
            trials: Newly updated trials to save.
            suppress_all_errors: Flag for `retry_on_exception` that makes
                the decorator suppress the thrown exception even if it
                occurred in all the retries (exception is still logged).

        Returns:
            bool: Whether the trials were saved.
        """
        if self.db_settings_set:
            save_updated_trials(
                experiment=experiment, trials=trials, db_settings=self.db_settings
            )
            return True
        return False

    @retry_on_exception(
        retries=3,
        default_return_on_suppression=False,
        exception_types=RETRY_EXCEPTION_TYPES,
    )
    def _save_generation_strategy_to_db_if_possible(
        self, generation_strategy: GenerationStrategy, suppress_all_errors: bool = False
    ) -> bool:
        """Saves given generation strategy if DB settings are set on this
        `WithDBSettingsBase` instance.

        Args:
            generation_strategy: Generation strategy to save in DB.
            suppress_all_errors: Flag for `retry_on_exception` that makes
                the decorator suppress the thrown exception even if it
                occurred in all the retries (exception is still logged).

        Returns:
            bool: Whether the experiment was saved.
        """
        if self.db_settings_set:
            save_generation_strategy(
                generation_strategy=generation_strategy, db_settings=self.db_settings
            )
            return True
        return False

    @retry_on_exception(
        retries=3,
        default_return_on_suppression=False,
        exception_types=RETRY_EXCEPTION_TYPES,
    )
    def _update_generation_strategy_in_db_if_possible(
        self,
        generation_strategy: GenerationStrategy,
        new_generator_runs: List[GeneratorRun],
        suppress_all_errors: bool = False,
    ) -> bool:
        """Updates the given generation strategy with new generator runs (and with
        new current generation step if applicable) if DB settings are set
        on this `WithDBSettingsBase` instance.

        Args:
            generation_strategy: Generation strategy to update in DB.
            new_generator_runs: New generator runs of this generation strategy
                since its last save.
            suppress_all_errors: Flag for `retry_on_exception` that makes
                the decorator suppress the thrown exception even if it
                occurred in all the retries (exception is still logged).
        Returns:
            bool: Whether the experiment was saved.
        """
        if self.db_settings_set:
            update_generation_strategy(
                generation_strategy=generation_strategy,
                generator_runs=new_generator_runs,
                db_settings=self.db_settings,
            )
            return True
        return False
