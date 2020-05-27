#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Type

from ax.core.base_trial import BaseTrial
from ax.core.experiment import Experiment
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.utils.common.executils import retry_on_exception
from ax.utils.common.typeutils import not_none


RETRY_EXCEPTION_TYPES: Tuple[Type[Exception], ...] = ()
try:  # We don't require SQLAlchemy by default.
    from sqlalchemy.exc import OperationalError
    from ax.storage.sqa_store.structs import DBSettings
    from ax.service.utils.storage import (  # noqa F401
        load_experiment_and_generation_strategy,
        save_experiment,
        save_generation_strategy,
        save_new_trial,
        save_updated_trial,
    )

    # We retry on `OperationalError` if saving to DB.
    RETRY_EXCEPTION_TYPES = (OperationalError,)
except ModuleNotFoundError:  # pragma: no cover
    DBSettings = None


class WithDBSettingsBase:
    """Helper class providing methods for saving changes made to an experiment
    if `db_settings` property is set to a non-None value on the instance.
    """

    _db_settings: Optional[DBSettings] = None

    def __init__(self, db_settings: Optional[DBSettings] = None) -> None:
        print(f"In WithDBSettings, db settings: {db_settings}")
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

    def _load_experiment_and_generation_strategy(
        self, experiment_name: str
    ) -> Tuple[Optional[Experiment], Optional[GenerationStrategy]]:
        """Loads experiment and its corresponding generation strategy from database
        if DB settings are set on this `WithDBSettingsBase` instance.

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
        """Saves given trial and generation strategy if DB settings are
        set on this `WithDBSettingsBase` instance.

        Returns:
            bool: Whether the experiment was saved.
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
    def _save_updated_trial_to_db_if_possible(
        self,
        experiment: Experiment,
        trial: BaseTrial,
        suppress_all_errors: bool = False,
    ) -> bool:
        """Saves attached experiment and generation strategy if DB settings are
        set on this `WithDBSettingsBase` instance.

        Returns:
            bool: Whether the experiment was saved.
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
    def _save_generation_strategy_to_db_if_possible(
        self, generation_strategy: GenerationStrategy, suppress_all_errors: bool = False
    ) -> bool:
        """Saves attached experiment and generation strategy if DB settings are
        set on this `WithDBSettingsBase` instance.

        Returns:
            bool: Whether the experiment was saved.
        """
        if self.db_settings_set:
            save_generation_strategy(
                generation_strategy=generation_strategy, db_settings=self.db_settings
            )
            return True
        return False
