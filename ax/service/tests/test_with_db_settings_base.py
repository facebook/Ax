#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import string
from typing import Tuple
from unittest.mock import patch

from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.service.utils.with_db_settings_base import WithDBSettingsBase
from ax.storage.sqa_store.db import init_test_engine_and_session_factory
from ax.storage.sqa_store.load import (
    _load_experiment,
    _load_generation_strategy_by_experiment_name,
)
from ax.storage.sqa_store.save import (
    _save_experiment,
    _save_generation_strategy,
    _save_or_update_trials,
)
from ax.storage.sqa_store.structs import DBSettings
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment,
    get_generator_run,
    get_simple_experiment,
)
from ax.utils.testing.modeling_stubs import get_generation_strategy


class TestWithDBSettingsBase(TestCase):
    """Tests saving/loading functionality of WithDBSettingsBase class."""

    def setUp(self):
        self.generation_strategy = get_generation_strategy(with_experiment=True)
        self.experiment = self.generation_strategy.experiment

        init_test_engine_and_session_factory(force_init=True)
        self.with_db_settings = WithDBSettingsBase(
            db_settings=DBSettings(url="sqlite://")
        )
        _save_experiment(
            self.experiment,
            encoder=self.with_db_settings.db_settings.encoder,
            decoder=self.with_db_settings.db_settings.decoder,
        )
        _save_generation_strategy(
            generation_strategy=self.generation_strategy,
            encoder=self.with_db_settings.db_settings.encoder,
            decoder=self.with_db_settings.db_settings.decoder,
        )

    def get_random_experiment(self) -> Experiment:
        """Get an Experiment instance with random name."""

        experiment = get_experiment()
        experiment_name = "".join(random.choice(string.ascii_letters) for i in range(8))
        experiment.name = experiment_name
        return experiment

    def get_random_generation_strategy(self) -> GenerationStrategy:
        """Get an GenerationStrategy instance with random name."""

        generation_strategy = get_generation_strategy()
        gs_name = "".join(random.choice(string.ascii_letters) for i in range(8))
        generation_strategy._name = gs_name
        return generation_strategy

    def init_experiment_and_generation_strategy(
        self, save_experiment: bool = True, save_generation_strategy: bool = True
    ) -> Tuple[Experiment, GenerationStrategy]:
        """Generate a random Experiment and associated generation_strategy"""

        generation_strategy = self.get_random_generation_strategy()
        experiment = self.get_random_experiment()
        generation_strategy.experiment = experiment

        if save_experiment:
            _save_experiment(
                experiment,
                encoder=self.with_db_settings.db_settings.encoder,
                decoder=self.with_db_settings.db_settings.decoder,
            )
        if save_generation_strategy:
            _save_generation_strategy(
                generation_strategy=generation_strategy,
                encoder=self.with_db_settings.db_settings.encoder,
                decoder=self.with_db_settings.db_settings.decoder,
            )
        return experiment, generation_strategy

    def test_get_experiment_and_generation_strategy_db_id(self):

        (
            exp_id,
            gen_id,
        ) = self.with_db_settings._get_experiment_and_generation_strategy_db_id(
            self.experiment.name
        )
        self.assertIsNotNone(exp_id)
        self.assertIsNotNone(gen_id)

    def test_save_experiment(self):
        experiment = self.get_random_experiment()
        saved = self.with_db_settings._save_experiment_to_db_if_possible(experiment)
        self.assertTrue(saved)
        loaded_experiment = _load_experiment(
            experiment.name, decoder=self.with_db_settings.db_settings.decoder
        )
        self.assertIsNotNone(loaded_experiment)
        self.assertEqual(experiment, loaded_experiment)

    def test_save_generation_strategy(self):
        experiment, generation_strategy = self.init_experiment_and_generation_strategy(
            save_generation_strategy=False
        )
        saved = self.with_db_settings._save_generation_strategy_to_db_if_possible(
            generation_strategy
        )
        self.assertTrue(saved)
        loaded_gs = _load_generation_strategy_by_experiment_name(
            experiment_name=experiment.name,
            decoder=self.with_db_settings.db_settings.decoder,
        )
        self.assertIsNotNone(loaded_gs)
        self.assertEqual(loaded_gs.name, generation_strategy.name)

    def test_save_load_experiment_and_generation_strategy(self):
        experiment, generation_strategy = self.init_experiment_and_generation_strategy(
            save_generation_strategy=False
        )
        db_exp, db_gs = self.with_db_settings._load_experiment_and_generation_strategy(
            experiment.name
        )
        self.assertIsNotNone(db_exp)
        self.assertIsNone(db_gs)

        experiment, generation_strategy = self.init_experiment_and_generation_strategy(
            save_experiment=False, save_generation_strategy=False
        )
        (
            exp_saved,
            gs_saved,
        ) = self.with_db_settings._maybe_save_experiment_and_generation_strategy(
            experiment, generation_strategy
        )
        self.assertTrue(exp_saved)
        self.assertTrue(gs_saved)

        db_exp, db_gs = self.with_db_settings._load_experiment_and_generation_strategy(
            experiment.name
        )
        self.assertIsNotNone(db_exp)
        self.assertEqual(db_exp, experiment)
        self.assertIsNotNone(db_gs)
        self.assertEqual(db_gs.name, generation_strategy.name)

        simple_experiment = get_simple_experiment()
        _save_experiment(
            simple_experiment,
            encoder=self.with_db_settings.db_settings.encoder,
            decoder=self.with_db_settings.db_settings.decoder,
        )
        with self.assertRaisesRegex(ValueError, "Service API only"):
            self.with_db_settings._load_experiment_and_generation_strategy(
                simple_experiment.name
            )

    def test_update_generation_strategy(self):
        _, generation_strategy = self.init_experiment_and_generation_strategy()

        generator_run = get_generator_run()
        self.assertIsNone(generator_run.db_id)
        updated = self.with_db_settings._update_generation_strategy_in_db_if_possible(
            generation_strategy, [generator_run]
        )
        self.assertTrue(updated)
        self.assertIsNotNone(generator_run.db_id)

    @patch(f"{WithDBSettingsBase.__module__}.STORAGE_MINI_BATCH_SIZE", 2)
    def test_update_generation_strategy_mini_batches(self):
        _, generation_strategy = self.init_experiment_and_generation_strategy()

        # Check with 1 GR.
        generator_run = get_generator_run()
        self.assertIsNone(generator_run.db_id)
        updated = self.with_db_settings._update_generation_strategy_in_db_if_possible(
            generation_strategy, [generator_run]
        )
        self.assertTrue(updated)
        self.assertIsNotNone(generator_run.db_id)

        # Check with multiple GRs, where their number % mini batch size is not 0.
        grs = [generator_run.clone() for _ in range(5)]
        for gr in grs:
            self.assertIsNone(gr._db_id)
        updated = self.with_db_settings._update_generation_strategy_in_db_if_possible(
            generation_strategy, grs
        )
        self.assertTrue(updated)
        for gr in grs:
            self.assertIsNotNone(gr.db_id)

    def test_save_new_trial(self):
        experiment, _ = self.init_experiment_and_generation_strategy(
            save_generation_strategy=False
        )

        exp = _load_experiment(
            experiment.name, decoder=self.with_db_settings.db_settings.decoder
        )
        trial = exp.new_trial()
        saved = self.with_db_settings._save_or_update_trial_in_db_if_possible(
            exp, trial
        )
        self.assertTrue(saved)
        exp = _load_experiment(
            experiment.name, decoder=self.with_db_settings.db_settings.decoder
        )
        self.assertEqual(len(exp.trials), 1)
        self.assertEqual(exp.trials[0].status, TrialStatus.CANDIDATE)

    def test_save_updated_trial(self):
        experiment, _ = self.init_experiment_and_generation_strategy(
            save_generation_strategy=False
        )

        exp = _load_experiment(
            experiment.name, decoder=self.with_db_settings.db_settings.decoder
        )
        trial = exp.new_trial()
        _save_or_update_trials(
            experiment=experiment,
            trials=[trial],
            encoder=self.with_db_settings.db_settings.encoder,
            decoder=self.with_db_settings.db_settings.decoder,
        )
        self.assertEqual(trial.status, TrialStatus.CANDIDATE)

        trial.mark_running(True)
        saved = self.with_db_settings._save_or_update_trial_in_db_if_possible(
            exp, trial
        )
        self.assertTrue(saved)
        exp = _load_experiment(
            experiment.name, decoder=self.with_db_settings.db_settings.decoder
        )
        self.assertEqual(len(exp.trials), 1)
        self.assertEqual(exp.trials[0].status, TrialStatus.RUNNING)

    @patch(f"{WithDBSettingsBase.__module__}.STORAGE_MINI_BATCH_SIZE", 2)
    def test_updated_trials_mini_batch(self):
        experiment, _ = self.init_experiment_and_generation_strategy(
            save_generation_strategy=False
        )

        # Check with 1 trial.
        trial = experiment.new_trial()
        self.assertIsNone(trial.db_id)
        self.with_db_settings._save_or_update_trials_in_db_if_possible(
            experiment=experiment,
            trials=[trial],
        )
        loaded_experiment = _load_experiment(
            experiment.name, decoder=self.with_db_settings.db_settings.decoder
        )
        self.assertEqual(
            loaded_experiment.trials.get(trial.index).status, TrialStatus.CANDIDATE
        )
        self.assertIsNotNone(trial.db_id)

        # Check with multiple trials, where their number % mini batch size is not 0.
        trials = [experiment.new_trial() for _ in range(5)]
        for t in trials:
            self.assertIsNone(t.db_id)

        trial.mark_running(no_runner_required=True)
        trials.append(trial)

        self.with_db_settings._save_or_update_trials_in_db_if_possible(
            experiment=experiment,
            trials=trials,
        )
        loaded_experiment = _load_experiment(
            experiment.name, decoder=self.with_db_settings.db_settings.decoder
        )
        # All trials except for the one we marked as running should be candidates.
        for t in trials:
            self.assertIsNotNone(t.db_id)
            if t.index != trial.index:
                self.assertEqual(t.status, TrialStatus.CANDIDATE)
            else:
                self.assertEqual(t.status, TrialStatus.RUNNING)

    def test_update_experiment_properties_in_db(self):
        experiment, _ = self.init_experiment_and_generation_strategy(
            save_generation_strategy=False
        )
        experiment._properties["test_property"] = True
        self.with_db_settings._update_experiment_properties_in_db(
            experiment_with_updated_properties=experiment
        )
        loaded_experiment = _load_experiment(
            experiment.name, decoder=self.with_db_settings.db_settings.decoder
        )
        self.assertEqual(loaded_experiment._properties, {"test_property": True})
