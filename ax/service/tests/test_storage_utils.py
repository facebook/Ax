#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.base_trial import TrialStatus
from ax.service.utils.storage import (
    load_experiment,
    load_experiment_and_generation_strategy,
    save_experiment,
    save_generation_strategy,
    save_new_trial,
    save_updated_trial,
)
from ax.storage.sqa_store.db import init_test_engine_and_session_factory
from ax.storage.sqa_store.structs import DBSettings
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment, get_simple_experiment
from ax.utils.testing.modeling_stubs import get_generation_strategy


class TestStorageUtils(TestCase):
    """Tests saving/loading functionality of AxClient."""

    def setUp(self):
        self.exp = get_experiment()
        init_test_engine_and_session_factory(force_init=True)
        self.db_settings = DBSettings(url="sqlite://")
        save_experiment(self.exp, self.db_settings)

    def test_save_load_experiment(self):
        load_experiment(self.exp.name, self.db_settings)
        simple_experiment = get_simple_experiment()
        save_experiment(simple_experiment, self.db_settings)
        with self.assertRaisesRegex(ValueError, "Service API only"):
            load_experiment(simple_experiment.name, self.db_settings)

    def test_save_load_experiment_and_generation_strategy(self):
        exp, gs = load_experiment_and_generation_strategy(
            self.exp.name, self.db_settings
        )
        self.assertIsNone(gs)
        gs = get_generation_strategy()
        gs._experiment = self.exp
        save_experiment(self.exp, self.db_settings)
        save_generation_strategy(gs, self.db_settings)
        exp, gs = load_experiment_and_generation_strategy(
            self.exp.name, self.db_settings
        )
        self.assertIsNotNone(gs)

    def test_save_load_new_trial(self):
        exp = load_experiment(self.exp.name, self.db_settings)
        trial = exp.new_trial()
        save_new_trial(exp, trial, self.db_settings)
        exp = load_experiment(self.exp.name, self.db_settings)
        self.assertEqual(len(exp.trials), 1)
        self.assertEqual(exp.trials[0].status, TrialStatus.CANDIDATE)

        trial.mark_running(True)
        save_updated_trial(exp, trial, self.db_settings)
        exp = load_experiment(self.exp.name, self.db_settings)
        self.assertEqual(exp.trials[0].status, TrialStatus.RUNNING)
