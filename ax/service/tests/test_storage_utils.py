#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from ax.service.utils.storage import (
    load_experiment,
    load_experiment_and_generation_strategy,
    save_experiment,
    save_experiment_and_generation_strategy,
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
        save_experiment_and_generation_strategy(self.exp, gs, self.db_settings)
        exp, gs = load_experiment_and_generation_strategy(
            self.exp.name, self.db_settings
        )
        self.assertIsNotNone(gs)
