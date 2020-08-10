#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.service.utils.with_db_settings_base import WithDBSettingsBase
from ax.storage.sqa_store.db import init_test_engine_and_session_factory
from ax.storage.sqa_store.save import _save_experiment, _save_generation_strategy
from ax.storage.sqa_store.structs import DBSettings
from ax.utils.common.testutils import TestCase
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
            self.experiment, encoder=self.with_db_settings.db_settings.encoder
        )
        _save_generation_strategy(
            generation_strategy=self.generation_strategy,
            encoder=self.with_db_settings.db_settings.encoder,
        )

    def test_get_experiment_and_generation_strategy_db_id(self):

        (
            exp_id,
            gen_id,
        ) = self.with_db_settings._get_experiment_and_generation_strategy_db_id(
            self.experiment.name
        )
        self.assertIsNotNone(exp_id)
        self.assertIsNotNone(gen_id)
