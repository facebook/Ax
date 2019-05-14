#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from ax.service.utils.storage import load_experiment, save_experiment
from ax.storage.sqa_store.db import init_test_engine_and_session_factory
from ax.storage.sqa_store.structs import DBSettings
from ax.utils.common.testutils import TestCase
from ax.utils.testing.fake import get_experiment, get_simple_experiment


class TestStorageUtils(TestCase):
    """Tests saving/loading functionality of AxClient."""

    def test_save_load_experiment(self):
        exp = get_experiment()
        init_test_engine_and_session_factory()
        db_settings = DBSettings(url="sqlite://")
        save_experiment(exp, db_settings)
        load_experiment(exp.name, db_settings)

        simple_experiment = get_simple_experiment()
        save_experiment(simple_experiment, db_settings)
        with self.assertRaisesRegex(ValueError, "Service API only"):
            load_experiment(simple_experiment.name, db_settings)
