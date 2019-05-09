#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from unittest.mock import MagicMock, Mock

from ax.service.utils.storage import initialize_db, load_experiment, save_experiment

# from ax.service.utils.storage import *
from ax.storage.sqa_store.db import init_test_engine_and_session_factory
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.storage.sqa_store.structs import DBSettings
from ax.utils.common.testutils import TestCase
from ax.utils.testing.fake import get_experiment, get_simple_experiment


def MockDBAPI():
    connection = Mock()

    def connect(*args, **kwargs):
        return connection

    return MagicMock(connect=Mock(side_effect=connect))


class TestStorageUtils(TestCase):
    """Tests saving/loading functionality of AxClient."""

    def test_initialize_db(self):
        db_settings = DBSettings(
            encoder=Encoder(config=SQAConfig()),
            decoder=Decoder(config=SQAConfig()),
            creator=lambda: MockDBAPI().connect(),
        )
        initialize_db(db_settings)

    def test_save_load_experiment(self):
        exp = get_experiment()
        init_test_engine_and_session_factory(force_init=True)
        db_settings = DBSettings(
            encoder=Encoder(config=SQAConfig()),
            decoder=Decoder(config=SQAConfig()),
            creator=None,
        )
        save_experiment(exp, db_settings)
        load_experiment(exp.name, db_settings)

        simple_experiment = get_simple_experiment()
        save_experiment(simple_experiment, db_settings)
        with self.assertRaisesRegex(ValueError, "Service API only"):
            load_experiment(simple_experiment.name, db_settings)
