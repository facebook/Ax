#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from tempfile import NamedTemporaryFile
from unittest.mock import patch

from ax.utils.common.logger import build_file_handler, get_logger
from ax.utils.common.testutils import TestCase


BASE_LOGGER_NAME = f"ax.{__name__}"


class LoggerTest(TestCase):
    def setUp(self):
        self.warning_string = "Test warning"

    def testLogger(self):
        logger = get_logger(BASE_LOGGER_NAME + ".testLogger")
        # Verify it doesn't crash
        logger.warning(self.warning_string)
        # Patch it, verify we actually called it
        patcher = patch.object(logger, "warning")
        mock_warning = patcher.start()
        logger.warning(self.warning_string)
        mock_warning.assert_called_once_with(self.warning_string)
        # Need to stop patcher, else in some environments (like pytest)
        # the mock will leak into other tests, since it's getting set
        # onto the python logger directly.
        patcher.stop()

    def testLoggerWithFile(self):
        with NamedTemporaryFile() as tf:
            logger = get_logger(BASE_LOGGER_NAME + ".testLoggerWithFile")
            logger.addHandler(build_file_handler(tf.name))
            logger.info(self.warning_string)
            output = str(tf.read())
            self.assertIn(BASE_LOGGER_NAME, output)
            self.assertIn(self.warning_string, output)
            tf.close()

    def testLoggerOutputNameWithFile(self):
        with NamedTemporaryFile() as tf:
            logger = get_logger(BASE_LOGGER_NAME + ".testLoggerOutputNameWithFile")
            logger.addHandler(build_file_handler(tf.name))
            logger = logging.LoggerAdapter(logger, {"output_name": "my_output_name"})
            logger.warning(self.warning_string)
            output = str(tf.read())
            self.assertIn("my_output_name", output)
            self.assertIn(self.warning_string, output)
            tf.close()
