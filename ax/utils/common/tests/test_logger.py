#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tempfile import NamedTemporaryFile
from unittest import mock

from ax.utils.common.logger import get_logger
from ax.utils.common.testutils import TestCase


class LoggerTest(TestCase):
    def setUp(self):
        self.warning_string = "Test warning"

    def testLogger(self):
        logger = get_logger(__name__)
        logger.warning = mock.MagicMock(name="warning")
        logger.warning(self.warning_string)
        logger.warning.assert_called_once_with(self.warning_string)

    def testLoggerWithFile(self):
        with NamedTemporaryFile(mode="w", delete=False) as tf:
            filepath = tf.name
            tf.close()
        logger = get_logger(name=__name__, filepath=filepath)
        logger.warning(self.warning_string)
        with open(filepath, "r") as log_file:
            contents = log_file.read()
            self.assertIsInstance(contents, str, msg=f"Type of contents: {type(contents)}.")
            self.assertIn(self.warning_string, contents, msg=f"Contents: {contents}.")
