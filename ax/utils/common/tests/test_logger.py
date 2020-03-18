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
        with NamedTemporaryFile(mode="w") as tf:
            logger = get_logger(name=__name__, filepath=tf.name)
            logger.warning(self.warning_string)
            with open(tf.name) as log_file:
                line = log_file.readline()
                self.assertIn(self.warning_string, line)
            tf.close()
