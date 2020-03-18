#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tempfile import NamedTemporaryFile
from unittest.mock import patch

from ax.utils.common.logger import get_logger
from ax.utils.common.testutils import TestCase


class LoggerTest(TestCase):
    def setUp(self):
        self.warning_string = "Test warning"

    def testLogger(self):
        logger = get_logger(__name__)
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
            logger = get_logger(__name__, tf.name)
            logger.warning(self.warning_string)
            self.assertIn(self.warning_string, str(tf.read()))
            tf.close()
