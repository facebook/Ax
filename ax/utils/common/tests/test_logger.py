#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

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
