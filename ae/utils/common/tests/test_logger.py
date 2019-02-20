#!/usr/bin/env python3

from unittest import mock

from ae.lazarus.ae.utils.common.logger import get_logger
from ae.lazarus.ae.utils.common.testutils import TestCase


class LoggerTest(TestCase):
    def setUp(self):
        self.warning_string = "Test warning"

    def testLogger(self):
        logger = get_logger(__name__)
        logger.warning = mock.MagicMock(name="warning")
        logger.warning(self.warning_string)
        logger.warning.assert_called_once_with(self.warning_string)
