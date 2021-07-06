#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

from ax.core.runner import Runner
from ax.utils.common.testutils import TestCase


class RunnerTest(TestCase):
    def testAbstractRunner(self):
        class DummyRunner(Runner):
            def run(self, trial):
                pass

        runner = DummyRunner()
        self.assertFalse(runner.staging_required)
        with self.assertRaises(NotImplementedError):
            runner.stop(trial=mock.Mock(), reason="")
        runner_clone = runner.clone()
        self.assertIsInstance(runner_clone, DummyRunner)
