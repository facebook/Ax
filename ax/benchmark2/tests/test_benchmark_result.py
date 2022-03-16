# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.benchmark2.benchmark_result import BenchmarkResult
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment


class TestBenchmarkResult(TestCase):
    def test_benchmark_result(self):
        exp = get_experiment()
        result = BenchmarkResult(experiment=exp)

        self.assertEqual(result.experiment, exp)
