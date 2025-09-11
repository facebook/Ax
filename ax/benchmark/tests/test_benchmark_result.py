# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.benchmark.benchmark_result import BenchmarkResult
from ax.utils.common.testutils import TestCase

from ax.utils.testing.core_stubs import get_experiment


class TestBenchmarkResult(TestCase):
    def test_benchmark_result_invalid_inputs(self) -> None:
        """
        Test that a BenchmarkResult cannot be specified with both an `experiment`
        and an `experiment_storage_id`.
        """
        with self.assertRaisesRegex(ValueError, "Cannot specify both an `experiment` "):
            BenchmarkResult(
                name="name",
                seed=0,
                inference_trace=[],
                oracle_trace=[],
                optimization_trace=[],
                score_trace=[],
                cost_trace=[],
                fit_time=0.0,
                gen_time=0.0,
                experiment=get_experiment(),
                experiment_storage_id="experiment_storage_id",
            )

        with self.assertRaisesRegex(
            ValueError, "Must provide an `experiment` or `experiment_storage_id`"
        ):
            BenchmarkResult(
                name="name",
                seed=0,
                inference_trace=[],
                oracle_trace=[],
                optimization_trace=[],
                score_trace=[],
                cost_trace=[],
                fit_time=0.0,
                gen_time=0.0,
            )
