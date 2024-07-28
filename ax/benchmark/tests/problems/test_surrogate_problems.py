# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import numpy as np
from ax.benchmark.benchmark import compute_score_trace
from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.utils.common.testutils import TestCase
from ax.utils.testing.benchmark_stubs import get_moo_surrogate, get_soo_surrogate


class TestSurrogateProblems(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # print max output so errors in 'repr' can be fully shown
        self.maxDiff = None

    def test_conforms_to_api(self) -> None:
        sbp = get_soo_surrogate()
        self.assertIsInstance(sbp, BenchmarkProblem)

        mbp = get_moo_surrogate()
        self.assertIsInstance(mbp, BenchmarkProblem)

    def test_repr(self) -> None:

        sbp = get_soo_surrogate()

        expected_repr = (
            "SOOSurrogateBenchmarkProblem(name='test', "
            "optimization_config=OptimizationConfig(objective=Objective(metric_name="
            '"branin", '
            "minimize=False), "
            "outcome_constraints=[]), num_trials=6, "
            "observe_noise_stds=True, has_ground_truth=True, "
            "tracking_metrics=[], optimal_value=0.0, is_noiseless=True)"
        )
        self.assertEqual(repr(sbp), expected_repr)

    def test_compute_score_trace(self) -> None:
        soo_problem = get_soo_surrogate()
        score_trace = compute_score_trace(
            np.arange(10),
            num_baseline_trials=5,
            problem=soo_problem,
        )
        self.assertTrue(np.isfinite(score_trace).all())

        moo_problem = get_moo_surrogate()

        score_trace = compute_score_trace(
            np.arange(10), num_baseline_trials=5, problem=moo_problem
        )
        self.assertTrue(np.isfinite(score_trace).all())
