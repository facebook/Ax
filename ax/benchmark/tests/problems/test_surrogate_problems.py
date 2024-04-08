# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import numpy as np
from ax.benchmark.benchmark import compute_score_trace
from ax.benchmark.benchmark_problem import BenchmarkProblemProtocol
from ax.core.runner import Runner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.benchmark_stubs import get_moo_surrogate, get_soo_surrogate


class TestSurrogateProblems(TestCase):
    def test_conforms_to_protocol(self) -> None:
        sbp = get_soo_surrogate()
        self.assertIsInstance(sbp, BenchmarkProblemProtocol)

        mbp = get_moo_surrogate()
        self.assertIsInstance(mbp, BenchmarkProblemProtocol)

    def test_lazy_instantiation(self) -> None:

        # test instantiation from init
        sbp = get_soo_surrogate()
        # test __repr__ method

        expected_repr = (
            "SOOSurrogateBenchmarkProblem(name=test, "
            "optimization_config=OptimizationConfig(objective=Objective(metric_name="
            '"branin", '
            "minimize=False), "
            "outcome_constraints=[]), num_trials=6, is_noiseless=True, "
            "observe_noise_stds=True, noise_stds=0.0, tracking_metrics=[])"
        )
        self.assertEqual(repr(sbp), expected_repr)

        self.assertIsNone(sbp._runner)
        # sets runner
        self.assertIsInstance(sbp.runner, Runner)

        self.assertIsNotNone(sbp._runner)
        self.assertIsNotNone(sbp.runner)

        # repeat for MOO
        sbp = get_moo_surrogate()

        self.assertIsNone(sbp._runner)
        # sets runner
        self.assertIsInstance(sbp.runner, Runner)

        self.assertIsNotNone(sbp._runner)
        self.assertIsNotNone(sbp.runner)

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
