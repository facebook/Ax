# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from logging import WARNING

import numpy as np
from ax.adapter.registry import Generators
from ax.benchmark.benchmark import benchmark_replication
from ax.benchmark.methods.modular_botorch import get_sobol_botorch_modular_acquisition
from ax.benchmark.methods.sobol import get_sobol_benchmark_method
from ax.benchmark.problems.registry import get_benchmark_problem
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import mock_botorch_optimize
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP
from pyre_extensions import none_throws


class TestMethods(TestCase):
    def _test_mbm_acquisition(self, batch_size: int) -> None:
        method = get_sobol_botorch_modular_acquisition(
            model_cls=SingleTaskGP,
            acquisition_cls=qKnowledgeGradient,
            batch_size=batch_size,
        )
        is_batched = batch_size > 1
        expected_name = "MBM::SingleTaskGP_qKnowledgeGradient" + (
            f"_q{batch_size}" if is_batched else ""
        )
        self.assertEqual(method.name, expected_name)
        gs = method.generation_strategy
        sobol, kg = gs._steps
        self.assertEqual(kg.generator, Generators.BOTORCH_MODULAR)
        model_kwargs = none_throws(kg.model_kwargs)
        self.assertEqual(model_kwargs["botorch_acqf_class"], qKnowledgeGradient)
        surrogate_spec = model_kwargs["surrogate_spec"]
        self.assertEqual(
            surrogate_spec.model_configs[0].botorch_model_class.__name__,
            "SingleTaskGP",
        )

    def test_mbm_acquisition(self) -> None:
        for batch_size in [1, 2]:
            with self.subTest(batch_size=batch_size):
                self._test_mbm_acquisition(batch_size=batch_size)

    @mock_botorch_optimize
    def _test_benchmark_replication_runs(
        self, batch_size: int, acqf_cls: type[AcquisitionFunction]
    ) -> None:
        problem = get_benchmark_problem(problem_key="ackley4")
        method = get_sobol_botorch_modular_acquisition(
            model_cls=SingleTaskGP,
            batch_size=batch_size,
            acquisition_cls=acqf_cls,
            num_sobol_trials=2,
            name="test",
        )
        n_sobol_trials = method.generation_strategy._steps[0].num_trials
        self.assertEqual(n_sobol_trials, 2)
        self.assertEqual(method.name, "test")
        # Only run one non-Sobol trial
        n_total_trials = n_sobol_trials + 1
        problem = get_benchmark_problem(
            problem_key="ackley4", num_trials=n_total_trials
        )
        result = benchmark_replication(
            problem=problem, method=method, seed=0, orchestrator_logging_level=WARNING
        )
        self.assertTrue(np.isfinite(result.score_trace).all())
        self.assertEqual(len(result.optimization_trace), n_total_trials)

        self.assertEqual(
            len(none_throws(result.experiment).arms_by_name),
            n_total_trials * batch_size,
        )

    def test_benchmark_replication_runs(self) -> None:
        with self.subTest(name="sequential LogEI"):
            self._test_benchmark_replication_runs(
                batch_size=1,
                acqf_cls=LogExpectedImprovement,
            )
        with self.subTest(name="sequential qLogEI"):
            self._test_benchmark_replication_runs(
                batch_size=1,
                acqf_cls=qLogExpectedImprovement,
            )
        with self.subTest(name="batch qLogEI"):
            self._test_benchmark_replication_runs(
                batch_size=2,
                acqf_cls=qLogExpectedImprovement,
            )

    def test_sobol(self) -> None:
        method = get_sobol_benchmark_method()
        self.assertEqual(method.name, "Sobol")
        gs = method.generation_strategy
        self.assertEqual(len(gs._steps), 1)
        self.assertEqual(gs._steps[0].generator, Generators.SOBOL)
