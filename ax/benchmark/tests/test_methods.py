# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import numpy as np

from ax.benchmark.benchmark import benchmark_replication
from ax.benchmark.methods.modular_botorch import get_sobol_botorch_modular_acquisition
from ax.benchmark.problems.registry import get_problem
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import fast_botorch_optimize
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient


class TestMethods(TestCase):
    def test_mbm_acquisition(self) -> None:
        method = get_sobol_botorch_modular_acquisition(
            acquisition_cls=qKnowledgeGradient,
            acquisition_options={"num_fantasies": 16},
        )
        self.assertEqual(method.name, "SOBOL+BOTORCH_MODULAR::qKnowledgeGradient")
        gs = method.generation_strategy
        sobol, kg = gs._steps
        self.assertEqual(kg.model, Models.BOTORCH_MODULAR)
        model_kwargs = kg.model_kwargs
        # pyre-fixme[16]: Optional type has no attribute `__getitem__`.
        self.assertEqual(model_kwargs["botorch_acqf_class"], qKnowledgeGradient)
        self.assertEqual(model_kwargs["acquisition_options"], {"num_fantasies": 16})

    @fast_botorch_optimize
    def test_benchmark_replication_runs(self) -> None:
        problem = get_problem(problem_name="ackley4")
        method = get_sobol_botorch_modular_acquisition(
            acquisition_cls=qKnowledgeGradient
        )
        n_sobol_trials = method.generation_strategy._steps[0].num_trials
        # Only run one non-Sobol trial
        problem = replace(problem, num_trials=n_sobol_trials + 1)
        result = benchmark_replication(problem=problem, method=method, seed=0)
        self.assertTrue(np.isfinite(result.score_trace).all())
