# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np

from ax.benchmark.benchmark import benchmark_replication
from ax.benchmark.benchmark_method import get_sequential_optimization_scheduler_options
from ax.benchmark.methods.modular_botorch import get_sobol_botorch_modular_acquisition
from ax.benchmark.problems.registry import get_problem
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import not_none
from ax.utils.testing.mock import fast_botorch_optimize
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.models.gp_regression import SingleTaskGP


class TestMethods(TestCase):
    def test_mbm_acquisition(self) -> None:
        method = get_sobol_botorch_modular_acquisition(
            model_cls=SingleTaskGP,
            acquisition_cls=qKnowledgeGradient,
            scheduler_options=get_sequential_optimization_scheduler_options(),
            distribute_replications=False,
        )
        self.assertEqual(method.name, "MBM::SingleTaskGP_qKnowledgeGradient")
        gs = method.generation_strategy
        sobol, kg = gs._steps
        self.assertEqual(kg.model, Models.BOTORCH_MODULAR)
        model_kwargs = not_none(kg.model_kwargs)
        self.assertEqual(model_kwargs["botorch_acqf_class"], qKnowledgeGradient)
        surrogate_spec = next(iter(model_kwargs["surrogate_specs"].values()))
        self.assertEqual(
            surrogate_spec.botorch_model_class.__name__,
            "SingleTaskGP",
        )

    @fast_botorch_optimize
    def test_benchmark_replication_runs(self) -> None:
        problem = get_problem(problem_name="ackley4")
        method = get_sobol_botorch_modular_acquisition(
            model_cls=SingleTaskGP,
            scheduler_options=get_sequential_optimization_scheduler_options(),
            acquisition_cls=LogExpectedImprovement,
            distribute_replications=False,
        )
        n_sobol_trials = method.generation_strategy._steps[0].num_trials
        # Only run one non-Sobol trial
        problem = get_problem(problem_name="ackley4", num_trials=n_sobol_trials + 1)
        result = benchmark_replication(problem=problem, method=method, seed=0)
        self.assertTrue(np.isfinite(result.score_trace).all())
