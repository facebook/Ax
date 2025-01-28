# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from unittest.mock import patch

import numpy as np
from ax.benchmark.benchmark import (
    benchmark_replication,
    get_benchmark_runner,
    get_benchmark_scheduler_options,
)
from ax.benchmark.methods.modular_botorch import get_sobol_botorch_modular_acquisition
from ax.benchmark.methods.sobol import get_sobol_benchmark_method
from ax.benchmark.problems.registry import get_problem
from ax.core.experiment import Experiment
from ax.modelbridge.registry import Models
from ax.service.scheduler import Scheduler
from ax.service.utils.best_point import (
    get_best_by_raw_objective_with_trial_index,
    get_best_parameters_from_model_predictions_with_trial_index,
)
from ax.utils.common.random import with_rng_seed
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
            distribute_replications=False,
        )
        is_batched = batch_size > 1
        expected_name = "MBM::SingleTaskGP_qKnowledgeGradient" + (
            f"_q{batch_size}" if is_batched else ""
        )
        self.assertEqual(method.name, expected_name)
        gs = method.generation_strategy
        sobol, kg = gs._steps
        self.assertEqual(kg.model, Models.BOTORCH_MODULAR)
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
        problem = get_problem(problem_key="ackley4")
        method = get_sobol_botorch_modular_acquisition(
            model_cls=SingleTaskGP,
            batch_size=batch_size,
            acquisition_cls=acqf_cls,
            num_sobol_trials=2,
            name="test",
            distribute_replications=False,
        )
        n_sobol_trials = method.generation_strategy._steps[0].num_trials
        self.assertEqual(n_sobol_trials, 2)
        self.assertEqual(method.name, "test")
        # Only run one non-Sobol trial
        n_total_trials = n_sobol_trials + 1
        problem = get_problem(problem_key="ackley4", num_trials=n_total_trials)
        result = benchmark_replication(problem=problem, method=method, seed=0)
        self.assertTrue(np.isfinite(result.score_trace).all())
        self.assertEqual(result.optimization_trace.shape, (n_total_trials,))

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
        method = get_sobol_benchmark_method(distribute_replications=False)
        self.assertEqual(method.name, "Sobol")
        gs = method.generation_strategy
        self.assertEqual(len(gs._steps), 1)
        self.assertEqual(gs._steps[0].model, Models.SOBOL)
        problem = get_problem(problem_key="ackley4", num_trials=3)
        result = benchmark_replication(problem=problem, method=method, seed=0)
        self.assertTrue(np.isfinite(result.score_trace).all())

    def _test_get_best_parameters(self, use_model_predictions: bool) -> None:
        problem = get_problem(problem_key="ackley4", num_trials=2, noise_std=1.0)

        method = get_sobol_botorch_modular_acquisition(
            model_cls=SingleTaskGP,
            acquisition_cls=qLogExpectedImprovement,
            distribute_replications=False,
            use_model_predictions_for_best_point=use_model_predictions,
            num_sobol_trials=1,
        )

        experiment = Experiment(
            name="test",
            search_space=problem.search_space,
            optimization_config=problem.optimization_config,
            runner=get_benchmark_runner(problem=problem),
        )

        scheduler = Scheduler(
            experiment=experiment,
            generation_strategy=method.generation_strategy.clone_reset(),
            options=get_benchmark_scheduler_options(method=method),
        )

        with with_rng_seed(seed=0):
            scheduler.run_n_trials(max_trials=problem.num_trials)

        # because the second trial is a BoTorch trial, the model should be used
        best_point_mixin_path = "ax.service.utils.best_point_mixin.best_point_utils."
        with patch(
            best_point_mixin_path
            + "get_best_parameters_from_model_predictions_with_trial_index",
            wraps=get_best_parameters_from_model_predictions_with_trial_index,
        ) as mock_get_best_parameters_from_predictions, patch(
            best_point_mixin_path + "get_best_by_raw_objective_with_trial_index",
            wraps=get_best_by_raw_objective_with_trial_index,
        ) as mock_get_best_by_raw_objective_with_trial_index:
            best_params = method.get_best_parameters(
                experiment=experiment,
                optimization_config=problem.optimization_config,
                n_points=1,
            )
        if use_model_predictions:
            mock_get_best_parameters_from_predictions.assert_called_once()
            # get_best_by_raw_objective_with_trial_index might be used as
            # fallback
        else:
            mock_get_best_parameters_from_predictions.assert_not_called()
            mock_get_best_by_raw_objective_with_trial_index.assert_called_once()
        self.assertEqual(len(best_params), 1)

    def test_get_best_parameters(self) -> None:
        for use_model_predictions in [False, True]:
            with self.subTest(f"{use_model_predictions=}"):
                self._test_get_best_parameters(
                    use_model_predictions=use_model_predictions
                )
