# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import patch

from ax.benchmark.problems.surrogate.lcbench.early_stopping import (
    BASELINE_VALUES,
    get_lcbench_early_stopping_benchmark_problem,
    OPTIMAL_VALUES,
)
from ax.benchmark.problems.surrogate.lcbench.utils import DEFAULT_METRIC_NAME
from ax.utils.common.testutils import TestCase
from ax.utils.testing.benchmark_stubs import get_mock_lcbench_data


class TestEarlyStoppingProblem(TestCase):
    def test_get_lcbench_early_stopping_problem(self) -> None:
        # Just test one problem for speed. We are mocking out the data load
        # anyway, so there is nothing to distinguish these problems from each
        # other

        observe_noise_sd = True
        num_trials = 4
        noise_std = 1.0
        seed = 27
        dataset_name = "credit-g"

        early_stopping_path = get_lcbench_early_stopping_benchmark_problem.__module__
        with patch(
            f"{early_stopping_path}.load_lcbench_data",
            return_value=get_mock_lcbench_data(),
        ) as mock_load_lcbench_data, patch(
            # Fitting a surrogate won't work with this small synthetic data
            f"{early_stopping_path}._create_surrogate_regressor"
        ) as mock_create_surrogate_regressor:
            problem = get_lcbench_early_stopping_benchmark_problem(
                dataset_name=dataset_name,
                observe_noise_sd=observe_noise_sd,
                num_trials=num_trials,
                constant_step_runtime=True,
                noise_std=noise_std,
                seed=seed,
            )

        mock_load_lcbench_data.assert_called_once()
        mock_load_lcbench_data_kwargs = mock_load_lcbench_data.call_args.kwargs
        self.assertEqual(mock_load_lcbench_data_kwargs["dataset_name"], dataset_name)
        create_surrogate_regressor_call_args = (
            mock_create_surrogate_regressor.call_args_list
        )
        self.assertEqual(len(create_surrogate_regressor_call_args), 2)
        self.assertEqual(create_surrogate_regressor_call_args[0].kwargs["seed"], seed)
        self.assertEqual(problem.noise_std, noise_std)
        self.assertEqual(
            problem.optimization_config.objective.metric.name, DEFAULT_METRIC_NAME
        )
        self.assertIsNone(problem.step_runtime_function)
        self.assertEqual(problem.optimal_value, OPTIMAL_VALUES[dataset_name])
        self.assertEqual(problem.baseline_value, BASELINE_VALUES[dataset_name])
