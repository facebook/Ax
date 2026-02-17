# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import MagicMock, patch

import numpy as np
import torch
from ax.adapter.torch import TorchAdapter
from ax.benchmark.benchmark_test_functions.surrogate import SurrogateTestFunction
from ax.benchmark.testing.benchmark_stubs import (
    get_adapter,
    get_saas_adapter,
    get_soo_surrogate_test_function,
)
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
)
from botorch.models.deterministic import PosteriorMeanModel
from botorch.sampling.pathwise.posterior_samplers import MatheronPathModel
from pyre_extensions import assert_is_instance


class TestSurrogateTestFunction(TestCase):
    def test_surrogate_test_function(self) -> None:
        # Construct a search space with log-scale parameters.
        for noise_std in (0.0, 0.1, {"dummy_metric": 0.2}):
            with self.subTest(noise_std=noise_std):
                surrogate = MagicMock()
                mock_mean = torch.tensor([[0.1234]], dtype=torch.double)
                surrogate.predict = MagicMock(return_value=(mock_mean, 0))
                surrogate.device = torch.device("cpu")
                surrogate.dtype = torch.double
                test_function = SurrogateTestFunction(
                    name="test test function",
                    outcome_names=["dummy metric"],
                    _surrogate=surrogate,
                )
                self.assertEqual(test_function.name, "test test function")
                self.assertIs(test_function.surrogate, surrogate)

    def test_equality(self) -> None:
        def _construct_test_function(name: str) -> SurrogateTestFunction:
            return SurrogateTestFunction(
                name=name,
                _surrogate=MagicMock(),
                outcome_names=["dummy_metric"],
            )

        runner_1 = _construct_test_function("test 1")
        runner_2 = _construct_test_function("test 2")
        runner_1a = _construct_test_function("test 1")
        self.assertEqual(runner_1, runner_1a)
        self.assertNotEqual(runner_1, runner_2)
        self.assertNotEqual(runner_1, 1)
        self.assertNotEqual(runner_1, None)

    def test_surrogate_model_types(self) -> None:
        """Test different surrogate model types: sample and mean."""
        experiment = get_branin_experiment(with_completed_trial=True)

        for surrogate_model_type in [MatheronPathModel, PosteriorMeanModel]:
            with self.subTest(surrogate_model_type=surrogate_model_type):
                adapter = get_adapter(experiment)

                test_function = SurrogateTestFunction(
                    name=f"test_{surrogate_model_type}_surrogate",
                    outcome_names=["branin"],
                    _surrogate=adapter,
                    surrogate_model_type=surrogate_model_type,
                    seed=42,
                )

                # Verify the surrogate type is set correctly
                self.assertEqual(
                    test_function.surrogate_model_type, surrogate_model_type
                )
                self.assertEqual(test_function.seed, 42)

                # Test evaluation
                test_params = {"x1": 0.5, "x2": 0.5}
                result = test_function.evaluate_true(test_params)

                # Ensure result is a tensor
                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.dtype, torch.double)
                self.assertEqual(result.shape, torch.Size([1]))  # One outcome

    def test_surrogate_model_types_with_random_seeds(self) -> None:
        """Test that different random seeds produce different results for samples."""
        experiment = get_branin_experiment(with_completed_trial=True)
        test_params = {"x1": 0.5, "x2": 0.5}

        results = []
        for seed in [0, 1, 2]:
            adapter = get_adapter(experiment)
            test_function = SurrogateTestFunction(
                name=f"test_sample_surrogate_seed_{seed}",
                outcome_names=["branin"],
                _surrogate=adapter,
                surrogate_model_type=MatheronPathModel,
                seed=seed,
            )

            result = test_function.evaluate_true(test_params)
            results.append(result.item())

        # Different seeds should produce different results for sample type
        self.assertFalse(
            all(r == results[0] for r in results[1:]),
            "Different random seeds should produce different sample results",
        )

    def test_mean_surrogate_consistency(self) -> None:
        """Test that mean surrogate type produces consistent results."""
        experiment = get_branin_experiment(with_completed_trial=True)
        test_params = {"x1": 0.5, "x2": 0.5}

        results = []
        # outcomes should be consistent since seed is fixed
        for i in range(3):
            adapter = get_adapter(experiment)
            test_function = SurrogateTestFunction(
                name=f"test_mean_surrogate_{i}",
                outcome_names=["branin"],
                _surrogate=adapter,
                surrogate_model_type=MatheronPathModel,
                seed=42,
            )

            result = test_function.evaluate_true(test_params)
            results.append(result.item())

        # Mean type should produce consistent results regardless of seed
        self.assertTrue(np.all(results[0] == np.array(results)))

    def test_surrogate_model_with_multiple_outcomes(self) -> None:
        """Test surrogate models with multiple outcome names."""
        experiment = get_branin_experiment_with_multi_objective(
            with_completed_trial=True
        )
        adapter = TorchAdapter(
            experiment=experiment,
            search_space=experiment.search_space,
            generator=BoTorchGenerator(),
            data=experiment.lookup_data(),
            transforms=[],
        )

        for surrogate_model_type in [MatheronPathModel, PosteriorMeanModel]:
            with self.subTest(surrogate_model_type=surrogate_model_type):
                test_function = SurrogateTestFunction(
                    name=f"test_multi_outcome_{surrogate_model_type}",
                    outcome_names=["branin_a", "branin_b"],
                    _surrogate=adapter,
                    surrogate_model_type=surrogate_model_type,
                )
                test_params = {"x1": 0.5, "x2": 0.5}
                result = test_function.evaluate_true(test_params)

                # Should return 2 outcomes
                self.assertEqual(result.shape, torch.Size([2]))

    def test_saas_surrogate_model(self) -> None:
        """Test surrogate test function with SaasFullyBayesianSingleTaskGP model."""
        experiment = get_branin_experiment(with_completed_trial=True)

        # Create adapter with SaasFullyBayesianSingleTaskGP model
        adapter = get_saas_adapter(experiment)

        for surrogate_model_type in [MatheronPathModel, PosteriorMeanModel]:
            with self.subTest(surrogate_model_type=surrogate_model_type):
                test_function = SurrogateTestFunction(
                    name=f"test_saas_surrogate_{surrogate_model_type}",
                    outcome_names=["branin"],
                    _surrogate=adapter,
                    surrogate_model_type=surrogate_model_type,
                    seed=123,
                )

                # Verify the surrogate type is set correctly
                self.assertEqual(
                    test_function.surrogate_model_type, surrogate_model_type
                )
                self.assertEqual(test_function.seed, 123)

                # Test evaluation
                test_params = {"x1": 0.5, "x2": 0.5}
                result = test_function.evaluate_true(test_params)

                # Ensure result is a tensor with correct properties
                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.dtype, torch.double)
                self.assertEqual(result.shape, torch.Size([1]))  # One outcome

    def test_lazy_instantiation(self) -> None:
        test_function = get_soo_surrogate_test_function()

        self.assertIsNone(test_function._surrogate)

        # Accessing `surrogate` sets datasets and surrogate
        self.assertIsInstance(test_function.surrogate, TorchAdapter)
        self.assertIsInstance(test_function._surrogate, TorchAdapter)

        with patch.object(
            test_function,
            "get_surrogate",
            wraps=test_function.get_surrogate,
        ) as mock_get_surrogate:
            test_function.surrogate
        mock_get_surrogate.assert_not_called()

    def test_instantiation_raises_with_missing_args(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "If `get_surrogate` is None, `_surrogate`"
        ):
            SurrogateTestFunction(name="test runner", outcome_names=[])

    def test_ensemble_sampling(self) -> None:
        """Test that ensemble sampling works correctly."""
        experiment = get_branin_experiment(with_completed_trial=True)
        adapter = get_saas_adapter(experiment)  # Creates ensemble model

        # Test with ensemble sampling enabled (default)
        test_function = SurrogateTestFunction(
            name="test_ensemble_sampling_enabled",
            outcome_names=["branin"],
            _surrogate=adapter,
            surrogate_model_type=PosteriorMeanModel,
            sample_from_ensemble=True,
        )

        # Access surrogate to trigger wrapping
        surrogate = test_function.surrogate
        # Access base_model through deterministic wrapper
        wrapped_model = assert_is_instance(surrogate.botorch_model, PosteriorMeanModel)

        # Check that exactly one model has weight 1.0 and others have weight 0.0
        weights = assert_is_instance(wrapped_model.ensemble_weights, torch.Tensor)
        self.assertEqual(weights.sum().item(), 1.0)
        self.assertEqual((weights == 1.0).sum().item(), 1)
        self.assertEqual((weights == 0.0).sum().item(), len(weights) - 1)

    def test_ensemble_no_sampling(self) -> None:
        """Test that ensemble weights remain unchanged when sampling is disabled."""
        experiment = get_branin_experiment(with_completed_trial=True)
        adapter = get_saas_adapter(experiment)  # Creates ensemble model

        # Test with ensemble sampling disabled
        test_function = SurrogateTestFunction(
            name="test_ensemble_sampling_disabled",
            outcome_names=["branin"],
            _surrogate=adapter,
            surrogate_model_type=PosteriorMeanModel,
            sample_from_ensemble=False,
        )

        # Access surrogate to trigger wrapping
        surrogate = test_function.surrogate
        self.assertIsNone(surrogate.botorch_model.ensemble_weights)
