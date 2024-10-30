# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import MagicMock, patch

import torch
from ax.benchmark.benchmark_test_functions.surrogate import SurrogateTestFunction
from ax.modelbridge.torch import TorchModelBridge
from ax.utils.common.testutils import TestCase
from ax.utils.testing.benchmark_stubs import get_soo_surrogate_test_function


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
                    _datasets=[],
                )
                self.assertEqual(test_function.name, "test test function")
                self.assertIs(test_function.surrogate, surrogate)

    def test_lazy_instantiation(self) -> None:
        test_function = get_soo_surrogate_test_function()

        self.assertIsNone(test_function._surrogate)
        self.assertIsNone(test_function._datasets)

        # Accessing `surrogate` sets datasets and surrogate
        self.assertIsInstance(test_function.surrogate, TorchModelBridge)
        self.assertIsInstance(test_function._surrogate, TorchModelBridge)
        self.assertIsInstance(test_function._datasets, list)

        # Accessing `datasets` also sets datasets and surrogate
        test_function = get_soo_surrogate_test_function()
        self.assertIsInstance(test_function.datasets, list)
        self.assertIsInstance(test_function._surrogate, TorchModelBridge)
        self.assertIsInstance(test_function._datasets, list)

        with patch.object(
            test_function,
            "get_surrogate_and_datasets",
            wraps=test_function.get_surrogate_and_datasets,
        ) as mock_get_surrogate_and_datasets:
            test_function.surrogate
        mock_get_surrogate_and_datasets.assert_not_called()

    def test_instantiation_raises_with_missing_args(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "If `get_surrogate_and_datasets` is None, `_surrogate` and "
        ):
            SurrogateTestFunction(name="test runner", outcome_names=[])

    def test_equality(self) -> None:
        def _construct_test_function(name: str) -> SurrogateTestFunction:
            return SurrogateTestFunction(
                name=name,
                _surrogate=MagicMock(),
                _datasets=[],
                outcome_names=["dummy_metric"],
            )

        runner_1 = _construct_test_function("test 1")
        runner_2 = _construct_test_function("test 2")
        runner_1a = _construct_test_function("test 1")
        self.assertEqual(runner_1, runner_1a)
        self.assertNotEqual(runner_1, runner_2)
        self.assertNotEqual(runner_1, 1)
