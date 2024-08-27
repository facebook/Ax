# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import MagicMock, patch

import torch
from ax.benchmark.runners.surrogate import SurrogateRunner
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.torch import TorchModelBridge
from ax.utils.common.testutils import TestCase
from ax.utils.testing.benchmark_stubs import get_soo_surrogate


class TestSurrogateRunner(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter("x", ParameterType.FLOAT, 0.0, 5.0),
                RangeParameter("y", ParameterType.FLOAT, 1.0, 10.0, log_scale=True),
                RangeParameter("z", ParameterType.INT, 1.0, 5.0, log_scale=True),
            ]
        )

    def test_surrogate_runner(self) -> None:
        # Construct a search space with log-scale parameters.
        for noise_std in (0.0, 0.1, {"dummy_metric": 0.2}):
            with self.subTest(noise_std=noise_std):
                surrogate = MagicMock()
                mock_mean = torch.tensor([[0.1234]], dtype=torch.double)
                surrogate.predict = MagicMock(return_value=(mock_mean, 0))
                surrogate.device = torch.device("cpu")
                surrogate.dtype = torch.double
                runner = SurrogateRunner(
                    name="test runner",
                    surrogate=surrogate,
                    datasets=[],
                    search_space=self.search_space,
                    outcome_names=["dummy_metric"],
                    noise_stds=noise_std,
                )
                self.assertEqual(runner.name, "test runner")
                self.assertIs(runner.surrogate, surrogate)
                self.assertEqual(runner.outcome_names, ["dummy_metric"])
                self.assertEqual(runner.noise_stds, noise_std)

    def test_lazy_instantiation(self) -> None:
        runner = get_soo_surrogate().runner

        self.assertIsNone(runner._surrogate)
        self.assertIsNone(runner._datasets)

        # Accessing `surrogate` sets datasets and surrogate
        self.assertIsInstance(runner.surrogate, TorchModelBridge)
        self.assertIsInstance(runner._surrogate, TorchModelBridge)
        self.assertIsInstance(runner._datasets, list)

        # Accessing `datasets` also sets datasets and surrogate
        runner = get_soo_surrogate().runner
        self.assertIsInstance(runner.datasets, list)
        self.assertIsInstance(runner._surrogate, TorchModelBridge)
        self.assertIsInstance(runner._datasets, list)

        with patch.object(
            runner,
            "get_surrogate_and_datasets",
            wraps=runner.get_surrogate_and_datasets,
        ) as mock_get_surrogate_and_datasets:
            runner.surrogate
        mock_get_surrogate_and_datasets.assert_not_called()

    def test_instantiation_raises_with_missing_args(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "If get_surrogate_and_datasets is not provided, surrogate and "
        ):
            SurrogateRunner(
                name="test runner",
                search_space=self.search_space,
                outcome_names=[],
                noise_stds=0.0,
            )

    def test_equality(self) -> None:

        def _construct_runner(name: str) -> SurrogateRunner:
            return SurrogateRunner(
                name=name,
                surrogate=MagicMock(),
                datasets=[],
                search_space=self.search_space,
                outcome_names=["dummy_metric"],
                noise_stds=0.0,
            )

        runner_1 = _construct_runner("test 1")
        runner_2 = _construct_runner("test 2")
        runner_1a = _construct_runner("test 1")
        self.assertEqual(runner_1, runner_1a)
        self.assertNotEqual(runner_1, runner_2)
        self.assertNotEqual(runner_1, 1)
