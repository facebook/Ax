# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import MagicMock

import torch
from ax.benchmark.problems.surrogate import SurrogateRunner
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.utils.common.testutils import TestCase


class TestSurrogateRunner(TestCase):
    def test_surrogate_runner(self) -> None:
        # Construct a search space with log-scale parameters.
        search_space = SearchSpace(
            parameters=[
                RangeParameter("x", ParameterType.FLOAT, 0.0, 5.0),
                RangeParameter("y", ParameterType.FLOAT, 1.0, 10.0, log_scale=True),
                RangeParameter("z", ParameterType.INT, 1.0, 5.0, log_scale=True),
            ]
        )
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
                    search_space=search_space,
                    outcome_names=["dummy_metric"],
                    noise_stds=noise_std,
                )
                self.assertEqual(runner.name, "test runner")
                self.assertIs(runner.surrogate, surrogate)
                self.assertEqual(runner.outcome_names, ["dummy_metric"])
                self.assertEqual(runner.noise_stds, noise_std)
