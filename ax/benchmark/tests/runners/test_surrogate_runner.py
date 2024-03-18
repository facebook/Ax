# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import MagicMock

import torch
from ax.benchmark.problems.surrogate import SurrogateRunner
from ax.core.arm import Arm
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.trial import Trial
from ax.modelbridge.transforms.int_to_float import IntToFloat
from ax.modelbridge.transforms.log import Log
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast, not_none


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

                # Check that the transforms are set up correctly.
                transforms = not_none(runner.transforms)
                self.assertEqual(len(transforms), 2)
                self.assertIsInstance(transforms[0], IntToFloat)
                self.assertIsInstance(transforms[1], Log)
                self.assertEqual(
                    checked_cast(IntToFloat, transforms[0]).transform_parameters, {"z"}
                )
                self.assertEqual(
                    checked_cast(Log, transforms[1]).transform_parameters, {"y", "z"}
                )
                # Check that evaluation works correctly with the transformed parameters.
                trial = Trial(experiment=MagicMock())
                trial.add_arm(Arm({"x": 2.5, "y": 10.0, "z": 1.0}, name="0_0"))
                run_output = runner.run(trial)
                self.assertEqual(run_output["outcome_names"], ["dummy_metric"])
                self.assertEqual(run_output["Ys_true"]["0_0"], [0.1234])
                self.assertEqual(
                    run_output["Ystds"]["0_0"],
                    [
                        (
                            noise_std
                            if not isinstance(noise_std, dict)
                            else noise_std["dummy_metric"]
                        )
                    ],
                )
                surrogate.predict.assert_called_once()
                X = surrogate.predict.call_args[1]["X"]
                self.assertTrue(
                    torch.allclose(
                        X, torch.tensor([[2.5, 1.0, 0.0]], dtype=torch.double)
                    )
                )
