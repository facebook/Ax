# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
        surrogate = MagicMock()
        surrogate.predict = MagicMock(return_value=(torch.zeros(1, 1), 0))
        # Construct a search space with log-scale parameters.
        search_space = SearchSpace(
            parameters=[
                RangeParameter("x", ParameterType.FLOAT, 0.0, 5.0),
                RangeParameter("y", ParameterType.FLOAT, 1.0, 10.0, log_scale=True),
                RangeParameter("z", ParameterType.INT, 1.0, 5.0, log_scale=True),
            ]
        )

        runner = SurrogateRunner(
            name="test runner",
            surrogate=surrogate,
            datasets=[],
            search_space=search_space,
            metric_names=["dummy metric"],
        )
        self.assertEqual(runner.name, "test runner")
        self.assertIs(runner.surrogate, surrogate)
        self.assertEqual(runner.metric_names, ["dummy metric"])
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
        # Check that the evaluation works correctly with the transformed parameters.
        trial = Trial(experiment=MagicMock())
        trial.add_arm(Arm({"x": 2.5, "y": 10.0, "z": 1.0}, name="0_0"))
        run_output = runner.run(trial)
        self.assertEqual(run_output["dummy metric"]["0_0"].item(), 0.0)
        surrogate.predict.assert_called_once()
        X = surrogate.predict.call_args[1]["X"]
        self.assertTrue(torch.allclose(X, torch.tensor([[2.5, 1.0, 0.0]])))
