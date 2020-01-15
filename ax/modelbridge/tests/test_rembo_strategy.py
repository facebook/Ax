#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import numpy as np
import torch
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.simple_experiment import SimpleExperiment
from ax.metrics.l2norm import L2NormMetric
from ax.modelbridge.strategies.rembo import HeSBOStrategy, REMBOStrategy
from ax.utils.common.testutils import TestCase
from ax.utils.measurement.synthetic_functions import hartmann6


def hartmann_evaluation_function(parameterization, weight=None):
    x = np.array([parameterization.get(f"x{i}") for i in range(6)])
    return {"hartmann6": (hartmann6(x), 0.0), "l2norm": (np.sqrt((x ** 2).sum()), 0.0)}


class REMBOStrategyTest(TestCase):
    @patch(
        "ax.models.torch.botorch_defaults.optimize_acqf",
        autospec=True,
        return_value=(
            torch.randn((2, 6), dtype=torch.double),
            torch.randn((2, 6), dtype=torch.double),
        ),
    )
    @patch("ax.models.torch.botorch_defaults.fit_gpytorch_model", autospec=True)
    def test_REMBOStrategy(self, mock_fit_gpytorch_model, mock_optimize_acqf):
        # Construct a high-D test experiment with multiple metrics
        hartmann_search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name=f"x{i}",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=1.0,
                )
                for i in range(20)
            ]
        )

        exp = SimpleExperiment(
            name="test",
            search_space=hartmann_search_space,
            evaluation_function=hartmann_evaluation_function,
            objective_name="hartmann6",
            minimize=True,
            outcome_constraints=[
                OutcomeConstraint(
                    metric=L2NormMetric(
                        name="l2norm",
                        param_names=[f"x{i}" for i in range(6)],
                        noise_sd=0.2,
                    ),
                    op=ComparisonOp.LEQ,
                    bound=1.25,
                    relative=False,
                )
            ],
        )

        # Instantiate the strategy
        gs = REMBOStrategy(D=20, d=6, k=4, init_per_proj=4)

        # Check that arms and data are correctly segmented by projection
        exp.new_batch_trial(generator_run=gs.gen(experiment=exp, n=2))
        self.assertEqual(len(gs.arms_by_proj[0]), 2)
        self.assertEqual(len(gs.arms_by_proj[1]), 0)

        exp.new_batch_trial(generator_run=gs.gen(experiment=exp, n=2))

        self.assertEqual(len(gs.arms_by_proj[0]), 2)
        self.assertEqual(len(gs.arms_by_proj[1]), 2)

        # Iterate until the first projection fits a GP
        for _ in range(4):
            exp.new_batch_trial(generator_run=gs.gen(experiment=exp, n=2))
            mock_fit_gpytorch_model.assert_not_called()

        self.assertEqual(len(gs.arms_by_proj[0]), 4)
        self.assertEqual(len(gs.arms_by_proj[1]), 4)
        self.assertEqual(len(gs.arms_by_proj[2]), 2)
        self.assertEqual(len(gs.arms_by_proj[3]), 2)

        # Keep iterating until GP is used for gen
        for i in range(4):
            # First two trials will go towards 3rd and 4th proj. getting enough
            if i < 1:  # data for GP.
                self.assertLess(len(gs.arms_by_proj[2]), 4)
            if i < 2:
                self.assertLess(len(gs.arms_by_proj[3]), 4)

            exp.new_batch_trial(generator_run=gs.gen(experiment=exp, n=2))
            if i < 2:
                mock_fit_gpytorch_model.assert_not_called()
            else:
                # After all proj. have > 4 arms' worth of data, GP can be fit.
                self.assertFalse(any(len(x) < 4 for x in gs.arms_by_proj.values()))
                mock_fit_gpytorch_model.assert_called()

        self.assertTrue(len(gs.model_transitions) > 0)
        gs2 = gs.clone_reset()
        self.assertEqual(gs2.D, 20)
        self.assertEqual(gs2.d, 6)

    def testHeSBOStrategy(self):
        gs = HeSBOStrategy(D=10, d=4, init_per_proj=2)
        self.assertEqual(gs.name, "HeSBO")
        self.assertEqual(len(gs.projections), 1)
        A, bounds_d = gs.projections[0]
        self.assertEqual(bounds_d, [(-1, 1)] * 4)
        z = torch.abs(A).sum(dim=1)
        self.assertTrue(torch.allclose(z, torch.ones(10, dtype=torch.double)))
        gs2 = gs.clone_reset()
        self.assertTrue(isinstance(gs2, HeSBOStrategy))
