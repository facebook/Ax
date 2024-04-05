#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from ax.core.experiment import Experiment
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.metrics.l2norm import L2NormMetric
from ax.modelbridge.strategies.rembo import HeSBOStrategy, REMBOStrategy
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.mock import fast_botorch_optimize


class REMBOStrategyTest(TestCase):
    @fast_botorch_optimize
    def test_REMBOStrategy(self) -> None:
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

        exp = Experiment(
            name="test",
            search_space=hartmann_search_space,
            optimization_config=OptimizationConfig(
                objective=Objective(
                    metric=Hartmann6Metric(
                        name="hartmann6", param_names=[f"x{i}" for i in range(6)]
                    ),
                    minimize=True,
                ),
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
            ),
            runner=SyntheticRunner(),
        )

        # Instantiate the strategy
        gs = REMBOStrategy(D=20, d=6, k=4, init_per_proj=4)

        # Check that arms and data are correctly segmented by projection
        exp.new_batch_trial(generator_run=gs.gen(experiment=exp, n=2)).run()
        self.assertEqual(len(gs.arms_by_proj[0]), 2)
        self.assertEqual(len(gs.arms_by_proj[1]), 0)

        exp.new_batch_trial(generator_run=gs.gen(experiment=exp, n=2)).run()

        self.assertEqual(len(gs.arms_by_proj[0]), 2)
        self.assertEqual(len(gs.arms_by_proj[1]), 2)

        # Iterate until the first projection fits a GP
        for _ in range(4):
            exp.new_batch_trial(generator_run=gs.gen(experiment=exp, n=2)).run()

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

            exp.new_batch_trial(generator_run=gs.gen(experiment=exp, n=2)).run()
            if i >= 2:
                self.assertFalse(any(len(x) < 4 for x in gs.arms_by_proj.values()))

        gs2 = gs.clone_reset()
        self.assertEqual(gs2.D, 20)
        self.assertEqual(gs2.d, 6)

    def test_HeSBOStrategy(self) -> None:
        gs = HeSBOStrategy(D=10, d=4, init_per_proj=2)
        self.assertEqual(gs.name, "HeSBO")
        self.assertEqual(len(gs.projections), 1)
        A, bounds_d = gs.projections[0]
        self.assertEqual(bounds_d, [(-1, 1)] * 4)
        z = torch.abs(A).sum(dim=1)
        self.assertTrue(torch.allclose(z, torch.ones(10, dtype=torch.double)))
        gs2 = gs.clone_reset()
        self.assertTrue(isinstance(gs2, HeSBOStrategy))
