#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from warnings import catch_warnings

import numpy as np
from ax.models.discrete.eb_ashr import EBAshr, no_model_feasibility_util
from ax.utils.common.testutils import TestCase


class EBAshrTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.Xs = [
            [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
        ]  # 5 arms, each of dimensionality 2
        self.Ys = [
            [1, -1, 2, -2, 0],
            [-1, 1, -2, 2, 0],
        ]  # two metrics evaluated on each of the five arms
        self.Yvars = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        self.parameter_values = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
        ]  # not used in best arm selection
        self.outcome_names = ["x", "y"]  # metric names, not used in best arm selection

    def test_EBAshr(self) -> None:
        generator = EBAshr()
        generator.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        arms, weights, gen_metadata = generator.gen(
            n=3,
            parameter_values=self.parameter_values,
            outcome_constraints=(np.array([[0.0, -1.0]]), np.array([[0.0]])),
            objective_weights=np.array([1, 0]),
        )

        self.assertEqual(arms, [[1, 1], [5, 5], [2, 2]])
        self.assertEqual(
            list(gen_metadata["regression"]), [False, False, True, True, False]
        )
        self.assertTrue(gen_metadata["prob_infeasibility"][0, 0] < 0.9)
        self.assertTrue(gen_metadata["prob_infeasibility"][1, 0] < 0.9)
        self.assertTrue(gen_metadata["prob_infeasibility"][2, 1] > 0.9)
        self.assertTrue(gen_metadata["prob_infeasibility"][3, 0] > 0.9)
        self.assertTrue(gen_metadata["prob_infeasibility"][4, 0] < 0.9)
        self.assertEqual(weights, [1.0] * 3)
        self.assertEqual(gen_metadata["best_x"], arms[0])

        with self.subTest("No variance"):
            generator.fit(
                Xs=self.Xs,
                Ys=self.Ys,
                Yvars=[[0] * len(self.Ys[0]) for _ in self.Ys],
                parameter_values=self.parameter_values,
                outcome_names=self.outcome_names,
            )
            arms, weights, gen_metadata = generator.gen(
                n=3,
                parameter_values=self.parameter_values,
                outcome_constraints=(np.array([[0.0, -1.0]]), np.array([[0.0]])),
                objective_weights=np.array([1, 0]),
            )
            self.assertEqual(sorted([tuple(a) for a in arms]), [(1, 1), (2, 2), (5, 5)])

            self.assertEqual(
                list(gen_metadata["regression"]), [True, True, True, True, False]
            )
            prob_infeasibility = gen_metadata["prob_infeasibility"]
            self.assertEqual(prob_infeasibility[0, 0], 0)
            self.assertEqual(prob_infeasibility[1, 0], 1)
            self.assertEqual(prob_infeasibility[2, 1], 1)
            self.assertEqual(prob_infeasibility[3, 0], 1)
            self.assertEqual(prob_infeasibility[4, 0], 0)

            self.assertEqual(weights, [1.0] * 3)
            self.assertEqual(gen_metadata["best_x"], arms[0])

        with catch_warnings(record=True) as warning_list:
            arms, weights, gen_metadata = generator.gen(
                n=3,
                parameter_values=self.parameter_values,
                outcome_constraints=None,
                objective_weights=np.array([1, -1]),
            )
        self.assertEqual(
            "In case of multi-objective adding metric values together might"
            " not lead to a meaningful result.",
            str(warning_list[0].message),
        )

    def test_no_model_feasibility_util(self) -> None:
        probabilities = no_model_feasibility_util(
            lb=1.0, ub=2.0, Y=np.array([0.0, 1.0, 1.5, 2.0, 3.0])
        )
        self.assertTrue((probabilities == [False, True, True, True, False]).all())
