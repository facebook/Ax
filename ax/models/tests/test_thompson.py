#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from warnings import catch_warnings

import numpy as np
from ax.exceptions.core import UnsupportedError
from ax.exceptions.model import ModelError
from ax.models.discrete.thompson import ThompsonSampler
from ax.utils.common.testutils import TestCase


class ThompsonSamplerTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.Xs = [[[1, 1], [2, 2], [3, 3], [4, 4]]]  # 4 arms, each of dimensionality 2
        self.Ys = [[1, 2, 3, 4]]
        self.Yvars = [[1, 1, 1, 1]]
        self.parameter_values = [[1, 2, 3, 4], [1, 2, 3, 4]]
        self.outcome_names = ["x", "y"]  # not used for regular TS

        self.multiple_metrics_Xs = [
            [[1, 1], [2, 2], [3, 3], [4, 4]],
            [[1, 1], [2, 2], [3, 3], [4, 4]],
        ]  # 2 metrics, 4 arms, each of dimensionality 2
        self.multiple_metrics_Ys = [[1, 2, 3, 4], [0, 0, 0, 1]]
        self.multiple_metrics_Yvars = [[1, 1, 1, 1], [1, 1, 1, 1]]

    def test_ThompsonSampler(self) -> None:
        generator = ThompsonSampler(min_weight=0.0)
        self.assertEqual(generator.topk, 1)
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
            objective_weights=np.ones(1),
        )
        self.assertEqual(arms, [[4, 4], [3, 3], [2, 2]])
        for weight, expected_weight in zip(
            weights, [3 * i for i in [0.725, 0.225, 0.05]]
        ):
            self.assertAlmostEqual(weight, expected_weight, 1)
        self.assertEqual(len(gen_metadata["arms_to_weights"]), 4)
        self.assertEqual(gen_metadata["best_x"], arms[0])

    def test_ThompsonSamplerValidation(self) -> None:
        generator = ThompsonSampler(min_weight=0.01)

        # all Xs are not the same
        with self.assertRaises(ValueError):
            generator.fit(
                Xs=[[[1, 1], [2, 2], [3, 3], [4, 4]], [[1, 1], [2, 2], [4, 4]]],
                Ys=self.Ys,
                Yvars=self.Yvars,
                parameter_values=self.parameter_values,
                outcome_names=self.outcome_names,
            )

        # multiple observations per parameterization
        with self.assertRaises(ValueError):
            generator.fit(
                Xs=[[[1, 1], [2, 2], [2, 2]]],
                Ys=self.Ys,
                Yvars=self.Yvars,
                parameter_values=self.parameter_values,
                outcome_names=self.outcome_names,
            )

        # these are not the same observations, so should not error
        generator.fit(
            Xs=[[[1, 1], [2.0, 2], [2, 2]]],
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )

        # requires objective weights
        with self.assertRaises(ValueError):
            generator.gen(5, self.parameter_values, objective_weights=None)

    def test_ThompsonSamplerTopKError(self) -> None:
        generator = ThompsonSampler(topk=5)
        with self.assertRaisesRegex(ModelError, r"ThompsonSampler `topk=\d+`"):
            generator.fit(
                Xs=self.Xs,
                Ys=self.Ys,
                Yvars=self.Yvars,
                parameter_values=self.parameter_values,
                outcome_names=self.outcome_names,
            )

    def test_TopTwo_alters_weights_vs_TopOne(self) -> None:
        np.random.seed(0)

        # Compare TTTS results to the vanilla TS
        ts1 = ThompsonSampler(num_samples=1000, min_weight=0.0, topk=1)
        ts1.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        _, _, gen_metadata1 = ts1.gen(
            n=4,
            parameter_values=self.parameter_values,
            objective_weights=np.ones(1),
        )
        full_w1 = gen_metadata1["arms_to_weights"]

        ts2 = ThompsonSampler(num_samples=1000, min_weight=0.0, topk=2)
        ts2.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        _, _, gen_metadata2 = ts2.gen(
            n=4,
            parameter_values=self.parameter_values,
            objective_weights=np.ones(1),
        )
        full_w2 = gen_metadata2["arms_to_weights"]

        # Sanity: same length
        self.assertEqual(len(full_w1), 4)
        self.assertEqual(len(full_w2), 4)

        # 1) Best arm (4) weight must drop under TTTS
        self.assertLess(full_w2[3], full_w1[3])

        # 2) Second-best arm (3) weight must rise
        self.assertGreater(full_w2[2], full_w1[2])

        # 3) Arm 2 (index 1) and arm 1 (index 0) both see P(2nd)>P(best)
        #   so they increase
        self.assertGreater(full_w2[1], full_w1[1])
        self.assertGreater(full_w2[0], full_w1[0])

        # 4) Monotonicity in the final TTTS distribution still holds
        self.assertTrue(full_w2[3] > full_w2[2] > full_w2[1] > full_w2[0])

    def test_ThompsonSamplerMinWeight(self) -> None:
        np.random.seed(0)
        generator = ThompsonSampler(min_weight=0.01)
        generator.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        arms, weights, _ = generator.gen(
            n=3,
            parameter_values=self.parameter_values,
            objective_weights=np.ones(1),
        )
        self.assertEqual(arms, [[4, 4], [3, 3], [2, 2]])
        for weight, expected_weight in zip(
            weights, [3 * i for i in [0.725, 0.225, 0.05]]
        ):
            self.assertAlmostEqual(weight, expected_weight, 1)

    def test_ThompsonSamplerUniformWeights(self) -> None:
        generator = ThompsonSampler(min_weight=0.0, uniform_weights=True)
        generator.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        arms, weights, _ = generator.gen(
            n=3,
            parameter_values=self.parameter_values,
            objective_weights=np.ones(1),
        )
        self.assertEqual(arms, [[4, 4], [3, 3], [2, 2]])
        for weight, expected_weight in zip(weights, [1.0, 1.0, 1.0]):
            self.assertAlmostEqual(weight, expected_weight, 1)

    def test_ThompsonSamplerInfeasible(self) -> None:
        generator = ThompsonSampler(min_weight=0.9)
        generator.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        with self.assertRaises(ModelError):
            generator.gen(
                n=3,
                parameter_values=self.parameter_values,
                objective_weights=np.ones(1),
            )

    def test_ThompsonSamplerOutcomeConstraints(self) -> None:
        generator = ThompsonSampler(min_weight=0.0)
        generator.fit(
            Xs=self.multiple_metrics_Xs,
            Ys=self.multiple_metrics_Ys,
            Yvars=self.multiple_metrics_Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        arms, weights, _ = generator.gen(
            n=4,
            parameter_values=self.parameter_values,
            objective_weights=np.array([1, 0]),
            outcome_constraints=(
                # pass in multiples of the same constraint
                # to ensure that shapes are correct for multiple constraints
                np.array([[0, 1], [0, 1], [0, 1]]),
                np.array([[1], [1], [1]]),
            ),
        )
        self.assertEqual(arms, [[3, 3], [4, 4], [2, 2], [1, 1]])
        for weight, expected_weight in zip(
            weights, [4 * i for i in [0.4, 0.4, 0.15, 0.05]]
        ):
            self.assertAlmostEqual(weight, expected_weight, delta=0.15)

    def test_ThompsonSamplerOutcomeConstraintsInfeasible(self) -> None:
        generator = ThompsonSampler(min_weight=0.0)
        generator.fit(
            Xs=self.multiple_metrics_Xs,
            Ys=self.multiple_metrics_Ys,
            Yvars=self.multiple_metrics_Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        with self.assertRaises(ModelError):
            generator.gen(
                n=3,
                parameter_values=self.parameter_values,
                objective_weights=np.ones(2),
                outcome_constraints=(np.array([[0, 1]]), np.array([[-10]])),
            )

    def test_ThompsonSamplerPredict(self) -> None:
        generator = ThompsonSampler(min_weight=0.0)
        generator.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        f, cov = generator.predict([[1, 1], [3, 3]])
        self.assertTrue(np.array_equal(f, np.array([[1], [3]])))
        self.assertTrue(np.array_equal(cov, np.ones((2, 1, 1))))

        with self.assertRaisesRegex(UnsupportedError, "out-of-sample"):
            generator.predict([[1, 2]])

    def test_ThompsonSamplerMultiObjectiveWarning(self) -> None:
        generator = ThompsonSampler(min_weight=0.0)
        generator.fit(
            Xs=self.multiple_metrics_Xs,
            Ys=self.multiple_metrics_Ys,
            Yvars=self.multiple_metrics_Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        with catch_warnings(record=True) as warning_list:
            arms, weights, _ = generator.gen(
                n=4,
                parameter_values=self.parameter_values,
                objective_weights=np.array([1, -1]),
                outcome_constraints=None,
            )
        self.assertEqual(
            "In case of multi-objective adding metric values together might"
            " not lead to a meaningful result.",
            str(warning_list[0].message),
        )

    def test_ThompsonSamplerNonPositiveN(self) -> None:
        generator = ThompsonSampler(min_weight=0.0)
        generator.fit(
            Xs=self.Xs,
            Ys=self.Ys,
            Yvars=self.Yvars,
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        for n in (-1, 0):
            with self.assertRaisesRegex(ValueError, "ThompsonSampler requires n > 0"):
                generator.gen(
                    n=n,
                    parameter_values=self.parameter_values,
                    objective_weights=np.ones(1),
                )
