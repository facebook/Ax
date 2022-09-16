#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from ax.exceptions.model import ModelError
from ax.models.discrete.thompson import ThompsonSampler
from ax.utils.common.testutils import TestCase


class ThompsonSamplerTest(TestCase):
    def setUp(self) -> None:
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

    def testThompsonSampler(self) -> None:
        generator = ThompsonSampler(min_weight=0.0)
        generator.fit(
            # pyre-fixme[6]: For 1st param expected `List[List[List[Union[None,
            #  bool, float, int, str]]]]` but got `List[List[List[int]]]`.
            Xs=self.Xs,
            # pyre-fixme[6]: For 2nd param expected `List[List[float]]` but got
            #  `List[List[int]]`.
            Ys=self.Ys,
            # pyre-fixme[6]: For 3rd param expected `List[List[float]]` but got
            #  `List[List[int]]`.
            Yvars=self.Yvars,
            # pyre-fixme[6]: For 4th param expected `List[List[Union[None, bool,
            #  float, int, str]]]` but got `List[List[int]]`.
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        arms, weights, gen_metadata = generator.gen(
            n=3,
            # pyre-fixme[6]: For 2nd param expected `List[List[Union[None, bool,
            #  float, int, str]]]` but got `List[List[int]]`.
            parameter_values=self.parameter_values,
            objective_weights=np.ones(1),
        )
        self.assertEqual(arms, [[4, 4], [3, 3], [2, 2]])
        for weight, expected_weight in zip(
            weights, [3 * i for i in [0.725, 0.225, 0.05]]
        ):
            self.assertAlmostEqual(weight, expected_weight, 1)
        self.assertEqual(len(gen_metadata["arms_to_weights"]), 4)

    def testThompsonSamplerValidation(self) -> None:
        generator = ThompsonSampler(min_weight=0.01)

        # all Xs are not the same
        with self.assertRaises(ValueError):
            generator.fit(
                Xs=[[[1, 1], [2, 2], [3, 3], [4, 4]], [[1, 1], [2, 2], [4, 4]]],
                # pyre-fixme[6]: For 2nd param expected `List[List[float]]` but got
                #  `List[List[int]]`.
                Ys=self.Ys,
                # pyre-fixme[6]: For 3rd param expected `List[List[float]]` but got
                #  `List[List[int]]`.
                Yvars=self.Yvars,
                # pyre-fixme[6]: For 4th param expected `List[List[Union[None, bool,
                #  float, int, str]]]` but got `List[List[int]]`.
                parameter_values=self.parameter_values,
                outcome_names=self.outcome_names,
            )

        # multiple observations per parameterization
        with self.assertRaises(ValueError):
            generator.fit(
                Xs=[[[1, 1], [2, 2], [2, 2]]],
                # pyre-fixme[6]: For 2nd param expected `List[List[float]]` but got
                #  `List[List[int]]`.
                Ys=self.Ys,
                # pyre-fixme[6]: For 3rd param expected `List[List[float]]` but got
                #  `List[List[int]]`.
                Yvars=self.Yvars,
                # pyre-fixme[6]: For 4th param expected `List[List[Union[None, bool,
                #  float, int, str]]]` but got `List[List[int]]`.
                parameter_values=self.parameter_values,
                outcome_names=self.outcome_names,
            )

        # these are not the same observations, so should not error
        generator.fit(
            Xs=[[[1, 1], [2.0, 2], [2, 2]]],
            # pyre-fixme[6]: For 2nd param expected `List[List[float]]` but got
            #  `List[List[int]]`.
            Ys=self.Ys,
            # pyre-fixme[6]: For 3rd param expected `List[List[float]]` but got
            #  `List[List[int]]`.
            Yvars=self.Yvars,
            # pyre-fixme[6]: For 4th param expected `List[List[Union[None, bool,
            #  float, int, str]]]` but got `List[List[int]]`.
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )

        # requires objective weights
        with self.assertRaises(ValueError):
            # pyre-fixme[6]: For 2nd param expected `List[List[Union[None, bool,
            #  float, int, str]]]` but got `List[List[int]]`.
            generator.gen(5, self.parameter_values, objective_weights=None)

    def testThompsonSamplerMinWeight(self) -> None:
        generator = ThompsonSampler(min_weight=0.01)
        generator.fit(
            # pyre-fixme[6]: For 1st param expected `List[List[List[Union[None,
            #  bool, float, int, str]]]]` but got `List[List[List[int]]]`.
            Xs=self.Xs,
            # pyre-fixme[6]: For 2nd param expected `List[List[float]]` but got
            #  `List[List[int]]`.
            Ys=self.Ys,
            # pyre-fixme[6]: For 3rd param expected `List[List[float]]` but got
            #  `List[List[int]]`.
            Yvars=self.Yvars,
            # pyre-fixme[6]: For 4th param expected `List[List[Union[None, bool,
            #  float, int, str]]]` but got `List[List[int]]`.
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        arms, weights, _ = generator.gen(
            n=5,
            # pyre-fixme[6]: For 2nd param expected `List[List[Union[None, bool,
            #  float, int, str]]]` but got `List[List[int]]`.
            parameter_values=self.parameter_values,
            objective_weights=np.ones(1),
        )
        self.assertEqual(arms, [[4, 4], [3, 3], [2, 2]])
        for weight, expected_weight in zip(
            weights, [3 * i for i in [0.725, 0.225, 0.05]]
        ):
            self.assertAlmostEqual(weight, expected_weight, 1)

    def testThompsonSamplerUniformWeights(self) -> None:
        generator = ThompsonSampler(min_weight=0.0, uniform_weights=True)
        generator.fit(
            # pyre-fixme[6]: For 1st param expected `List[List[List[Union[None,
            #  bool, float, int, str]]]]` but got `List[List[List[int]]]`.
            Xs=self.Xs,
            # pyre-fixme[6]: For 2nd param expected `List[List[float]]` but got
            #  `List[List[int]]`.
            Ys=self.Ys,
            # pyre-fixme[6]: For 3rd param expected `List[List[float]]` but got
            #  `List[List[int]]`.
            Yvars=self.Yvars,
            # pyre-fixme[6]: For 4th param expected `List[List[Union[None, bool,
            #  float, int, str]]]` but got `List[List[int]]`.
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        arms, weights, _ = generator.gen(
            n=3,
            # pyre-fixme[6]: For 2nd param expected `List[List[Union[None, bool,
            #  float, int, str]]]` but got `List[List[int]]`.
            parameter_values=self.parameter_values,
            objective_weights=np.ones(1),
        )
        self.assertEqual(arms, [[4, 4], [3, 3], [2, 2]])
        for weight, expected_weight in zip(weights, [1.0, 1.0, 1.0]):
            self.assertAlmostEqual(weight, expected_weight, 1)

    def testThompsonSamplerInfeasible(self) -> None:
        generator = ThompsonSampler(min_weight=0.9)
        generator.fit(
            # pyre-fixme[6]: For 1st param expected `List[List[List[Union[None,
            #  bool, float, int, str]]]]` but got `List[List[List[int]]]`.
            Xs=self.Xs,
            # pyre-fixme[6]: For 2nd param expected `List[List[float]]` but got
            #  `List[List[int]]`.
            Ys=self.Ys,
            # pyre-fixme[6]: For 3rd param expected `List[List[float]]` but got
            #  `List[List[int]]`.
            Yvars=self.Yvars,
            # pyre-fixme[6]: For 4th param expected `List[List[Union[None, bool,
            #  float, int, str]]]` but got `List[List[int]]`.
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        with self.assertRaises(ModelError):
            generator.gen(
                n=3,
                # pyre-fixme[6]: For 2nd param expected `List[List[Union[None, bool,
                #  float, int, str]]]` but got `List[List[int]]`.
                parameter_values=self.parameter_values,
                objective_weights=np.ones(1),
            )

    def testThompsonSamplerOutcomeConstraints(self) -> None:
        generator = ThompsonSampler(min_weight=0.0)
        generator.fit(
            # pyre-fixme[6]: For 1st param expected `List[List[List[Union[None,
            #  bool, float, int, str]]]]` but got `List[List[List[int]]]`.
            Xs=self.multiple_metrics_Xs,
            # pyre-fixme[6]: For 2nd param expected `List[List[float]]` but got
            #  `List[List[int]]`.
            Ys=self.multiple_metrics_Ys,
            # pyre-fixme[6]: For 3rd param expected `List[List[float]]` but got
            #  `List[List[int]]`.
            Yvars=self.multiple_metrics_Yvars,
            # pyre-fixme[6]: For 4th param expected `List[List[Union[None, bool,
            #  float, int, str]]]` but got `List[List[int]]`.
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        arms, weights, _ = generator.gen(
            n=4,
            # pyre-fixme[6]: For 2nd param expected `List[List[Union[None, bool,
            #  float, int, str]]]` but got `List[List[int]]`.
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

    def testThompsonSamplerOutcomeConstraintsInfeasible(self) -> None:
        generator = ThompsonSampler(min_weight=0.0)
        generator.fit(
            # pyre-fixme[6]: For 1st param expected `List[List[List[Union[None,
            #  bool, float, int, str]]]]` but got `List[List[List[int]]]`.
            Xs=self.multiple_metrics_Xs,
            # pyre-fixme[6]: For 2nd param expected `List[List[float]]` but got
            #  `List[List[int]]`.
            Ys=self.multiple_metrics_Ys,
            # pyre-fixme[6]: For 3rd param expected `List[List[float]]` but got
            #  `List[List[int]]`.
            Yvars=self.multiple_metrics_Yvars,
            # pyre-fixme[6]: For 4th param expected `List[List[Union[None, bool,
            #  float, int, str]]]` but got `List[List[int]]`.
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        with self.assertRaises(ModelError):
            generator.gen(
                n=3,
                # pyre-fixme[6]: For 2nd param expected `List[List[Union[None, bool,
                #  float, int, str]]]` but got `List[List[int]]`.
                parameter_values=self.parameter_values,
                objective_weights=np.ones(2),
                outcome_constraints=(np.array([[0, 1]]), np.array([[-10]])),
            )

    def testThompsonSamplerPredict(self) -> None:
        generator = ThompsonSampler(min_weight=0.0)
        generator.fit(
            # pyre-fixme[6]: For 1st param expected `List[List[List[Union[None,
            #  bool, float, int, str]]]]` but got `List[List[List[int]]]`.
            Xs=self.Xs,
            # pyre-fixme[6]: For 2nd param expected `List[List[float]]` but got
            #  `List[List[int]]`.
            Ys=self.Ys,
            # pyre-fixme[6]: For 3rd param expected `List[List[float]]` but got
            #  `List[List[int]]`.
            Yvars=self.Yvars,
            # pyre-fixme[6]: For 4th param expected `List[List[Union[None, bool,
            #  float, int, str]]]` but got `List[List[int]]`.
            parameter_values=self.parameter_values,
            outcome_names=self.outcome_names,
        )
        f, cov = generator.predict([[1, 1], [3, 3]])
        self.assertTrue(np.array_equal(f, np.array([[1], [3]])))
        self.assertTrue(np.array_equal(cov, np.ones((2, 1, 1))))

        with self.assertRaises(ValueError):
            generator.predict([[1, 2]])
