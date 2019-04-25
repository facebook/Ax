#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.utils.common.testutils import TestCase
from ax.utils.testing.fake import (
    get_arms,
    get_model_predictions,
    get_model_predictions_per_arm,
    get_optimization_config,
    get_search_space,
)


GENERATOR_RUN_STR = "GeneratorRun(3 arms, total weight 3.0)"
GENERATOR_RUN_STR_PLUS_1 = "GeneratorRun(3 arms, total weight 4.0)"


class GeneratorRunTest(TestCase):
    def setUp(self):
        self.model_predictions = get_model_predictions()
        self.optimization_config = get_optimization_config()
        self.search_space = get_search_space()

        self.arms = get_arms()
        self.weights = [2, 1, 1]
        self.unweighted_run = GeneratorRun(
            arms=self.arms,
            optimization_config=self.optimization_config,
            search_space=self.search_space,
            model_predictions=self.model_predictions,
            fit_time=4.0,
            gen_time=10.0,
        )
        self.weighted_run = GeneratorRun(
            arms=self.arms,
            weights=self.weights,
            optimization_config=self.optimization_config,
            search_space=self.search_space,
            model_predictions=self.model_predictions,
        )

    def testInit(self):
        self.assertEqual(
            len(self.unweighted_run.optimization_config.outcome_constraints),
            len(self.optimization_config.outcome_constraints),
        )
        self.assertEqual(
            len(self.unweighted_run.search_space.parameters),
            len(self.search_space.parameters),
        )
        self.assertEqual(str(self.unweighted_run), GENERATOR_RUN_STR)
        self.assertIsNotNone(self.unweighted_run.time_created)
        self.assertEqual(self.unweighted_run.generator_run_type, None)
        self.assertEqual(self.unweighted_run.fit_time, 4.0)
        self.assertEqual(self.unweighted_run.gen_time, 10.0)

        with self.assertRaises(ValueError):
            GeneratorRun(
                arms=self.arms,
                weights=[],
                optimization_config=self.optimization_config,
                search_space=self.search_space,
            )

    def testClone(self):
        weighted_run2 = self.weighted_run.clone()
        self.assertEqual(
            self.weighted_run.optimization_config, weighted_run2.optimization_config
        )
        weighted_run2.arms[0].name = "bogus_name"
        self.assertNotEqual(self.weighted_run.arms, weighted_run2.arms)

    def testMergeDuplicateArm(self):
        arms = self.arms + [self.arms[0]]
        run = GeneratorRun(
            arms=arms,
            optimization_config=self.optimization_config,
            search_space=self.search_space,
            model_predictions=self.model_predictions,
        )
        self.assertEqual(str(run), GENERATOR_RUN_STR_PLUS_1)

    def testIndex(self):
        self.assertIsNone(self.unweighted_run.index)
        self.unweighted_run.index = 1
        with self.assertRaises(ValueError):
            self.unweighted_run.index = 2

    def testModelPredictions(self):
        self.assertEqual(self.unweighted_run.model_predictions, get_model_predictions())
        self.assertEqual(
            self.unweighted_run.model_predictions_by_arm,
            get_model_predictions_per_arm(),
        )
        run_no_model_predictions = GeneratorRun(
            arms=self.arms,
            weights=self.weights,
            optimization_config=get_optimization_config(),
            search_space=get_search_space(),
        )
        self.assertIsNone(run_no_model_predictions.model_predictions)
        self.assertIsNone(run_no_model_predictions.model_predictions_by_arm)

    def testEq(self):
        self.assertEqual(self.unweighted_run, self.unweighted_run)

        arms = [
            Arm(parameters={"w": 0.5, "x": 15, "y": "foo", "z": False}),
            Arm(parameters={"w": 1.4, "x": 2, "y": "bar", "z": True}),
        ]
        unweighted_run_2 = GeneratorRun(
            arms=arms,
            optimization_config=self.optimization_config,
            search_space=self.search_space,
        )
        self.assertNotEqual(self.unweighted_run, unweighted_run_2)

    def testParamDf(self):
        param_df = self.unweighted_run.param_df
        self.assertEqual(len(param_df), len(self.arms))

    def testBestArm(self):
        generator_run = GeneratorRun(
            arms=self.arms,
            weights=self.weights,
            optimization_config=get_optimization_config(),
            search_space=get_search_space(),
            best_arm_predictions=(self.arms[0], ({"a": 1.0}, {"a": {"a": 2.0}})),
        )
        self.assertEqual(
            generator_run.best_arm_predictions,
            (self.arms[0], ({"a": 1.0}, {"a": {"a": 2.0}})),
        )
