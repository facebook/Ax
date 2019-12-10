#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import numpy as np
from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.simple_experiment import SimpleExperiment, TEvaluationOutcome
from ax.core.types import TParameterization
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_search_space


def _get_sum(parameterization: TParameterization) -> float:
    param_names = list(parameterization.keys())
    if any(param_name not in param_names for param_name in ["x1", "x2"]):
        raise ValueError("Parametrization does not contain x1 or x2")
    x1, x2 = parameterization["x1"], parameterization["x2"]
    return x1 + x2


def sum_evaluation_function(
    parameterization: TParameterization, weight: Optional[float] = None
) -> TEvaluationOutcome:
    sum = _get_sum(parameterization)
    return {"sum": (sum, 0.0)}


def sum_evaluation_function_numpy(
    parameterization: TParameterization, weight: Optional[float] = None
) -> TEvaluationOutcome:
    sum = np.float32(_get_sum(parameterization))
    return {"sum": (sum, np.float32(0.0))}


def sum_evaluation_function_v2(
    parameterization: TParameterization, weight: Optional[float] = None
) -> TEvaluationOutcome:
    sum = _get_sum(parameterization)
    return (sum, 0.0)


def sum_evaluation_function_v2_numpy(
    parameterization: TParameterization, weight: Optional[float] = None
) -> TEvaluationOutcome:
    sum = np.float32(_get_sum(parameterization))
    return (sum, np.float32(0.0))


def sum_evaluation_function_v3(
    parameterization: TParameterization, weight: Optional[float] = None
) -> TEvaluationOutcome:
    return _get_sum(parameterization)


def sum_evaluation_function_v3_numpy(
    parameterization: TParameterization, weight: Optional[float] = None
) -> TEvaluationOutcome:
    return np.float32(_get_sum(parameterization))


def sum_evaluation_function_v4(
    parameterization: TParameterization,
) -> TEvaluationOutcome:
    return _get_sum(parameterization)


def sum_evaluation_function_v4_numpy(
    parameterization: TParameterization,
) -> TEvaluationOutcome:
    return np.float32(_get_sum(parameterization))


class SimpleExperimentTest(TestCase):
    def setUp(self) -> None:
        self.experiment = SimpleExperiment(
            name="test_branin",
            search_space=get_branin_search_space(),
            evaluation_function=sum_evaluation_function,
            objective_name="sum",
        )
        self.arms = [
            Arm(parameters={"x1": 0.75, "x2": 1}),
            Arm(parameters={"x1": 2, "x2": 7}),
            Arm(parameters={"x1": 10, "x2": 8}),
            Arm(parameters={"x1": -2, "x2": 10}),
        ]

    def testBasic(self) -> None:
        self.assertTrue(self.experiment.is_simple_experiment)
        trial = self.experiment.new_trial()
        with self.assertRaises(NotImplementedError):
            trial.runner = SyntheticRunner()
        with self.assertRaises(NotImplementedError):
            self.experiment.add_tracking_metric(Metric(name="test"))
        with self.assertRaises(NotImplementedError):
            self.experiment.update_tracking_metric(Metric(name="test"))
        self.assertTrue(self.experiment.eval_trial(trial).df.empty)
        batch = self.experiment.new_batch_trial()
        batch.add_arm(Arm(parameters={"x1": 5, "x2": 10}))
        self.assertEqual(self.experiment.eval_trial(batch).df["mean"][0], 15)
        self.experiment.new_batch_trial().add_arm(Arm(parameters={"x1": 15, "x2": 25}))
        self.assertAlmostEqual(self.experiment.eval().df["mean"][1], 40)
        self.assertEqual(batch.fetch_data().df["mean"][0], 15)
        self.assertAlmostEqual(self.experiment.fetch_data().df["mean"][1], 40)

    def testTrial(self) -> None:
        for i in range(len(self.arms)):
            self.experiment.new_trial(generator_run=GeneratorRun(arms=[self.arms[i]]))
        self.assertFalse(self.experiment.eval().df.empty)

    def testUnimplementedEvaluationFunction(self) -> None:
        experiment = SimpleExperiment(
            name="test_branin",
            search_space=get_branin_search_space(),
            objective_name="sum",
        )
        with self.assertRaises(Exception):
            experiment.evaluation_function(parameterization={})

        experiment.evaluation_function = sum_evaluation_function

    def testEvaluationFunctionNumpy(self) -> None:
        experiment = SimpleExperiment(
            name="test_branin",
            search_space=get_branin_search_space(),
            objective_name="sum",
            evaluation_function=sum_evaluation_function_numpy,
        )

        for i in range(len(self.arms)):
            experiment.new_trial(generator_run=GeneratorRun(arms=[self.arms[i]]))
        self.assertFalse(experiment.eval().df.empty)

    def testEvaluationFunctionV2(self) -> None:
        experiment = SimpleExperiment(
            name="test_branin",
            search_space=get_branin_search_space(),
            objective_name="sum",
            evaluation_function=sum_evaluation_function_v2,
        )

        for i in range(len(self.arms)):
            experiment.new_trial(generator_run=GeneratorRun(arms=[self.arms[i]]))
        self.assertFalse(experiment.eval().df.empty)

    def testEvaluationFunctionV2Numpy(self) -> None:
        experiment = SimpleExperiment(
            name="test_branin",
            search_space=get_branin_search_space(),
            objective_name="sum",
            evaluation_function=sum_evaluation_function_v2_numpy,
        )

        for i in range(len(self.arms)):
            experiment.new_trial(generator_run=GeneratorRun(arms=[self.arms[i]]))
        self.assertFalse(experiment.eval().df.empty)

    def testEvaluationFunctionV3(self) -> None:
        experiment = SimpleExperiment(
            name="test_branin",
            search_space=get_branin_search_space(),
            objective_name="sum",
            evaluation_function=sum_evaluation_function_v3,
        )

        for i in range(len(self.arms)):
            experiment.new_trial(generator_run=GeneratorRun(arms=[self.arms[i]]))
        self.assertFalse(experiment.eval().df.empty)

    def testEvaluationFunctionV3Numpy(self) -> None:
        experiment = SimpleExperiment(
            name="test_branin",
            search_space=get_branin_search_space(),
            objective_name="sum",
            evaluation_function=sum_evaluation_function_v3_numpy,
        )

        for i in range(len(self.arms)):
            experiment.new_trial(generator_run=GeneratorRun(arms=[self.arms[i]]))
        self.assertFalse(experiment.eval().df.empty)

    def testEvaluationFunctionV4(self) -> None:
        experiment = SimpleExperiment(
            name="test_branin",
            search_space=get_branin_search_space(),
            objective_name="sum",
            evaluation_function=sum_evaluation_function_v4,
        )

        for i in range(len(self.arms)):
            experiment.new_trial(generator_run=GeneratorRun(arms=[self.arms[i]]))
        self.assertFalse(experiment.eval().df.empty)

    def testEvaluationFunctionV4Numpy(self) -> None:
        experiment = SimpleExperiment(
            name="test_branin",
            search_space=get_branin_search_space(),
            objective_name="sum",
            evaluation_function=sum_evaluation_function_v4_numpy,
        )

        for i in range(len(self.arms)):
            experiment.new_trial(generator_run=GeneratorRun(arms=[self.arms[i]]))
        self.assertFalse(experiment.eval().df.empty)

    def testOptionalObjectiveName(self) -> None:
        experiment = SimpleExperiment(
            name="test_branin",
            search_space=get_branin_search_space(),
            evaluation_function=sum_evaluation_function_v2,
        )

        for i in range(len(self.arms)):
            experiment.new_trial(generator_run=GeneratorRun(arms=[self.arms[i]]))
        self.assertFalse(experiment.eval().df.empty)
