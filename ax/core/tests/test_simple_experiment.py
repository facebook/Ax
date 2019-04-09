#!/usr/bin/env python3
from typing import Optional

from ax.core.arm import Arm
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.simple_experiment import SimpleExperiment, TEvaluationOutcome
from ax.core.types import TParameterization
from ax.runners.synthetic import SyntheticRunner
from ax.tests.fake import get_branin_search_space
from ax.utils.common.testutils import TestCase


def sum_evaluation_function(
    parameterization: TParameterization, weight: Optional[float] = None
) -> TEvaluationOutcome:
    param_names = list(parameterization.keys())
    if any(param_name not in param_names for param_name in ["x1", "x2"]):
        raise ValueError("Parametrization does not contain x1 or x2")
    x1, x2 = parameterization["x1"], parameterization["x2"]
    return {"sum": (x1 + x2, 0.0)}


class SimpleExperimentTest(TestCase):
    def setUp(self):
        self.experiment = SimpleExperiment(
            name="test_branin",
            search_space=get_branin_search_space(),
            evaluation_function=sum_evaluation_function,
            objective_name="sum",
        )
        self.arms = [
            Arm(params={"x1": 0.75, "x2": 1}),
            Arm(params={"x1": 2, "x2": 7}),
            Arm(params={"x1": 10, "x2": 8}),
            Arm(params={"x1": -2, "x2": 10}),
        ]

    def test_basic(self):
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
        batch.add_arm(Arm(params={"x1": 5, "x2": 10}))
        self.assertEqual(self.experiment.eval_trial(batch).df["mean"][0], 15)
        self.experiment.new_batch_trial().add_arm(Arm(params={"x1": 15, "x2": 25}))
        self.assertAlmostEqual(self.experiment.eval().df["mean"][1], 40)
        self.assertEqual(
            self.experiment.fetch_trial_data(batch.index).df["mean"][0], 15
        )
        self.assertAlmostEqual(self.experiment.fetch_data().df["mean"][1], 40)

    def test_trial(self):
        for i in range(len(self.arms)):
            self.experiment.new_trial(generator_run=GeneratorRun(arms=[self.arms[i]]))
        self.assertFalse(self.experiment.eval().df.empty)

    def test_unimplemented_evaluation_function(self):
        experiment = SimpleExperiment(
            name="test_branin",
            search_space=get_branin_search_space(),
            objective_name="sum",
        )
        with self.assertRaises(Exception):
            experiment.evaluation_function(parameters={})

        experiment.evaluation_function = sum_evaluation_function
