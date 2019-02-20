#!/usr/bin/env python3
from typing import Optional

from ae.lazarus.ae.core.condition import Condition
from ae.lazarus.ae.core.generator_run import GeneratorRun
from ae.lazarus.ae.core.simple_experiment import SimpleExperiment, TEvaluationOutcome
from ae.lazarus.ae.core.types.types import TParameterization
from ae.lazarus.ae.tests.fake import get_branin_search_space
from ae.lazarus.ae.utils.common.testutils import TestCase


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
        self.conditions = [
            Condition(params={"x1": 0.75, "x2": 1}),
            Condition(params={"x1": 2, "x2": 7}),
            Condition(params={"x1": 10, "x2": 8}),
            Condition(params={"x1": -2, "x2": 10}),
        ]

    def test_basic(self):
        trial = self.experiment.new_trial()
        self.assertTrue(self.experiment.eval_trial(trial).df.empty)
        batch = self.experiment.new_batch_trial()
        batch.add_condition(Condition(params={"x1": 5, "x2": 10}))
        self.assertEqual(self.experiment.eval_trial(batch).df["mean"][0], 15)
        self.experiment.new_batch_trial().add_condition(
            Condition(params={"x1": 15, "x2": 25})
        )
        self.assertAlmostEqual(self.experiment.eval().df["mean"][1], 40)

        # fetch_data -> eval, fetch_trial_data -> eval_trial
        self.assertEqual(
            self.experiment.fetch_trial_data(batch.index).df["mean"][0], 15
        )
        self.assertAlmostEqual(self.experiment.fetch_data().df["mean"][1], 40)

    def test_trial(self):
        for i in range(len(self.conditions)):
            self.experiment.new_trial(
                generator_run=GeneratorRun(conditions=[self.conditions[i]])
            )
        self.assertFalse(self.experiment.eval().df.empty)
