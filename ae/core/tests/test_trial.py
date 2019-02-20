#!/usr/bin/env python3

from unittest.mock import patch

import pandas as pd
from ae.lazarus.ae.core.base_trial import TrialStatus
from ae.lazarus.ae.core.data import Data
from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.core.generator_run import GeneratorRun
from ae.lazarus.ae.tests.fake import get_conditions, get_experiment, get_objective
from ae.lazarus.ae.utils.common.testutils import TestCase


TEST_DATA = Data(
    df=pd.DataFrame(
        [
            {
                "condition_name": "0_0",
                "metric_name": get_objective().metric.name,
                "mean": 1.0,
                "sem": 2.0,
                "trial_index": 0,
            }
        ]
    )
)


class TrialTest(TestCase):
    def setUp(self):
        self.experiment = get_experiment()
        self.trial = self.experiment.new_trial()
        self.condition = get_conditions()[0]
        self.trial.add_condition(self.condition)

    def test_eq(self):
        new_trial = self.experiment.new_trial()
        self.assertNotEqual(self.trial, new_trial)

    def test_basic_properties(self):
        self.assertEqual(self.experiment, self.trial.experiment)
        self.assertEqual(self.trial.index, 0)
        self.assertEqual(self.trial.status, TrialStatus.CANDIDATE)
        self.assertIsNotNone(self.trial.time_created)
        self.assertEqual(self.trial.conditions_by_name["0_0"], self.trial.condition)
        self.assertEqual(self.trial.conditions, [self.condition])

        # Test empty conditions
        with self.assertRaises(AttributeError):
            self.experiment.new_trial().condition_weights
        with self.assertRaises(AttributeError):
            self.experiment.new_trial().abandoned_conditions

    def test_adding_new_trials(self):
        new_condition = get_conditions()[1]
        new_trial = self.experiment.new_trial(
            generator_run=GeneratorRun(conditions=[new_condition])
        )
        with self.assertRaises(ValueError):
            self.experiment.new_trial(
                generator_run=GeneratorRun(conditions=get_conditions())
            )
        self.assertEqual(new_trial.conditions_by_name["1_0"], new_condition)
        with self.assertRaises(KeyError):
            self.trial.conditions_by_name["1_0"]

    def test_add_trial_same_condition(self):
        # Check that adding new condition w/out name works correctly.
        new_trial1 = self.experiment.new_trial(
            generator_run=GeneratorRun(
                conditions=[self.condition.clone(clear_name=True)]
            )
        )
        self.assertEqual(new_trial1.condition.name, self.trial.condition.name)
        self.assertFalse(new_trial1.condition is self.trial.condition)
        # Check that adding new condition with name works correctly.
        new_trial2 = self.experiment.new_trial(
            generator_run=GeneratorRun(conditions=[self.condition.clone()])
        )
        self.assertEqual(new_trial2.condition.name, self.trial.condition.name)
        self.assertFalse(new_trial2.condition is self.trial.condition)
        condition_wrong_name = self.condition.clone(clear_name=True)
        condition_wrong_name.name = "wrong_name"
        with self.assertRaises(ValueError):
            new_trial2 = self.experiment.new_trial(
                generator_run=GeneratorRun(conditions=[condition_wrong_name])
            )

    def test_abandonment(self):
        self.assertFalse(self.trial.is_abandoned)
        self.trial.mark_abandoned(reason="testing")
        self.assertTrue(self.trial.is_abandoned)

    @patch(
        f"{Experiment.__module__}.{Experiment.__name__}.fetch_trial_data",
        return_value=TEST_DATA,
    )
    def test_objective_mean(self, _mock):
        self.assertEqual(self.trial.objective_mean, 1.0)

    @patch(
        f"{Experiment.__module__}.{Experiment.__name__}.fetch_trial_data",
        return_value=Data(),
    )
    def test_objective_mean_empty_df(self, _mock):
        self.assertIsNone(self.trial.objective_mean)

    def testRepr(self):
        repr_ = "Trial(experiment_name='test', index=0, status=TrialStatus.CANDIDATE)"
        self.assertEqual(str(self.trial), repr_)
