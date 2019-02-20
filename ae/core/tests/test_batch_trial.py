#!/usr/bin/env python3

from unittest.mock import PropertyMock, patch

import numpy as np
from ae.lazarus.ae.core.base_trial import TrialStatus
from ae.lazarus.ae.core.condition import Condition
from ae.lazarus.ae.core.generator_run import GeneratorRun
from ae.lazarus.ae.runners.synthetic import SyntheticRunner
from ae.lazarus.ae.tests.fake import (
    get_abandoned_condition,
    get_condition,
    get_condition_weights,
    get_conditions,
    get_experiment,
    get_weights,
)
from ae.lazarus.ae.utils.common.testutils import TestCase


class BatchTrialTest(TestCase):
    def setUp(self):
        self.experiment = get_experiment()
        self.experiment.status_quo = None
        self.batch = self.experiment.new_batch_trial()
        conditions = get_conditions()
        weights = get_weights()
        self.status_quo = conditions[0]
        self.sq_weight = weights[0]
        self.conditions = conditions[1:]
        self.weights = weights[1:]
        self.batch.add_conditions_and_weights(
            conditions=self.conditions, weights=self.weights
        )

    def testEq(self):
        new_batch_trial = self.experiment.new_batch_trial()
        self.assertNotEqual(self.batch, new_batch_trial)

        abandoned_condition = get_abandoned_condition()
        abandoned_condition_2 = get_abandoned_condition()
        self.assertEqual(abandoned_condition, abandoned_condition_2)

    def testBasicProperties(self):
        self.assertEqual(self.experiment, self.batch.experiment)
        self.assertEqual(self.batch.index, 0)
        self.assertEqual(self.batch.status, TrialStatus.CANDIDATE)
        self.assertIsNotNone(self.batch.time_created)
        self.assertEqual(self.batch.conditions_by_name["0_0"], self.batch.conditions[0])
        self.assertEqual(self.batch.conditions_by_name["0_1"], self.batch.conditions[1])

        # Test empty conditions
        self.assertEqual(len(self.experiment.new_batch_trial().abandoned_conditions), 0)

    def testUndefinedSetters(self):
        with self.assertRaises(NotImplementedError):
            self.batch.condition_weights = get_condition_weights()

        with self.assertRaises(NotImplementedError):
            self.batch.status = TrialStatus.RUNNING

    def testBasicSetter(self):
        self.batch.runner = SyntheticRunner()
        self.assertIsNotNone(self.batch.runner)

        self.batch.trial_type = None
        self.assertIsNone(self.batch.trial_type)

        # Default experiment only supports None as trial_type
        with self.assertRaises(ValueError):
            self.batch.trial_type = ""

    def testAddCondition(self):
        self.assertEqual(len(self.batch.conditions), len(self.conditions))
        self.assertEqual(len(self.batch.generator_run_structs), 1)
        self.assertEqual(sum(self.batch.weights), sum(self.weights))

        condition_params = get_condition().params
        condition_params["w"] = 5.0
        self.batch.add_condition(Condition(condition_params), 3)

        self.assertEqual(self.batch.conditions_by_name["0_2"], self.batch.conditions[2])
        self.assertEqual(len(self.batch.conditions), len(self.conditions) + 1)
        self.assertEqual(len(self.batch.generator_run_structs), 2)
        self.assertEqual(sum(self.batch.weights), sum(self.weights) + 3)

    def testAddGeneratorRun(self):
        self.assertEqual(len(self.batch.conditions), len(self.conditions))
        self.assertEqual(len(self.batch.generator_run_structs), 1)
        self.assertEqual(sum(self.batch.weights), sum(self.weights))

        # one of these conditions already exists on the BatchTrial,
        # so we should just update its weight
        new_conditions = [
            Condition(params={"w": 0.75, "x": 1, "y": "foo", "z": True}),
            Condition(params={"w": 1.4, "x": 5, "y": "bar", "z": False}),
        ]
        new_weights = [0.75, 0.25]
        gr = GeneratorRun(conditions=new_conditions, weights=new_weights)
        self.batch.add_generator_run(gr, 2.0)

        self.assertEqual(len(self.batch.conditions), len(self.conditions) + 1)
        self.assertEqual(len(self.batch.generator_run_structs), 2)
        self.assertEqual(sum(self.batch.weights), sum(self.weights) + 2)

    def testStatusQuoOverlap(self):
        tot_weight = sum(self.batch.weights)
        new_sq = Condition(params={"w": 0.95, "x": 1, "y": "foo", "z": True})

        # Set status quo to existing condition
        self.batch.set_status_quo(self.conditions[0], self.sq_weight)
        self.assertTrue(self.batch.status_quo == self.conditions[0])
        self.assertEqual(self.batch.status_quo.name, self.conditions[0].name)
        self.assertEqual(
            self.batch.condition_weights[self.conditions[0]],
            self.weights[0] + self.sq_weight,
        )
        self.assertEqual(sum(self.batch.weights), tot_weight + self.sq_weight)

        # Set status quo to new condition, add it
        self.batch.set_status_quo(new_sq, self.sq_weight)
        self.assertEqual(self.batch.status_quo.name, "status_quo_0")
        self.batch.add_conditions_and_weights([new_sq])
        self.assertEqual(
            self.batch.generator_run_structs[1].generator_run.conditions[0].name,
            "status_quo_0",
        )

    def testStatusQuo(self):
        tot_weight = sum(self.batch.weights)
        num_conditions = len(self.batch.conditions)
        avg_weight = float(tot_weight) / num_conditions
        new_sq = Condition(params={"w": 0.95, "x": 1, "y": "foo", "z": True})

        # Test negative weight
        with self.assertRaises(ValueError):
            self.batch.set_status_quo(new_sq, -1)

        # Set status quo to new condition
        self.batch.set_status_quo(new_sq, self.sq_weight)
        self.assertTrue(self.batch.status_quo == new_sq)
        self.assertEqual(self.batch.status_quo.name, "status_quo_0")
        self.assertEqual(sum(self.batch.weights), tot_weight + self.sq_weight)
        # sq weight should be ignored when sq is None
        self.batch.set_status_quo(None)
        self.assertEqual(sum(self.batch.weights), tot_weight)

        # Verify experiment status quo gets set on init
        self.experiment.status_quo = self.status_quo
        batch2 = self.batch.clone()
        self.assertEqual(batch2.status_quo, self.experiment.status_quo)
        self.assertEqual(batch2._status_quo_weight, avg_weight)

        # Try setting sq to existing arm with different name
        with self.assertRaises(ValueError):
            self.batch.set_status_quo(
                Condition(self.conditions[0].params, name="new_name")
            )

    def testBatchLifecycle(self):
        self.batch.runner = SyntheticRunner()
        self.batch.run()
        self.assertEqual(self.batch.status, TrialStatus.STAGED)
        self.assertIsNotNone(self.batch.time_staged)
        self.assertTrue(self.batch.status.is_deployed)
        self.assertTrue(self.batch.status.expecting_data)

        # Cannot change conditions or runner once run
        with self.assertRaises(ValueError):
            self.batch.add_conditions_and_weights(
                conditions=self.conditions, weights=self.weights
            )

        with self.assertRaises(ValueError):
            self.batch.runner = None

        self.batch.mark_running()
        self.assertEqual(self.batch.status, TrialStatus.RUNNING)
        self.assertIsNotNone(self.batch.time_run_started)

        self.batch.mark_completed()
        self.assertEqual(self.batch.status, TrialStatus.COMPLETED)
        self.assertIsNotNone(self.batch.time_completed)
        self.assertTrue(self.batch.status.is_terminal)

        # Cannot change status after BatchTrial is completed
        with self.assertRaises(ValueError):
            self.batch.mark_staged()

        with self.assertRaises(ValueError):
            self.batch.mark_completed()

        with self.assertRaises(ValueError):
            self.batch.mark_running()

        with self.assertRaises(ValueError):
            self.batch.mark_abandoned()

        with self.assertRaises(ValueError):
            self.batch.mark_failed()

    def testAbandonBatchTrial(self):
        reason = "BatchTrial behaved poorly"
        self.batch.mark_abandoned(reason)

        self.assertEqual(self.batch.status, TrialStatus.ABANDONED)
        self.assertIsNotNone(self.batch.time_completed)
        self.assertEqual(self.batch.abandoned_reason, reason)

    def testFailedBatchTrial(self):
        self.batch.runner = SyntheticRunner()
        self.batch.run()
        self.batch.mark_running()
        self.batch.mark_failed()

        self.assertEqual(self.batch.status, TrialStatus.FAILED)
        self.assertIsNotNone(self.batch.time_completed)

    def testAbandonCondition(self):
        condition = self.batch.conditions[0]
        reason = "Bad condition"
        self.batch.mark_condition_abandoned(condition, reason)
        self.assertEqual(len(self.batch.abandoned_conditions), 1)
        self.assertEqual(self.batch.abandoned_conditions[0], condition)

        self.assertEqual(len(self.batch.abandoned_conditions_metadata), 1)
        metadata = self.batch.abandoned_conditions_metadata[0]
        self.assertEqual(metadata.reason, reason)
        self.assertEqual(metadata.name, condition.name)

        # Fail to abandon condition not in BatchTrial
        with self.assertRaises(ValueError):
            self.batch.mark_condition_abandoned(
                Condition(params={"x": 3, "y": "fooz", "z": False})
            )

    def testClone(self):
        new_batch_trial = self.batch.clone()
        self.assertEqual(len(new_batch_trial.generator_run_structs), 1)
        self.assertEqual(len(new_batch_trial.conditions), 2)
        self.assertEqual(new_batch_trial.runner, self.batch.runner)
        self.assertEqual(new_batch_trial.trial_type, self.batch.trial_type)

    def testRunner(self):
        # Verify BatchTrial without runner will fail
        with self.assertRaises(ValueError):
            self.batch.run()

        # Verify mark running without runner will fail
        with self.assertRaises(ValueError):
            self.batch.mark_running()

        self.batch.runner = SyntheticRunner()
        self.batch.run()
        self.assertEqual(self.batch.deployed_name, "test_0")
        self.assertNotEqual(len(self.batch.run_metadata.keys()), 0)
        self.assertEqual(self.batch.status, TrialStatus.STAGED)

        # Verify setting runner on experiment but not BatchTrial
        # Also mock staging_required to be false
        staging_mock = PropertyMock()
        with patch.object(SyntheticRunner, "staging_required", staging_mock):
            mock_runner = SyntheticRunner()
            staging_mock.return_value = False

            self.experiment.runner = mock_runner
            b2 = self.experiment.new_batch_trial()
            b2.run()
            self.assertEqual(b2.deployed_name, "test_1")
            self.assertEqual(b2.status, TrialStatus.RUNNING)

    def testIsFactorial(self):
        self.assertFalse(self.batch.is_factorial)

        new_batch_trial = self.experiment.new_batch_trial()
        new_batch_trial.add_conditions_and_weights(
            conditions=[Condition(params={"x": 1})]
        )
        self.assertFalse(new_batch_trial.is_factorial)

        new_batch_trial = self.experiment.new_batch_trial()
        new_batch_trial.add_conditions_and_weights(
            conditions=[
                Condition(params={"x": 1, "y": 1}),
                Condition(params={"x": 2, "y": 2}),
                Condition(params={"x": 1, "y": 2}),
                Condition(params={"x": 2, "y": 1}),
            ]
        )
        self.assertTrue(new_batch_trial.is_factorial)

    def testNormalizedConditionWeights(self):
        new_batch_trial = self.experiment.new_batch_trial()
        parameterizations = [
            {"y": 0.25, "x": 0.75, "z": 75},
            {"y": 0.375, "x": 0.375, "z": 63},
        ]
        conditions = [Condition(params=p) for i, p in enumerate(parameterizations)]
        new_batch_trial.add_conditions_and_weights(
            conditions=conditions, weights=[2, 1]
        )

        # test normalizing to 1
        condition_weights = new_batch_trial.normalized_condition_weights()
        self.assertEqual(list(condition_weights.keys()), conditions)
        self.assertTrue(np.allclose(list(condition_weights.values()), [2 / 3, 1 / 3]))

        # test normalizing to 100
        condition_weights = new_batch_trial.normalized_condition_weights(total=100)
        self.assertEqual(list(condition_weights.keys()), conditions)
        self.assertTrue(
            np.allclose(list(condition_weights.values()), [200 / 3, 100 / 3])
        )

        # test normalizing with truncation
        condition_weights = new_batch_trial.normalized_condition_weights(
            total=1, trunc_digits=2
        )
        self.assertEqual(list(condition_weights.keys()), conditions)
        self.assertTrue(np.allclose(list(condition_weights.values()), [0.67, 0.33]))

    def testAddGeneratorRunValidation(self):
        new_batch_trial = self.experiment.new_batch_trial()
        new_conditions = [
            Condition(name="0_1", params={"w": 0.75, "x": 1, "y": "foo", "z": True}),
            Condition(name="0_2", params={"w": 0.75, "x": 1, "y": "foo", "z": True}),
        ]
        gr = GeneratorRun(conditions=new_conditions)
        with self.assertRaises(ValueError):
            new_batch_trial.add_generator_run(gr)

    def testRepr(self):
        repr_ = (
            "BatchTrial(experiment_name='test', index=0, status=TrialStatus.CANDIDATE)"
        )  # noqa
        self.assertEqual(str(self.batch), repr_)
