#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from unittest.mock import PropertyMock, patch

import numpy as np
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun, GeneratorRunType
from ax.core.parameter import FixedParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_abandoned_arm,
    get_arm,
    get_arm_weights1,
    get_arms,
    get_experiment,
    get_weights,
)


class BatchTrialTest(TestCase):
    def setUp(self):
        self.experiment = get_experiment()
        self.experiment.status_quo = None
        self.batch = self.experiment.new_batch_trial()
        arms = get_arms()
        weights = get_weights()
        self.status_quo = arms[0]
        self.sq_weight = weights[0]
        self.arms = arms[1:]
        self.weights = weights[1:]
        self.batch.add_arms_and_weights(arms=self.arms, weights=self.weights)

    def testEq(self):
        new_batch_trial = self.experiment.new_batch_trial()
        self.assertNotEqual(self.batch, new_batch_trial)

        abandoned_arm = get_abandoned_arm()
        abandoned_arm_2 = get_abandoned_arm()
        self.assertEqual(abandoned_arm, abandoned_arm_2)

    def testBasicProperties(self):
        self.assertEqual(self.experiment, self.batch.experiment)
        self.assertEqual(self.batch.index, 0)
        self.assertEqual(self.batch.status, TrialStatus.CANDIDATE)
        self.assertIsNotNone(self.batch.time_created)
        self.assertEqual(self.batch.arms_by_name["0_0"], self.batch.arms[0])
        self.assertEqual(self.batch.arms_by_name["0_1"], self.batch.arms[1])
        self.assertEqual(
            self.batch.generator_run_structs[0].generator_run.generator_run_type,
            GeneratorRunType.MANUAL.name,
        )

        # Test empty arms
        self.assertEqual(len(self.experiment.new_batch_trial().abandoned_arms), 0)

    def testUndefinedSetters(self):
        with self.assertRaises(NotImplementedError):
            self.batch.arm_weights = get_arm_weights1()

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

    def testAddArm(self):
        self.assertEqual(len(self.batch.arms), len(self.arms))
        self.assertEqual(len(self.batch.generator_run_structs), 1)
        self.assertEqual(sum(self.batch.weights), sum(self.weights))

        arm_parameters = get_arm().parameters
        arm_parameters["w"] = 5.0
        self.batch.add_arm(Arm(arm_parameters), 3)

        self.assertEqual(self.batch.arms_by_name["0_2"], self.batch.arms[2])
        self.assertEqual(len(self.batch.arms), len(self.arms) + 1)
        self.assertEqual(len(self.batch.generator_run_structs), 2)
        self.assertEqual(sum(self.batch.weights), sum(self.weights) + 3)

    def testAddGeneratorRun(self):
        self.assertEqual(len(self.batch.arms), len(self.arms))
        self.assertEqual(len(self.batch.generator_run_structs), 1)
        self.assertEqual(sum(self.batch.weights), sum(self.weights))

        # one of these arms already exists on the BatchTrial,
        # so we should just update its weight
        new_arms = [
            Arm(parameters={"w": 0.75, "x": 1, "y": "foo", "z": True}),
            Arm(parameters={"w": 1.4, "x": 5, "y": "bar", "z": False}),
        ]
        new_weights = [0.75, 0.25]
        gr = GeneratorRun(arms=new_arms, weights=new_weights)
        self.batch.add_generator_run(gr, 2.0)

        self.assertEqual(len(self.batch.arms), len(self.arms) + 1)
        self.assertEqual(len(self.batch.generator_run_structs), 2)
        self.assertEqual(sum(self.batch.weights), sum(self.weights) + 2)

    def testInitWithGeneratorRun(self):
        generator_run = GeneratorRun(arms=self.arms, weights=self.weights)
        batch = self.experiment.new_batch_trial(generator_run=generator_run)
        batch.add_arms_and_weights(arms=self.arms, weights=self.weights)
        self.assertEqual(self.batch.arms_by_name["0_0"], self.batch.arms[0])
        self.assertEqual(self.batch.arms_by_name["0_1"], self.batch.arms[1])
        self.assertEqual(len(batch.arms), len(self.arms))
        self.assertEqual(len(self.batch.generator_run_structs), 1)

    def testStatusQuoOverlap(self):
        new_sq = Arm(parameters={"w": 0.95, "x": 1, "y": "foo", "z": True})
        # Set status quo to existing arm
        self.batch.set_status_quo_with_weight(self.arms[0], self.sq_weight)
        # Status quo weight is set to the average of other arms' weights.
        # In this case, there are only two arms: 0_0 (SQ) and 0_1 (not SQ).
        # So their weights are equal, as weight(0_0) = avg(weight(0_1)).
        self.assertEqual(self.batch.weights[0], self.batch.weights[1])
        self.assertTrue(self.batch.status_quo.parameters == self.arms[0].parameters)
        self.assertEqual(self.batch.status_quo.name, self.batch.arms[0].name)
        self.assertEqual(self.batch.arm_weights[self.batch.arms[0]], self.sq_weight)
        self.assertEqual(sum(self.batch.weights), self.weights[1] + self.sq_weight)

        # Set status quo to new arm, add it
        self.batch.set_status_quo_with_weight(new_sq, self.sq_weight)
        self.assertEqual(self.batch.status_quo.name, "status_quo_0")
        self.batch.add_arms_and_weights([new_sq])
        self.assertEqual(
            self.batch.generator_run_structs[1].generator_run.arms[0].name,
            "status_quo_0",
        )

    def testStatusQuo(self):
        tot_weight = sum(self.batch.weights)
        new_sq = Arm(parameters={"w": 0.95, "x": 1, "y": "foo", "z": True})

        # Test negative weight
        with self.assertRaises(ValueError):
            self.batch.set_status_quo_with_weight(new_sq, -1)

        # Test that directly setting the status quo raises an error
        with self.assertRaises(NotImplementedError):
            self.batch.status_quo = new_sq

        # Set status quo to new arm
        self.batch.set_status_quo_with_weight(new_sq, self.sq_weight)
        self.assertTrue(self.batch.status_quo == new_sq)
        self.assertEqual(self.batch.status_quo.name, "status_quo_0")
        self.assertEqual(sum(self.batch.weights), tot_weight + self.sq_weight)
        # sq weight should be ignored when sq is None
        self.batch.unset_status_quo()
        self.assertEqual(sum(self.batch.weights), tot_weight)

        # Verify experiment status quo gets set on init
        self.experiment.status_quo = self.status_quo
        batch2 = self.batch.clone()
        self.assertEqual(batch2.status_quo, self.experiment.status_quo)

        # Since optimize_for_power was not set, the weight override should not be
        # And the status quo shoudl not appear in arm_weights
        self.assertIsNone(batch2._status_quo_weight_override)
        self.assertTrue(batch2.status_quo not in batch2.arm_weights)
        self.assertEqual(sum(batch2.weights), sum(self.weights))

        # Try setting sq to existing arm with different name
        with self.assertRaises(ValueError):
            self.batch.set_status_quo_with_weight(
                Arm(new_sq.parameters, name="new_name"), 1
            )

    def testStatusQuoOptimizeForPower(self):
        self.experiment.status_quo = self.status_quo
        batch = self.experiment.new_batch_trial(optimize_for_power=True)
        self.assertEqual(batch._status_quo_weight_override, 1)

        self.experiment.status_quo = None
        with self.assertRaises(ValueError):
            batch = self.experiment.new_batch_trial(optimize_for_power=True)

        batch.add_arms_and_weights(arms=self.arms, weights=self.weights)
        expected_status_quo_weight = math.sqrt(sum(self.weights))
        self.assertTrue(
            math.isclose(batch._status_quo_weight_override, expected_status_quo_weight)
        )
        self.assertTrue(
            math.isclose(
                batch.arm_weights[batch.status_quo], expected_status_quo_weight
            )
        )

    def testBatchLifecycle(self):
        staging_mock = PropertyMock()
        with patch.object(SyntheticRunner, "staging_required", staging_mock):
            mock_runner = SyntheticRunner()
            staging_mock.return_value = True
            self.batch.runner = mock_runner
            self.batch.run()
            self.assertEqual(self.batch.status, TrialStatus.STAGED)
            self.assertIsNotNone(self.batch.time_staged)
            self.assertTrue(self.batch.status.is_deployed)
            self.assertFalse(self.batch.status.expecting_data)

            # Cannot change arms or runner once run
            with self.assertRaises(ValueError):
                self.batch.add_arms_and_weights(arms=self.arms, weights=self.weights)

            with self.assertRaises(ValueError):
                self.batch.runner = None

            # Cannot run batch that was already run
            with self.assertRaises(ValueError):
                self.batch.run()

            self.batch.mark_running()
            self.assertEqual(self.batch.status, TrialStatus.RUNNING)
            self.assertIsNotNone(self.batch.time_run_started)
            self.assertTrue(self.batch.status.expecting_data)

            self.batch.complete()
            # Cannot complete that which is already completed
            with self.assertRaises(ValueError):
                self.batch.complete()

            # Verify trial is completed
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
        self.batch.mark_failed()

        self.assertEqual(self.batch.status, TrialStatus.FAILED)
        self.assertIsNotNone(self.batch.time_completed)

    def testAbandonArm(self):
        arm = self.batch.arms[0]
        reason = "Bad arm"
        self.batch.mark_arm_abandoned(arm.name, reason)
        self.assertEqual(len(self.batch.abandoned_arms), 1)
        self.assertEqual(self.batch.abandoned_arms[0], arm)

        self.assertEqual(len(self.batch.abandoned_arms_metadata), 1)
        metadata = self.batch.abandoned_arms_metadata[0]
        self.assertEqual(metadata.reason, reason)
        self.assertEqual(metadata.name, arm.name)

        # Fail to abandon arm not in BatchTrial
        with self.assertRaises(ValueError):
            self.batch.mark_arm_abandoned(
                Arm(parameters={"x": 3, "y": "fooz", "z": False})
            )

    def testClone(self):
        new_batch_trial = self.batch.clone()
        self.assertEqual(len(new_batch_trial.generator_run_structs), 1)
        self.assertEqual(len(new_batch_trial.arms), 2)
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
        self.assertEqual(self.batch.status, TrialStatus.RUNNING)

        # Verify setting runner on experiment but not BatchTrial
        # Also mock staging_required to be false
        staging_mock = PropertyMock()
        with patch.object(SyntheticRunner, "staging_required", staging_mock):
            mock_runner = SyntheticRunner()
            staging_mock.return_value = True

            self.experiment.runner = mock_runner
            b2 = self.experiment.new_batch_trial()
            b2.run()
            self.assertEqual(b2.deployed_name, "test_1")
            self.assertEqual(b2.status, TrialStatus.STAGED)

    def testIsFactorial(self):
        self.assertFalse(self.batch.is_factorial)

        # Insufficient factors
        small_experiment = Experiment(
            name="small_test",
            search_space=SearchSpace([FixedParameter("a", ParameterType.INT, 4)]),
        )
        small_trial = small_experiment.new_batch_trial().add_arm(Arm({"a": 4}))
        self.assertFalse(small_trial.is_factorial)

        new_batch_trial = self.experiment.new_batch_trial()
        new_batch_trial.add_arms_and_weights(
            arms=[
                Arm(parameters={"w": 0.75, "x": 1, "y": "foo", "z": True}),
                Arm(parameters={"w": 0.75, "x": 2, "y": "foo", "z": True}),
                Arm(parameters={"w": 0.77, "x": 1, "y": "foo", "z": True}),
            ]
        )
        self.assertFalse(new_batch_trial.is_factorial)

        new_batch_trial = self.experiment.new_batch_trial()
        new_batch_trial.add_arms_and_weights(
            arms=[
                Arm(parameters={"w": 0.77, "x": 1, "y": "foo", "z": True}),
                Arm(parameters={"w": 0.77, "x": 2, "y": "foo", "z": True}),
                Arm(parameters={"w": 0.75, "x": 1, "y": "foo", "z": True}),
                Arm(parameters={"w": 0.75, "x": 2, "y": "foo", "z": True}),
            ]
        )
        self.assertTrue(new_batch_trial.is_factorial)

    def testNormalizedArmWeights(self):
        new_batch_trial = self.experiment.new_batch_trial()
        parameterizations = [
            {"w": 0.75, "x": 1, "y": "foo", "z": True},
            {"w": 0.77, "x": 2, "y": "foo", "z": True},
        ]
        arms = [Arm(parameters=p) for i, p in enumerate(parameterizations)]
        new_batch_trial.add_arms_and_weights(arms=arms, weights=[2, 1])

        # test normalizing to 1
        arm_weights = new_batch_trial.normalized_arm_weights()
        # self.assertEqual(list(arm_weights.keys()), arms)
        batch_arm_parameters = [arm.parameters for arm in list(arm_weights.keys())]
        arm_parameters = [arm.parameters for arm in arms]
        self.assertEqual(batch_arm_parameters, arm_parameters)
        self.assertTrue(np.allclose(list(arm_weights.values()), [2 / 3, 1 / 3]))

        # test normalizing to 100
        arm_weights = new_batch_trial.normalized_arm_weights(total=100)
        batch_arm_parameters = [arm.parameters for arm in list(arm_weights.keys())]
        arm_parameters = [arm.parameters for arm in arms]
        self.assertEqual(batch_arm_parameters, arm_parameters)
        self.assertTrue(np.allclose(list(arm_weights.values()), [200 / 3, 100 / 3]))

        # test normalizing with truncation
        arm_weights = new_batch_trial.normalized_arm_weights(total=1, trunc_digits=2)
        batch_arm_parameters = [arm.parameters for arm in list(arm_weights.keys())]
        arm_parameters = [arm.parameters for arm in arms]
        self.assertEqual(batch_arm_parameters, arm_parameters)
        self.assertTrue(np.allclose(list(arm_weights.values()), [0.67, 0.33]))

    def testAddGeneratorRunValidation(self):
        new_batch_trial = self.experiment.new_batch_trial()
        new_arms = [
            Arm(name="0_1", parameters={"w": 0.75, "x": 1, "y": "foo", "z": True}),
            Arm(name="0_2", parameters={"w": 0.75, "x": 1, "y": "foo", "z": True}),
        ]
        gr = GeneratorRun(arms=new_arms)
        with self.assertRaises(ValueError):
            new_batch_trial.add_generator_run(gr)

    def testSetStatusQuoAndOptimizePower(self):
        batch_trial = self.experiment.new_batch_trial()
        status_quo = Arm(
            name="status_quo", parameters={"w": 0.0, "x": 1, "y": "foo", "z": True}
        )

        # Test adding status quo and optimizing power on empty batch
        batch_trial.set_status_quo_and_optimize_power(status_quo)
        self.assertEqual(batch_trial.arm_weights[status_quo], 1.0)

        # Test adding status quo and optimizing power on non-empty batch
        batch_trial = self.experiment.new_batch_trial()
        parameterizations = [
            {"w": 0.75, "x": 1, "y": "foo", "z": True},
            {"w": 0.77, "x": 2, "y": "foo", "z": True},
        ]
        arms = [Arm(parameters=p) for i, p in enumerate(parameterizations)]
        batch_trial.add_arms_and_weights(arms=arms)
        batch_trial.set_status_quo_and_optimize_power(status_quo)
        self.assertEqual(batch_trial.arm_weights[status_quo], np.sqrt(2))

        # Test adding status quo and optimizing power when trial already
        # has a status quo
        batch_trial = self.experiment.new_batch_trial()
        batch_trial.set_status_quo_with_weight(status_quo, 1)
        self.assertEqual(batch_trial.arm_weights[status_quo], 1.0)
        batch_trial.add_arms_and_weights(arms=arms)
        batch_trial.set_status_quo_and_optimize_power(status_quo)
        self.assertEqual(batch_trial.arm_weights[status_quo], np.sqrt(2))
        # Since status quo is not in the generator runs, all of its weight
        # comes from _status_quo_weight_override
        self.assertEqual(batch_trial._status_quo_weight_override, np.sqrt(2))

        # Test adding status quo and optimizing power when status quo
        # is in the generator runs
        batch_trial = self.experiment.new_batch_trial()
        parameterizations = [
            {"w": 0.75, "x": 1, "y": "foo", "z": True},
            {"w": 0.77, "x": 2, "y": "foo", "z": True},
            {"w": 0.0, "x": 1, "y": "foo", "z": True},
        ]
        arms = [Arm(parameters=p) for i, p in enumerate(parameterizations)]
        batch_trial.add_arms_and_weights(arms=arms)
        batch_trial.set_status_quo_and_optimize_power(status_quo)
        self.assertEqual(batch_trial.arm_weights[status_quo], np.sqrt(2))
        # Since status quo has a weight of 1 in the generator runs, only part of
        # its weight comes from _status_quo_weight_override
        self.assertEqual(batch_trial._status_quo_weight_override, np.sqrt(2))

    def testRepr(self):
        repr_ = "BatchTrial(experiment_name='test', index=0, status=TrialStatus.CANDIDATE)"  # noqa
        self.assertEqual(str(self.batch), repr_)
