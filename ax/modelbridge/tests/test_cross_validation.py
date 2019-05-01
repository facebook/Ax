#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from unittest import mock

import numpy as np
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.modelbridge.cross_validation import (
    CVResult,
    compute_diagnostics,
    cross_validate,
)
from ax.utils.common.testutils import TestCase


class CrossValidationTest(TestCase):
    def setUp(self):
        self.training_data = [
            Observation(
                features=ObservationFeatures(parameters={"x": 2.0}, trial_index=0),
                data=ObservationData(
                    means=np.array([2.0, 4.0]),
                    covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
                    metric_names=["a", "b"],
                ),
                arm_name="1_1",
            ),
            Observation(
                features=ObservationFeatures(parameters={"x": 2.0}, trial_index=1),
                data=ObservationData(
                    means=np.array([3.0, 5.0, 6.0]),
                    covariance=np.array(
                        [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
                    ),
                    metric_names=["a", "b", "a"],
                ),
                arm_name="1_1",
            ),
            Observation(
                features=ObservationFeatures(parameters={"x": 3.0}),
                data=ObservationData(
                    means=np.array([7.0, 8.0]),
                    covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
                    metric_names=["a", "b"],
                ),
                arm_name="1_2",
            ),
            Observation(
                features=ObservationFeatures(parameters={"x": 4.0}),
                data=ObservationData(
                    means=np.array([9.0, 10.0]),
                    covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
                    metric_names=["a", "b"],
                ),
                arm_name="1_3",
            ),
        ]
        self.observation_data = [
            ObservationData(
                means=np.array([2.0, 1.0]),
                covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
                metric_names=["a", "b"],
            )
        ] * 4

    def testCrossValidate(self):
        # Prepare input and output data
        ma = mock.MagicMock()
        ma.get_training_data = mock.MagicMock(
            "ax.modelbridge.base.ModelBridge.get_training_data",
            autospec=True,
            return_value=self.training_data,
        )
        ma.cross_validate = mock.MagicMock(
            "ax.modelbridge.base.ModelBridge.cross_validate",
            autospec=True,
            return_value=self.observation_data,
        )
        # Do cross validation
        with self.assertRaises(ValueError):
            cross_validate(model=ma, folds=4)
        with self.assertRaises(ValueError):
            cross_validate(model=ma, folds=0)
        # First 2-fold
        result = cross_validate(model=ma, folds=2)
        self.assertEqual(len(result), 4)
        # Check that ModelBridge.cross_validate was called correctly.
        z = ma.cross_validate.mock_calls
        self.assertEqual(len(z), 2)
        train = [
            [obs.features.parameters["x"] for obs in r[2]["cv_training_data"]]
            for r in z
        ]
        test = [[obsf.parameters["x"] for obsf in r[2]["cv_test_points"]] for r in z]
        # Test no overlap between train and test sets, and all points used
        for i in range(2):
            self.assertEqual(len(set(train[i]).intersection(test[i])), 0)
            self.assertEqual(len(train[i]) + len(test[i]), 4)
        # Test all points used as test points
        all_test = np.hstack(test)
        self.assertTrue(
            np.array_equal(sorted(all_test), np.array([2.0, 2.0, 3.0, 4.0]))
        )

        # Test LOO
        result = cross_validate(model=ma, folds=-1)
        self.assertEqual(len(result), 4)
        z = ma.cross_validate.mock_calls[2:]
        self.assertEqual(len(z), 3)
        train = [
            [obs.features.parameters["x"] for obs in r[2]["cv_training_data"]]
            for r in z
        ]
        test = [[obsf.parameters["x"] for obsf in r[2]["cv_test_points"]] for r in z]
        # Test no overlap between train and test sets, and all points used
        for i in range(3):
            self.assertEqual(len(set(train[i]).intersection(test[i])), 0)
            self.assertEqual(len(train[i]) + len(test[i]), 4)
        # Test all points used as test points
        all_test = np.hstack(test)
        self.assertTrue(
            np.array_equal(sorted(all_test), np.array([2.0, 2.0, 3.0, 4.0]))
        )
        # Test selector

        def test_selector(obs):
            return obs.features.parameters["x"] != 4.0

        result = cross_validate(model=ma, folds=-1, test_selector=test_selector)
        self.assertEqual(len(result), 3)
        z = ma.cross_validate.mock_calls[5:]
        self.assertEqual(len(z), 2)
        all_test = np.hstack(
            [[obsf.parameters["x"] for obsf in r[2]["cv_test_points"]] for r in z]
        )
        self.assertTrue(np.array_equal(sorted(all_test), np.array([2.0, 2.0, 3.0])))

    def testComputeDiagnostics(self):
        # Construct CVResults
        result = []
        for i, obs in enumerate(self.training_data):
            result.append(CVResult(observed=obs, predicted=self.observation_data[i]))
        # Compute diagnostics
        diag = compute_diagnostics(result=result)
        for v in diag.values():
            self.assertEqual(set(v.keys()), {"a", "b"})
        # Check for correct computation, relative to manually computed result
        self.assertAlmostEqual(diag["MAPE"]["a"], 0.4984126984126984)
        self.assertAlmostEqual(diag["Total raw effect"]["a"], 3.5)
        self.assertAlmostEqual(diag["Total raw effect"]["b"], 1.5)
