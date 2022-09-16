#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
from unittest import mock

import numpy as np
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.modelbridge.cross_validation import (
    assess_model_fit,
    compute_diagnostics,
    cross_validate,
    cross_validate_by_trial,
    CVDiagnostics,
    CVResult,
    has_good_opt_config_model_fit,
    SingleDiagnosticBestModelSelector,
)
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class CrossValidationTest(TestCase):
    def setUp(self) -> None:
        self.training_data = [
            Observation(
                # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
                features=ObservationFeatures(parameters={"x": 2.0}, trial_index=0),
                data=ObservationData(
                    means=np.array([2.0, 4.0]),
                    covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
                    metric_names=["a", "b"],
                ),
                arm_name="1_1",
            ),
            Observation(
                # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
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
                # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
                features=ObservationFeatures(parameters={"x": 4.0}, trial_index=2),
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
        self.diagnostics: List[CVDiagnostics] = [
            {"Fisher exact test p": {"y_a": 0.0, "y_b": 0.4}},
            {"Fisher exact test p": {"y_a": 0.1, "y_b": 0.1}},
            {"Fisher exact test p": {"y_a": 0.5, "y_b": 0.6}},
        ]

    def testCrossValidate(self) -> None:
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
        # pyre-fixme[6]: For 1st param expected `Collection[ndarray]` but got
        #  `List[List[typing.Any]]`.
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
        # pyre-fixme[6]: For 1st param expected `Collection[ndarray]` but got
        #  `List[List[typing.Any]]`.
        all_test = np.hstack(test)
        self.assertTrue(
            np.array_equal(sorted(all_test), np.array([2.0, 2.0, 3.0, 4.0]))
        )
        # Test selector

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def test_selector(obs):
            return obs.features.parameters["x"] != 4.0

        result = cross_validate(model=ma, folds=-1, test_selector=test_selector)
        self.assertEqual(len(result), 3)
        z = ma.cross_validate.mock_calls[5:]
        self.assertEqual(len(z), 2)
        all_test = np.hstack(
            # pyre-fixme[6]: For 1st param expected `Collection[ndarray]` but got
            #  `List[List[typing.Any]]`.
            [[obsf.parameters["x"] for obsf in r[2]["cv_test_points"]] for r in z]
        )
        self.assertTrue(np.array_equal(sorted(all_test), np.array([2.0, 2.0, 3.0])))

    def testCrossValidateByTrial(self) -> None:
        # With only 1 trial
        ma = mock.MagicMock()
        ma.get_training_data = mock.MagicMock(
            "ax.modelbridge.base.ModelBridge.get_training_data",
            autospec=True,
            return_value=self.training_data[1:3],
        )
        with self.assertRaises(ValueError):
            cross_validate_by_trial(model=ma)
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
        # Non-existent trial
        with self.assertRaises(ValueError):
            cross_validate_by_trial(model=ma, trial=10)

        # Working
        result = cross_validate_by_trial(model=ma)
        self.assertEqual(len(result), 1)

        # Check that ModelBridge.cross_validate was called correctly.
        z = ma.cross_validate.mock_calls
        self.assertEqual(len(z), 1)
        train_trials = [obs.features.trial_index for obs in z[0][2]["cv_training_data"]]
        test_trials = [obsf.trial_index for obsf in z[0][2]["cv_test_points"]]
        self.assertEqual(train_trials, [0, 1])
        self.assertEqual(test_trials, [2])

        # Check result is correct
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].observed.features.trial_index, 2)

    def test_cross_validate_gives_a_useful_error_for_model_with_no_data(self) -> None:
        exp = get_branin_experiment()
        sobol = Models.SOBOL(experiment=exp, search_space=exp.search_space)
        with self.assertRaisesRegex(ValueError, "no training data"):
            cross_validate(model=sobol)

    # pyre-fixme[3]: Return type must be annotated.
    def test_cross_validate_raises_not_implemented_error_for_non_cv_model_with_data(
        self,
    ):
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run().complete()
        sobol = Models.SOBOL(
            experiment=exp, search_space=exp.search_space, data=exp.fetch_data()
        )
        with self.assertRaises(NotImplementedError):
            cross_validate(model=sobol)

    def testComputeDiagnostics(self) -> None:
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
        self.assertAlmostEqual(diag["Log likelihood"]["a"], -50.09469266602336)
        self.assertAlmostEqual(diag["Log likelihood"]["b"], -25.82334285505847)

    def testAssessModelFit(self) -> None:
        # Construct diagnostics
        result = []
        for i, obs in enumerate(self.training_data):
            result.append(CVResult(observed=obs, predicted=self.observation_data[i]))
        diag = compute_diagnostics(result=result)
        for v in diag.values():
            self.assertEqual(set(v.keys()), {"a", "b"})
        # Check for correct computation, relative to manually computed result
        self.assertAlmostEqual(diag["Fisher exact test p"]["a"], 0.10)
        self.assertAlmostEqual(diag["Fisher exact test p"]["b"], 0.166666666)

        assess_model_fit_result = assess_model_fit(
            diagnostics=diag, significance_level=0.05
        )
        self.assertTrue("a" in assess_model_fit_result.bad_fit_metrics_to_fisher_score)
        self.assertTrue("b" in assess_model_fit_result.bad_fit_metrics_to_fisher_score)
        assess_model_fit_result = assess_model_fit(
            diagnostics=diag, significance_level=0.15
        )
        self.assertTrue("a" in assess_model_fit_result.good_fit_metrics_to_fisher_score)
        self.assertTrue("b" in assess_model_fit_result.bad_fit_metrics_to_fisher_score)
        assess_model_fit_result = assess_model_fit(
            diagnostics=diag, significance_level=0.2
        )
        self.assertTrue("a" in assess_model_fit_result.good_fit_metrics_to_fisher_score)
        self.assertTrue("b" in assess_model_fit_result.good_fit_metrics_to_fisher_score)

    def testHasGoodOptConfigModelFit(self) -> None:
        # Construct diagnostics
        result = []
        for i, obs in enumerate(self.training_data):
            result.append(CVResult(observed=obs, predicted=self.observation_data[i]))
        diag = compute_diagnostics(result=result)
        assess_model_fit_result = assess_model_fit(
            diagnostics=diag,
            significance_level=0.05,
        )

        # Test single objective
        optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric("a"))
        )
        has_good_fit = has_good_opt_config_model_fit(
            optimization_config=optimization_config,
            assess_model_fit_result=assess_model_fit_result,
        )
        self.assertFalse(has_good_fit)

        # Test multi objective
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(metrics=[Metric("a"), Metric("b")])
        )
        has_good_fit = has_good_opt_config_model_fit(
            optimization_config=optimization_config,
            assess_model_fit_result=assess_model_fit_result,
        )
        self.assertFalse(has_good_fit)

        # Test constraints
        optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric("a")),
            outcome_constraints=[
                OutcomeConstraint(metric=Metric("b"), op=ComparisonOp.GEQ, bound=0.1)
            ],
        )
        has_good_fit = has_good_opt_config_model_fit(
            optimization_config=optimization_config,
            assess_model_fit_result=assess_model_fit_result,
        )
        self.assertFalse(has_good_fit)

    def testSingleDiagnosticBestModelSelector_min_mean(self) -> None:
        s = SingleDiagnosticBestModelSelector(
            diagnostic="Fisher exact test p",
            criterion=min,
            metric_aggregation=np.mean,
        )
        self.assertEqual(s.best_diagnostic(self.diagnostics), 1)

    def testSingleDiagnosticBestModelSelector_min_min(self) -> None:
        s = SingleDiagnosticBestModelSelector(
            diagnostic="Fisher exact test p",
            criterion=min,
            metric_aggregation=min,
        )
        self.assertEqual(s.best_diagnostic(self.diagnostics), 0)

    def testSingleDiagnosticBestModelSelector_max_mean(self) -> None:
        s = SingleDiagnosticBestModelSelector(
            diagnostic="Fisher exact test p",
            criterion=max,
            metric_aggregation=np.mean,
        )
        self.assertEqual(s.best_diagnostic(self.diagnostics), 2)
