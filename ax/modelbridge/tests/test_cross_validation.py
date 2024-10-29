#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
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
from ax.core.parameter import FixedParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.modelbridge.cross_validation import (
    assess_model_fit,
    compute_diagnostics,
    cross_validate,
    cross_validate_by_trial,
    CVDiagnostics,
    CVResult,
    has_good_opt_config_model_fit,
)
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.modeling_stubs import get_observation1trans, get_observation2trans


class CrossValidationTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
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
        self.observation_data_transformed_result = [get_observation1trans().data] * 4
        self.transformed_cv_input_dict = {
            "cv_training_data": [get_observation2trans()],
            "cv_test_points": [get_observation1trans().features],
            "search_space": SearchSpace(
                [FixedParameter("x", ParameterType.FLOAT, 8.0)]
            ),
        }
        self.diagnostics: list[CVDiagnostics] = [
            {"Fisher exact test p": {"y_a": 0.0, "y_b": 0.4}},
            {"Fisher exact test p": {"y_a": 0.1, "y_b": 0.1}},
            {"Fisher exact test p": {"y_a": 0.5, "y_b": 0.6}},
        ]

    def test_CrossValidate(self) -> None:
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
        ma._transform_inputs_for_cv = mock.MagicMock(
            "ax.modelbridge.base.ModelBridge._transform_inputs_for_cv",
            autospec=True,
            return_value=list(self.transformed_cv_input_dict.values()),
        )
        ma._cross_validate = mock.MagicMock(
            "ax.modelbridge.base.ModelBridge._cross_validate",
            autospec=True,
            return_value=self.observation_data_transformed_result,
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
        # Test LOO in transformed space
        result = cross_validate(model=ma, folds=-1, untransform=False)
        result_predicted_obs_data = [cv_result.predicted for cv_result in result]
        self.assertEqual(
            result_predicted_obs_data, self.observation_data_transformed_result
        )
        # Check that ModelBridge._transform_inputs_for_cv was called correctly.
        z = ma._transform_inputs_for_cv.mock_calls
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
        # Test ModelBridge._cross_validate was called correctly.
        z = ma._cross_validate.mock_calls
        self.assertEqual(len(z), 3)
        ma._cross_validate.assert_called_with(
            **self.transformed_cv_input_dict, use_posterior_predictive=False
        )

        # Test selector

        def test_selector(obs: Observation) -> bool:
            return obs.features.parameters["x"] != 4.0

        result = cross_validate(model=ma, folds=-1, test_selector=test_selector)
        self.assertEqual(len(result), 3)
        z = ma.cross_validate.mock_calls[5:]
        self.assertEqual(len(z), 2)
        all_test = np.hstack(
            [[obsf.parameters["x"] for obsf in r[2]["cv_test_points"]] for r in z]
        )
        self.assertTrue(np.array_equal(sorted(all_test), np.array([2.0, 2.0, 3.0])))

        # test observation noise
        for untransform in (True, False):
            result = cross_validate(
                model=ma,
                folds=-1,
                use_posterior_predictive=True,
                untransform=untransform,
            )
            if untransform:
                mock_cv = ma.cross_validate
            else:
                mock_cv = ma._cross_validate
            call_kwargs = mock_cv.mock_calls[-1].kwargs
            self.assertTrue(call_kwargs["use_posterior_predictive"])

    def test_CrossValidateByTrial(self) -> None:
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

        mock_cv = ma.cross_validate
        call_kwargs = mock_cv.mock_calls[-1].kwargs
        self.assertFalse(call_kwargs["use_posterior_predictive"])

        # test observation noise
        result = cross_validate_by_trial(model=ma, use_posterior_predictive=True)
        call_kwargs = mock_cv.mock_calls[-1].kwargs
        self.assertTrue(call_kwargs["use_posterior_predictive"])

    def test_cross_validate_gives_a_useful_error_for_model_with_no_data(self) -> None:
        exp = get_branin_experiment()
        sobol = Models.SOBOL(experiment=exp, search_space=exp.search_space)
        with self.assertRaisesRegex(ValueError, "no training data"):
            cross_validate(model=sobol)

    @mock_botorch_optimize
    def test_cross_validate_catches_warnings(self) -> None:
        exp = get_branin_experiment(with_batch=True, with_completed_batch=True)
        model = Models.BOTORCH_MODULAR(
            experiment=exp, search_space=exp.search_space, data=exp.fetch_data()
        )
        for untransform in [False, True]:
            with warnings.catch_warnings(record=True) as ws:
                cross_validate(model=model, untransform=untransform)
                self.assertEqual(len(ws), 0)

    def test_cross_validate_raises_not_implemented_error_for_non_cv_model_with_data(
        self,
    ) -> None:
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run().complete()
        sobol = Models.SOBOL(
            experiment=exp, search_space=exp.search_space, data=exp.fetch_data()
        )
        with self.assertRaises(NotImplementedError):
            cross_validate(model=sobol)

    def test_ComputeDiagnostics(self) -> None:
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
        self.assertAlmostEqual(diag["MAPE"]["b"], 0.8312499999999999)
        self.assertAlmostEqual(
            diag["wMAPE"]["a"],
            sum([0.0, 1.0, 4.0, 5.0, 7.0]) / sum([2, 3, 6, 7, 9]),
        )
        self.assertAlmostEqual(
            diag["wMAPE"]["b"], sum([3.0, 4.0, 7.0, 9.0]) / sum([4, 5, 8, 10])
        )
        self.assertAlmostEqual(diag["Total raw effect"]["a"], 3.5)
        self.assertAlmostEqual(diag["Total raw effect"]["b"], 1.5)
        self.assertAlmostEqual(diag["Log likelihood"]["a"], -50.09469266602336)
        self.assertAlmostEqual(diag["Log likelihood"]["b"], -25.82334285505847)
        self.assertEqual(diag["MSE"]["a"], 18.2)
        self.assertEqual(diag["MSE"]["b"], 38.75)

    def test_AssessModelFit(self) -> None:
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

    def test_HasGoodOptConfigModelFit(self) -> None:
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
            objective=Objective(metric=Metric("a"), minimize=True)
        )
        has_good_fit = has_good_opt_config_model_fit(
            optimization_config=optimization_config,
            assess_model_fit_result=assess_model_fit_result,
        )
        self.assertFalse(has_good_fit)

        # Test multi objective
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(
                objectives=[
                    Objective(Metric("a"), minimize=False),
                    Objective(Metric("b"), minimize=False),
                ]
            )
        )
        has_good_fit = has_good_opt_config_model_fit(
            optimization_config=optimization_config,
            assess_model_fit_result=assess_model_fit_result,
        )
        self.assertFalse(has_good_fit)

        # Test constraints
        optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric("a"), minimize=False),
            outcome_constraints=[
                OutcomeConstraint(metric=Metric("b"), op=ComparisonOp.GEQ, bound=0.1)
            ],
        )
        has_good_fit = has_good_opt_config_model_fit(
            optimization_config=optimization_config,
            assess_model_fit_result=assess_model_fit_result,
        )
        self.assertFalse(has_good_fit)
