#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from collections.abc import Iterable
from unittest import mock

import numpy as np
from ax.adapter.cross_validation import (
    assess_model_fit,
    compute_diagnostics,
    cross_validate,
    CVData,
    CVDiagnostics,
    CVResult,
    gen_trial_split,
    has_good_opt_config_model_fit,
    logger,
)
from ax.adapter.data_utils import ExperimentData
from ax.adapter.registry import Generators, MBM_X_trans, Y_trans
from ax.adapter.torch import TorchAdapter
from ax.adapter.transforms.transform_to_new_sq import TransformToNewSQ
from ax.adapter.transforms.unit_x import UnitX
from ax.core import ObservationFeatures
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.observation import Observation, ObservationData
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp, TParameterization
from ax.exceptions.core import UnsupportedError
from ax.exceptions.model import CrossValidationError
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_experiment_with_observations,
    get_search_space_for_range_value,
)
from ax.utils.testing.mock import (
    mock_botorch_optimize,
    mock_botorch_optimize_context_manager,
)
from botorch.exceptions.warnings import InputDataWarning


class CrossValidationTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # pyre-ignore [9] Pyre is too picky with union types.
        parameterizations: list[TParameterization] = [
            {"x": x} for x in [2.0, 2.0, 3.0, 4.0]
        ]
        means = [[2.0, 4.0], [3.0, 5.0], [7.0, 8.0], [9.0, 10.0]]
        sems = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
        self.experiment = get_experiment_with_observations(
            observations=means,
            sems=sems,
            search_space=get_search_space_for_range_value(min=0.0, max=10.0),
            parameterizations=parameterizations,
        )
        with mock_botorch_optimize_context_manager():
            self.adapter = TorchAdapter(
                experiment=self.experiment,
                generator=BoTorchGenerator(),
                transforms=[UnitX],
            )
        self.training_data = self.adapter.get_training_data()
        self.training_obs = self.training_data.convert_to_list_of_observations()
        self.observation_data = ObservationData(
            means=np.array([2.0, 1.0]),
            covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
            metric_signatures=["m1", "m2"],
        )
        self.cv_results = [
            CVResult(observed=obs, predicted=self.observation_data)
            for obs in self.training_obs
        ]
        self.diagnostics: list[CVDiagnostics] = [
            {"Fisher exact test p": {"y_m1": 0.0, "y_m2": 0.4}},
            {"Fisher exact test p": {"y_m1": 0.1, "y_m2": 0.1}},
            {"Fisher exact test p": {"y_m1": 0.5, "y_m2": 0.6}},
        ]

    def test_cross_validate_base(self) -> None:
        # Do cross validation
        with self.assertRaisesRegex(ValueError, "which is less than 4 folds"):
            cross_validate(model=self.adapter, folds=4)
        with self.assertRaisesRegex(ValueError, "Folds must be"):
            cross_validate(model=self.adapter, folds=0)
        # First 2-fold
        with mock.patch.object(
            self.adapter, "cross_validate", wraps=self.adapter.cross_validate
        ) as mock_cv:
            result = cross_validate(model=self.adapter, folds=2)
        self.assertEqual(len(result), 4)
        # Check that Adapter.cross_validate was called correctly.
        z = mock_cv.mock_calls
        self.assertEqual(len(z), 2)
        train = [r[2]["cv_training_data"].arm_data["x"].tolist() for r in z]
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
        with mock.patch.object(
            self.adapter, "cross_validate", wraps=self.adapter.cross_validate
        ) as mock_cv:
            result = cross_validate(model=self.adapter, folds=-1)
        self.assertEqual(len(result), 4)
        z = mock_cv.mock_calls
        self.assertEqual(len(z), 3)
        train = [r[2]["cv_training_data"].arm_data["x"].tolist() for r in z]
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
        with (
            mock.patch.object(
                self.adapter,
                "_transform_inputs_for_cv",
                wraps=self.adapter._transform_inputs_for_cv,
            ) as mock_transform_cv,
            mock.patch.object(
                self.adapter,
                "_cross_validate",
                side_effect=lambda **kwargs: [self.observation_data]
                * len(kwargs["cv_test_points"]),
            ) as mock_cv,
        ):
            result = cross_validate(model=self.adapter, folds=-1, untransform=False)
        result_predicted_obs_data = [cv_result.predicted for cv_result in result]
        self.assertEqual(result_predicted_obs_data, [self.observation_data] * 4)
        # Check that Adapter._transform_inputs_for_cv was called correctly.
        z = mock_transform_cv.mock_calls
        self.assertEqual(len(z), 3)
        train = [call.kwargs["cv_training_data"].arm_data["x"].tolist() for call in z]
        test = [
            [obsf.parameters["x"] for obsf in call.kwargs["cv_test_points"]]
            for call in z
        ]
        # Test no overlap between train and test sets, and all points used
        for i in range(3):
            self.assertEqual(len(set(train[i]).intersection(test[i])), 0)
            self.assertEqual(len(train[i]) + len(test[i]), 4)
        # Test all points used as test points -- these are transformed after call.
        all_test = np.hstack(test)
        self.assertTrue(
            np.array_equal(sorted(all_test), np.array([0.2, 0.2, 0.3, 0.4]))
        )
        # Test Adapter._cross_validate was called correctly.
        self.assertEqual(mock_cv.call_count, 3)
        transform = self.adapter.transforms["UnitX"]
        # Compare against arbitrary call since the call ordering depends on
        # the order of arm names, which is not deterministic.
        expected_call = mock.call(
            cv_training_data=transform.transform_experiment_data(
                ExperimentData(
                    arm_data=self.training_data.arm_data.iloc[:-1].copy(),
                    observation_data=self.training_data.observation_data.iloc[
                        :-1
                    ].copy(),
                )
            ),
            cv_test_points=transform.transform_observation_features(
                [
                    ObservationFeatures(
                        parameters={"x": 4.0},
                        trial_index=3,
                        metadata=self.training_data.arm_data.iloc[-1]["metadata"],
                    )
                ]
            ),
            search_space=transform.transform_search_space(
                self.adapter._search_space.clone()
            ),
            use_posterior_predictive=False,
        )
        self.assertTrue(expected_call in mock_cv.mock_calls)

    def test_cross_validate_w_test_selector(self) -> None:
        def test_selector(obs: Observation) -> bool:
            return obs.features.parameters["x"] != 4.0

        with mock.patch.object(
            self.adapter, "cross_validate", wraps=self.adapter.cross_validate
        ) as mock_cv:
            result = cross_validate(
                model=self.adapter, folds=-1, test_selector=test_selector
            )
        self.assertEqual(len(result), 3)
        z = mock_cv.mock_calls
        self.assertEqual(len(z), 2)
        all_test = np.hstack(
            [[obsf.parameters["x"] for obsf in r[2]["cv_test_points"]] for r in z]
        )
        self.assertTrue(np.array_equal(sorted(all_test), np.array([2.0, 2.0, 3.0])))

        # test observation noise
        for untransform in (True, False):
            with mock.patch.object(
                self.adapter, "_cross_validate", wraps=self.adapter._cross_validate
            ) as mock_cv:
                result = cross_validate(
                    model=self.adapter,
                    folds=-1,
                    use_posterior_predictive=True,
                    untransform=untransform,
                )
            call_kwargs = mock_cv.call_args.kwargs
            self.assertTrue(call_kwargs["use_posterior_predictive"])

    def test_cross_validate_w_fold_generator(self) -> None:
        for train_trials, test_trial, exp_train_trials in [
            (None, 3, {0, 1, 2}),
            ([0, 1], 2, {0, 1}),
        ]:

            def fold_generator(training_data: ExperimentData) -> Iterable[CVData]:
                return gen_trial_split(
                    training_data=training_data,
                    train_trials=train_trials,  # noqa B023
                    test_trials=[test_trial],  # noqa B023
                )

            with mock.patch.object(
                self.adapter, "cross_validate", wraps=self.adapter.cross_validate
            ) as mock_cv:
                result = cross_validate(
                    model=self.adapter, fold_generator=fold_generator
                )
            self.assertEqual(len(result), 1)
            z = mock_cv.mock_calls
            self.assertEqual(len(z), 1)
            self.assertEqual(z[0][2]["cv_test_points"][0].trial_index, test_trial)
            self.assertEqual(
                set(
                    z[0][2]["cv_training_data"].arm_data.index.get_level_values(
                        "trial_index"
                    )
                ),
                exp_train_trials,
            )

        # Test errors
        def fold_generator(training_data: ExperimentData) -> Iterable[CVData]:
            return gen_trial_split(training_data=training_data, test_trials=[])

        with self.assertRaisesRegex(ValueError, "No test trials provided"):
            cross_validate(model=self.adapter, fold_generator=fold_generator)

        def fold_generator(training_data: ExperimentData) -> Iterable[CVData]:
            return gen_trial_split(training_data=training_data, test_trials=[5])

        with self.assertRaisesRegex(ValueError, "not all in training data"):
            cross_validate(model=self.adapter, fold_generator=fold_generator)

        def fold_generator(training_data: ExperimentData) -> Iterable[CVData]:
            return gen_trial_split(training_data=training_data, test_trials=[5])

        with self.assertRaisesRegex(ValueError, "not all in training data"):
            cross_validate(model=self.adapter, fold_generator=fold_generator)

        def fold_generator(training_data: ExperimentData) -> Iterable[CVData]:
            return gen_trial_split(
                training_data=training_data, train_trials=[0, 1], test_trials=[1]
            )

        with self.assertRaisesRegex(ValueError, "Test and train trials overlap"):
            cross_validate(model=self.adapter, fold_generator=fold_generator)

        def fold_generator(training_data: ExperimentData) -> Iterable[CVData]:
            return gen_trial_split(
                training_data=training_data, test_trials=[0, 1, 2, 3]
            )

        with self.assertRaisesRegex(ValueError, "All trials in data"):
            cross_validate(model=self.adapter, fold_generator=fold_generator)

    def test_cross_validate_with_data_reducing_transforms(self) -> None:
        # With transforms like TransformToNewSQ, the number of observations
        # and predictions may not match (because transforms throw away some data).
        # This checks that cross_validate handles this correctly for LOOCV
        # and errors out for non-LOO CV.
        # Experiment has multiple batch trials each with status quo arm.
        experiment = get_branin_experiment(
            with_status_quo=True, with_completed_batch=True, num_batch_trial=3
        )
        adapter = TorchAdapter(
            experiment=experiment,
            generator=BoTorchGenerator(),
            transforms=MBM_X_trans + [TransformToNewSQ] + Y_trans,
        )
        # With untransform=True (default), it just works.
        with self.assertNoLogs(logger=logger):
            res = cross_validate(model=adapter, folds=-1)
        # SQ arm is repeated 3 times, so we add +2 for that.
        self.assertEqual(len(res), len(experiment.arms_by_name) + 2)

        # With untransform=False, LOOCV should work and log a warning.
        with self.assertLogs(logger=logger):
            res = cross_validate(model=adapter, folds=-1, untransform=False)
        # We only have one result for SQ arm here, due to TransformToNewSQ.
        self.assertEqual(len(res), len(experiment.arms_by_name))

        # 2-fold CV should error out.
        with self.assertRaisesRegex(
            CrossValidationError,
            "fewer test observations than predictions",
        ):
            cross_validate(model=adapter, folds=2, untransform=False)

    def test_cross_validate_gives_a_useful_error_for_insufficient_data(self) -> None:
        # Sobol with no data and torch with only one point.
        exp_empty = get_branin_experiment()
        exp = get_branin_experiment(with_completed_trial=True)
        for adapter in [
            Generators.SOBOL(experiment=exp_empty),
            Generators.BOTORCH_MODULAR(experiment=exp),
        ]:
            with self.assertRaisesRegex(UnsupportedError, "at least two in-design"):
                cross_validate(model=adapter)

    @mock_botorch_optimize
    def test_cross_validate_catches_warnings(self) -> None:
        exp = get_branin_experiment(with_batch=True, with_completed_batch=True)
        model = Generators.BOTORCH_MODULAR(
            experiment=exp, search_space=exp.search_space, data=exp.fetch_data()
        )
        for untransform in [False, True]:
            with warnings.catch_warnings(record=True) as ws:
                cross_validate(model=model, untransform=untransform)
                self.assertFalse(any(w.category == InputDataWarning for w in ws))

    def test_cross_validate_raises_not_implemented_error_for_non_cv_model_with_data(
        self,
    ) -> None:
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run().complete()
        sobol = Generators.SOBOL(
            experiment=exp, search_space=exp.search_space, data=exp.fetch_data()
        )
        with self.assertRaises(NotImplementedError):
            cross_validate(model=sobol)

    def test_compute_diagnostics(self) -> None:
        # Compute diagnostics
        diag = compute_diagnostics(result=self.cv_results)
        for v in diag.values():
            self.assertEqual(set(v.keys()), {"m1", "m2"})
        # Check for correct computation, relative to manually computed result
        self.assertAlmostEqual(diag["MAPE"]["m1"], 0.4563492063492064)
        self.assertAlmostEqual(diag["MAPE"]["m2"], 0.8312499999999999)
        self.assertAlmostEqual(
            diag["wMAPE"]["m1"],
            sum([0.0, 1.0, 5.0, 7.0]) / sum([2, 3, 7, 9]),
        )
        self.assertAlmostEqual(
            diag["wMAPE"]["m2"], sum([3.0, 4.0, 7.0, 9.0]) / sum([4, 5, 8, 10])
        )
        self.assertAlmostEqual(diag["Total raw effect"]["m1"], 3.5)
        self.assertAlmostEqual(diag["Total raw effect"]["m2"], 1.5)
        self.assertAlmostEqual(diag["Log likelihood"]["m1"], -41.175754132818696)
        self.assertAlmostEqual(diag["Log likelihood"]["m2"], -25.82334285505847)
        self.assertEqual(diag["MSE"]["m1"], 18.75)
        self.assertEqual(diag["MSE"]["m2"], 38.75)

    def test_assess_model_fit(self) -> None:
        # Construct diagnostics
        diag = compute_diagnostics(result=self.cv_results)
        for v in diag.values():
            self.assertEqual(set(v.keys()), {"m1", "m2"})
        # Check for correct computation, relative to manually computed result
        self.assertAlmostEqual(diag["Fisher exact test p"]["m1"], 0.16666, places=4)
        self.assertAlmostEqual(diag["Fisher exact test p"]["m2"], 0.16666, places=4)

        diag["Fisher exact test p"]["m1"] = 0.1  # differentiate for testing.
        assess_model_fit_result = assess_model_fit(
            diagnostics=diag, significance_level=0.05
        )
        self.assertTrue("m1" in assess_model_fit_result.bad_fit_metrics_to_fisher_score)
        self.assertTrue("m2" in assess_model_fit_result.bad_fit_metrics_to_fisher_score)
        assess_model_fit_result = assess_model_fit(
            diagnostics=diag, significance_level=0.15
        )
        self.assertTrue(
            "m1" in assess_model_fit_result.good_fit_metrics_to_fisher_score
        )
        self.assertTrue("m2" in assess_model_fit_result.bad_fit_metrics_to_fisher_score)
        assess_model_fit_result = assess_model_fit(
            diagnostics=diag, significance_level=0.2
        )
        self.assertTrue(
            "m1" in assess_model_fit_result.good_fit_metrics_to_fisher_score
        )
        self.assertTrue(
            "m2" in assess_model_fit_result.good_fit_metrics_to_fisher_score
        )

    def test_has_good_opt_config_model_fit(self) -> None:
        # Construct diagnostics
        diag = compute_diagnostics(result=self.cv_results)
        assess_model_fit_result = assess_model_fit(
            diagnostics=diag,
            significance_level=0.05,
        )

        # Test single objective
        optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric("m1"), minimize=True)
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
                    Objective(Metric("m1"), minimize=False),
                    Objective(Metric("m2"), minimize=False),
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
            objective=Objective(metric=Metric("m1"), minimize=False),
            outcome_constraints=[
                OutcomeConstraint(metric=Metric("m2"), op=ComparisonOp.GEQ, bound=0.1)
            ],
        )
        has_good_fit = has_good_opt_config_model_fit(
            optimization_config=optimization_config,
            assess_model_fit_result=assess_model_fit_result,
        )
        self.assertFalse(has_good_fit)
