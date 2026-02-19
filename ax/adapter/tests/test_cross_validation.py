#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from collections.abc import Iterable
from itertools import product
from unittest import mock

import numpy as np
import torch
from ax.adapter.cross_validation import (
    _efficient_loo_cross_validate,
    _fold_cross_validate,
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
from ax.adapter.transforms.standardize_y import StandardizeY
from ax.adapter.transforms.transform_to_new_sq import TransformToNewSQ
from ax.adapter.transforms.unit_x import UnitX
from ax.core import ObservationFeatures
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.observation import Observation, ObservationData
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.trial import Trial
from ax.core.types import ComparisonOp, TParameterization
from ax.exceptions.core import UnsupportedError
from ax.exceptions.model import CrossValidationError
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.generators.torch.botorch_modular.surrogate import Surrogate, SurrogateSpec
from ax.generators.torch.botorch_modular.utils import ModelConfig
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
from botorch.cross_validation import CVResults, efficient_loo_cv, ensemble_loo_cv
from botorch.exceptions.warnings import InputDataWarning
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.robust_relevance_pursuit_model import (
    RobustRelevancePursuitSingleTaskGP,
)
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator
from pandas import DataFrame

# Number of in-design points created by _create_adapter_with_out_of_design_points()
_OOD_ADAPTER_IN_DESIGN_COUNT = 3


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
            cross_validate(adapter=self.adapter, folds=4)
        with self.assertRaisesRegex(ValueError, "Folds must be"):
            cross_validate(adapter=self.adapter, folds=0)
        # First 2-fold
        with mock.patch.object(
            self.adapter, "cross_validate", wraps=self.adapter.cross_validate
        ) as mock_cv:
            result = cross_validate(adapter=self.adapter, folds=2)
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

        # Test LOO - use naive CV path by mocking efficient LOO
        with (
            mock.patch(
                "ax.adapter.cross_validation._efficient_loo_cross_validate",
                side_effect=ValueError("Force fallback to naive CV"),
            ),
            mock.patch.object(
                self.adapter, "cross_validate", wraps=self.adapter.cross_validate
            ) as mock_cv,
        ):
            result = cross_validate(adapter=self.adapter, folds=-1)
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
        # Test LOO in transformed space - use naive path by mocking efficient LOO
        with (
            mock.patch(
                "ax.adapter.cross_validation._efficient_loo_cross_validate",
                side_effect=ValueError("Force fallback to naive CV"),
            ),
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
            result = cross_validate(adapter=self.adapter, folds=-1, untransform=False)
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
                adapter=self.adapter, folds=-1, test_selector=test_selector
            )
        self.assertEqual(len(result), 3)
        z = mock_cv.mock_calls
        self.assertEqual(len(z), 2)
        all_test = np.hstack(
            [[obsf.parameters["x"] for obsf in r[2]["cv_test_points"]] for r in z]
        )
        self.assertTrue(np.array_equal(sorted(all_test), np.array([2.0, 2.0, 3.0])))

        # test observation noise - use naive path by disabling efficient LOO
        for untransform in (True, False):
            with (
                mock.patch(
                    "ax.adapter.cross_validation._efficient_loo_cross_validate",
                    side_effect=ValueError("Force fallback to naive CV"),
                ),
                mock.patch.object(
                    self.adapter, "_cross_validate", wraps=self.adapter._cross_validate
                ) as mock_cv,
            ):
                result = cross_validate(
                    adapter=self.adapter,
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
                    adapter=self.adapter, fold_generator=fold_generator
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
            cross_validate(adapter=self.adapter, fold_generator=fold_generator)

        def fold_generator(training_data: ExperimentData) -> Iterable[CVData]:
            return gen_trial_split(training_data=training_data, test_trials=[5])

        with self.assertRaisesRegex(ValueError, "not all in training data"):
            cross_validate(adapter=self.adapter, fold_generator=fold_generator)

        def fold_generator(training_data: ExperimentData) -> Iterable[CVData]:
            return gen_trial_split(training_data=training_data, test_trials=[5])

        with self.assertRaisesRegex(ValueError, "not all in training data"):
            cross_validate(adapter=self.adapter, fold_generator=fold_generator)

        def fold_generator(training_data: ExperimentData) -> Iterable[CVData]:
            return gen_trial_split(
                training_data=training_data, train_trials=[0, 1], test_trials=[1]
            )

        with self.assertRaisesRegex(ValueError, "Test and train trials overlap"):
            cross_validate(adapter=self.adapter, fold_generator=fold_generator)

        def fold_generator(training_data: ExperimentData) -> Iterable[CVData]:
            return gen_trial_split(
                training_data=training_data, test_trials=[0, 1, 2, 3]
            )

        with self.assertRaisesRegex(ValueError, "All trials in data"):
            cross_validate(adapter=self.adapter, fold_generator=fold_generator)

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
            res = cross_validate(adapter=adapter, folds=-1)
        # SQ arm is repeated 3 times, so we add +2 for that.
        self.assertEqual(len(res), len(experiment.arms_by_name) + 2)

        # With untransform=False, LOOCV should work and log a warning.
        with self.assertLogs(logger=logger):
            res = cross_validate(adapter=adapter, folds=-1, untransform=False)
        # We only have one result for SQ arm here, due to TransformToNewSQ.
        self.assertEqual(len(res), len(experiment.arms_by_name))

        # 2-fold CV should error out.
        with self.assertRaisesRegex(
            CrossValidationError,
            "fewer test observations than predictions",
        ):
            cross_validate(adapter=adapter, folds=2, untransform=False)

    def test_cross_validate_gives_a_useful_error_for_insufficient_data(self) -> None:
        # Sobol with no data and torch with only one point.
        exp_empty = get_branin_experiment()
        exp = get_branin_experiment(with_completed_trial=True)
        for adapter in [
            Generators.SOBOL(experiment=exp_empty),
            Generators.BOTORCH_MODULAR(experiment=exp),
        ]:
            with self.assertRaisesRegex(UnsupportedError, "at least two in-design"):
                cross_validate(adapter=adapter)

    @mock_botorch_optimize
    def test_cross_validate_catches_warnings(self) -> None:
        exp = get_branin_experiment(with_batch=True, with_completed_batch=True)
        model = Generators.BOTORCH_MODULAR(
            experiment=exp, search_space=exp.search_space, data=exp.fetch_data()
        )
        for untransform in [False, True]:
            with warnings.catch_warnings(record=True) as ws:
                cross_validate(adapter=model, untransform=untransform)
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
            cross_validate(adapter=sobol)

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
        # Kendall tau rank correlation (NaN because y_pred is constant)
        self.assertTrue(np.isnan(diag["Kendall tau rank correlation"]["m1"]))
        self.assertTrue(np.isnan(diag["Kendall tau rank correlation"]["m2"]))

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

    def test_efficient_loo_cv_is_attempted(self) -> None:
        """Test that efficient LOO CV is attempted only when all conditions are met."""
        # Setup adapter with a BoTorchGenerator
        with (
            mock.patch(
                "botorch.cross_validation.efficient_loo_cv"
            ) as mock_efficient_loo,
            mock.patch("botorch.cross_validation.ensemble_loo_cv"),
        ):
            # Create mock LOO results
            # Create a mock posterior
            mock_mean = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
            mock_var = torch.tensor([[0.1], [0.1], [0.1], [0.1]])
            mock_mvn = MultivariateNormal(
                mean=mock_mean.squeeze(-1),
                covariance_matrix=DiagLinearOperator(mock_var.squeeze(-1)),
            )
            mock_posterior = GPyTorchPosterior(distribution=mock_mvn)

            # Get the surrogate model from the adapter
            surrogate = self.adapter.generator.surrogate
            model = surrogate.model

            mock_loo_results = CVResults(
                model=model,
                posterior=mock_posterior,
                observed_Y=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
                observed_Yvar=None,
            )
            mock_efficient_loo.return_value = mock_loo_results

            # Run cross_validate which will call _cross_validate for each fold
            result = cross_validate(adapter=self.adapter, folds=-1)

            # Verify we get results (either from efficient or fallback path)
            self.assertEqual(len(result), 4)

        # Test conditions that should prevent efficient LOO CV from being used
        # Each tuple: (kwargs_override, adapter_override, description)
        # pyre-ignore[9]: Type is correct for cross_validate kwargs
        conditions_preventing_efficient_loo: list[
            tuple[dict[str, object], TorchAdapter | None, str]
        ] = [
            ({"folds": 2}, None, "folds != -1"),
            ({"test_selector": lambda _: True}, None, "test_selector provided"),
        ]

        def _fold_gen(td: ExperimentData) -> Iterable[CVData]:
            return gen_trial_split(td, test_trials=[0])

        conditions_preventing_efficient_loo.append(
            ({"fold_generator": _fold_gen}, None, "fold_generator provided")
        )

        # Add refit_on_cv=True condition with separate adapter
        with mock_botorch_optimize_context_manager():
            adapter_refit = TorchAdapter(
                experiment=self.experiment,
                generator=BoTorchGenerator(refit_on_cv=True),
                transforms=[UnitX],
            )
        conditions_preventing_efficient_loo.append(
            ({}, adapter_refit, "refit_on_cv=True")
        )

        # Add auxiliary experiments condition
        # We test that the condition is checked correctly by mocking
        # get_training_data to avoid needing a fully functional adapter
        exp_with_aux = mock.MagicMock()
        exp_with_aux.auxiliary_experiments_by_purpose = {"some_purpose": ["aux_exp"]}
        adapter_with_aux = mock.MagicMock(spec=TorchAdapter)
        adapter_with_aux._experiment = exp_with_aux
        adapter_with_aux.generator = BoTorchGenerator()

        # For adapter with aux experiments, directly verify the condition check
        # rather than running through the full cross_validate path
        with (
            self.subTest(condition="has auxiliary experiments"),
            mock.patch(
                "ax.adapter.cross_validation._efficient_loo_cross_validate"
            ) as mock_efficient,
            mock.patch("ax.adapter.cross_validation._fold_cross_validate") as mock_fold,
        ):
            mock_fold.return_value = []
            cross_validate(adapter=adapter_with_aux)
            self.assertFalse(
                mock_efficient.called,
                "Efficient LOO should not be called when has auxiliary experiments",
            )

        for kwargs, adapter_override, desc in conditions_preventing_efficient_loo:
            adapter = adapter_override or self.adapter
            with (
                self.subTest(condition=desc),
                mock.patch(
                    "ax.adapter.cross_validation._efficient_loo_cross_validate"
                ) as mock_efficient,
            ):
                # pyre-ignore[6]: kwargs is properly typed for cross_validate
                cross_validate(adapter=adapter, **kwargs)
                self.assertFalse(
                    mock_efficient.called,
                    f"Efficient LOO should not be called when {desc}",
                )

        # Test logger when efficient LOO fails even though all conditions were met
        with self.subTest(condition="efficient LOO fails with exception"):
            with (
                mock.patch(
                    "ax.adapter.cross_validation._efficient_loo_cross_validate"
                ) as mock_efficient,
                mock.patch(
                    "ax.adapter.cross_validation._fold_cross_validate"
                ) as mock_fold,
                mock.patch("ax.adapter.cross_validation.logger") as mock_logger,
            ):
                # Force efficient LOO to fail
                mock_efficient.side_effect = ValueError("Test failure reason")
                mock_fold.return_value = []

                # Run cross_validate - should fall back to fold CV
                cross_validate(adapter=self.adapter, folds=-1)

                # Verify efficient LOO was attempted
                self.assertTrue(mock_efficient.called)
                # Verify fold CV was used as fallback
                self.assertTrue(mock_fold.called)
                # Verify the failure was logged
                mock_logger.debug.assert_called_once()
                log_message = mock_logger.debug.call_args[0][0]
                self.assertIn("Efficient LOO CV failed", log_message)
                self.assertIn("Test failure reason", log_message)

    def test_efficient_loo_cv_matches_naive(self) -> None:
        """End-to-end test: Ax.Adapter.cross_validate returns same results
        whether using efficient LOO CV or naive implementation.

        With refit_on_cv=False, both approaches should produce mathematically
        identical results because:
        1. Same hyperparameters are used (no refitting)
        2. Same LOO training/test splits (with unique arm names)
        3. Same posterior computation

        Tests all combinations of:
        - untransform: True and False
        - use_posterior_predictive: True and False
        - with_out_of_design_points: True and False

        This test uses StandardizeY (a Y-transform) to verify that the efficient
        LOO CV path correctly handles observation transforms. The Y-transform is
        critical because it changes the Y values between original and transformed
        space, and bugs in transform/untransform handling would cause observations
        and predictions to be compared in different spaces.

        It also tests out-of-design filtering by using expand_model_space=False
        to prevent automatic expansion of the model space bounds.
        """
        # Test all configurations
        for untransform, use_posterior_predictive, with_ood in product(
            [True, False], [True, False], [False, True]
        ):
            with mock_botorch_optimize_context_manager():
                if with_ood:
                    adapter = _create_adapter_with_out_of_design_points()
                    expected_count = _OOD_ADAPTER_IN_DESIGN_COUNT
                else:
                    adapter = _create_adapter_with_all_in_design_points()
                    expected_count = None

            with self.subTest(
                with_out_of_design=with_ood,
                untransform=untransform,
                use_posterior_predictive=use_posterior_predictive,
            ):
                self._test_efficient_loo_cv_matches_naive(
                    adapter=adapter,
                    untransform=untransform,
                    use_posterior_predictive=use_posterior_predictive,
                    expected_in_design_count=expected_count,
                )

    def _test_efficient_loo_cv_matches_naive(
        self,
        adapter: TorchAdapter,
        untransform: bool,
        use_posterior_predictive: bool,
        expected_in_design_count: int | None,
    ) -> None:
        """Run efficient vs naive CV and compare results.

        Args:
            adapter: The TorchAdapter to test.
            untransform: Whether to untransform predictions to original space.
            use_posterior_predictive: Whether to use posterior predictive.
            expected_in_design_count: Expected number of in-design points,
                or None if all points are in-design.
        """
        # Verify OOD setup if expected
        if expected_in_design_count is not None:
            all_data = adapter.get_training_data(filter_in_design=False)
            in_design_data = adapter.get_training_data(filter_in_design=True)
            self.assertGreater(
                len(all_data.arm_data),
                len(in_design_data.arm_data),
                "Test setup error: expected some out-of-design points",
            )
            self.assertEqual(
                len(in_design_data.arm_data),
                expected_in_design_count,
                f"Test setup error: expected {expected_in_design_count} in-design "
                "points",
            )

        # Run naive CV (by forcing fallback)
        with (
            mock.patch(
                "ax.adapter.cross_validation._efficient_loo_cross_validate",
                side_effect=ValueError("Force fallback to naive CV"),
            ),
            mock.patch(
                "ax.adapter.cross_validation._fold_cross_validate",
                wraps=_fold_cross_validate,
            ) as mock_naive_cv,
        ):
            result_naive = cross_validate(
                adapter=adapter,
                folds=-1,
                untransform=untransform,
                use_posterior_predictive=use_posterior_predictive,
            )

            # Verify naive path was used
            self.assertTrue(mock_naive_cv.called, "Naive CV not called")

        # Run efficient CV
        with (
            mock.patch(
                "ax.adapter.cross_validation._efficient_loo_cross_validate",
                wraps=_efficient_loo_cross_validate,
            ) as mock_efficient,
            mock.patch(
                "ax.adapter.cross_validation._fold_cross_validate",
            ) as mock_naive,
        ):
            result_efficient = cross_validate(
                adapter=adapter,
                folds=-1,
                untransform=untransform,
                use_posterior_predictive=use_posterior_predictive,
            )

            # Verify efficient path was used successfully
            self.assertTrue(mock_efficient.called, "Efficient LOO CV not called")
            self.assertFalse(
                mock_naive.called,
                "Naive CV was called (efficient failed)",
            )

        # Verify result counts match
        self.assertEqual(len(result_efficient), len(result_naive))

        # Verify OOD filtering if expected
        if expected_in_design_count is not None:
            self.assertEqual(
                len(result_efficient),
                expected_in_design_count,
                "Should only include in-design points",
            )

        # Sort for consistent comparison
        def sort_key(cv_result: CVResult) -> tuple[float, ...]:
            return tuple(cv_result.observed.data.means.tolist())

        result_efficient_sorted = sorted(result_efficient, key=sort_key)
        result_naive_sorted = sorted(result_naive, key=sort_key)

        # Verify observations are in correct space (only for non-OOD case
        # where we have StandardizeY with controlled Y values)
        if expected_in_design_count is None:
            for cv_result in result_efficient_sorted:
                obs_means = cv_result.observed.data.means
                if untransform:
                    self.assertTrue(
                        np.all(obs_means > 5.0),
                        f"untransform=True: expected original space, got {obs_means}",
                    )
                else:
                    self.assertTrue(
                        np.all(np.abs(obs_means) < 3.0),
                        f"untransform=False: expected standardized, got {obs_means}",
                    )

        # Compare predictions
        for cv_efficient, cv_naive in zip(
            result_efficient_sorted, result_naive_sorted, strict=True
        ):
            np.testing.assert_array_equal(
                cv_efficient.observed.data.means,
                cv_naive.observed.data.means,
            )
            np.testing.assert_allclose(
                cv_efficient.predicted.means,
                cv_naive.predicted.means,
                rtol=1e-4,
                atol=1e-6,
                err_msg="Predicted means don't match",
            )
            np.testing.assert_allclose(
                cv_efficient.predicted.covariance,
                cv_naive.predicted.covariance,
                rtol=1e-4,
                atol=1e-6,
                err_msg="Predicted covariances don't match",
            )

    def test_efficient_loo_cv_with_robust_relevance_pursuit_model(self) -> None:
        """Test that RobustRelevancePursuitSingleTaskGP uses efficient LOO CV.

        This test verifies that:
        1) An Adapter with a RobustRelevancePursuitSingleTaskGP surrogate can
           execute CV successfully using the efficient implementation.
        2) If the efficient implementation fails, the entire CV fails because
           the robust relevance pursuit model doesn't support the regular CV path
           (due to state incompatibility when refitting).
        """
        # Create a simple experiment with data
        experiment = get_branin_experiment(with_batch=True, with_completed_batch=True)

        # Create adapter with RobustRelevancePursuitSingleTaskGP
        adapter = TorchAdapter(
            experiment=experiment,
            generator=BoTorchGenerator(
                surrogate=Surrogate(
                    surrogate_spec=SurrogateSpec(
                        model_configs=[
                            ModelConfig(
                                botorch_model_class=RobustRelevancePursuitSingleTaskGP,
                            )
                        ],
                    ),
                ),
            ),
            transforms=[UnitX],
        )

        # Part 1: Verify that efficient LOO CV works with this model
        # The efficient implementation should be called and succeed
        with mock.patch(
            "botorch.cross_validation.efficient_loo_cv",
            wraps=efficient_loo_cv,
        ) as mock_efficient_loo:
            result = cross_validate(adapter=adapter, folds=-1)

            # Verify we got results
            self.assertGreater(len(result), 0)

            # Verify efficient_loo_cv was called (at least once per unique fold)
            self.assertTrue(mock_efficient_loo.called)

        # Part 2: Verify that if efficient implementation fails, CV fails entirely
        # because RobustRelevancePursuitSingleTaskGP doesn't support naive CV
        # (due to state_dict size mismatch when the model is refitted with LOO data)
        with mock.patch(
            "ax.adapter.cross_validation._efficient_loo_cross_validate",
            side_effect=ValueError("Simulated efficient LOO CV failure"),
        ):
            # The naive CV path should fail for RobustRelevancePursuitSingleTaskGP
            # because it uses SparseOutlierGaussianLikelihood which has state
            # (raw_rho) that changes during fitting and can't be transferred
            # to a model fitted on different data
            with self.assertRaises((ValueError, RuntimeError)):
                cross_validate(adapter=adapter, folds=-1)

    def test_efficient_loo_cv_with_fully_bayesian_model(self) -> None:
        """Test that FullyBayesianSAAS models use efficient LOO CV via ensemble_loo_cv.

        This test verifies that:
        1) An Adapter with a SaasFullyBayesianSingleTaskGP surrogate triggers
           the efficient LOO CV path.
        2) The ensemble_loo_cv function is used (not efficient_loo_cv) because
           SaasFullyBayesianSingleTaskGP has _is_ensemble=True.
        3) The efficient and naive implementations produce matching results.
        """
        # Create a simple experiment with data
        experiment = get_branin_experiment(with_batch=True, with_completed_batch=True)

        # Create adapter with SaasFullyBayesianSingleTaskGP
        adapter = TorchAdapter(
            experiment=experiment,
            generator=BoTorchGenerator(
                surrogate=Surrogate(
                    surrogate_spec=SurrogateSpec(
                        model_configs=[
                            ModelConfig(
                                botorch_model_class=SaasFullyBayesianSingleTaskGP,
                            )
                        ],
                    ),
                )
            ),
            transforms=[UnitX],
        )

        # We need to mock the MCMC fitting to avoid running actual NUTS sampling
        # which is very slow. Instead, we'll inject mock MCMC samples.
        surrogate = adapter.generator.surrogate  # pyre-ignore[16]
        model = surrogate.model

        # Verify the model is a SaasFullyBayesianSingleTaskGP
        self.assertIsInstance(model, SaasFullyBayesianSingleTaskGP)

        # Get training data shape info
        train_X = model.train_inputs[0]
        d = train_X.shape[-1]
        num_models = 4  # Number of MCMC samples

        # Create mock MCMC samples
        tkwargs = {"dtype": train_X.dtype, "device": train_X.device}
        mcmc_samples = {
            "lengthscale": torch.rand(num_models, 1, d, **tkwargs),
            "outputscale": torch.rand(num_models, **tkwargs),
            "mean": torch.randn(num_models, **tkwargs),
            "noise": torch.rand(num_models, 1, **tkwargs) * 0.1 + 0.01,
        }
        model.load_mcmc_samples(mcmc_samples)

        # Verify the model is an ensemble model
        self.assertTrue(model._is_ensemble)

        # Part 1: Run cross_validate with efficient LOO CV disabled first
        # (by making _efficient_loo_cross_validate raise a ValueError so it uses naive)
        with mock.patch(
            "ax.adapter.cross_validation._efficient_loo_cross_validate",
            side_effect=ValueError("Force fallback to naive CV"),
        ):
            result_naive = cross_validate(adapter=adapter, folds=-1)

        # Part 2: Run cross_validate with efficient LOO CV enabled (default path)
        # Also verify that ensemble_loo_cv is called
        with mock.patch(
            "botorch.cross_validation.ensemble_loo_cv",
            wraps=ensemble_loo_cv,
        ) as mock_ensemble_loo:
            result_efficient = cross_validate(adapter=adapter, folds=-1)

            # Verify ensemble_loo_cv was called (at least once per unique fold)
            self.assertTrue(mock_ensemble_loo.called)

        # Part 3: Compare the predictions from both methods
        # Both should return the same number of results
        self.assertEqual(len(result_efficient), len(result_naive))

        # Sort both results by observed means to ensure consistent comparison
        # (ordering may differ between efficient and naive implementations)
        def sort_key(cv_result: CVResult) -> tuple[float, ...]:
            return tuple(cv_result.observed.data.means.tolist())

        result_efficient_sorted = sorted(result_efficient, key=sort_key)
        result_naive_sorted = sorted(result_naive, key=sort_key)

        for cv_efficient, cv_naive in zip(
            result_efficient_sorted, result_naive_sorted, strict=True
        ):
            # The observed values should be identical
            np.testing.assert_array_equal(
                cv_efficient.observed.data.means,
                cv_naive.observed.data.means,
            )

            # The predicted means should be very close
            np.testing.assert_allclose(
                cv_efficient.predicted.means,
                cv_naive.predicted.means,
                rtol=1e-4,
                atol=1e-6,
                err_msg="Efficient and naive LOO CV predicted means don't match",
            )

            # The predicted covariances should be very close
            np.testing.assert_allclose(
                cv_efficient.predicted.covariance,
                cv_naive.predicted.covariance,
                rtol=1e-4,
                atol=1e-6,
                err_msg="Efficient and naive LOO CV predicted covariances don't match",
            )


def _create_adapter_with_all_in_design_points() -> TorchAdapter:
    """Create a test adapter where all points are in-design.

    Creates an experiment with 4 unique parameterizations and Y values
    with clear variation for StandardizeY testing.

    Returns:
        A TorchAdapter with all points in-design.
    """
    # pyre-ignore [9]: Pyre is too picky with union types.
    parameterizations: list[TParameterization] = [
        {"x": x} for x in [1.0, 2.0, 3.0, 4.0]
    ]
    # Use Y values with clear variation for StandardizeY
    means = [[10.0, 20.0], [20.0, 30.0], [30.0, 40.0], [40.0, 50.0]]
    sems = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
    experiment = get_experiment_with_observations(
        observations=means,
        sems=sems,
        search_space=get_search_space_for_range_value(min=0.0, max=10.0),
        parameterizations=parameterizations,
    )

    return TorchAdapter(
        experiment=experiment,
        generator=BoTorchGenerator(refit_on_cv=False),
        transforms=[UnitX, StandardizeY],
    )


def _create_adapter_with_out_of_design_points() -> TorchAdapter:
    """Create a test adapter with out-of-design points.

    Creates a branin experiment with 3 in-design trials and 1 out-of-design
    trial (outside the search space bounds). Uses expand_model_space=False
    to prevent automatic expansion of model space bounds.

    Returns:
        A TorchAdapter with 3 in-design points and 1 out-of-design point.
    """
    # Create branin experiment with manually added OOD trial
    experiment = get_branin_experiment(with_batch=False)

    # Add in-design trials
    # NOTE: The number of in-design points must match _OOD_ADAPTER_IN_DESIGN_COUNT
    in_design_points = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    assert len(in_design_points) == _OOD_ADAPTER_IN_DESIGN_COUNT
    for i, (x1, x2) in enumerate(in_design_points):
        trial = experiment.new_trial()
        arm = Arm(parameters={"x1": x1, "x2": x2}, name=f"in_design_{i}")
        trial.add_arm(arm)
        trial.run()
        trial.mark_completed()

    # Add out-of-design trial (outside bounds)
    ood_trial = experiment.new_trial()
    ood_arm = Arm(parameters={"x1": 100.0, "x2": 100.0}, name="out_of_design")
    ood_trial.add_arm(ood_arm)
    ood_trial.run()
    ood_trial.mark_completed()

    # Attach data
    branin_metric = experiment.metrics["branin"]
    metric_signature = branin_metric.signature
    data_rows = []
    for trial_index, trial in experiment.trials.items():
        if isinstance(trial, Trial) and trial.arm is not None:
            data_rows.append(
                {
                    "arm_name": trial.arm.name,
                    "trial_index": trial_index,
                    "metric_name": "branin",
                    "metric_signature": metric_signature,
                    "mean": float(trial_index) + 1.0,
                    "sem": 0.1,
                }
            )
    experiment.attach_data(Data(df=DataFrame(data_rows)))

    return TorchAdapter(
        experiment=experiment,
        generator=BoTorchGenerator(refit_on_cv=False),
        transforms=[UnitX, StandardizeY],
        expand_model_space=False,
    )
