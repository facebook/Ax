#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Sized
from contextlib import ExitStack
from typing import Any
from unittest import mock

import numpy as np
import torch
from ax.adapter.adapter_utils import _binary_pref_to_comp_pair, _consolidate_comparisons
from ax.adapter.base import Adapter
from ax.adapter.cross_validation import cross_validate
from ax.adapter.registry import Cont_X_trans, MBM_X_trans
from ax.adapter.torch import TorchAdapter
from ax.adapter.transforms.one_hot import OneHot
from ax.adapter.transforms.standardize_y import StandardizeY
from ax.adapter.transforms.unit_x import UnitX
from ax.core.arm import Arm
from ax.core.auxiliary import AuxiliaryExperiment, AuxiliaryExperimentPurpose
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig, PreferenceOptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint, ScalarizedOutcomeConstraint
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace, SearchSpaceDigest
from ax.core.types import ComparisonOp
from ax.exceptions.core import DataRequiredError
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
from ax.generators.torch.botorch_modular.surrogate import Surrogate, SurrogateSpec
from ax.generators.torch.botorch_modular.utils import ModelConfig
from ax.generators.torch_base import TorchGenerator, TorchGenResults
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.stats.model_fit_stats import MSE
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
    get_data,
    get_experiment_with_observations,
    get_search_space_for_range_value,
    get_search_space_for_range_values,
)
from ax.utils.testing.mock import mock_botorch_optimize
from ax.utils.testing.preference_stubs import get_pbo_experiment
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.preference import (
    AnalyticExpectedUtilityOfBestOption,
    qExpectedUtilityOfBestOption,
)
from botorch.models import SingleTaskGP
from botorch.models.map_saas import AdditiveMapSaasSingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.utils.datasets import (
    ContextualDataset,
    MultiTaskDataset,
    RankingDataset,
    SupervisedDataset,
)
from pandas import DataFrame
from pyre_extensions import assert_is_instance, none_throws


class TorchAdapterTest(TestCase):
    @mock_botorch_optimize
    def test_TorchAdapter(self, device: torch.device | None = None) -> None:
        tkwargs: dict[str, Any] = {"dtype": torch.double, "device": device}
        # Construct an experiment with known data.
        feature_names = ["x1", "x2", "x3"]
        search_space = get_search_space_for_range_values(
            min=0.0, max=5.0, parameter_names=feature_names
        )
        opt_config = OptimizationConfig(
            objective=Objective(metric=Metric("y1"), minimize=True),
            outcome_constraints=[
                OutcomeConstraint(
                    metric=Metric("y2"), op=ComparisonOp.GEQ, bound=0.0, relative=False
                )
            ],
        )
        experiment = Experiment(
            search_space=search_space, optimization_config=opt_config, name="test"
        )
        X = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], **tkwargs)
        for x in X.tolist():
            experiment.new_trial().add_arm(
                Arm(parameters=dict(zip(feature_names, x)))
            ).mark_running(no_runner_required=True).mark_completed()
        experiment.attach_data(
            data=Data(
                df=DataFrame.from_records(
                    {
                        "trial_index": [0, 0, 1, 1],
                        "metric_name": ["y1", "y2", "y1", "y2"],
                        "arm_name": ["0_0", "0_0", "1_0", "1_0"],
                        "mean": [3.0, 2.0, 1.0, 0.0],
                        "sem": [3.0, 1e-4, 2.0, 1e-3],
                        "metric_signature": ["y1", "y2", "y1", "y2"],
                    }
                )
            )
        )
        # Construct the adapter and test key methods.
        adapter = TorchAdapter(
            experiment=experiment,
            generator=BoTorchGenerator(),
            torch_device=device,
            fit_on_init=False,
        )
        self.assertTrue(adapter.can_predict)
        self.assertTrue(adapter.can_model_in_sample)
        self.assertEqual(adapter.device, device)
        self.assertIsNone(adapter._last_experiment_data)
        experiment_data = adapter.get_training_data()
        # Test `_fit`.
        X = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], **tkwargs)
        datasets = [
            SupervisedDataset(
                X=X,
                Y=torch.tensor([[3.0], [1.0]], **tkwargs),
                Yvar=torch.tensor([[9.0], [4.0]], **tkwargs),
                feature_names=feature_names,
                outcome_names=["y1"],
                group_indices=torch.tensor([0, 1], device=device),
            ),
            SupervisedDataset(
                X=X,
                Y=torch.tensor([[2.0], [0.0]], **tkwargs),
                Yvar=torch.tensor([[1e-8], [1e-6]], **tkwargs),
                feature_names=feature_names,
                outcome_names=["y2"],
                group_indices=torch.tensor([0, 1], device=device),
            ),
        ]
        observation_features = [
            ObservationFeatures(parameters=dict(zip(feature_names, Xi.tolist())))
            for Xi in X
        ]

        generator = adapter.generator
        with mock.patch.object(generator, "fit", wraps=generator.fit) as mock_fit:
            adapter._fit(search_space=search_space, experiment_data=experiment_data)
        generator_fit_args = mock_fit.call_args.kwargs
        self.assertEqual(generator_fit_args["datasets"], datasets)
        expected_ssd = SearchSpaceDigest(
            feature_names=feature_names, bounds=[(0, 5)] * 3
        )
        self.assertEqual(generator_fit_args["search_space_digest"], expected_ssd)
        self.assertEqual(
            generator_fit_args["candidate_metadata"],
            [[{Keys.TRIAL_COMPLETION_TIMESTAMP: mock.ANY}] * 2] * 2,
        )
        self.assertEqual(adapter._last_experiment_data, experiment_data)

        with mock.patch(f"{TorchAdapter.__module__}.logger.debug") as mock_logger:
            adapter._fit(search_space=search_space, experiment_data=experiment_data)
        mock_logger.assert_called_once_with(
            "The experiment data is identical to the last experiment data "
            "used to fit the generator. Skipping generator fitting."
        )

        # Test `_predict`
        pred_means = [3.0, 2.0]
        pred_var = [4.0, 3.0]
        predict_return_value = (
            torch.tensor([pred_means], **tkwargs),
            torch.tensor([[[pred_var[0], 0.0], [0.0, pred_var[1]]]], **tkwargs),
        )
        pr_obs_data_expected = ObservationData(
            metric_signatures=["y1", "y2"],
            means=np.array(pred_means),
            covariance=np.diag(pred_var),
        )
        with mock.patch.object(
            generator, "predict", return_value=predict_return_value
        ) as mock_predict:
            pr_obs_data = adapter._predict(
                observation_features=observation_features[:1]
            )
        self.assertTrue(torch.equal(mock_predict.mock_calls[0][2]["X"], X[:1]))
        self.assertEqual(pr_obs_data, [pr_obs_data_expected])

        # Test `_gen`
        gen_return_value = TorchGenResults(
            points=torch.tensor([[1.0, 2.0, 3.0]], **tkwargs),
            weights=torch.tensor([1.0], **tkwargs),
            gen_metadata={"foo": 99},
        )
        best_point_return_value = torch.tensor([1.0, 2.0, 3.0], **tkwargs)
        opt_config = OptimizationConfig(
            objective=Objective(metric=Metric("y1"), minimize=False),
        )
        pending_observations = {
            "y2": [ObservationFeatures(parameters={"x1": 1.0, "x2": 2.0, "x3": 3.0})]
        }
        with ExitStack() as es, mock.patch.object(
            generator, "gen", return_value=gen_return_value
        ) as mock_gen:
            es.enter_context(
                mock.patch.object(
                    generator, "best_point", return_value=best_point_return_value
                )
            )
            es.enter_context(
                mock.patch(
                    f"{TorchAdapter.__module__}.TorchAdapter."
                    "_array_callable_to_tensor_callable",
                    return_value=torch.round,
                )
            )
            es.enter_context(
                # silence a warning about inability to generate unique candidates
                mock.patch(f"{Adapter.__module__}.logger.warning")
            )
            gen_run = adapter.gen(
                n=3,
                search_space=search_space,
                optimization_config=opt_config,
                pending_observations=pending_observations,
                fixed_features=ObservationFeatures(parameters={"x2": 3.0}),
                model_gen_options={"option": "yes"},
            )
        gen_args = mock_gen.mock_calls[0][2]
        self.assertEqual(gen_args["n"], 3)
        self.assertEqual(gen_args["search_space_digest"], expected_ssd)
        gen_opt_config = gen_args["torch_opt_config"]
        self.assertTrue(
            torch.equal(
                gen_opt_config.objective_weights,
                torch.tensor([1.0, 0.0], **tkwargs),
            )
        )
        self.assertIsNone(gen_opt_config.outcome_constraints)
        self.assertIsNone(gen_opt_config.linear_constraints)
        self.assertEqual(gen_opt_config.fixed_features, {1: 3.0})
        X_pending_y1, X_pending_y2 = gen_opt_config.pending_observations
        self.assertTrue(torch.equal(X_pending_y1, torch.tensor([], **tkwargs)))
        self.assertTrue(
            torch.equal(X_pending_y2, torch.tensor([[1.0, 2.0, 3.0]], **tkwargs))
        )
        self.assertEqual(gen_opt_config.model_gen_options, {"option": "yes"})
        self.assertIs(gen_opt_config.rounding_func, torch.round)
        self.assertFalse(gen_opt_config.is_moo)
        self.assertEqual(gen_opt_config.opt_config_metrics, opt_config.metrics)
        self.assertEqual(gen_args["search_space_digest"].target_values, {})
        self.assertEqual(len(gen_run.arms), 1)
        self.assertEqual(gen_run.arms[0].parameters, {"x1": 1.0, "x2": 2.0, "x3": 3.0})
        self.assertEqual(gen_run.weights, [1.0])
        self.assertEqual(gen_run.fit_time, 0.0)
        self.assertEqual(gen_run.gen_metadata, {"foo": 99})

        # Test `_cross_validate`
        cv_obs_data_expected = ObservationData(
            metric_signatures=["y1", "y2"],
            means=np.array([3.0, 2.0]),
            covariance=np.diag([4.0, 3.0]),
        )
        cv_test_points = [
            ObservationFeatures(parameters={"x1": 1.0, "x2": 3.0, "x3": 2.0})
        ]
        X_test = torch.tensor([[1.0, 3.0, 2.0]], **tkwargs)

        with mock.patch.object(
            generator,
            "cross_validate",
            return_value=predict_return_value,
        ) as mock_cross_validate:
            cv_obs_data = adapter._cross_validate(
                search_space=search_space,
                cv_training_data=experiment_data,
                cv_test_points=cv_test_points,
            )
        generator_cv_args = mock_cross_validate.mock_calls[0][2]
        self.assertEqual(generator_cv_args["datasets"], datasets)
        self.assertTrue(torch.equal(generator_cv_args["X_test"], X_test))
        self.assertEqual(generator_cv_args["search_space_digest"], expected_ssd)
        self.assertEqual(cv_obs_data, [cv_obs_data_expected])

        # Transform observations
        # This functionality is likely to be deprecated (T134940274)
        # so this is not a thorough test.
        adapter.transform_observations(
            observations=[
                Observation(features=cv_test_points[0], data=cv_obs_data_expected)
            ]
        )

        # Transform observation features
        obsf = [ObservationFeatures(parameters={"x": 1.0, "y": 2.0})]
        adapter.parameters = ["x", "y"]
        X = adapter._transform_observation_features(observation_features=obsf)
        self.assertTrue(torch.equal(X, torch.tensor([[1.0, 2.0]], **tkwargs)))

    def test_TorchAdapter_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_TorchAdapter(device=torch.device("cuda"))

    @mock_botorch_optimize
    def test_evaluate_acquisition_function(self) -> None:
        experiment = get_branin_experiment(with_completed_trial=True)
        adapter = TorchAdapter(
            experiment=experiment,
            generator=BoTorchGenerator(),
            transforms=[UnitX, StandardizeY],
        )
        obsf = ObservationFeatures(parameters={"x1": 1.0, "x2": 2.0})

        # Check for value error when optimization config is not set.
        with mock.patch.object(
            adapter, "_optimization_config", None
        ), self.assertRaisesRegex(ValueError, "optimization_config"):
            adapter.evaluate_acquisition_function(observation_features=[obsf])

        mock_acq_val = 5.0
        with mock.patch.object(
            adapter, "_evaluate_acquisition_function", return_value=[mock_acq_val]
        ) as mock_eval:
            acqf_vals = adapter.evaluate_acquisition_function(
                observation_features=[obsf]
            )
        self.assertEqual(acqf_vals, [mock_acq_val])
        mock_eval.assert_called_once()
        # Check that the private method was called with transformed obsf.
        # Bounds for branin are [-5.0, 10.0] and [0.0, 15.0].
        expected_X = [6.0 / 15.0, 2.0 / 15.0]
        self.assertEqual(
            mock_eval.call_args.kwargs["observation_features"],
            [
                [
                    ObservationFeatures(
                        parameters={"x1": expected_X[0], "x2": expected_X[1]}
                    )
                ]
            ],
        )

        # Check calls down to the acquisition function.
        acqf_path = "botorch.acquisition.logei.qLogNoisyExpectedImprovement.forward"
        with mock.patch(
            acqf_path, return_value=torch.tensor([mock_acq_val], dtype=torch.double)
        ) as mock_acqf:
            acqf_vals = adapter.evaluate_acquisition_function(
                observation_features=[obsf]
            )
        self.assertEqual(acqf_vals, [mock_acq_val])
        mock_acqf.assert_called_once()
        expected_tensor = torch.tensor([[expected_X]], dtype=torch.double)
        self.assertAllClose(mock_acqf.call_args.kwargs["X"], expected_tensor)

        # Test evaluating at multiple points.
        # Case 1: List[ObsFeat, ObsFeat], should be 2 x 1 x d.
        with mock.patch(
            acqf_path,
            return_value=torch.tensor([mock_acq_val, mock_acq_val], dtype=torch.double),
        ) as mock_acqf:
            acqf_vals = adapter.evaluate_acquisition_function(
                observation_features=[obsf, obsf.clone()]
            )
        mock_acqf.assert_called_once()
        self.assertAllClose(
            mock_acqf.call_args.kwargs["X"], expected_tensor.repeat(2, 1, 1)
        )
        # Case 2: List[List[ObsFeat, ObsFeat]], should be 1 x 2 x d.
        with mock.patch(
            acqf_path,
            return_value=torch.tensor([mock_acq_val, mock_acq_val], dtype=torch.double),
        ) as mock_acqf:
            acqf_vals = adapter.evaluate_acquisition_function(
                observation_features=[[obsf, obsf.clone()]]
            )
        mock_acqf.assert_called_once()
        self.assertAllClose(
            mock_acqf.call_args.kwargs["X"], expected_tensor.repeat(1, 2, 1)
        )

    def test_best_point(self) -> None:
        search_space = get_search_space_for_range_value()
        oc = OptimizationConfig(
            objective=Objective(metric=Metric("a"), minimize=False),
            outcome_constraints=[],
        )
        exp = Experiment(search_space=search_space, optimization_config=oc, name="test")
        exp.new_trial().add_arm(Arm(parameters={"x": 1.0})).mark_running(
            no_runner_required=True
        ).mark_completed()
        exp.attach_data(get_data(metric_name="a", num_non_sq_arms=1, include_sq=False))
        adapter = TorchAdapter(
            experiment=exp,
            generator=TorchGenerator(),
            transforms=[OneHot, UnitX],
        )
        self.assertEqual(
            list(adapter.transforms.keys()),
            ["FillMissingParameters", "Cast", "OneHot", "UnitX"],
        )

        mean = 1.0
        cov = 2.0
        predict_return_value = ({"m": [mean]}, {"m": {"m": [cov]}})
        best_point_value = 0.6
        gen_return_value = TorchGenResults(
            points=torch.tensor([[1.0]]), weights=torch.tensor([1.0])
        )
        with mock.patch(
            f"{TorchGenerator.__module__}.TorchGenerator.best_point",
            return_value=torch.tensor([best_point_value]),
            autospec=True,
        ), mock.patch.object(adapter, "predict", return_value=predict_return_value):
            with mock.patch.object(
                adapter.generator, "gen", return_value=gen_return_value
            ):
                run = adapter.gen(n=1, optimization_config=oc)

            _, model_predictions = none_throws(adapter.model_best_point())

        arm, predictions = none_throws(run.best_arm_predictions)
        predictions = none_throws(predictions)
        model_predictions = none_throws(model_predictions)
        # UnitX removes 1 and divides by 5. Reversing here.
        self.assertEqual(arm.parameters.keys(), {"x"})
        self.assertAlmostEqual(
            float(arm.parameters["x"]), (best_point_value * 5.0) + 1.0, places=5
        )
        # 1.0 in transformed space is 6.0 in original space.
        self.assertEqual(run.arms[0].parameters, {"x": 6.0})
        self.assertEqual(predictions[0], {"m": mean})
        self.assertEqual(predictions[1], {"m": {"m": cov}})
        self.assertEqual(model_predictions[0], {"m": mean})
        self.assertEqual(model_predictions[1], {"m": {"m": cov}})

        # test optimization config validation - raise error when
        # ScalarizedOutcomeConstraint contains a metric that is not in the outcomes
        with self.assertRaisesRegex(ValueError, "as a relative constraint."):
            adapter.gen(
                n=1,
                optimization_config=OptimizationConfig(
                    objective=Objective(metric=Metric("a"), minimize=False),
                    outcome_constraints=[
                        ScalarizedOutcomeConstraint(
                            metrics=[Metric("wrong_metric_name")],
                            weights=[1.0],
                            op=ComparisonOp.LEQ,
                            bound=0,
                        )
                    ],
                ),
            )

        with mock.patch(
            f"{TorchGenerator.__module__}.TorchGenerator.best_point",
            side_effect=NotImplementedError,
            autospec=True,
        ):
            res = adapter.model_best_point()
        self.assertIsNone(res)

    @mock_botorch_optimize
    def test_importances(self) -> None:
        experiment = get_branin_experiment_with_multi_objective(
            with_completed_trial=True
        )
        adapter = TorchAdapter(experiment=experiment, generator=BoTorchGenerator())
        # generator doesn't have enough data for training, so equal importances.
        self.assertEqual(
            adapter.feature_importances("branin_a"), {"x1": 0.5, "x2": 0.5}
        )
        self.assertEqual(
            adapter.feature_importances("branin_b"), {"x1": 0.5, "x2": 0.5}
        )

    def test_candidate_metadata_propagation(self) -> None:
        exp = get_branin_experiment(with_status_quo=True, with_completed_batch=True)
        # Check that the metadata is correctly re-added to observation
        # features during `fit`.
        preexisting_batch_gr = exp.trials[0].generator_runs[0]
        preexisting_batch_gr._candidate_metadata_by_arm_signature = {
            preexisting_batch_gr.arms[0].signature: {
                "preexisting_batch_cand_metadata": "some_value"
            }
        }
        generator = TorchGenerator()
        with mock.patch.object(
            generator, "fit", wraps=generator.fit
        ) as mock_generator_fit:
            adapter = TorchAdapter(experiment=exp, generator=generator)

        datasets = mock_generator_fit.call_args.kwargs.get("datasets")
        X_expected = torch.tensor(
            [list(arm.parameters.values()) for arm in exp.trials[0].arms],
            dtype=torch.double,
        )
        for dataset in datasets:
            self.assertTrue(torch.equal(dataset.X, X_expected))

        candidate_metadata = mock_generator_fit.call_args.kwargs.get(
            "candidate_metadata"
        )
        self.assertEqual(len(candidate_metadata), 1)
        self.assertEqual(len(candidate_metadata[0]), len(exp.trials[0].arms))
        self.assertEqual(
            candidate_metadata[0][0],
            {
                "preexisting_batch_cand_metadata": "some_value",
                Keys.TRIAL_COMPLETION_TIMESTAMP: mock.ANY,
            },
        )

        # Check that `gen` correctly propagates the metadata to the GR.
        candidate_metadata = [
            {"some_key": "some_value_0"},
            {"some_key": "some_value_1"},
        ]
        gen_results = TorchGenResults(
            points=torch.tensor([[1, 2], [2, 3]]),
            weights=torch.tensor([1.0, 2.0]),
            candidate_metadata=candidate_metadata,
        )
        with mock.patch.object(generator, "gen", return_value=gen_results):
            gr = adapter.gen(n=1)
        self.assertEqual(
            gr.candidate_metadata_by_arm_signature,
            {
                gr.arms[0].signature: candidate_metadata[0],
                gr.arms[1].signature: candidate_metadata[1],
            },
        )

        # Check that `None` candidate metadata is handled correctly.
        gen_results = TorchGenResults(
            points=torch.tensor([[2, 4], [3, 5]]),
            weights=torch.tensor([1.0, 2.0]),
            candidate_metadata=None,
        )
        with mock.patch.object(generator, "gen", return_value=gen_results):
            gr = adapter.gen(n=1)
        self.assertIsNone(gr.candidate_metadata_by_arm_signature)

        # Check that no candidate metadata is handled correctly.
        exp = get_branin_experiment(with_status_quo=True, with_completed_trial=True)
        generator = TorchGenerator()
        with mock.patch.object(
            generator, "fit", wraps=generator.fit
        ) as mock_generator_fit:
            adapter = TorchAdapter(experiment=exp, generator=generator)
        with mock.patch.object(generator, "gen", return_value=gen_results):
            gr = adapter.gen(n=1)
        # This should be None since gen_results doesn't include any metadata.
        self.assertIsNone(gr.candidate_metadata_by_arm_signature)

    def test_fit_tracking_metrics(self) -> None:
        exp = get_experiment_with_observations(
            observations=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            with_tracking_metrics=True,
        )
        for fit_tracking_metrics in (True, False):
            generator = TorchGenerator()
            with mock.patch.object(
                generator, "fit", wraps=generator.fit
            ) as mock_generator_fit:
                adapter = TorchAdapter(
                    experiment=exp,
                    search_space=exp.search_space,
                    data=exp.lookup_data(),
                    generator=generator,
                    transforms=[],
                    fit_tracking_metrics=fit_tracking_metrics,
                )
            mock_generator_fit.assert_called_once()
            call_kwargs = mock_generator_fit.call_args.kwargs
            if fit_tracking_metrics:
                expected_outcomes = ["m1", "m2", "m3"]
            else:
                expected_outcomes = ["m1", "m2"]
            self.assertEqual(adapter.outcomes, expected_outcomes)
            self.assertEqual(len(call_kwargs["datasets"]), len(expected_outcomes))

    def test_convert_experiment_data(self) -> None:
        feature_names = ["x0", "x1", "x2"]
        search_space = get_search_space_for_range_values(
            min=0.0, max=5.0, parameter_names=feature_names
        )
        raw_X = torch.rand(10, 3) * 5
        raw_X[:, -1].round_()  # Make sure last column is integer.
        raw_X[0, -1] = 0  # Make sure task value 0 exists.
        raw_Y = torch.sin(raw_X).sum(-1, keepdim=True)
        experiment = get_experiment_with_observations(
            parameterizations=[
                {f"x{i}": x_[i].item() for i in range(3)} for x_ in raw_X
            ],
            observations=raw_Y.tolist(),
            search_space=search_space,
        )
        adapter = TorchAdapter(experiment=experiment, generator=BoTorchGenerator())
        metric_signatures = ["m1"]
        experiment_data = adapter.get_training_data()
        for use_task, expected_class in (
            (True, MultiTaskDataset),
            (False, SupervisedDataset),
        ):
            search_space_digest = SearchSpaceDigest(
                feature_names=feature_names,
                bounds=[(0.0, 5.0)] * 3,
                ordinal_features=[2],
                discrete_choices={2: list(range(0, 11))},
                task_features=[2] if use_task else [],
                target_values={2: 0} if use_task else {},  # pyre-ignore
            )
            converted_datasets, ordered_outcomes, _ = adapter._convert_experiment_data(
                experiment_data=experiment_data,
                outcomes=metric_signatures,
                parameters=feature_names,
                search_space_digest=search_space_digest,
            )
            self.assertEqual(len(converted_datasets), 1)
            dataset = none_throws(converted_datasets[0])
            self.assertIs(dataset.__class__, expected_class)
            if use_task:
                sort_idx = torch.argsort(raw_X[:, -1])
                expected_X = raw_X[sort_idx]
                expected_Y = raw_Y[sort_idx]
            else:
                expected_X = raw_X
                expected_Y = raw_Y
            self.assertTrue(torch.equal(dataset.X, expected_X.to(torch.double)))
            self.assertTrue(torch.equal(dataset.Y, expected_Y))
            self.assertIsNone(dataset.Yvar)
            self.assertEqual(dataset.feature_names, feature_names)
            self.assertEqual(dataset.outcome_names, metric_signatures)
            self.assertEqual(ordered_outcomes, metric_signatures)

            with self.assertRaisesRegex(DataRequiredError, "no corresponding data"):
                adapter._convert_experiment_data(
                    experiment_data=experiment_data,
                    outcomes=metric_signatures + ["extra"],
                    parameters=feature_names,
                    search_space_digest=search_space_digest,
                )

    def test_convert_experiment_data_with_conflicting_names(self) -> None:
        """Test that _convert_experiment_data handles parameter name
        and metric name conflicts."""
        feature_names = ["m1", "x0", "x1"]  # m1 is both a feature and metric
        search_space = get_search_space_for_range_values(
            min=0.0, max=5.0, parameter_names=feature_names
        )
        raw_X = torch.rand(5, 3) * 5

        raw_m1_Y = torch.sin(raw_X).sum(-1, keepdim=True)
        raw_Y = torch.cat([raw_m1_Y, raw_m1_Y + 1.0], dim=1)

        experiment = get_experiment_with_observations(
            parameterizations=[
                {f"{feature_names[i]}": x_[i].item() for i in range(3)} for x_ in raw_X
            ],
            observations=raw_Y.tolist(),
            search_space=search_space,
        )
        adapter = TorchAdapter(experiment=experiment, generator=TorchGenerator())

        metric_names = ["m1", "m2"]
        experiment_data = adapter.get_training_data()

        search_space_digest = SearchSpaceDigest(
            feature_names=feature_names,
            bounds=[(0.0, 5.0)] * 3,
        )

        # This should work without errors despite the name conflict
        converted_datasets, ordered_outcomes, _ = adapter._convert_experiment_data(
            experiment_data=experiment_data,
            outcomes=metric_names,
            parameters=feature_names,
            search_space_digest=search_space_digest,
        )

        # Verify the datasets were created correctly
        self.assertEqual(len(converted_datasets), 2)
        self.assertEqual(len(ordered_outcomes), 2)
        self.assertIn("m1", ordered_outcomes)
        self.assertIn("m2", ordered_outcomes)

        # Check that all datasets have the correct feature names and shapes
        for dataset in converted_datasets:
            self.assertEqual(dataset.feature_names, feature_names)
            self.assertEqual(dataset.X.shape[1], 3)
            self.assertEqual(dataset.Y.shape[1], 1)
            # Verify we have data for all 5 observations
            self.assertEqual(dataset.X.shape[0], 5)
            self.assertEqual(dataset.Y.shape[0], 5)

    def test_convert_contextual_observations(self) -> None:
        raw_X = torch.rand(10, 3) * 5
        raw_X[:, -1].round_()  # Make sure last column is integer.
        raw_X[0, -1] = 0  # Make sure task value 0 exists.
        raw_Y = torch.sin(raw_X).sum(-1, keepdim=True).expand(-1, 4)
        feature_names = ["x0", "x1", "x2"]
        metric_signatures = ["y", "y:c0", "y:c1", "y:c2"]
        parameter_decomposition = {f"c{i}": [f"x{i}"] for i in range(3)}
        metric_decomposition = {f"c{i}": [f"y:c{i}"] for i in range(3)}

        search_space = get_search_space_for_range_values(
            min=0.0, max=5.0, parameter_names=feature_names
        )
        # Make an optimization config that includes all metrics.
        opt_config = OptimizationConfig(
            objective=Objective(metric=Metric("y"), minimize=True),
            outcome_constraints=[
                OutcomeConstraint(
                    metric=Metric(f"y:c{i}"), op=ComparisonOp.GEQ, bound=0
                )
                for i in range(3)
            ],
        )
        experiment = get_experiment_with_observations(
            parameterizations=[
                {f"x{i}": x_[i].item() for i in range(3)} for x_ in raw_X
            ],
            observations=raw_Y.tolist(),
            search_space=search_space,
            optimization_config=opt_config,
        )
        experiment._properties = {
            "parameter_decomposition": parameter_decomposition,
            "metric_decomposition": metric_decomposition,
        }
        adapter = TorchAdapter(experiment=experiment, generator=BoTorchGenerator())
        experiment_data = adapter.get_training_data()
        converted_datasets, ordered_outcomes, _ = adapter._convert_experiment_data(
            experiment_data=experiment_data,
            outcomes=metric_signatures,
            parameters=feature_names,
            search_space_digest=SearchSpaceDigest(
                feature_names=feature_names,
                bounds=[(0.0, 5.0)] * 3,
                ordinal_features=[2],
                discrete_choices={2: list(range(0, 11))},
            ),
        )
        self.assertEqual(len(converted_datasets), 2)
        expected_outcomes = list(converted_datasets[0].outcome_names)
        expected_outcomes.extend(list(converted_datasets[1].outcome_names))
        self.assertEqual(ordered_outcomes, expected_outcomes)
        for dataset in converted_datasets:
            self.assertIsInstance(dataset, ContextualDataset)
            self.assertEqual(dataset.feature_names, feature_names)
            self.assertDictEqual(
                assert_is_instance(dataset, ContextualDataset).parameter_decomposition,
                parameter_decomposition,
            )
            if len(dataset.outcome_names) == 1:
                self.assertListEqual(dataset.outcome_names, ["y"])
                self.assertTrue(torch.equal(dataset.X, raw_X))
                self.assertTrue(torch.equal(dataset.Y, raw_Y[:, :1]))
            else:
                self.assertListEqual(dataset.outcome_names, ["y:c0", "y:c1", "y:c2"])
                self.assertListEqual(
                    assert_is_instance(dataset, ContextualDataset).context_buckets,
                    ["c0", "c1", "c2"],
                )
                self.assertDictEqual(
                    none_throws(
                        assert_is_instance(
                            dataset, ContextualDataset
                        ).metric_decomposition
                    ),
                    metric_decomposition,
                )
                self.assertTrue(torch.equal(dataset.X, raw_X))
                self.assertTrue(torch.equal(dataset.Y, raw_Y[:, 1:]))
        # Test _get_fit_args handling of outcome names
        adapter._fit_tracking_metrics = True
        converted_datasets2, _, _ = adapter._get_fit_args(
            search_space=search_space,
            experiment_data=experiment_data,
            update_outcomes_and_parameters=True,
        )
        self.assertEqual(adapter.outcomes, expected_outcomes)
        self.assertEqual(converted_datasets, converted_datasets2)
        # Check that outcomes are not updated when
        # `update_outcomes_and_parameters` is False
        adapter._get_fit_args(
            search_space=search_space,
            experiment_data=experiment_data,
            update_outcomes_and_parameters=False,
        )
        self.assertEqual(adapter.outcomes, expected_outcomes)

    @mock_botorch_optimize
    def test_gen_metadata_untransform(self) -> None:
        experiment = get_experiment_with_observations(
            observations=[[0.0, 1.0], [2.0, 3.0]]
        )
        generator = BoTorchGenerator()
        adapter = TorchAdapter(experiment=experiment, generator=generator)
        for additional_metadata in (
            {},
            {"objective_thresholds": None},
            {"objective_thresholds": torch.tensor([0.0, 0.0])},
        ):
            gen_return_value = TorchGenResults(
                points=torch.tensor([[1.0, 2.0]]),
                weights=torch.tensor([1.0]),
                gen_metadata={Keys.EXPECTED_ACQF_VAL: [1.0], **additional_metadata},
            )
            with mock.patch.object(
                adapter,
                "_untransform_objective_thresholds",
                wraps=adapter._untransform_objective_thresholds,
            ) as mock_untransform, mock.patch.object(
                generator,
                "gen",
                return_value=gen_return_value,
            ):
                adapter.gen(n=1)
            if additional_metadata.get("objective_thresholds", None) is None:
                mock_untransform.assert_not_called()
            else:
                mock_untransform.assert_called_once()

    @mock_botorch_optimize
    def test_gen_with_expanded_parameter_space(self) -> None:
        # Test that an expanded search space with range and unordered choice
        # parameters can still generate (when using the default transforms).
        search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x1",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=1.0,
                ),
                ChoiceParameter(
                    name="x2",
                    parameter_type=ParameterType.FLOAT,
                    values=[0.0, 1.0, 2.0],
                    is_ordered=False,
                ),
            ]
        )
        experiment = get_experiment_with_observations(
            observations=[[0.0, 1.0], [2.0, 3.0]], search_space=search_space
        )
        # Attach a trial from outside of the search space.
        trial = experiment.new_trial(
            generator_run=GeneratorRun(
                arms=[Arm(parameters={"x1": 1.5, "x2": 0.5}, name="manual")]
            )
        )
        data = Data(
            df=DataFrame.from_records(
                [
                    {
                        "arm_name": "manual",
                        "metric_name": metric,
                        "mean": o,
                        "sem": None,
                        "trial_index": trial.index,
                        "metric_signature": metric,
                    }
                    for metric, o in (("m1", 0.2), ("m2", 0.5))
                ]
            )
        )
        experiment.attach_data(data)
        trial.run().complete()
        adapter = TorchAdapter(
            experiment=experiment, generator=BoTorchGenerator(), transforms=MBM_X_trans
        )
        # Check the expanded model space. Range is expanded, Choice is not.
        model_space = adapter._model_space
        self.assertEqual(
            model_space.parameters["x1"],
            RangeParameter(
                name="x1", lower=0.0, upper=1.5, parameter_type=ParameterType.FLOAT
            ),
        )
        self.assertEqual(model_space.parameters["x2"], search_space.parameters["x2"])
        self.assertNotEqual(adapter._model_space, adapter._search_space)
        # Generate candidates.
        gr = adapter.gen(n=3)
        self.assertEqual(sum(gr.weights), 3)

    @mock_botorch_optimize
    def test_predict_with_posterior_predictive(self) -> None:
        # Checks that noise is added when using posterior predictive.
        exp = get_experiment_with_observations([[1.0], [1.5], [2.0]])
        adapter = TorchAdapter(
            experiment=exp,
            generator=BoTorchGenerator(),
        )
        obs_ft = ObservationFeatures(parameters={"x": 0.0, "y": 0.0})
        mean_default, cov_default = adapter.predict(observation_features=[obs_ft])
        mean_predictive, cov_predictive = adapter.predict(
            observation_features=[obs_ft], use_posterior_predictive=True
        )
        # Check that means are close.
        self.assertAlmostEqual(mean_default["m1"][0], mean_predictive["m1"][0])
        # Check that variance is larger.
        self.assertGreater(cov_predictive["m1"]["m1"], cov_default["m1"]["m1"])

    @mock_botorch_optimize
    def test_fitting_auxiliary_experiment_dataset(self) -> None:
        pref_metrics = ["metric2", "metric3"]
        metric_names = ["metric1", "metric2", "metric3"]

        pe_exp_with_data = get_pbo_experiment(
            num_parameters=len(pref_metrics),
            num_experimental_metrics=0,
            parameter_names=pref_metrics,
            num_experimental_trials=0,
            num_preference_trials=3,
            num_preference_trials_w_repeated_arm=5,
            unbounded_search_space=True,
            experiment_name="pe_exp",
        )

        pref_opt_config = PreferenceOptimizationConfig(
            objective=MultiObjective(
                objectives=[
                    Objective(metric=Metric(name=pref_m), minimize=False)
                    for pref_m in pref_metrics
                ]
            ),
            preference_profile_name=pe_exp_with_data.name,
        )

        exp = get_pbo_experiment(
            num_parameters=4,
            num_experimental_metrics=3,
            tracking_metric_names=metric_names,
            num_experimental_trials=4,
            num_preference_trials=0,
            num_preference_trials_w_repeated_arm=0,
            experiment_name="bo_exp",
            optimization_config=pref_opt_config,
        )
        exp.add_auxiliary_experiment(
            purpose=AuxiliaryExperimentPurpose.PE_EXPERIMENT,
            auxiliary_experiment=AuxiliaryExperiment(experiment=pe_exp_with_data),
        )

        surrogate_specs = [
            # Default, minimum surrogate spec
            SurrogateSpec(),
            # Correctly specified surrogate spec with model selection
            SurrogateSpec(
                model_configs=[
                    ModelConfig(
                        botorch_model_class=SingleTaskGP,
                        outcome_transform_classes=[Standardize],
                        name="STGP",
                    ),
                    ModelConfig(
                        botorch_model_class=AdditiveMapSaasSingleTaskGP,
                        outcome_transform_classes=[Standardize],
                        name="SAAS",
                    ),
                ],
                metric_to_model_configs={
                    Keys.PAIRWISE_PREFERENCE_QUERY.value: [
                        ModelConfig(
                            botorch_model_class=PairwiseGP,
                            mll_class=PairwiseLaplaceMarginalLogLikelihood,
                            input_transform_classes=[Normalize],
                        )
                    ]
                },
                eval_criterion=MSE,
            ),
            # We should handle default preference model fallback when unspecified
            SurrogateSpec(
                model_configs=[
                    ModelConfig(
                        botorch_model_class=SingleTaskGP,
                        outcome_transform_classes=[Standardize],
                        name="STGP",
                    ),
                    ModelConfig(
                        botorch_model_class=AdditiveMapSaasSingleTaskGP,
                        outcome_transform_classes=[Standardize],
                        name="SAAS",
                    ),
                ],
                eval_criterion=MSE,
            ),
        ]

        for surrogate_spec in surrogate_specs:
            adapter = TorchAdapter(
                experiment=exp,
                data=exp.lookup_data(),
                generator=BoTorchGenerator(
                    surrogate_spec=surrogate_spec,
                ),
            )

            generator = assert_is_instance(adapter.generator, BoTorchGenerator)
            # With PE data, we should use a model list by default
            self.assertIsInstance(generator.surrogate.model, ModelListGP)
            # 3 outcomes + 1 aux experiments = 4 datasets
            self.assertEqual(
                len(assert_is_instance(generator.surrogate.model.models, Sized)), 3
            )
            # using PairwiseGP for the preference dataset
            self.assertIsInstance(
                generator.surrogate._submodels[(Keys.PAIRWISE_PREFERENCE_QUERY.value,)],
                PairwiseGP,
            )

            # Checking CV and gen works correctly
            cross_validate(adapter)
            adapter.gen(n=2)

    def test_fitting_auxiliary_experiment_empty_dataset(self) -> None:
        pref_metrics = ["metric2", "metric3"]
        metric_names = ["metric1", "metric2", "metric3"]

        empty_pe_exp = get_pbo_experiment(
            num_parameters=len(pref_metrics),
            num_experimental_metrics=0,
            parameter_names=pref_metrics,
            num_experimental_trials=0,
            num_preference_trials=0,
            num_preference_trials_w_repeated_arm=0,
            unbounded_search_space=True,
            experiment_name="pe_exp",
        )

        pref_opt_config = PreferenceOptimizationConfig(
            objective=MultiObjective(
                objectives=[
                    Objective(metric=Metric(name=pref_m), minimize=False)
                    for pref_m in pref_metrics
                ]
            ),
            preference_profile_name=empty_pe_exp.name,
        )

        exp = get_pbo_experiment(
            num_parameters=4,
            num_experimental_metrics=3,
            tracking_metric_names=metric_names,
            num_experimental_trials=4,
            num_preference_trials=0,
            num_preference_trials_w_repeated_arm=0,
            experiment_name="bo_exp",
            optimization_config=pref_opt_config,
        )
        exp.add_auxiliary_experiment(
            purpose=AuxiliaryExperimentPurpose.PE_EXPERIMENT,
            auxiliary_experiment=AuxiliaryExperiment(experiment=empty_pe_exp),
        )

        surrogate_specs = [
            # Default, minimum surrogate spec
            SurrogateSpec(),
            # Correctly specified surrogate spec with model selection
            SurrogateSpec(
                model_configs=[
                    ModelConfig(
                        botorch_model_class=SingleTaskGP,
                        outcome_transform_classes=[Standardize],
                        name="STGP",
                    ),
                    ModelConfig(
                        botorch_model_class=AdditiveMapSaasSingleTaskGP,
                        outcome_transform_classes=[Standardize],
                        name="SAAS",
                    ),
                ],
                metric_to_model_configs={
                    Keys.PAIRWISE_PREFERENCE_QUERY.value: [
                        ModelConfig(
                            botorch_model_class=PairwiseGP,
                            mll_class=PairwiseLaplaceMarginalLogLikelihood,
                            input_transform_classes=[Normalize],
                        )
                    ]
                },
                eval_criterion=MSE,
            ),
            # We should handle default preference model fallback when unspecified
            SurrogateSpec(
                model_configs=[
                    ModelConfig(
                        botorch_model_class=SingleTaskGP,
                        outcome_transform_classes=[Standardize],
                        name="STGP",
                    ),
                    ModelConfig(
                        botorch_model_class=AdditiveMapSaasSingleTaskGP,
                        outcome_transform_classes=[Standardize],
                        name="SAAS",
                    ),
                ],
                eval_criterion=MSE,
            ),
        ]

        for surrogate_spec in surrogate_specs:
            with self.assertRaisesRegex(
                DataRequiredError,
                "No data found in the auxiliary preference exploration experiment.",
            ):
                TorchAdapter(
                    experiment=exp,
                    data=exp.lookup_data(),
                    generator=BoTorchGenerator(
                        surrogate_spec=surrogate_spec,
                    ),
                )

    @mock_botorch_optimize
    def test_pairwise_preference_generator(self) -> None:
        experiment = get_pbo_experiment()
        surrogate = Surrogate(
            surrogate_spec=SurrogateSpec(
                model_configs=[
                    ModelConfig(
                        botorch_model_class=PairwiseGP,
                        mll_class=PairwiseLaplaceMarginalLogLikelihood,
                        input_transform_classes=[Normalize],
                        input_transform_options={
                            "Normalize": {"d": len(experiment.parameters)}
                        },
                    )
                ]
            )
        )

        cases = [
            (qLogNoisyExpectedImprovement, None, 3),
            (qExpectedUtilityOfBestOption, None, 3),
            (
                AnalyticExpectedUtilityOfBestOption,
                # Analytic Acqfs do not support pending points and sequential opt
                {"optimizer_kwargs": {"sequential": False}},
                2,  # analytic EUBO only supports n=2
            ),
        ]
        for botorch_acqf_class, model_gen_options, n in cases:
            pmb = TorchAdapter(
                experiment=experiment,
                generator=BoTorchGenerator(
                    botorch_acqf_class=botorch_acqf_class,
                    surrogate=surrogate,
                ),
                optimization_config=OptimizationConfig(
                    Objective(
                        Metric(Keys.PAIRWISE_PREFERENCE_QUERY.value), minimize=False
                    )
                ),
                fit_tracking_metrics=False,
            )
            # Can generate candidates correctly
            # pyre-ignore: Incompatible parameter type [6]
            generator_run = pmb.gen(n=n, model_gen_options=model_gen_options)
            self.assertEqual(len(generator_run.arms), n)

        parameter_names = list(experiment.parameters.keys())
        outcomes = [assert_is_instance(Keys.PAIRWISE_PREFERENCE_QUERY.value, str)]

        datasets, _, candidate_metadata = pmb._convert_experiment_data(
            experiment_data=pmb._training_data,
            outcomes=outcomes,
            parameters=parameter_names,
            search_space_digest=None,
        )
        self.assertTrue(len(datasets) == 1)
        self.assertIsInstance(datasets[0], RankingDataset)
        self.assertIsNotNone(candidate_metadata)

        # Test individual helper methods
        X = torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [1.0, 2.0, 3.0], [2.1, 3.1, 4.1]]
        )
        Y = torch.tensor([[1, 0, 0, 1]])
        expected_X = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [2.1, 3.1, 4.1]])
        ordered_Y = torch.tensor([[0, 1], [3, 2]])
        expected_Y = torch.tensor([[0, 1], [2, 0]])

        # `_binary_pref_to_comp_pair`.
        comp_pair_Y = _binary_pref_to_comp_pair(Y=Y)
        self.assertTrue(torch.equal(comp_pair_Y, ordered_Y))

        # test `_binary_pref_to_comp_pair` with invalid data
        bad_Y = torch.tensor([[1, 1, 0, 0]])
        with self.assertRaises(ValueError):
            _binary_pref_to_comp_pair(Y=bad_Y)

        # `_consolidate_comparisons`.
        consolidated_X, consolidated_Y = _consolidate_comparisons(X=X, Y=comp_pair_Y)
        self.assertTrue(torch.equal(consolidated_X, expected_X))
        self.assertTrue(torch.equal(consolidated_Y, expected_Y))

        with self.assertRaises(ValueError):
            _consolidate_comparisons(
                X=X.expand(2, *X.shape), Y=comp_pair_Y.expand(2, *comp_pair_Y.shape)
            )

    def test_get_transformed_model_gen_args_with_target_point(self) -> None:
        # Test that _get_transformed_model_gen_args correctly processes target_point

        # Setup: create adapter with target arm in optimization config
        experiment = get_branin_experiment(with_completed_trial=True)
        pruning_target_parameterization = Arm(parameters={"x1": -5.0, "x2": 15.0})
        optimization_config = none_throws(
            experiment.optimization_config
        ).clone_with_args(
            pruning_target_parameterization=pruning_target_parameterization
        )

        adapter = TorchAdapter(
            generator=TorchGenerator(),
            experiment=experiment,
            transforms=Cont_X_trans,
        )

        # Execute: call _get_transformed_gen_args then _get_transformed_model_gen_args
        base_gen_args = adapter._get_transformed_gen_args(
            search_space=experiment.search_space,
            optimization_config=optimization_config,
            pending_observations={},
        )

        search_space_digest, torch_opt_config = adapter._get_transformed_model_gen_args(
            search_space=base_gen_args.search_space,
            pending_observations=base_gen_args.pending_observations,
            fixed_features=base_gen_args.fixed_features,
            optimization_config=base_gen_args.optimization_config,
        )

        # Assert: confirm pruning_target_point is correctly extracted and transformed
        self.assertIsNotNone(torch_opt_config.pruning_target_point)
        expected_target = torch.tensor([0.0, 1.0], dtype=torch.double)
        torch.testing.assert_close(
            torch_opt_config.pruning_target_point, expected_target
        )

    def test_get_transformed_model_gen_args_no_target_point(self) -> None:
        # Test that _get_transformed_model_gen_args handles
        # pruning_target_parameterization=None correctly

        # Setup: create adapter without target arm (default case)
        experiment = get_branin_experiment(with_completed_trial=True)
        adapter = TorchAdapter(
            generator=TorchGenerator(),
            experiment=experiment,
            transforms=Cont_X_trans,
        )

        # Execute: call _get_transformed_gen_args then _get_transformed_model_gen_args
        base_gen_args = adapter._get_transformed_gen_args(
            search_space=experiment.search_space,
            optimization_config=none_throws(experiment.optimization_config),
            pending_observations={},
        )

        search_space_digest, torch_opt_config = adapter._get_transformed_model_gen_args(
            search_space=base_gen_args.search_space,
            pending_observations=base_gen_args.pending_observations,
            fixed_features=base_gen_args.fixed_features,
            optimization_config=base_gen_args.optimization_config,
        )

        # Assert: confirm target_point is None when no pruning_target_parameterization
        #  is provided
        self.assertIsNone(torch_opt_config.pruning_target_point)

    def test_get_transformed_model_gen_args_with_sq_as_target(self) -> None:
        # Test that _get_transformed_model_gen_args correctly processes the status quo
        # as the target point
        experiment = get_branin_experiment(
            with_completed_trial=True, with_status_quo=True
        )

        adapter = TorchAdapter(
            generator=TorchGenerator(), experiment=experiment, transforms=Cont_X_trans
        )
        oc = none_throws(experiment.optimization_config).clone()
        # Execute: call _get_transformed_gen_args then _get_transformed_model_gen_args
        base_gen_args = adapter._get_transformed_gen_args(
            search_space=experiment.search_space,
            optimization_config=oc,
            pending_observations={},
        )

        _, torch_opt_config = adapter._get_transformed_model_gen_args(
            search_space=base_gen_args.search_space,
            pending_observations=base_gen_args.pending_observations,
            fixed_features=base_gen_args.fixed_features,
            optimization_config=base_gen_args.optimization_config,
        )

        # Assert: confirm pruning_target_point is correctly extracted and transformed
        self.assertIsNotNone(torch_opt_config.pruning_target_point)
        expected_target = torch.tensor([1 / 3.0, 0.0], dtype=torch.double)
        torch.testing.assert_close(
            torch_opt_config.pruning_target_point, expected_target
        )
