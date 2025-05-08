#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from typing import Any
from unittest import mock
from unittest.mock import Mock

import numpy as np
import torch
from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.metric import Metric
from ax.core.objective import Objective, ScalarizedObjective
from ax.core.observation import (
    Observation,
    ObservationData,
    ObservationFeatures,
    observations_from_data,
)
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.core.parameter import FixedParameter, ParameterType, RangeParameter
from ax.core.parameter_constraint import SumConstraint
from ax.core.search_space import SearchSpace
from ax.core.utils import get_target_trial_index
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.exceptions.model import ModelError
from ax.metrics.branin import BraninMetric
from ax.modelbridge.base import (
    _combine_multiple_status_quo_observations,
    Adapter,
    clamp_observation_features,
    DataLoaderConfig,
    gen_arms,
    GenResults,
    logger,
    unwrap_observation_data,
)
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.registry import Y_trans
from ax.modelbridge.transforms.fill_missing_parameters import FillMissingParameters
from ax.modelbridge.transforms.standardize_y import StandardizeY
from ax.models.base import Generator
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data_batch,
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
    get_branin_experiment_with_timestamp_map_metric,
    get_branin_optimization_config,
    get_experiment,
    get_experiment_with_observations,
    get_experiment_with_repeated_arms,
    get_map_metric,
    get_non_monolithic_branin_moo_data,
    get_optimization_config_no_constraints,
    get_search_space_for_range_value,
    get_search_space_for_value,
)
from ax.utils.testing.modeling_stubs import (
    get_experiment_for_value,
    get_observation1,
    get_observation1trans,
    get_observation2,
    get_observation2trans,
    transform_1,
    transform_2,
)
from botorch.exceptions.warnings import InputDataWarning
from botorch.models.utils.assorted import validate_input_scaling
from pyre_extensions import assert_is_instance, none_throws


class BaseAdapterTest(TestCase):
    @mock.patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=([get_observation1(), get_observation2()]),
    )
    @mock.patch(
        "ax.modelbridge.base.gen_arms",
        autospec=True,
        return_value=([Arm(parameters={})], None),
    )
    @mock.patch("ax.modelbridge.base.Adapter._fit", autospec=True)
    def test_Adapter(
        self, mock_fit: Mock, mock_gen_arms: Mock, mock_observations_from_data: Mock
    ) -> None:
        # Test that on init transforms are stored and applied in the correct order
        transforms = [transform_1, transform_2]
        exp = get_experiment_for_value()
        adapter = Adapter(experiment=exp, model=Generator(), transforms=transforms)
        self.assertFalse(adapter._experiment_has_immutable_search_space_and_opt_config)
        self.assertEqual(
            list(adapter.transforms.keys()), ["Cast", "transform_1", "transform_2"]
        )
        fit_args = mock_fit.mock_calls[0][2]
        self.assertTrue(fit_args["search_space"] == get_search_space_for_value(8.0))
        self.assertTrue(fit_args["observations"] == [])
        self.assertTrue(mock_observations_from_data.called)

        # Test prediction with arms.
        with self.assertRaisesRegex(
            UserInputError, "Input to predict must be a list of `ObservationFeatures`."
        ):
            # pyre-ignore[6]: Intentionally wrong argument type.
            adapter.predict([Arm(parameters={"x": 1.0})])

        # Test errors on prediction.
        adapter._predict = mock.MagicMock(
            "ax.modelbridge.base.Adapter._predict",
            autospec=True,
            side_effect=ValueError("Predict failed"),
        )
        obs_features = [get_observation2().features]
        with self.assertRaisesRegex(ValueError, "Predict failed"):
            adapter.predict(obs_features)

        # Test that transforms are applied correctly on predict
        mock_predict = mock.MagicMock(
            "ax.modelbridge.base.Adapter._predict",
            autospec=True,
            return_value=[get_observation2trans().data],
        )
        adapter._predict = mock_predict
        adapter.predict(obs_features)
        # Observation features sent to _predict are un-transformed afterwards
        mock_predict.assert_called_with(
            observation_features=obs_features, use_posterior_predictive=False
        )

        # Test transforms applied on gen
        adapter._gen = mock.MagicMock(
            "ax.modelbridge.base.Adapter._gen",
            autospec=True,
            return_value=GenResults(
                observation_features=[get_observation1trans().features], weights=[2]
            ),
        )
        oc = get_optimization_config_no_constraints()
        adapter._set_kwargs_to_save(
            model_key="TestModel", model_kwargs={}, bridge_kwargs={}
        )
        # Test input error when generating 0 candidates.
        with self.assertRaisesRegex(UserInputError, "Attempted to generate"):
            adapter.gen(n=0)
        gr = adapter.gen(
            n=1,
            search_space=get_search_space_for_value(),
            optimization_config=oc,
            pending_observations={"a": [get_observation2().features]},
            fixed_features=ObservationFeatures({"x": 5}),
        )
        self.assertEqual(gr._model_key, "TestModel")
        # pyre-fixme[16]: Callable `_gen` has no attribute `assert_called_with`.
        adapter._gen.assert_called_with(
            n=1,
            search_space=SearchSpace([FixedParameter("x", ParameterType.FLOAT, 8.0)]),
            optimization_config=oc,
            pending_observations={"a": [get_observation2trans().features]},
            fixed_features=ObservationFeatures({"x": 36}),
            model_gen_options=None,
        )
        mock_gen_arms.assert_called_with(
            arms_by_signature={}, observation_features=[get_observation1().features]
        )

        # Gen with no pending observations and no fixed features
        adapter.gen(
            n=1, search_space=get_search_space_for_value(), optimization_config=None
        )
        adapter._gen.assert_called_with(
            n=1,
            search_space=SearchSpace([FixedParameter("x", ParameterType.FLOAT, 8.0)]),
            optimization_config=None,
            pending_observations={},
            fixed_features=ObservationFeatures(parameters={}),
            model_gen_options=None,
        )

        # Gen with multi-objective optimization config.
        oc2 = OptimizationConfig(
            objective=ScalarizedObjective(
                metrics=[Metric(name="test_metric"), Metric(name="test_metric_2")]
            )
        )
        adapter.gen(
            n=1, search_space=get_search_space_for_value(), optimization_config=oc2
        )
        adapter._gen.assert_called_with(
            n=1,
            search_space=SearchSpace([FixedParameter("x", ParameterType.FLOAT, 8.0)]),
            optimization_config=oc2,
            pending_observations={},
            fixed_features=ObservationFeatures(parameters={}),
            model_gen_options=None,
        )

        # Test transforms applied on cross_validate and the warning is suppressed.
        called = False

        def warn_and_return_mock_obs(
            *args: Any, **kwargs: Any
        ) -> list[ObservationData]:
            nonlocal called
            called = True
            validate_input_scaling(
                # Dummy non-scaled values to trigger warnings if checks are enabled.
                train_X=torch.randn(2, 5),
                train_Y=torch.rand(2, 1),
            )
            return [get_observation1trans().data]

        mock_cv = mock.MagicMock(
            "ax.modelbridge.base.Adapter._cross_validate",
            autospec=True,
            side_effect=warn_and_return_mock_obs,
        )
        adapter._cross_validate = mock_cv
        cv_training_data = [get_observation2()]
        cv_test_points = [get_observation1().features]

        # Test transforms applied on cv_training_data, cv_test_points
        (
            transformed_cv_training_data,
            transformed_cv_test_points,
            transformed_ss,
        ) = adapter._transform_inputs_for_cv(
            cv_training_data=cv_training_data, cv_test_points=cv_test_points
        )
        self.assertEqual(transformed_cv_training_data, [get_observation2trans()])
        self.assertEqual(transformed_cv_test_points, [get_observation1trans().features])
        self.assertEqual(
            transformed_ss, SearchSpace([FixedParameter("x", ParameterType.FLOAT, 8.0)])
        )

        with warnings.catch_warnings(record=True) as ws:
            cv_predictions = adapter.cross_validate(
                cv_training_data=cv_training_data, cv_test_points=cv_test_points
            )
        self.assertTrue(called)
        self.assertFalse(any(w.category is InputDataWarning for w in ws))

        mock_cv.assert_called_with(
            search_space=SearchSpace([FixedParameter("x", ParameterType.FLOAT, 8.0)]),
            cv_training_data=[get_observation2trans()],
            cv_test_points=[get_observation1().features],  # untransformed after
            use_posterior_predictive=False,
        )
        self.assertTrue(cv_predictions == [get_observation1().data])

        # Test use_posterior_predictive in CV
        adapter.cross_validate(
            cv_training_data=cv_training_data,
            cv_test_points=cv_test_points,
            use_posterior_predictive=True,
        )

        mock_cv.assert_called_with(
            search_space=SearchSpace([FixedParameter("x", ParameterType.FLOAT, 8.0)]),
            cv_training_data=[get_observation2trans()],
            cv_test_points=[get_observation1().features],  # untransformed after
            use_posterior_predictive=True,
        )

        # Test stored training data
        obs = adapter.get_training_data()
        self.assertTrue(obs == [get_observation1(), get_observation2()])
        self.assertEqual(adapter.metric_names, {"a", "b"})
        self.assertIsNone(adapter.status_quo)
        self.assertTrue(adapter.model_space == get_search_space_for_value())
        self.assertEqual(adapter.training_in_design, [False, False])

        with self.assertRaises(ValueError):
            adapter.training_in_design = [True, True, False]

        with self.assertRaises(ValueError):
            adapter.training_in_design = [True, True, False]

        # Test feature_importances
        with self.assertRaises(NotImplementedError):
            adapter.feature_importances("a")

        # Test transform observation features
        with mock.patch(
            "ax.modelbridge.base.Adapter._transform_observation_features",
            autospec=True,
        ) as mock_tr:
            adapter.transform_observation_features([get_observation2().features])
        mock_tr.assert_called_with(adapter, [get_observation2trans().features])

        # Test that fit is not called when fit_on_init = False.
        mock_fit.reset_mock()
        adapter = Adapter(experiment=exp, model=Generator(), fit_on_init=False)
        self.assertEqual(mock_fit.call_count, 0)

        # Test error when fit_tracking_metrics is False and optimization
        # config is not specified.
        with self.assertRaisesRegex(UserInputError, "fit_tracking_metrics"):
            Adapter(experiment=exp, model=Generator(), fit_tracking_metrics=False)

        # Test error when fit_tracking_metrics is False and optimization
        # config is updated to include new metrics.
        adapter = Adapter(
            experiment=exp,
            model=Generator(),
            optimization_config=oc,
            fit_tracking_metrics=False,
        )
        new_oc = OptimizationConfig(
            objective=Objective(metric=Metric(name="test_metric2"), minimize=False),
        )
        with self.assertRaisesRegex(UnsupportedError, "fit_tracking_metrics"):
            adapter.gen(n=1, optimization_config=new_oc)

    @mock.patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=([get_observation1(), get_observation2()]),
    )
    @mock.patch(
        "ax.modelbridge.base.gen_arms",
        autospec=True,
        return_value=([Arm(parameters={})], None),
    )
    @mock.patch("ax.modelbridge.base.Adapter._fit", autospec=True)
    def test_repeat_candidates(self, _: Mock, __: Mock, ___: Mock) -> None:
        adapter = Adapter(
            experiment=get_experiment_for_value(),
            model=Generator(),
        )
        # mock _gen to return 1 result
        adapter._gen = mock.MagicMock(
            "ax.modelbridge.base.Adapter._gen",
            autospec=True,
            return_value=GenResults(
                observation_features=[get_observation1trans().features], weights=[2]
            ),
        )
        adapter._set_kwargs_to_save(
            model_key="TestModel", model_kwargs={}, bridge_kwargs={}
        )
        with self.assertLogs("ax", level="INFO") as cm:
            adapter.gen(
                n=2,
            )
            self.assertTrue(
                any(
                    "was not able to generate 2 unique candidates" in x
                    for x in cm.output
                ),
                cm.output,
            )

        with self.assertLogs("ax", level="INFO") as cm:
            adapter.gen(
                n=1,
            )
            get_logger("ax").info("log to prevent error if there are no other logs")
            self.assertFalse(
                any(
                    "was not able to generate 2 unique candidates" in x
                    for x in cm.output
                ),
                cm.output,
            )

    @mock.patch(
        "ax.modelbridge.base.gen_arms",
        autospec=True,
        return_value=([Arm(parameters={"x1": 0.0, "x2": 0.0})], None),
    )
    def test_with_status_quo(self, _: Mock) -> None:
        # Test init with a status quo.
        exp = get_branin_experiment(
            with_trial=True,
            with_status_quo=True,
            with_completed_trial=True,
        )
        adapter = Adapter(
            experiment=exp,
            model=Generator(),
            transforms=Y_trans,
        )
        self.assertIsNotNone(adapter.status_quo)
        self.assertEqual(adapter.status_quo.features.parameters, {"x1": 0.0, "x2": 0.0})

    @mock.patch("ax.modelbridge.base.Adapter._fit", autospec=True)
    @mock.patch("ax.modelbridge.base.Adapter._gen", autospec=True)
    def test_timing(self, _: Mock, __: Mock) -> None:
        search_space = get_search_space_for_value()
        experiment = Experiment(search_space=search_space)
        adapter = Adapter(experiment=experiment, model=Generator(), fit_on_init=False)
        self.assertEqual(adapter.fit_time, 0.0)
        adapter._fit_if_implemented(
            search_space=search_space, observations=[], time_so_far=3.0
        )
        adapter._fit_if_implemented(
            search_space=search_space, observations=[], time_so_far=2.0
        )
        adapter._fit_if_implemented(
            search_space=search_space, observations=[], time_so_far=1.0
        )
        self.assertAlmostEqual(adapter.fit_time, 6.0, places=1)
        self.assertAlmostEqual(adapter.fit_time_since_gen, 6.0, places=1)
        adapter.gen(1)
        self.assertAlmostEqual(adapter.fit_time, 6.0, places=1)
        self.assertAlmostEqual(adapter.fit_time_since_gen, 0.0, places=1)

    @mock.patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=([get_observation1(), get_observation2()]),
    )
    def test_ood_gen(self, _) -> None:
        # Test fit_out_of_design by returning OOD candidates
        ss = SearchSpace([RangeParameter("x", ParameterType.FLOAT, 0.0, 1.0)])
        experiment = Experiment(search_space=ss)
        adapter = Adapter(
            experiment=experiment,
            model=Generator(),
            fit_out_of_design=True,
        )
        obs = ObservationFeatures(parameters={"x": 3.0})
        adapter._gen = mock.MagicMock(
            "ax.modelbridge.base.Adapter._gen",
            autospec=True,
            return_value=GenResults(observation_features=[obs], weights=[2]),
        )
        gr = adapter.gen(n=1)
        self.assertEqual(gr.arms[0].parameters, obs.parameters)

        # Test clamping arms by setting fit_out_of_design=False
        adapter = Adapter(
            experiment=experiment,
            model=Generator(),
            fit_out_of_design=False,
        )
        obs = ObservationFeatures(parameters={"x": 3.0})
        adapter._gen = mock.MagicMock(
            "ax.modelbridge.base.Adapter._gen",
            autospec=True,
            return_value=GenResults(observation_features=[obs], weights=[2]),
        )
        gr = adapter.gen(n=1)
        self.assertEqual(gr.arms[0].parameters, {"x": 1.0})

    def test_set_status_quo(self) -> None:
        exp = get_experiment_for_value()
        exp._status_quo = Arm(parameters={"x": 3.0}, name="0_0")
        with self.assertLogs(logger="ax", level="WARNING") as logs:
            adapter = Adapter(experiment=exp, model=Generator())
        self.assertTrue(
            any("is not present in the training data" in log for log in logs.output)
        )
        # Status quo name is set but status quo itself is not, since there is no data.
        self.assertEqual(adapter.status_quo_name, "0_0")
        self.assertIsNone(adapter.status_quo)

        for num_batch_trial in (1, 2):
            # Experiment with status quo in num_batch_trial trials. Only one completed.
            exp = get_branin_experiment(
                with_batch=True,
                with_status_quo=True,
                num_batch_trial=num_batch_trial,
                with_completed_batch=True,
            )
            adapter = Adapter(experiment=exp, model=Generator())
            # Status quo is set with the target trial index.
            self.assertEqual(
                none_throws(adapter.status_quo).features.trial_index,
                get_target_trial_index(experiment=exp),
            )
            # Status quo data by trial extracts the data from all trials.
            self.assertEqual(
                set(none_throws(adapter.status_quo_data_by_trial).keys()),
                set(range(num_batch_trial)),
            )
            # Status quo name is set.
            self.assertEqual(adapter._status_quo_name, none_throws(exp.status_quo).name)

    def test_status_quo_for_non_monolithic_data(self) -> None:
        exp = get_branin_experiment_with_multi_objective(with_status_quo=True)
        sobol_generator = get_sobol(
            search_space=exp.search_space,
        )
        sobol_run = sobol_generator.gen(n=5)
        exp.new_batch_trial(
            sobol_run, add_status_quo_arm=False
        ).set_status_quo_with_weight(status_quo=exp.status_quo, weight=1.0).run()

        # create data where metrics vary in start and end times
        data = get_non_monolithic_branin_moo_data()
        with warnings.catch_warnings(record=True) as ws, mock.patch.object(
            exp, "lookup_data", return_value=data
        ):
            adapter = Adapter(experiment=exp, model=Generator(), data=data)
        # Check that we get warnings about start and end time columns being discarded.
        self.assertTrue(any("start_time" in str(w.message) for w in ws))
        self.assertTrue(any("end_time" in str(w.message) for w in ws))
        # Check that SQ is set.
        self.assertEqual(adapter.status_quo_name, "status_quo")
        self.assertIsNotNone(adapter.status_quo)

    def test_set_status_quo_with_multiple_trials_with_status_quo(self) -> None:
        exp = get_experiment_with_repeated_arms(with_data=True)
        exp.status_quo = assert_is_instance(exp.trials[1], BatchTrial).status_quo
        adapter = Adapter(experiment=exp, model=Generator())
        # Check that for experiments with many trials the status quo is set
        # to the value of the status quo of the last trial (target trial).
        self.assertEqual(
            adapter.status_quo,
            Observation(
                features=ObservationFeatures(
                    parameters={"w": 0.85, "x": 1, "y": "baz", "z": False},
                    trial_index=get_target_trial_index(experiment=exp),
                    metadata={},
                ),
                data=ObservationData(
                    means=np.array([2.0, 4.0]),
                    covariance=np.array([[1.0, 0.0], [0.0, 16.0]]),
                    metric_names=["a", "b"],
                ),
                arm_name="0_0",
            ),
        )

    def test_set_status_quo_with_multiple_observations(self) -> None:
        # Test for the case where the status quo arm has multiple observations
        # for the target trial. This happens with MapData.

        # Prevent data from being filtered down.
        data_loader_config = DataLoaderConfig(
            latest_rows_per_group=None, fit_only_completed_map_metrics=False
        )

        # Case 1: Experiment has an optimization config with single map key.
        exp = get_branin_experiment_with_timestamp_map_metric(with_status_quo=True)
        # Add a second map metric, and a non-map metric.
        exp.optimization_config = none_throws(exp.optimization_config).clone_with_args(
            outcome_constraints=[
                OutcomeConstraint(
                    metric=get_map_metric("branin_map_constraint"),
                    op=ComparisonOp.LEQ,
                    bound=5.0,
                ),
                OutcomeConstraint(
                    metric=BraninMetric(
                        name="branin_constraint",
                        param_names=["x1", "x2"],
                        lower_is_better=True,
                    ),
                    op=ComparisonOp.LEQ,
                    bound=5.0,
                ),
            ]
        )
        # Attach a trial with status quo & fetch some data.
        exp.new_trial().add_arm(exp.status_quo).run()
        for _ in range(3):
            exp.fetch_data()
        for additional_fetch in (False, True):
            if additional_fetch:
                # Fetch constraint metric an additional time. This will lead to two
                # separate observations for the status quo arm.
                exp.fetch_data(
                    metrics=[exp.metrics["branin_map_constraint"]],
                    combine_with_last_data=True,
                )
            with self.assertNoLogs(logger=logger, level="WARN"), mock.patch(
                "ax.modelbridge.base._combine_multiple_status_quo_observations",
                wraps=_combine_multiple_status_quo_observations,
            ) as mock_combine:
                adapter = Adapter(
                    experiment=exp,
                    model=Generator(),
                    data_loader_config=data_loader_config,
                )
            mock_combine.assert_called_once()
            call_kwargs = mock_combine.call_args.kwargs
            self.assertEqual(
                len(call_kwargs["status_quo_observations"]), 3 + additional_fetch
            )
            if additional_fetch:
                # Last observation should only include the constraint metric.
                self.assertEqual(
                    set(call_kwargs["status_quo_observations"][-1].data.metric_names),
                    {"branin_map_constraint"},
                )
            self.assertEqual(call_kwargs["map_key"], "timestamp")
            opt_config_metrics = set(none_throws(exp.optimization_config).metrics)
            self.assertEqual(call_kwargs["metrics"], opt_config_metrics)
            adapter_sq = none_throws(adapter.status_quo)
            self.assertEqual(
                adapter_sq.features.parameters, none_throws(exp.status_quo).parameters
            )
            self.assertEqual(
                adapter_sq.features.trial_index, get_target_trial_index(experiment=exp)
            )
            self.assertTrue(
                set(adapter_sq.data.metric_names).issuperset(opt_config_metrics)
            )

        # Case 2: Experiment has an optimization config with !=1 map keys.
        for num_map_keys in (0, 2):
            with mock.patch(
                "ax.modelbridge.base.extract_map_keys_from_opt_config",
                return_value=set(["timestamp", "step"][:num_map_keys]),
            ) as mock_extract, self.assertLogs(
                logger=logger, level="WARN"
            ) as mock_logs:
                adapter = Adapter(
                    experiment=exp,
                    model=Generator(),
                    data_loader_config=data_loader_config,
                )
            mock_extract.assert_called_once()
            self.assertIsNone(adapter.status_quo)
            self.assertTrue(
                any(
                    f"optimization config includes {num_map_keys} map keys" in log
                    for log in mock_logs.output
                )
            )

        # Case 3: Experiment doesn't have an optimization config.
        metrics = none_throws(exp.optimization_config).metrics.values()
        exp._optimization_config = None
        # Attach as tracking metric to prevent data filtering.
        for m in metrics:
            exp.add_tracking_metric(m)
        with self.assertLogs(logger=logger, level="WARN") as mock_logs:
            adapter = Adapter(
                experiment=exp, model=Generator(), data_loader_config=data_loader_config
            )
        self.assertIsNone(adapter.status_quo)
        self.assertTrue(
            any(
                "the Adapter does not have an optimization config" in log
                for log in mock_logs.output
            )
        )

    def test_transform_observations(self) -> None:
        """
        This functionality is unused, even in the subclass where it is implemented.
        """
        adapter = Adapter(experiment=get_experiment_for_value(), model=Generator())
        with self.assertRaises(NotImplementedError):
            adapter.transform_observations([])
        with self.assertRaises(NotImplementedError):
            adapter.transform_observations([])

    def test_UnwrapObservationData(self) -> None:
        observation_data = [get_observation1().data, get_observation2().data]
        f, cov = unwrap_observation_data(observation_data)
        self.assertEqual(f["a"], [2.0, 2.0])
        self.assertEqual(f["b"], [4.0, 1.0])
        self.assertEqual(cov["a"]["a"], [1.0, 2.0])
        self.assertEqual(cov["b"]["b"], [4.0, 5.0])
        self.assertEqual(cov["a"]["b"], [2.0, 3.0])
        self.assertEqual(cov["b"]["a"], [3.0, 4.0])
        # Check that errors if metric mismatch
        od3 = ObservationData(
            metric_names=["a"], means=np.array([2.0]), covariance=np.array([[4.0]])
        )
        with self.assertRaises(ValueError):
            unwrap_observation_data(observation_data + [od3])

    def test_GenArms(self) -> None:
        p1 = {"x": 0, "y": 1}
        p2 = {"x": 4, "y": 8}
        observation_features = [
            # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, bool,
            #  float, int, str]]` but got `Dict[str, int]`.
            ObservationFeatures(parameters=p1),
            # pyre-fixme[6]: For 1st param expected `Dict[str, Union[None, bool,
            #  float, int, str]]` but got `Dict[str, int]`.
            ObservationFeatures(parameters=p2),
        ]
        arms, candidate_metadata = gen_arms(observation_features=observation_features)
        self.assertEqual(arms[0].parameters, p1)
        self.assertIsNone(candidate_metadata)

        arm = Arm(name="1_1", parameters=p1)
        arms_by_signature = {arm.signature: arm}
        observation_features[0].metadata = {"some_key": "some_val_0"}
        observation_features[1].metadata = {"some_key": "some_val_1"}
        arms, candidate_metadata = gen_arms(
            observation_features=observation_features,
            arms_by_signature=arms_by_signature,
        )
        self.assertEqual(arms[0].name, "1_1")
        self.assertEqual(
            candidate_metadata,
            {
                arms[0].signature: {"some_key": "some_val_0"},
                arms[1].signature: {"some_key": "some_val_1"},
            },
        )

    @mock.patch(
        "ax.modelbridge.base.Adapter._gen",
        autospec=True,
        return_value=GenResults(
            observation_features=[get_observation1trans().features], weights=[2]
        ),
    )
    def test_GenWithDefaults(self, mock_gen: Mock) -> None:
        exp = get_experiment_for_value()
        exp.optimization_config = get_optimization_config_no_constraints()
        ss = get_search_space_for_range_value()
        adapter = Adapter(experiment=exp, model=Generator(), search_space=ss)
        adapter.gen(1)
        mock_gen.assert_called_with(
            adapter,
            n=1,
            search_space=ss,
            fixed_features=ObservationFeatures(parameters={}),
            model_gen_options=None,
            optimization_config=OptimizationConfig(
                objective=Objective(metric=Metric("test_metric"), minimize=False),
                outcome_constraints=[],
            ),
            pending_observations={},
        )

    @mock.patch(
        "ax.modelbridge.base.Adapter._gen",
        autospec=True,
        return_value=GenResults(
            observation_features=[get_observation1trans().features], weights=[2]
        ),
    )
    def test_gen_on_experiment_with_imm_ss_and_opt_conf(self, _) -> None:
        exp = get_experiment_for_value()
        exp._properties[Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF] = True
        exp.optimization_config = get_optimization_config_no_constraints()
        adapter = Adapter(experiment=exp, model=Generator())
        self.assertTrue(adapter._experiment_has_immutable_search_space_and_opt_config)
        gr = adapter.gen(1)
        self.assertIsNone(gr.optimization_config)
        self.assertIsNone(gr.search_space)

    def test_ClampObservationFeaturesNearBounds(self) -> None:
        cases = [
            (
                ObservationFeatures(
                    parameters={"w": 1.0, "x": 2, "y": "foo", "z": True}
                ),
                ObservationFeatures(
                    parameters={"w": 1.0, "x": 2, "y": "foo", "z": True}
                ),
            ),
            (
                ObservationFeatures(
                    parameters={"w": 0.0, "x": 2, "y": "foo", "z": True}
                ),
                ObservationFeatures(
                    parameters={"w": 0.5, "x": 2, "y": "foo", "z": True}
                ),
            ),
            (
                ObservationFeatures(
                    parameters={"w": 100.0, "x": 2, "y": "foo", "z": True}
                ),
                ObservationFeatures(
                    parameters={"w": 5.5, "x": 2, "y": "foo", "z": True}
                ),
            ),
            (
                ObservationFeatures(
                    parameters={"w": 1.0, "x": 0, "y": "foo", "z": True}
                ),
                ObservationFeatures(
                    parameters={"w": 1.0, "x": 1, "y": "foo", "z": True}
                ),
            ),
            (
                ObservationFeatures(
                    parameters={"w": 1.0, "x": 11, "y": "foo", "z": True}
                ),
                ObservationFeatures(
                    parameters={"w": 1.0, "x": 10, "y": "foo", "z": True}
                ),
            ),
        ]
        search_space = get_experiment().search_space
        for obs_ft, expected_obs_ft in cases:
            actual_obs_ft = clamp_observation_features([obs_ft], search_space)
            self.assertEqual(actual_obs_ft[0], expected_obs_ft)

    @mock.patch("ax.modelbridge.base.Adapter._fit", autospec=True)
    def test_FillMissingParameters(self, mock_fit: Mock) -> None:
        # Create experiment with arms from two search spaces
        ss1 = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
                )
            ],
        )
        ss2 = SearchSpace(
            parameters=[
                RangeParameter(
                    name="x1", parameter_type=ParameterType.FLOAT, lower=-5, upper=10
                ),
                RangeParameter(
                    name="x2", parameter_type=ParameterType.FLOAT, lower=0, upper=15
                ),
            ],
        )
        sq_arm = Arm(name="status_quo", parameters={"x1": None})
        experiment = Experiment(
            name="test",
            search_space=ss1,
            optimization_config=get_branin_optimization_config(),
            status_quo=sq_arm,
            is_test=True,
        )
        generator1 = get_sobol(search_space=ss1)
        gr1 = generator1.gen(n=5)
        generator2 = get_sobol(search_space=ss2)
        gr2 = generator2.gen(n=5)
        sq_vals = {"x1": 5.0, "x2": 5.0}
        for gr in [gr1, gr2]:
            trial = experiment.new_batch_trial(add_status_quo_arm=True)
            trial.add_generator_run(gr)
            trial.mark_running(no_runner_required=True)
            trial.mark_completed()
            experiment.attach_data(
                get_branin_data_batch(batch=trial, fill_vals=sq_vals)
            )
        # Fit model without filling missing parameters
        m = Adapter(experiment=experiment, model=Generator())
        self.assertEqual(
            [t.__name__ for t in m._raw_transforms],  # pyre-ignore[16]
            ["Cast"],
        )
        # Check that SQ and all trial 1 are OOD
        arm_names = [obs.arm_name for obs in m.get_training_data()]
        ood_arms = [a for i, a in enumerate(arm_names) if not m.training_in_design[i]]
        self.assertEqual(
            set(ood_arms), {"status_quo", "1_0", "1_1", "1_2", "1_3", "1_4"}
        )
        # Fit with filling missing parameters
        m = Adapter(
            experiment=experiment,
            model=Generator(),
            search_space=ss2,
            transforms=[],  # FillMissingParameters added by default.
            transform_configs={"FillMissingParameters": {"fill_values": sq_vals}},
        )
        self.assertEqual(
            [t.__name__ for t in m._raw_transforms], ["FillMissingParameters", "Cast"]
        )
        # All arms are in design now
        self.assertEqual(sum(m.training_in_design), 12)
        # Check the arms with missing values were correctly filled
        fit_args = mock_fit.mock_calls[1][2]
        for obs in fit_args["observations"]:
            if obs.arm_name == "status_quo":
                self.assertEqual(obs.features.parameters, sq_vals)
            elif obs.arm_name[0] == "0":
                # These arms were all missing x2
                self.assertEqual(obs.features.parameters["x2"], sq_vals["x2"])

    def test_SetModelSpace(self) -> None:
        # Set up experiment
        experiment = get_branin_experiment()
        # SQ values are OOD
        sq_vals = {"x1": 5.0, "x2": 20.0}
        # SQ is specified OOD
        experiment.status_quo = Arm(
            name="status_quo", parameters={"x1": None, "x2": None}
        )
        gr = get_sobol(search_space=experiment.search_space).gen(n=5)
        trial = experiment.new_batch_trial()
        trial.add_generator_run(gr)
        trial.add_arm(Arm(name="custom", parameters={"x1": -20, "x2": 18.0}))
        trial.add_arm(experiment.status_quo)
        trial.mark_running(no_runner_required=True)
        experiment.attach_data(get_branin_data_batch(batch=trial, fill_vals=sq_vals))
        trial.mark_completed()
        # Make search space with a parameter constraint
        ss = experiment.search_space.clone()
        ss.set_parameter_constraints(
            [
                SumConstraint(
                    parameters=list(ss.parameters.values()),
                    is_upper_bound=True,
                    bound=30.0,
                )
            ]
        )

        # Check that SQ and custom are OOD
        m = Adapter(
            experiment=experiment,
            model=Generator(),
            search_space=ss,
            expand_model_space=False,
        )
        arm_names = [obs.arm_name for obs in m.get_training_data()]
        ood_arms = [a for i, a in enumerate(arm_names) if not m.training_in_design[i]]
        self.assertEqual(set(ood_arms), {"status_quo", "custom"})
        self.assertEqual(m.model_space.parameters["x1"].lower, -5.0)  # pyre-ignore[16]
        self.assertEqual(m.model_space.parameters["x2"].upper, 15.0)  # pyre-ignore[16]
        self.assertEqual(len(m.model_space.parameter_constraints), 1)

        # With expand model space, custom is not OOD, and model space is expanded
        m = Adapter(
            experiment=experiment,
            model=Generator(),
            search_space=ss,
        )
        arm_names = [obs.arm_name for obs in m.get_training_data()]
        ood_arms = [a for i, a in enumerate(arm_names) if not m.training_in_design[i]]
        self.assertEqual(set(ood_arms), {"status_quo"})
        self.assertEqual(m.model_space.parameters["x1"].lower, -20.0)
        self.assertEqual(m.model_space.parameters["x2"].upper, 18.0)
        self.assertEqual(m.model_space.parameter_constraints, [])

        # With fill values, SQ is also in design, and x2 is further expanded
        m = Adapter(
            experiment=experiment,
            model=Generator(),
            search_space=ss,
            transforms=[FillMissingParameters],
            transform_configs={"FillMissingParameters": {"fill_values": sq_vals}},
        )
        self.assertEqual(sum(m.training_in_design), 7)
        self.assertEqual(m.model_space.parameters["x2"].upper, 20)
        self.assertEqual(m.model_space.parameter_constraints, [])

    @mock.patch(
        "ax.modelbridge.base.observations_from_data", wraps=observations_from_data
    )
    def test_fit_only_completed_map_metrics(
        self, mock_observations_from_data: Mock
    ) -> None:
        # _prepare_observations is called in the constructor and itself calls
        # observations_from_data with expanded statuses to include.
        experiment = get_experiment_for_value()
        experiment.status_quo = Arm(name="1_1", parameters={"x": 3.0})
        adapter = Adapter(
            experiment=experiment,
            model=Generator(),
            data=MapData(),
            data_loader_config=DataLoaderConfig(
                fit_only_completed_map_metrics=False,
            ),
        )
        kwargs = mock_observations_from_data.call_args.kwargs
        self.assertEqual(
            kwargs["statuses_to_include_map_metric"],
            adapter._data_loader_config.statuses_to_fit,
        )
        # assert `latest_rows_per_group` is 1
        self.assertEqual(kwargs["latest_rows_per_group"], 1)
        mock_observations_from_data.reset_mock()

        # With fit_only_completed_map_metrics=True, statuses to fit is limited.
        Adapter(
            experiment=experiment,
            model=Generator(),
            data_loader_config=DataLoaderConfig(
                fit_only_completed_map_metrics=True,
            ),
        )
        kwargs = mock_observations_from_data.call_args.kwargs
        self.assertEqual(
            kwargs["statuses_to_include_map_metric"], {TrialStatus.COMPLETED}
        )

    def test_data_extraction_from_experiment(self) -> None:
        # Checks that data is extracted from experiment both on __init__ and
        # in _process_and_transform_data, if it is not provided.
        exp = get_experiment_for_value()
        lookup_patch = mock.patch.object(
            exp, "lookup_data", return_value=exp.lookup_data()
        ).start()
        adapter = Adapter(experiment=exp, model=Generator())
        lookup_patch.assert_called_once()
        lookup_patch.reset_mock()
        adapter._process_and_transform_data(experiment=exp)
        lookup_patch.assert_called_once()
        lookup_patch.reset_mock()
        # Not called if data is provided.
        adapter = Adapter(experiment=exp, model=Generator(), data=MapData())
        adapter._process_and_transform_data(experiment=exp, data=MapData())
        lookup_patch.assert_not_called()

    def test_predict(self) -> None:
        # Construct an experiment with observations having std dev = 2.0
        np_obs = np.random.randn(5, 1)
        np_obs = np_obs / np.std(np_obs, ddof=1) * 2.0
        np_obs = np_obs - np.mean(np_obs)

        experiment = get_experiment_with_observations(observations=np_obs.tolist())
        adapter = Adapter(
            experiment=experiment,
            model=Generator(),
            transforms=[StandardizeY],
        )

        def mock_predict(
            observation_features: list[ObservationFeatures],
            use_posterior_predictive: bool = False,
        ) -> list[ObservationData]:
            return [
                ObservationData(
                    metric_names=["m1"], means=np.ones((1)), covariance=np.ones((1, 1))
                )
                for _ in observation_features
            ]

        obs_features = [
            ObservationFeatures(parameters={"x": 3.0, "y": 4.0}) for _ in range(3)
        ]
        with mock.patch.object(
            adapter, "_predict", side_effect=mock_predict
        ) as mock_pred:
            f, _ = adapter.predict(observation_features=obs_features)
        mock_pred.assert_called_once_with(
            observation_features=obs_features, use_posterior_predictive=False
        )
        # Check that the predictions were un-transformed with std dev = 2.0.
        self.assertTrue(np.allclose(f["m1"], np.ones(3) * 2.0))

        # Test for error if an observation is dropped.
        with mock.patch.object(
            adapter, "_predict", side_effect=mock_predict
        ), self.assertRaisesRegex(ModelError, "Predictions resulted in fewer"):
            adapter.predict(
                observation_features=[
                    ObservationFeatures(parameters={"x": 3.0, "y": 4.0}),
                    ObservationFeatures(parameters={"x": 3.0, "y": None}),
                ]
            )
