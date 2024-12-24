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
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective, ScalarizedObjective
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import FixedParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.modelbridge.base import (
    clamp_observation_features,
    gen_arms,
    GenResults,
    ModelBridge,
    unwrap_observation_data,
)
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.registry import Models, Y_trans
from ax.modelbridge.transforms.fill_missing_parameters import FillMissingParameters
from ax.models.base import Model
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data_batch,
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
    get_branin_optimization_config,
    get_experiment,
    get_experiment_with_repeated_arms,
    get_non_monolithic_branin_moo_data,
    get_optimization_config_no_constraints,
    get_search_space_for_range_value,
    get_search_space_for_range_values,
    get_search_space_for_value,
)
from ax.utils.testing.modeling_stubs import (
    get_experiment_for_value,
    get_observation1,
    get_observation1trans,
    get_observation2,
    get_observation2trans,
    get_observation_status_quo0,
    get_observation_status_quo1,
    transform_1,
    transform_2,
)
from botorch.exceptions.warnings import InputDataWarning
from pyre_extensions import none_throws


class BaseModelBridgeTest(TestCase):
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
    @mock.patch("ax.modelbridge.base.ModelBridge._fit", autospec=True)
    def test_ModelBridge(
        self, mock_fit: Mock, mock_gen_arms: Mock, mock_observations_from_data: Mock
    ) -> None:
        # Test that on init transforms are stored and applied in the correct order
        transforms = [transform_1, transform_2]
        exp = get_experiment_for_value()
        ss = get_search_space_for_value()
        modelbridge = ModelBridge(
            search_space=ss,
            model=Model(),
            # pyre-fixme[6]: For 3rd param expected
            #  `Optional[List[Type[Transform]]]` but got `List[Type[Union[transform_1,
            #  transform_2]]]`.
            transforms=transforms,
            experiment=exp,
            # pyre-fixme[6]: For 5th param expected `Optional[Data]` but got `int`.
            data=0,
        )
        self.assertFalse(
            modelbridge._experiment_has_immutable_search_space_and_opt_config
        )
        self.assertEqual(
            list(modelbridge.transforms.keys()), ["Cast", "transform_1", "transform_2"]
        )
        fit_args = mock_fit.mock_calls[0][2]
        self.assertTrue(fit_args["search_space"] == get_search_space_for_value(8.0))
        self.assertTrue(fit_args["observations"] == [])
        self.assertTrue(mock_observations_from_data.called)

        # Test deprecation error on update.
        with self.assertRaisesRegex(DeprecationWarning, "ModelBridge.update"):
            modelbridge.update(Mock(), Mock())

        # Test prediction with arms.
        with self.assertRaisesRegex(
            UserInputError, "Input to predict must be a list of `ObservationFeatures`."
        ):
            # pyre-ignore[6]: Intentionally wrong argument type.
            modelbridge.predict([Arm(parameters={"x": 1.0})])

        # Test prediction on out of design features.
        modelbridge._predict = mock.MagicMock(
            "ax.modelbridge.base.ModelBridge._predict",
            autospec=True,
            side_effect=ValueError("Out of Design"),
        )
        # This point is in design, and thus failures in predict are legitimate.
        with mock.patch.object(
            ModelBridge, "model_space", return_value=get_search_space_for_range_values
        ):
            with self.assertRaises(ValueError):
                modelbridge.predict([get_observation2().features])

        # This point is out of design, and not in training data.
        with self.assertRaises(ValueError):
            modelbridge.predict([get_observation_status_quo0().features])

        # Now it's in the training data.
        with mock.patch.object(
            ModelBridge,
            "get_training_data",
            return_value=[get_observation_status_quo0()],
        ):
            # Return raw training value.
            self.assertEqual(
                modelbridge.predict([get_observation_status_quo0().features]),
                unwrap_observation_data([get_observation_status_quo0().data]),
            )

        # Test that transforms are applied correctly on predict
        modelbridge._predict = mock.MagicMock(
            "ax.modelbridge.base.ModelBridge._predict",
            autospec=True,
            return_value=[get_observation2trans().data],
        )
        modelbridge.predict([get_observation2().features])
        # Observation features sent to _predict are un-transformed afterwards
        # pyre-fixme[16]: Callable `_predict` has no attribute `assert_called_with`.
        modelbridge._predict.assert_called_with([get_observation2().features])

        # Check that _single_predict is equivalent here.
        modelbridge._single_predict([get_observation2().features])
        # Observation features sent to _predict are un-transformed afterwards
        modelbridge._predict.assert_called_with([get_observation2().features])

        # Test transforms applied on gen
        modelbridge._gen = mock.MagicMock(
            "ax.modelbridge.base.ModelBridge._gen",
            autospec=True,
            return_value=GenResults(
                observation_features=[get_observation1trans().features], weights=[2]
            ),
        )
        oc = get_optimization_config_no_constraints()
        modelbridge._set_kwargs_to_save(
            model_key="TestModel", model_kwargs={}, bridge_kwargs={}
        )
        # Test input error when generating 0 candidates.
        with self.assertRaisesRegex(UserInputError, "Attempted to generate"):
            modelbridge.gen(n=0)
        gr = modelbridge.gen(
            n=1,
            search_space=get_search_space_for_value(),
            optimization_config=oc,
            pending_observations={"a": [get_observation2().features]},
            fixed_features=ObservationFeatures({"x": 5}),
        )
        self.assertEqual(gr._model_key, "TestModel")
        # pyre-fixme[16]: Callable `_gen` has no attribute `assert_called_with`.
        modelbridge._gen.assert_called_with(
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
        modelbridge.gen(
            n=1, search_space=get_search_space_for_value(), optimization_config=None
        )
        modelbridge._gen.assert_called_with(
            n=1,
            search_space=SearchSpace([FixedParameter("x", ParameterType.FLOAT, 8.0)]),
            optimization_config=None,
            pending_observations={},
            fixed_features=None,
            model_gen_options=None,
        )

        # Gen with multi-objective optimization config.
        oc2 = OptimizationConfig(
            objective=ScalarizedObjective(
                metrics=[Metric(name="test_metric"), Metric(name="test_metric_2")]
            )
        )
        modelbridge.gen(
            n=1, search_space=get_search_space_for_value(), optimization_config=oc2
        )
        modelbridge._gen.assert_called_with(
            n=1,
            search_space=SearchSpace([FixedParameter("x", ParameterType.FLOAT, 8.0)]),
            optimization_config=oc2,
            pending_observations={},
            fixed_features=None,
            model_gen_options=None,
        )

        # Test transforms applied on cross_validate and the warning is suppressed.
        called = False

        def warn_and_return_mock_obs(
            *args: Any, **kwargs: Any
        ) -> list[ObservationData]:
            nonlocal called
            called = True
            warnings.warn(
                "Data (outcome observations) is not standardized",
                InputDataWarning,
                stacklevel=2,
            )
            return [get_observation1trans().data]

        modelbridge._cross_validate = mock.MagicMock(
            "ax.modelbridge.base.ModelBridge._cross_validate",
            autospec=True,
            side_effect=warn_and_return_mock_obs,
        )
        cv_training_data = [get_observation2()]
        cv_test_points = [get_observation1().features]

        # Test transforms applied on cv_training_data, cv_test_points
        (
            transformed_cv_training_data,
            transformed_cv_test_points,
            transformed_ss,
        ) = modelbridge._transform_inputs_for_cv(
            cv_training_data=cv_training_data, cv_test_points=cv_test_points
        )
        self.assertEqual(transformed_cv_training_data, [get_observation2trans()])
        self.assertEqual(transformed_cv_test_points, [get_observation1trans().features])
        self.assertEqual(
            transformed_ss, SearchSpace([FixedParameter("x", ParameterType.FLOAT, 8.0)])
        )

        with warnings.catch_warnings(record=True) as ws:
            cv_predictions = modelbridge.cross_validate(
                cv_training_data=cv_training_data, cv_test_points=cv_test_points
            )
        self.assertTrue(called)
        self.assertFalse(any(w.category is InputDataWarning for w in ws))

        # pyre-fixme[16]: Callable `_cross_validate` has no attribute
        #  `assert_called_with`.
        modelbridge._cross_validate.assert_called_with(
            search_space=SearchSpace([FixedParameter("x", ParameterType.FLOAT, 8.0)]),
            cv_training_data=[get_observation2trans()],
            cv_test_points=[get_observation1().features],  # untransformed after
            use_posterior_predictive=False,
        )
        self.assertTrue(cv_predictions == [get_observation1().data])

        # Test use_posterior_predictive in CV
        modelbridge.cross_validate(
            cv_training_data=cv_training_data,
            cv_test_points=cv_test_points,
            use_posterior_predictive=True,
        )

        modelbridge._cross_validate.assert_called_with(
            search_space=SearchSpace([FixedParameter("x", ParameterType.FLOAT, 8.0)]),
            cv_training_data=[get_observation2trans()],
            cv_test_points=[get_observation1().features],  # untransformed after
            use_posterior_predictive=True,
        )

        # Test stored training data
        obs = modelbridge.get_training_data()
        self.assertTrue(obs == [get_observation1(), get_observation2()])
        self.assertEqual(modelbridge.metric_names, {"a", "b"})
        self.assertIsNone(modelbridge.status_quo)
        self.assertTrue(modelbridge.model_space == get_search_space_for_value())
        self.assertEqual(modelbridge.training_in_design, [False, False])

        with self.assertRaises(ValueError):
            modelbridge.training_in_design = [True, True, False]

        with self.assertRaises(ValueError):
            modelbridge.training_in_design = [True, True, False]

        # Test feature_importances
        with self.assertRaises(NotImplementedError):
            modelbridge.feature_importances("a")

        # Test transform observation features
        with mock.patch(
            "ax.modelbridge.base.ModelBridge._transform_observation_features",
            autospec=True,
        ) as mock_tr:
            modelbridge.transform_observation_features([get_observation2().features])
        mock_tr.assert_called_with(modelbridge, [get_observation2trans().features])

        # Test that fit is not called when fit_on_init = False.
        mock_fit.reset_mock()
        modelbridge = ModelBridge(
            search_space=ss,
            model=Model(),
            fit_on_init=False,
        )
        self.assertEqual(mock_fit.call_count, 0)

        # Test error when fit_tracking_metrics is False and optimization
        # config is not specified.
        with self.assertRaisesRegex(UserInputError, "fit_tracking_metrics"):
            ModelBridge(
                search_space=ss,
                model=Model(),
                fit_tracking_metrics=False,
            )

        # Test error when fit_tracking_metrics is False and optimization
        # config is updated to include new metrics.
        modelbridge = ModelBridge(
            search_space=ss,
            model=Model(),
            optimization_config=oc,
            fit_tracking_metrics=False,
        )
        new_oc = OptimizationConfig(
            objective=Objective(metric=Metric(name="test_metric2"), minimize=False),
        )
        with self.assertRaisesRegex(UnsupportedError, "fit_tracking_metrics"):
            modelbridge.gen(n=1, optimization_config=new_oc)

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
    @mock.patch("ax.modelbridge.base.ModelBridge._fit", autospec=True)
    def test_repeat_candidates(
        self, mock_fit: Mock, mock_gen_arms: Mock, mock_observations_from_data: Mock
    ) -> None:
        modelbridge = ModelBridge(
            search_space=get_search_space_for_value(),
            model=Model(),
            experiment=get_experiment_for_value(),
        )
        # mock _gen to return 1 result
        modelbridge._gen = mock.MagicMock(
            "ax.modelbridge.base.ModelBridge._gen",
            autospec=True,
            return_value=GenResults(
                observation_features=[get_observation1trans().features], weights=[2]
            ),
        )
        modelbridge._set_kwargs_to_save(
            model_key="TestModel", model_kwargs={}, bridge_kwargs={}
        )
        with self.assertLogs("ax", level="INFO") as cm:
            modelbridge.gen(
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
            modelbridge.gen(
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
    @mock.patch("ax.modelbridge.base.ModelBridge._fit", autospec=True)
    def test_with_status_quo(self, mock_fit: Mock, mock_gen_arms: Mock) -> None:
        # Test init with a status quo.
        exp = get_branin_experiment(
            with_trial=True,
            with_status_quo=True,
            with_completed_trial=True,
        )
        modelbridge = ModelBridge(
            search_space=exp.search_space,
            model=Model(),
            transforms=Y_trans,
            experiment=exp,
            data=exp.lookup_data(),
        )
        self.assertIsNotNone(modelbridge.status_quo)
        self.assertEqual(
            modelbridge.status_quo.features.parameters, {"x1": 0.0, "x2": 0.0}
        )

    @mock.patch("ax.modelbridge.base.ModelBridge._fit", autospec=True)
    @mock.patch("ax.modelbridge.base.ModelBridge._gen", autospec=True)
    def test_timing(self, mock_fit: Mock, mock_gen: Mock) -> None:
        search_space = get_search_space_for_value()
        modelbridge = ModelBridge(
            search_space=search_space, model=Model(), fit_on_init=False
        )
        self.assertEqual(modelbridge.fit_time, 0.0)
        modelbridge._fit_if_implemented(
            search_space=search_space, observations=[], time_so_far=3.0
        )
        modelbridge._fit_if_implemented(
            search_space=search_space, observations=[], time_so_far=2.0
        )
        modelbridge._fit_if_implemented(
            search_space=search_space, observations=[], time_so_far=1.0
        )
        self.assertAlmostEqual(modelbridge.fit_time, 6.0, places=1)
        self.assertAlmostEqual(modelbridge.fit_time_since_gen, 6.0, places=1)
        modelbridge.gen(1)
        self.assertAlmostEqual(modelbridge.fit_time, 6.0, places=1)
        self.assertAlmostEqual(modelbridge.fit_time_since_gen, 0.0, places=1)

    @mock.patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=([get_observation1(), get_observation2()]),
    )
    def test_ood_gen(self, _) -> None:
        # Test fit_out_of_design by returning OOD candidats
        exp = get_experiment_for_value()
        ss = SearchSpace([RangeParameter("x", ParameterType.FLOAT, 0.0, 1.0)])
        modelbridge = ModelBridge(
            search_space=ss,
            model=Model(),
            transforms=[],
            experiment=exp,
            # pyre-fixme[6]: For 5th param expected `Optional[Data]` but got `int`.
            data=0,
            fit_out_of_design=True,
        )
        obs = ObservationFeatures(parameters={"x": 3.0})
        modelbridge._gen = mock.MagicMock(
            "ax.modelbridge.base.ModelBridge._gen",
            autospec=True,
            return_value=GenResults(observation_features=[obs], weights=[2]),
        )
        gr = modelbridge.gen(n=1)
        self.assertEqual(gr.arms[0].parameters, obs.parameters)

        # Test clamping arms by setting fit_out_of_design=False
        modelbridge = ModelBridge(
            search_space=ss,
            model=Model(),
            transforms=[],
            experiment=exp,
            # pyre-fixme[6]: For 5th param expected `Optional[Data]` but got `int`.
            data=0,
            fit_out_of_design=False,
        )
        obs = ObservationFeatures(parameters={"x": 3.0})
        modelbridge._gen = mock.MagicMock(
            "ax.modelbridge.base.ModelBridge._gen",
            autospec=True,
            return_value=GenResults(observation_features=[obs], weights=[2]),
        )
        gr = modelbridge.gen(n=1)
        self.assertEqual(gr.arms[0].parameters, {"x": 1.0})

    @mock.patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=([get_observation1()]),
    )
    @mock.patch("ax.modelbridge.base.ModelBridge._fit", autospec=True)
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def test_SetStatusQuo(self, mock_fit, mock_observations_from_data):
        # NOTE: If empty data object is not passed, observations are not
        # extracted, even with mock.
        modelbridge = ModelBridge(
            search_space=get_search_space_for_value(),
            model=0,
            experiment=get_experiment_for_value(),
            data=Data(),
            status_quo_name="1_1",
        )
        self.assertEqual(modelbridge.status_quo, get_observation1())
        self.assertEqual(modelbridge.status_quo_name, "1_1")

        # Alternatively, we can specify by features
        modelbridge = ModelBridge(
            get_search_space_for_value(),
            0,
            [],
            get_experiment_for_value(),
            # pyre-fixme[6]: For 5th param expected `Optional[Data]` but got `int`.
            0,
            status_quo_features=get_observation1().features,
        )
        self.assertEqual(modelbridge.status_quo, get_observation1())
        self.assertEqual(modelbridge.status_quo_name, "1_1")

        # Alternatively, we can specify on experiment
        # Put a dummy arm with SQ name 1_1 on the dummy experiment.
        exp = get_experiment_for_value()
        sq = Arm(name="1_1", parameters={"x": 3.0})
        exp._status_quo = sq
        # Check that we set SQ to arm 1_1
        # pyre-fixme[6]: For 5th param expected `Optional[Data]` but got `int`.
        modelbridge = ModelBridge(get_search_space_for_value(), 0, [], exp, 0)
        self.assertEqual(modelbridge.status_quo, get_observation1())
        self.assertEqual(modelbridge.status_quo_name, "1_1")

        # Errors if features and name both specified
        with self.assertRaises(ValueError):
            modelbridge = ModelBridge(
                get_search_space_for_value(),
                0,
                [],
                exp,
                # pyre-fixme[6]: For 5th param expected `Optional[Data]` but got `int`.
                0,
                status_quo_features=get_observation1().features,
                status_quo_name="1_1",
            )

        # Left as None if features or name don't exist
        modelbridge = ModelBridge(
            get_search_space_for_value(),
            0,
            [],
            exp,
            # pyre-fixme[6]: For 5th param expected `Optional[Data]` but got `int`.
            0,
            status_quo_name="1_0",
        )
        self.assertIsNone(modelbridge.status_quo)
        self.assertIsNone(modelbridge.status_quo_name)
        modelbridge = ModelBridge(
            get_search_space_for_value(),
            0,
            [],
            get_experiment_for_value(),
            # pyre-fixme[6]: For 5th param expected `Optional[Data]` but got `int`.
            0,
            status_quo_features=ObservationFeatures(parameters={"x": 3.0, "y": 10.0}),
        )
        self.assertIsNone(modelbridge.status_quo)

    @mock.patch(
        "ax.modelbridge.base.ModelBridge._gen",
        autospec=True,
    )
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def test_status_quo_for_non_monolithic_data(self, mock_gen):
        mock_gen.return_value = GenResults(
            observation_features=[
                ObservationFeatures(
                    parameters={"x1": float(i), "x2": float(i)}, trial_index=1
                )
                for i in range(5)
            ],
            weights=[1] * 5,
        )
        exp = get_branin_experiment_with_multi_objective(with_status_quo=True)
        sobol = Models.SOBOL(search_space=exp.search_space)
        exp.new_batch_trial(sobol.gen(5)).set_status_quo_and_optimize_power(
            status_quo=exp.status_quo
        ).run()

        # create data where metrics vary in start and end times
        data = get_non_monolithic_branin_moo_data()
        with warnings.catch_warnings(record=True) as ws:
            bridge = ModelBridge(
                experiment=exp,
                data=data,
                model=Model(),
                search_space=exp.search_space,
            )
        # just testing it doesn't error
        bridge.gen(5)
        self.assertTrue(any("start_time" in str(w.message) for w in ws))
        self.assertTrue(any("end_time" in str(w.message) for w in ws))
        # pyre-fixme[16]: Optional type has no attribute `arm_name`.
        self.assertEqual(bridge.status_quo.arm_name, "status_quo")

    @mock.patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=(
            [
                get_observation_status_quo0(),
                get_observation_status_quo1(),
                get_observation1(),
                get_observation2(),
            ]
        ),
    )
    @mock.patch("ax.modelbridge.base.ModelBridge._fit", autospec=True)
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def test_SetStatusQuoMultipleObs(self, mock_fit, mock_observations_from_data):
        exp = get_experiment_with_repeated_arms(2)

        trial_index = 1
        status_quo_features = ObservationFeatures(
            # pyre-fixme[16]: `BaseTrial` has no attribute `status_quo`.
            parameters=exp.trials[trial_index].status_quo.parameters,
            trial_index=trial_index,
        )
        modelbridge = ModelBridge(
            get_search_space_for_value(),
            0,
            [],
            exp,
            # pyre-fixme[6]: For 5th param expected `Optional[Data]` but got `int`.
            0,
            status_quo_features=status_quo_features,
        )
        # Check that for experiments with many trials the status quo is set
        # to the value of the status quo of the last trial.
        if len(exp.trials) >= 1:
            self.assertEqual(modelbridge.status_quo, get_observation_status_quo1())

    def test_transform_observations(self) -> None:
        """
        This functionality is unused, even in the subclass where it is implemented.
        """
        ss = get_search_space_for_value()
        modelbridge = ModelBridge(search_space=ss, model=Model())
        with self.assertRaises(NotImplementedError):
            modelbridge.transform_observations([])
        with self.assertRaises(NotImplementedError):
            modelbridge.transform_observations([])

    @mock.patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=([get_observation1(), get_observation1()]),
    )
    @mock.patch("ax.modelbridge.base.ModelBridge._fit", autospec=True)
    def test_SetTrainingDataDupFeatures(
        self, mock_fit: Mock, mock_observations_from_data: Mock
    ) -> None:
        # Throws an error if repeated features in observations.
        with self.assertRaises(ValueError):
            ModelBridge(
                get_search_space_for_value(),
                0,
                [],
                get_experiment_for_value(),
                # pyre-fixme[6]: For 5th param expected `Optional[Data]` but got `int`.
                0,
                status_quo_name="1_1",
            )

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
        "ax.modelbridge.base.ModelBridge._gen",
        autospec=True,
        return_value=GenResults(
            observation_features=[get_observation1trans().features], weights=[2]
        ),
    )
    @mock.patch(
        "ax.modelbridge.base.ModelBridge.predict", autospec=True, return_value=None
    )
    def test_GenWithDefaults(self, _, mock_gen: Mock) -> None:
        exp = get_experiment_for_value()
        exp.optimization_config = get_optimization_config_no_constraints()
        ss = get_search_space_for_range_value()
        modelbridge = ModelBridge(
            search_space=ss, model=Model(), transforms=[], experiment=exp
        )
        modelbridge.gen(1)
        mock_gen.assert_called_with(
            modelbridge,
            n=1,
            search_space=ss,
            fixed_features=None,
            model_gen_options=None,
            optimization_config=OptimizationConfig(
                objective=Objective(metric=Metric("test_metric"), minimize=False),
                outcome_constraints=[],
            ),
            pending_observations={},
        )

    @mock.patch(
        "ax.modelbridge.base.ModelBridge._gen",
        autospec=True,
        return_value=GenResults(
            observation_features=[get_observation1trans().features], weights=[2]
        ),
    )
    @mock.patch(
        "ax.modelbridge.base.ModelBridge.predict", autospec=True, return_value=None
    )
    # pyre-fixme[3]: Return type must be annotated.
    def test_gen_on_experiment_with_imm_ss_and_opt_conf(self, _, __):
        exp = get_experiment_for_value()
        exp._properties[Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF] = True
        exp.optimization_config = get_optimization_config_no_constraints()
        ss = get_search_space_for_range_value()
        modelbridge = ModelBridge(
            search_space=ss, model=Model(), transforms=[], experiment=exp
        )
        self.assertTrue(
            modelbridge._experiment_has_immutable_search_space_and_opt_config
        )
        gr = modelbridge.gen(1)
        self.assertIsNone(gr.optimization_config)
        self.assertIsNone(gr.search_space)

    def test_set_status_quo(self) -> None:
        # experiment with single status quo in trial
        exp = get_branin_experiment(
            with_batch=True,
            with_status_quo=True,
            num_batch_trial=1,
            with_completed_batch=True,
        )
        modelbridge = ModelBridge(
            search_space=exp.search_space,
            experiment=exp,
            model=Model,
            data=exp.lookup_data(),
        )

        # we are able to set status_quo_data_by_trial when multiple
        # status_quos present in each trial
        self.assertIsNotNone(modelbridge.status_quo_data_by_trial)
        # status_quo is set
        self.assertIsNotNone(modelbridge.status_quo)
        # Status quo name is logged
        self.assertEqual(modelbridge._status_quo_name, none_throws(exp.status_quo).name)

        # experiment with multiple status quos in different trials
        exp = get_branin_experiment(
            with_batch=True,
            with_status_quo=True,
            num_batch_trial=2,
            with_completed_batch=True,
        )
        modelbridge = ModelBridge(
            search_space=exp.search_space,
            experiment=exp,
            model=Model,
            data=exp.lookup_data(),
        )
        # we are able to set status_quo_data_by_trial when multiple
        # status_quos present in each trial
        self.assertIsNotNone(modelbridge.status_quo_data_by_trial)
        # status_quo is not set
        self.assertIsNone(modelbridge.status_quo)
        # Status quo name can still be logged
        self.assertEqual(modelbridge._status_quo_name, none_throws(exp.status_quo).name)

        # a unique status_quo can be identified (by trial index)
        # if status_quo_features is specified
        status_quo_features = ObservationFeatures(
            parameters=none_throws(exp.status_quo).parameters,
            trial_index=0,
        )
        modelbridge = ModelBridge(
            search_space=exp.search_space,
            experiment=exp,
            model=Model,
            data=exp.lookup_data(),
            status_quo_features=status_quo_features,
        )
        self.assertIsNotNone(modelbridge.status_quo)


class testClampObservationFeatures(TestCase):
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

    @mock.patch("ax.modelbridge.base.ModelBridge._fit", autospec=True)
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
            trial = experiment.new_batch_trial(optimize_for_power=True)
            trial.add_generator_run(gr)
            trial.mark_running(no_runner_required=True)
            trial.mark_completed()
            experiment.attach_data(
                get_branin_data_batch(batch=trial, fill_vals=sq_vals)
            )
        # Fit model without filling missing parameters
        m = ModelBridge(
            search_space=ss1,
            model=None,
            experiment=experiment,
            data=experiment.lookup_data(),
        )
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
        m = ModelBridge(
            search_space=ss2,
            model=None,
            experiment=experiment,
            data=experiment.lookup_data(),
            transforms=[FillMissingParameters],
            transform_configs={"FillMissingParameters": {"fill_values": sq_vals}},
        )
        self.assertEqual(
            [t.__name__ for t in m._raw_transforms], ["Cast", "FillMissingParameters"]
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
        data = experiment.lookup_data()

        # Check that SQ and custom are OOD
        m = ModelBridge(
            search_space=experiment.search_space,
            model=None,
            experiment=experiment,
            data=data,
            expand_model_space=False,
        )
        arm_names = [obs.arm_name for obs in m.get_training_data()]
        ood_arms = [a for i, a in enumerate(arm_names) if not m.training_in_design[i]]
        self.assertEqual(set(ood_arms), {"status_quo", "custom"})
        self.assertEqual(m.model_space.parameters["x1"].lower, -5.0)  # pyre-ignore[16]
        self.assertEqual(m.model_space.parameters["x2"].upper, 15.0)  # pyre-ignore[16]

        # With expand model space, custom is not OOD, and model space is expanded
        m = ModelBridge(
            search_space=experiment.search_space,
            model=None,
            experiment=experiment,
            data=data,
        )
        arm_names = [obs.arm_name for obs in m.get_training_data()]
        ood_arms = [a for i, a in enumerate(arm_names) if not m.training_in_design[i]]
        self.assertEqual(set(ood_arms), {"status_quo"})
        self.assertEqual(m.model_space.parameters["x1"].lower, -20.0)
        self.assertEqual(m.model_space.parameters["x2"].upper, 18.0)

        # With fill values, SQ is also in design, and x2 is further expanded
        m = ModelBridge(
            search_space=experiment.search_space,
            model=None,
            experiment=experiment,
            data=data,
            transforms=[FillMissingParameters],
            transform_configs={"FillMissingParameters": {"fill_values": sq_vals}},
        )
        self.assertEqual(sum(m.training_in_design), 7)
        self.assertEqual(m.model_space.parameters["x2"].upper, 20)
