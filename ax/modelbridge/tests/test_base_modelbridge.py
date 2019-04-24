#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from unittest import mock

import numpy as np
import pandas as pd
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective, ScalarizedObjective
from ax.core.observation import (
    Observation,
    ObservationData,
    ObservationFeatures,
    observations_from_data,
)
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import FixedParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.base import ModelBridge, gen_arms, unwrap_observation_data
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.log import Log
from ax.utils.common.testutils import TestCase


def search_space_for_value(val: float = 3.0) -> SearchSpace:
    return SearchSpace([FixedParameter("x", ParameterType.FLOAT, val)])


def search_space_for_range_value(min: float = 3.0, max: float = 6.0) -> SearchSpace:
    return SearchSpace([RangeParameter("x", ParameterType.FLOAT, min, max)])


def search_space_for_range_values(min: float = 3.0, max: float = 6.0) -> SearchSpace:
    return SearchSpace(
        [
            RangeParameter("x", ParameterType.FLOAT, min, max),
            RangeParameter("y", ParameterType.FLOAT, min, max),
        ]
    )


def get_experiment() -> Experiment:
    return Experiment(search_space_for_value(), "test")


def get_optimization_config() -> OptimizationConfig:
    return OptimizationConfig(objective=Objective(metric=Metric("test_metric")))


def observation1() -> Observation:
    return Observation(
        features=ObservationFeatures(parameters={"x": 2.0, "y": 10.0}, trial_index=0),
        data=ObservationData(
            means=np.array([2.0, 4.0]),
            covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
            metric_names=["a", "b"],
        ),
        arm_name="1_1",
    )


def observation1trans() -> Observation:
    return Observation(
        features=ObservationFeatures(parameters={"x": 9.0, "y": 10.0}, trial_index=0),
        data=ObservationData(
            means=np.array([9.0, 25.0]),
            covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
            metric_names=["a", "b"],
        ),
        arm_name="1_1",
    )


def observation2() -> Observation:
    return Observation(
        features=ObservationFeatures(parameters={"x": 3.0, "y": 2.0}, trial_index=1),
        data=ObservationData(
            means=np.array([2.0, 1.0]),
            covariance=np.array([[2.0, 3.0], [4.0, 5.0]]),
            metric_names=["a", "b"],
        ),
        arm_name="1_1",
    )


def observation2trans() -> Observation:
    return Observation(
        features=ObservationFeatures(parameters={"x": 16.0, "y": 2.0}, trial_index=1),
        data=ObservationData(
            means=np.array([9.0, 4.0]),
            covariance=np.array([[2.0, 3.0], [4.0, 5.0]]),
            metric_names=["a", "b"],
        ),
        arm_name="1_1",
    )


# Prepare mock transforms
class t1(Transform):
    def transform_search_space(self, ss):
        new_ss = ss.clone()
        new_ss.parameters["x"]._value += 1.0
        return new_ss

    def transform_optimization_config(
        self, optimization_config, modelbridge, fixed_features
    ):
        return (
            optimization_config + 1
            if isinstance(optimization_config, int)
            else optimization_config
        )

    def transform_observation_features(self, x):
        for obsf in x:
            if "x" in obsf.parameters:
                obsf.parameters["x"] += 1
        return x

    def transform_observation_data(self, x, y):
        for obsd in x:
            obsd.means += 1
        return x

    def untransform_observation_features(self, x):
        for obsf in x:
            obsf.parameters["x"] -= 1
        return x

    def untransform_observation_data(self, x, y):
        for obsd in x:
            obsd.means -= 1
        return x


class t2(Transform):
    def transform_search_space(self, ss):
        new_ss = ss.clone()
        new_ss.parameters["x"]._value *= 2.0
        return new_ss

    def transform_optimization_config(
        self, optimization_config, modelbridge, fixed_features
    ):
        return (
            optimization_config ** 2
            if isinstance(optimization_config, int)
            else optimization_config
        )

    def transform_observation_features(self, x):
        for obsf in x:
            if "x" in obsf.parameters:
                obsf.parameters["x"] = obsf.parameters["x"] ** 2
        return x

    def transform_observation_data(self, x, y):
        for obsd in x:
            obsd.means = obsd.means ** 2
        return x

    def untransform_observation_features(self, x):
        for obsf in x:
            obsf.parameters["x"] = np.sqrt(obsf.parameters["x"])
        return x

    def untransform_observation_data(self, x, y):
        for obsd in x:
            obsd.means = np.sqrt(obsd.means)
        return x


class BaseModelBridgeTest(TestCase):
    @mock.patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=([observation1(), observation2()]),
    )
    @mock.patch(
        "ax.modelbridge.base.gen_arms", autospec=True, return_value=[Arm(parameters={})]
    )
    @mock.patch("ax.modelbridge.base.ModelBridge._fit", autospec=True)
    def testModelBridge(self, mock_fit, mock_gen_arms, mock_observations_from_data):
        # Test that on init transforms are stored and applied in the correct order
        transforms = [t1, t2]
        exp = get_experiment()
        modelbridge = ModelBridge(search_space_for_value(), 0, transforms, exp, 0)
        self.assertEqual(list(modelbridge.transforms.keys()), ["t1", "t2"])
        fit_args = mock_fit.mock_calls[0][2]
        self.assertTrue(fit_args["search_space"] == search_space_for_value(8.0))
        self.assertTrue(
            fit_args["observation_features"]
            == [observation1trans().features, observation2trans().features]
        )
        self.assertTrue(
            fit_args["observation_data"]
            == [observation1trans().data, observation2trans().data]
        )
        self.assertTrue(mock_observations_from_data.called)

        # Test that transforms are applied correctly on predict
        modelbridge._predict = mock.MagicMock(
            "ax.modelbridge.base.ModelBridge._predict",
            autospec=True,
            return_value=[observation2trans().data],
        )

        modelbridge.predict([observation2().features])
        # Observation features sent to _predict are un-transformed afterwards
        modelbridge._predict.assert_called_with([observation2().features])

        # Test transforms applied on gen
        modelbridge._gen = mock.MagicMock(
            "ax.modelbridge.base.ModelBridge._gen",
            autospec=True,
            return_value=([observation1trans().features], [2], None),
        )
        oc = OptimizationConfig(objective=Objective(metric=Metric(name="test_metric")))
        modelbridge.gen(
            n=1,
            search_space=search_space_for_value(),
            optimization_config=oc,
            pending_observations={"a": [observation2().features]},
            fixed_features=ObservationFeatures({"x": 5}),
        )
        modelbridge._gen.assert_called_with(
            n=1,
            search_space=SearchSpace([FixedParameter("x", ParameterType.FLOAT, 8.0)]),
            optimization_config=oc,
            pending_observations={"a": [observation2trans().features]},
            fixed_features=ObservationFeatures({"x": 36}),
            model_gen_options=None,
        )
        mock_gen_arms.assert_called_with(
            arms_by_signature={}, observation_features=[observation1().features]
        )

        # Gen with no pending observations and no fixed features
        modelbridge.gen(
            n=1, search_space=search_space_for_value(), optimization_config=None
        )
        modelbridge._gen.assert_called_with(
            n=1,
            search_space=SearchSpace([FixedParameter("x", ParameterType.FLOAT, 8.0)]),
            optimization_config=None,
            pending_observations={},
            fixed_features=ObservationFeatures({}),
            model_gen_options=None,
        )

        # Gen with multi-objective optimization config.
        oc2 = OptimizationConfig(
            objective=ScalarizedObjective(
                metrics=[Metric(name="test_metric"), Metric(name="test_metric_2")]
            )
        )
        modelbridge.gen(
            n=1, search_space=search_space_for_value(), optimization_config=oc2
        )
        modelbridge._gen.assert_called_with(
            n=1,
            search_space=SearchSpace([FixedParameter("x", ParameterType.FLOAT, 8.0)]),
            optimization_config=oc2,
            pending_observations={},
            fixed_features=ObservationFeatures({}),
            model_gen_options=None,
        )

        # Test transforms applied on cross_validate
        modelbridge._cross_validate = mock.MagicMock(
            "ax.modelbridge.base.ModelBridge._cross_validate",
            autospec=True,
            return_value=[observation1trans().data],
        )
        cv_training_data = [observation2()]
        cv_test_points = [observation1().features]
        cv_predictions = modelbridge.cross_validate(
            cv_training_data=cv_training_data, cv_test_points=cv_test_points
        )
        modelbridge._cross_validate.assert_called_with(
            obs_feats=[observation2trans().features],
            obs_data=[observation2trans().data],
            cv_test_points=[observation1().features],  # untransformed after
        )
        self.assertTrue(cv_predictions == [observation1().data])

        # Test stored training data
        obs = modelbridge.get_training_data()
        self.assertTrue(obs == [observation1(), observation2()])
        self.assertEqual(modelbridge.metric_names, {"a", "b"})
        self.assertIsNone(modelbridge.status_quo)
        self.assertTrue(modelbridge.model_space == search_space_for_value())
        self.assertEqual(modelbridge.training_in_design, [True, True])

        modelbridge.training_in_design = [True, False]
        with self.assertRaises(ValueError):
            modelbridge.training_in_design = [True, True, False]

        ood_obs = modelbridge.out_of_design_data()
        self.assertTrue(ood_obs == unwrap_observation_data([observation2().data]))

    @mock.patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=([observation1()]),
    )
    @mock.patch("ax.modelbridge.base.ModelBridge._fit", autospec=True)
    def testSetStatusQuo(self, mock_fit, mock_observations_from_data):
        modelbridge = ModelBridge(
            search_space_for_value(), 0, [], get_experiment(), 0, status_quo_name="1_1"
        )
        self.assertEqual(modelbridge.status_quo, observation1())

        # Alternatively, we can specify by features
        modelbridge = ModelBridge(
            search_space_for_value(),
            0,
            [],
            get_experiment(),
            0,
            status_quo_features=observation1().features,
        )
        self.assertEqual(modelbridge.status_quo, observation1())

        # Alternatively, we can specify on experiment
        # Put a dummy arm with SQ name 1_1 on the dummy experiment.
        exp = get_experiment()
        sq = Arm(name="1_1", parameters={"x": 3.0})
        exp._status_quo = sq
        # Check that we set SQ to arm 1_1
        modelbridge = ModelBridge(search_space_for_value(), 0, [], exp, 0)
        self.assertEqual(modelbridge.status_quo, observation1())

        # Errors if features and name both specified
        with self.assertRaises(ValueError):
            modelbridge = ModelBridge(
                search_space_for_value(),
                0,
                [],
                exp,
                0,
                status_quo_features=observation1().features,
                status_quo_name="1_1",
            )

        # Left as None if features or name don't exist
        modelbridge = ModelBridge(
            search_space_for_value(), 0, [], exp, 0, status_quo_name="1_0"
        )
        self.assertIsNone(modelbridge.status_quo)
        modelbridge = ModelBridge(
            search_space_for_value(),
            0,
            [],
            get_experiment(),
            0,
            status_quo_features=ObservationFeatures(parameters={"x": 3.0, "y": 10.0}),
        )
        self.assertIsNone(modelbridge.status_quo)

    @mock.patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=([observation1(), observation2()]),
    )
    @mock.patch("ax.modelbridge.base.ModelBridge._fit", autospec=True)
    def testSetStatusQuoMultipleObs(self, mock_fit, mock_observations_from_data):
        modelbridge = ModelBridge(
            search_space_for_value(), 0, [], get_experiment(), 0, status_quo_name="1_1"
        )
        # SQ not set if multiple feature sets for SQ arm.
        self.assertIsNone(modelbridge.status_quo)

    @mock.patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=([observation1(), observation1()]),
    )
    @mock.patch("ax.modelbridge.base.ModelBridge._fit", autospec=True)
    def testSetTrainingDataDupFeatures(self, mock_fit, mock_observations_from_data):
        # Throws an error if repeated features in observations.
        with self.assertRaises(ValueError):
            ModelBridge(
                search_space_for_value(),
                0,
                [],
                get_experiment(),
                0,
                status_quo_name="1_1",
            )

    def testUnwrapObservationData(self):
        observation_data = [observation1().data, observation2().data]
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

    def testGenArms(self):
        p1 = {"x": 0, "y": 1}
        p2 = {"x": 4, "y": 8}
        observation_features = [
            ObservationFeatures(parameters=p1),
            ObservationFeatures(parameters=p2),
        ]
        arms = gen_arms(observation_features=observation_features)
        self.assertEqual(arms[0].parameters, p1)

        arm = Arm(name="1_1", parameters=p1)
        arms_by_signature = {arm.signature: arm}
        arms = gen_arms(
            observation_features=observation_features,
            arms_by_signature=arms_by_signature,
        )
        self.assertEqual(arms[0].name, "1_1")

    @mock.patch(
        "ax.modelbridge.base.ModelBridge._gen",
        autospec=True,
        return_value=([observation1trans().features], [2], None),
    )
    @mock.patch(
        "ax.modelbridge.base.ModelBridge.predict", autospec=True, return_value=None
    )
    def testGenWithDefaults(self, _, mock_gen):
        exp = get_experiment()
        exp.optimization_config = get_optimization_config()
        ss = search_space_for_range_value()
        modelbridge = ModelBridge(ss, None, [], exp)
        modelbridge.gen(1)
        mock_gen.assert_called_with(
            modelbridge,
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
        "ax.modelbridge.base.ModelBridge._gen",
        autospec=True,
        side_effect=[
            ([observation1trans().features], [2], None),
            ([observation2trans().features], [2], None),
            ([observation2().features], [2], None),
        ],
    )
    @mock.patch("ax.modelbridge.base.ModelBridge._update", autospec=True)
    def test_update(self, _mock_update, _mock_gen):
        exp = get_experiment()
        exp.optimization_config = get_optimization_config()
        ss = search_space_for_range_values()
        exp.search_space = ss
        modelbridge = ModelBridge(ss, None, [Log], exp)
        exp.new_trial(generator_run=modelbridge.gen(1))
        modelbridge._set_training_data(
            observations_from_data(
                data=Data(
                    pd.DataFrame(
                        [
                            {
                                "arm_name": "0_0",
                                "metric_name": "m1",
                                "mean": 3.0,
                                "sem": 1.0,
                            }
                        ]
                    )
                ),
                experiment=exp,
            )
        )
        exp.new_trial(generator_run=modelbridge.gen(1))
        modelbridge.update(
            data=Data(
                pd.DataFrame(
                    [{"arm_name": "1_0", "metric_name": "m1", "mean": 5.0, "sem": 0.0}]
                )
            ),
            experiment=exp,
        )
        exp.new_trial(generator_run=modelbridge.gen(1))
        # Trying to update with unrecognised metric should error.
        with self.assertRaisesRegex(ValueError, "Unrecognised metric"):
            modelbridge.update(
                data=Data(
                    pd.DataFrame(
                        [
                            {
                                "arm_name": "1_0",
                                "metric_name": "m2",
                                "mean": 5.0,
                                "sem": 0.0,
                            }
                        ]
                    )
                ),
                experiment=exp,
            )
