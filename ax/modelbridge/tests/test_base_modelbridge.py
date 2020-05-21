#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import numpy as np
import pandas as pd
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.metric import Metric
from ax.core.objective import Objective, ScalarizedObjective
from ax.core.observation import (
    ObservationData,
    ObservationFeatures,
    observations_from_data,
)
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import FixedParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.modelbridge.base import ModelBridge, gen_arms, unwrap_observation_data
from ax.modelbridge.transforms.log import Log
from ax.models.base import Model
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_experiment_with_repeated_arms,
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
    def testModelBridge(self, mock_fit, mock_gen_arms, mock_observations_from_data):
        # Test that on init transforms are stored and applied in the correct order
        transforms = [transform_1, transform_2]
        exp = get_experiment_for_value()
        ss = get_search_space_for_value()
        modelbridge = ModelBridge(
            search_space=ss,
            model=Model(),
            transforms=transforms,
            experiment=exp,
            data=0,
        )
        self.assertEqual(
            list(modelbridge.transforms.keys()), ["Cast", "transform_1", "transform_2"]
        )
        fit_args = mock_fit.mock_calls[0][2]
        self.assertTrue(fit_args["search_space"] == get_search_space_for_value(8.0))
        self.assertTrue(fit_args["observation_features"] == [])
        self.assertTrue(fit_args["observation_data"] == [])
        self.assertTrue(mock_observations_from_data.called)

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
        modelbridge._predict.assert_called_with([get_observation2().features])

        # Check that _single_predict is equivalent here.
        modelbridge._single_predict([get_observation2().features])
        # Observation features sent to _predict are un-transformed afterwards
        modelbridge._predict.assert_called_with([get_observation2().features])

        # Test transforms applied on gen
        modelbridge._gen = mock.MagicMock(
            "ax.modelbridge.base.ModelBridge._gen",
            autospec=True,
            return_value=([get_observation1trans().features], [2], None, {}),
        )
        oc = OptimizationConfig(objective=Objective(metric=Metric(name="test_metric")))
        modelbridge._set_kwargs_to_save(
            model_key="TestModel", model_kwargs={}, bridge_kwargs={}
        )
        gr = modelbridge.gen(
            n=1,
            search_space=get_search_space_for_value(),
            optimization_config=oc,
            pending_observations={"a": [get_observation2().features]},
            fixed_features=ObservationFeatures({"x": 5}),
        )
        self.assertEqual(gr._model_key, "TestModel")
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
            n=1, search_space=get_search_space_for_value(), optimization_config=oc2
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
            return_value=[get_observation1trans().data],
        )
        cv_training_data = [get_observation2()]
        cv_test_points = [get_observation1().features]
        cv_predictions = modelbridge.cross_validate(
            cv_training_data=cv_training_data, cv_test_points=cv_test_points
        )
        modelbridge._cross_validate.assert_called_with(
            obs_feats=[get_observation2trans().features],
            obs_data=[get_observation2trans().data],
            cv_test_points=[get_observation1().features],  # untransformed after
        )
        self.assertTrue(cv_predictions == [get_observation1().data])

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

    @mock.patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=([get_observation1()]),
    )
    @mock.patch("ax.modelbridge.base.ModelBridge._fit", autospec=True)
    def testSetStatusQuo(self, mock_fit, mock_observations_from_data):
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

        # Alternatively, we can specify by features
        modelbridge = ModelBridge(
            get_search_space_for_value(),
            0,
            [],
            get_experiment_for_value(),
            0,
            status_quo_features=get_observation1().features,
        )
        self.assertEqual(modelbridge.status_quo, get_observation1())

        # Alternatively, we can specify on experiment
        # Put a dummy arm with SQ name 1_1 on the dummy experiment.
        exp = get_experiment_for_value()
        sq = Arm(name="1_1", parameters={"x": 3.0})
        exp._status_quo = sq
        # Check that we set SQ to arm 1_1
        modelbridge = ModelBridge(get_search_space_for_value(), 0, [], exp, 0)
        self.assertEqual(modelbridge.status_quo, get_observation1())

        # Errors if features and name both specified
        with self.assertRaises(ValueError):
            modelbridge = ModelBridge(
                get_search_space_for_value(),
                0,
                [],
                exp,
                0,
                status_quo_features=get_observation1().features,
                status_quo_name="1_1",
            )

        # Left as None if features or name don't exist
        modelbridge = ModelBridge(
            get_search_space_for_value(), 0, [], exp, 0, status_quo_name="1_0"
        )
        self.assertIsNone(modelbridge.status_quo)
        modelbridge = ModelBridge(
            get_search_space_for_value(),
            0,
            [],
            get_experiment_for_value(),
            0,
            status_quo_features=ObservationFeatures(parameters={"x": 3.0, "y": 10.0}),
        )
        self.assertIsNone(modelbridge.status_quo)

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
    def testSetStatusQuoMultipleObs(self, mock_fit, mock_observations_from_data):
        exp = get_experiment_with_repeated_arms(2)

        trial_index = 1
        status_quo_features = ObservationFeatures(
            parameters=exp.trials[trial_index].status_quo.parameters,
            trial_index=trial_index,
        )
        modelbridge = ModelBridge(
            get_search_space_for_value(),
            0,
            [],
            exp,
            0,
            status_quo_features=status_quo_features,
        )
        # Check that for experiments with many trials the status quo is set
        # to the value of the status quo of the last trial.
        if len(exp.trials) >= 1:
            self.assertEqual(modelbridge.status_quo, get_observation_status_quo1())

    @mock.patch(
        "ax.modelbridge.base.observations_from_data",
        autospec=True,
        return_value=([get_observation1(), get_observation1()]),
    )
    @mock.patch("ax.modelbridge.base.ModelBridge._fit", autospec=True)
    def testSetTrainingDataDupFeatures(self, mock_fit, mock_observations_from_data):
        # Throws an error if repeated features in observations.
        with self.assertRaises(ValueError):
            ModelBridge(
                get_search_space_for_value(),
                0,
                [],
                get_experiment_for_value(),
                0,
                status_quo_name="1_1",
            )

    def testUnwrapObservationData(self):
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

    def testGenArms(self):
        p1 = {"x": 0, "y": 1}
        p2 = {"x": 4, "y": 8}
        observation_features = [
            ObservationFeatures(parameters=p1),
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
        return_value=([get_observation1trans().features], [2], None, {}),
    )
    @mock.patch(
        "ax.modelbridge.base.ModelBridge.predict", autospec=True, return_value=None
    )
    def testGenWithDefaults(self, _, mock_gen):
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
            ([get_observation1trans().features], [2], None, {}),
            ([get_observation2trans().features], [2], None, {}),
            ([get_observation2().features], [2], None, {}),
        ],
    )
    @mock.patch("ax.modelbridge.base.ModelBridge._update", autospec=True)
    def test_update(self, _mock_update, _mock_gen):
        exp = get_experiment_for_value()
        exp.optimization_config = get_optimization_config_no_constraints()
        ss = get_search_space_for_range_values()
        exp.search_space = ss
        modelbridge = ModelBridge(
            search_space=ss, model=Model(), transforms=[Log], experiment=exp
        )
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
            ),
            ss,
        )
        exp.new_trial(generator_run=modelbridge.gen(1))
        modelbridge.update(
            new_data=Data(
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
                new_data=Data(
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
