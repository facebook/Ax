#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import numpy as np
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.modelbridge.discrete import DiscreteModelBridge, _get_parameter_values
from ax.models.discrete_base import DiscreteModel
from ax.utils.common.testutils import TestCase


class DiscreteModelBridgeTest(TestCase):
    def setUp(self):
        self.parameters = [
            ChoiceParameter("x", ParameterType.FLOAT, values=[0, 1]),
            ChoiceParameter("y", ParameterType.STRING, values=["foo", "bar"]),
            FixedParameter("z", ParameterType.BOOL, value=True),
        ]
        parameter_constraints = []

        self.search_space = SearchSpace(self.parameters, parameter_constraints)

        self.observation_features = [
            ObservationFeatures(parameters={"x": 0, "y": "foo", "z": True}),
            ObservationFeatures(parameters={"x": 1, "y": "foo", "z": True}),
            ObservationFeatures(parameters={"x": 1, "y": "bar", "z": True}),
        ]
        self.observation_data = [
            ObservationData(
                metric_names=["a", "b"],
                means=np.array([1.0, -1.0]),
                covariance=np.array([[1.0, 4.0], [4.0, 6.0]]),
            ),
            ObservationData(
                metric_names=["a", "b"],
                means=np.array([2.0, -2.0]),
                covariance=np.array([[2.0, 5.0], [5.0, 7.0]]),
            ),
            ObservationData(
                metric_names=["a"], means=np.array([3.0]), covariance=np.array([[3.0]])
            ),
        ]
        self.observations = [
            Observation(
                features=self.observation_features[i],
                data=self.observation_data[i],
                arm_name=str(i),
            )
            for i in range(3)
        ]
        self.pending_observations = {
            "b": [ObservationFeatures(parameters={"x": 0, "y": "foo", "z": True})]
        }
        self.model_gen_options = {"option": "yes"}

    @mock.patch(
        "ax.modelbridge.discrete.DiscreteModelBridge.__init__", return_value=None
    )
    def testFit(self, mock_init):
        ma = DiscreteModelBridge()
        ma._training_data = self.observations
        model = mock.create_autospec(DiscreteModel, instance=True)
        ma._fit(
            model, self.search_space, self.observation_features, self.observation_data
        )
        self.assertEqual(ma.parameters, ["x", "y", "z"])
        self.assertEqual(sorted(ma.outcomes), ["a", "b"])
        Xs = {
            "a": [[0, "foo", True], [1, "foo", True], [1, "bar", True]],
            "b": [[0, "foo", True], [1, "foo", True]],
        }
        Ys = {"a": [[1.0], [2.0], [3.0]], "b": [[-1.0], [-2.0]]}
        Yvars = {"a": [[1.0], [2.0], [3.0]], "b": [[6.0], [7.0]]}
        parameter_values = [[0.0, 1.0], ["foo", "bar"], [True]]
        model_fit_args = model.fit.mock_calls[0][2]
        for i, x in enumerate(model_fit_args["Xs"]):
            self.assertEqual(x, Xs[ma.outcomes[i]])
        for i, y in enumerate(model_fit_args["Ys"]):
            self.assertEqual(y, Ys[ma.outcomes[i]])
        for i, v in enumerate(model_fit_args["Yvars"]):
            self.assertEqual(v, Yvars[ma.outcomes[i]])
        self.assertEqual(model_fit_args["parameter_values"], parameter_values)

        sq_feat = ObservationFeatures({})
        sq_data = self.observation_data[0]
        with self.assertRaises(ValueError):
            ma._fit(
                model,
                self.search_space,
                self.observation_features + [sq_feat],
                self.observation_data + [sq_data],
            )

    @mock.patch(
        "ax.modelbridge.discrete.DiscreteModelBridge.__init__", return_value=None
    )
    def testPredict(self, mock_init):
        ma = DiscreteModelBridge()
        model = mock.MagicMock(DiscreteModel, autospec=True, instance=True)
        model.predict.return_value = (
            np.array([[1.0, -1], [2.0, -2]]),
            np.stack(
                (np.array([[1.0, 4.0], [4.0, 6]]), np.array([[2.0, 5.0], [5.0, 7]]))
            ),
        )
        ma.model = model
        ma.parameters = ["x", "y", "z"]
        ma.outcomes = ["a", "b"]
        observation_data = ma._predict(self.observation_features)
        X = [[0, "foo", True], [1, "foo", True], [1, "bar", True]]
        self.assertTrue(model.predict.mock_calls[0][2]["X"], X)
        for i, od in enumerate(observation_data):
            self.assertEqual(od, self.observation_data[i])

    @mock.patch(
        "ax.modelbridge.discrete.DiscreteModelBridge.__init__", return_value=None
    )
    def testGen(self, mock_init):
        # Test with constraints
        optimization_config = OptimizationConfig(
            objective=Objective(Metric("a"), minimize=True),
            outcome_constraints=[
                OutcomeConstraint(Metric("b"), ComparisonOp.GEQ, 2, False)
            ],
        )
        ma = DiscreteModelBridge()
        model = mock.MagicMock(DiscreteModel, autospec=True, instance=True)
        model.gen.return_value = ([[0.0, 2.0, 3.0], [1.0, 1.0, 3.0]], [1.0, 2.0], {})
        ma.model = model
        ma.parameters = ["x", "y", "z"]
        ma.outcomes = ["a", "b"]
        observation_features, weights, best_observation, _ = ma._gen(
            n=3,
            search_space=self.search_space,
            optimization_config=optimization_config,
            pending_observations=self.pending_observations,
            fixed_features=ObservationFeatures({}),
            model_gen_options=self.model_gen_options,
        )
        gen_args = model.gen.mock_calls[0][2]
        self.assertEqual(gen_args["n"], 3)
        self.assertEqual(
            gen_args["parameter_values"], [[0.0, 1.0], ["foo", "bar"], [True]]
        )
        self.assertTrue(
            np.array_equal(gen_args["objective_weights"], np.array([-1.0, 0.0]))
        )
        self.assertTrue(
            np.array_equal(gen_args["outcome_constraints"][0], np.array([[0.0, -1.0]]))
        )
        self.assertTrue(
            np.array_equal(gen_args["outcome_constraints"][1], np.array([[-2]]))
        )
        self.assertEqual(gen_args["pending_observations"][0], [])
        self.assertEqual(gen_args["pending_observations"][1], [[0, "foo", True]])
        self.assertEqual(gen_args["model_gen_options"], {"option": "yes"})
        self.assertEqual(
            observation_features[0].parameters, {"x": 0.0, "y": 2.0, "z": 3.0}
        )
        self.assertEqual(
            observation_features[1].parameters, {"x": 1.0, "y": 1.0, "z": 3.0}
        )
        self.assertEqual(weights, [1.0, 2.0])

        # Test with no constraints, no fixed feature, no pending observations
        search_space = SearchSpace(self.parameters[:2])
        optimization_config.outcome_constraints = []
        ma.parameters = ["x", "y"]
        ma._gen(
            n=3,
            search_space=search_space,
            optimization_config=optimization_config,
            pending_observations={},
            fixed_features=ObservationFeatures({}),
            model_gen_options={},
        )
        gen_args = model.gen.mock_calls[1][2]
        self.assertEqual(gen_args["parameter_values"], [[0.0, 1.0], ["foo", "bar"]])
        self.assertIsNone(gen_args["outcome_constraints"])
        self.assertIsNone(gen_args["pending_observations"])

        # Test validation
        optimization_config = OptimizationConfig(
            objective=Objective(Metric("a"), minimize=False),
            outcome_constraints=[
                OutcomeConstraint(Metric("b"), ComparisonOp.GEQ, 2, True)
            ],
        )
        with self.assertRaises(ValueError):
            ma._gen(
                n=3,
                search_space=search_space,
                optimization_config=optimization_config,
                pending_observations={},
                fixed_features=ObservationFeatures({}),
                model_gen_options={},
            )

    @mock.patch(
        "ax.modelbridge.discrete.DiscreteModelBridge.__init__", return_value=None
    )
    def testCrossValidate(self, mock_init):
        ma = DiscreteModelBridge()
        model = mock.MagicMock(DiscreteModel, autospec=True, instance=True)
        model.cross_validate.return_value = (
            np.array([[1.0, -1], [2.0, -2]]),
            np.stack(
                (np.array([[1.0, 4.0], [4.0, 6]]), np.array([[2.0, 5.0], [5.0, 7]]))
            ),
        )
        ma.model = model
        ma.parameters = ["x", "y", "z"]
        ma.outcomes = ["a", "b"]
        observation_data = ma._cross_validate(
            self.observation_features, self.observation_data, self.observation_features
        )
        Xs = [
            [[0, "foo", True], [1, "foo", True], [1, "bar", True]],
            [[0, "foo", True], [1, "foo", True]],
        ]
        Ys = [[[1.0], [2.0], [3.0]], [[-1.0], [-2.0]]]
        Yvars = [[[1.0], [2.0], [3.0]], [[6.0], [7.0]]]
        Xtest = [[0, "foo", True], [1, "foo", True], [1, "bar", True]]
        # Transform to arrays:
        model_cv_args = model.cross_validate.mock_calls[0][2]
        for i, x in enumerate(model_cv_args["Xs_train"]):
            self.assertEqual(x, Xs[i])
        for i, y in enumerate(model_cv_args["Ys_train"]):
            self.assertEqual(y, Ys[i])
        for i, v in enumerate(model_cv_args["Yvars_train"]):
            self.assertEqual(v, Yvars[i])
        self.assertEqual(model_cv_args["X_test"], Xtest)
        # Transform from arrays:
        for i, od in enumerate(observation_data):
            self.assertEqual(od, self.observation_data[i])

    def testGetParameterValues(self):
        parameter_values = _get_parameter_values(self.search_space, ["x", "y", "z"])
        self.assertEqual(parameter_values, [[0.0, 1.0], ["foo", "bar"], [True]])
        search_space = SearchSpace(self.parameters)
        search_space._parameters["x"] = RangeParameter(
            "x", ParameterType.FLOAT, 0.1, 0.4
        )
        with self.assertRaises(ValueError):
            _get_parameter_values(search_space, ["x", "y", "z"])
