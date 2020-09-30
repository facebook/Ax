#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from unittest import mock

import numpy as np
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.parameter_constraint import OrderConstraint, SumConstraint
from ax.core.search_space import SearchSpace
from ax.modelbridge.modelbridge_utils import get_bounds_and_task
from ax.modelbridge.numpy import NumpyModelBridge
from ax.models.numpy_base import NumpyModel
from ax.utils.common.testutils import TestCase


class NumpyModelBridgeTest(TestCase):
    def setUp(self):
        x = RangeParameter("x", ParameterType.FLOAT, lower=0, upper=1)
        y = RangeParameter(
            "y", ParameterType.FLOAT, lower=1, upper=2, is_fidelity=True, target_value=2
        )
        z = RangeParameter("z", ParameterType.FLOAT, lower=0, upper=5)
        self.parameters = [x, y, z]
        parameter_constraints = [
            OrderConstraint(x, y),
            SumConstraint([x, z], False, 3.5),
        ]

        self.search_space = SearchSpace(self.parameters, parameter_constraints)

        self.observation_features = [
            ObservationFeatures(parameters={"x": 0.2, "y": 1.2, "z": 3}),
            ObservationFeatures(parameters={"x": 0.4, "y": 1.4, "z": 3}),
            ObservationFeatures(parameters={"x": 0.6, "y": 1.6, "z": 3}),
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
            "b": [ObservationFeatures(parameters={"x": 0.6, "y": 1.6, "z": 3})]
        }
        self.model_gen_options = {"option": "yes"}

    @mock.patch("ax.modelbridge.numpy.NumpyModelBridge.__init__", return_value=None)
    def testFitAndUpdate(self, mock_init):
        sq_feat = ObservationFeatures({})
        sq_data = self.observation_data[2]
        sq_obs = Observation(
            features=ObservationFeatures({}),
            data=self.observation_data[2],
            arm_name="status_quo",
        )
        ma = NumpyModelBridge()
        ma._training_data = self.observations + [sq_obs]
        model = mock.create_autospec(NumpyModel, instance=True)
        # No out of design points allowed in direct calls to fit.
        with self.assertRaises(ValueError):
            ma._fit(
                model,
                self.search_space,
                self.observation_features + [sq_feat],
                self.observation_data + [sq_data],
            )
        ma._fit(
            model, self.search_space, self.observation_features, self.observation_data
        )
        self.assertEqual(ma.parameters, ["x", "y", "z"])
        self.assertEqual(sorted(ma.outcomes), ["a", "b"])
        Xs = {
            "a": np.array([[0.2, 1.2, 3.0], [0.4, 1.4, 3.0], [0.6, 1.6, 3.0]]),
            "b": np.array([[0.2, 1.2, 3.0], [0.4, 1.4, 3.0]]),
        }
        Ys = {"a": np.array([[1.0], [2.0], [3.0]]), "b": np.array([[-1.0], [-2.0]])}
        Yvars = {"a": np.array([[1.0], [2.0], [3.0]]), "b": np.array([[6.0], [7.0]])}
        bounds = [(0.0, 1.0), (1.0, 2.0), (0.0, 5.0)]
        model_fit_args = model.fit.mock_calls[0][2]
        for i, x in enumerate(model_fit_args["Xs"]):
            self.assertTrue(np.array_equal(x, Xs[ma.outcomes[i]]))
        for i, y in enumerate(model_fit_args["Ys"]):
            self.assertTrue(np.array_equal(y, Ys[ma.outcomes[i]]))
        for i, v in enumerate(model_fit_args["Yvars"]):
            self.assertTrue(np.array_equal(v, Yvars[ma.outcomes[i]]))
        self.assertEqual(model_fit_args["bounds"], bounds)
        self.assertEqual(model_fit_args["feature_names"], ["x", "y", "z"])

        # And update
        ma._update(
            observation_features=self.observation_features,
            observation_data=self.observation_data,
        )
        # Calling _update requires passing ALL data.
        model_update_args = model.update.mock_calls[0][2]
        for i, x in enumerate(model_update_args["Xs"]):
            self.assertTrue(np.array_equal(x, Xs[ma.outcomes[i]]))
        for i, y in enumerate(model_update_args["Ys"]):
            self.assertTrue(np.array_equal(y, Ys[ma.outcomes[i]]))
        for i, v in enumerate(model_update_args["Yvars"]):
            self.assertTrue(np.array_equal(v, Yvars[ma.outcomes[i]]))

    @mock.patch(
        "ax.modelbridge.numpy.NumpyModelBridge._model_predict",
        return_value=(
            np.array([[1.0, -1], [2.0, -2]]),
            np.stack(
                (np.array([[1.0, 4.0], [4.0, 6]]), np.array([[2.0, 5.0], [5.0, 7]]))
            ),
        ),
        autospec=True,
    )
    @mock.patch("ax.modelbridge.numpy.NumpyModelBridge.__init__", return_value=None)
    def testPredict(self, mock_init, mock_predict):
        ma = NumpyModelBridge()
        ma.parameters = ["x", "y", "z"]
        ma.outcomes = ["a", "b"]
        observation_data = ma._predict(self.observation_features)
        X = np.array([[0.2, 1.2, 3.0], [0.4, 1.4, 3.0], [0.6, 1.6, 3]])
        self.assertTrue(np.array_equal(mock_predict.mock_calls[0][2]["X"], X))
        for i, od in enumerate(observation_data):
            self.assertEqual(od, self.observation_data[i])

    @mock.patch(
        "ax.modelbridge.numpy.NumpyModelBridge._model_gen",
        autospec=True,
        return_value=(
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 3.0]]),
            np.array([1.0, 2.0]),
            {},
            [],
        ),
    )
    @mock.patch(
        "ax.modelbridge.numpy.NumpyModelBridge._model_best_point",
        autospec=True,
        return_value=None,
    )
    @mock.patch("ax.modelbridge.numpy.NumpyModelBridge.__init__", return_value=None)
    def testGen(self, mock_init, mock_best_point, mock_gen):
        # Test with constraints
        optimization_config = OptimizationConfig(
            objective=Objective(Metric("a"), minimize=True),
            outcome_constraints=[
                OutcomeConstraint(Metric("b"), ComparisonOp.GEQ, 2, False)
            ],
        )
        ma = NumpyModelBridge()
        ma.parameters = ["x", "y", "z"]
        ma.outcomes = ["a", "b"]
        ma.transforms = OrderedDict()
        observation_features, weights, best_obsf, _ = ma._gen(
            n=3,
            search_space=self.search_space,
            optimization_config=optimization_config,
            pending_observations=self.pending_observations,
            fixed_features=ObservationFeatures({"z": 3.0}),
            model_gen_options=self.model_gen_options,
        )
        gen_args = mock_gen.mock_calls[0][2]
        self.assertEqual(gen_args["n"], 3)
        self.assertEqual(gen_args["bounds"], [(0.0, 1.0), (1.0, 2.0), (0.0, 5.0)])
        self.assertTrue(
            np.array_equal(gen_args["objective_weights"], np.array([-1.0, 0.0]))
        )
        self.assertTrue(
            np.array_equal(gen_args["outcome_constraints"][0], np.array([[0.0, -1.0]]))
        )
        self.assertTrue(
            np.array_equal(gen_args["outcome_constraints"][1], np.array([[-2]]))
        )
        self.assertTrue(
            np.array_equal(
                gen_args["linear_constraints"][0],
                np.array([[1.0, -1, 0.0], [-1.0, 0.0, -1.0]]),
            )
        )
        self.assertTrue(
            np.array_equal(gen_args["linear_constraints"][1], np.array([[0.0], [-3.5]]))
        )
        self.assertEqual(gen_args["fixed_features"], {2: 3.0})
        self.assertTrue(
            np.array_equal(gen_args["pending_observations"][0], np.array([]))
        )
        self.assertTrue(
            np.array_equal(
                gen_args["pending_observations"][1], np.array([[0.6, 1.6, 3.0]])
            )
        )
        self.assertEqual(gen_args["model_gen_options"], {"option": "yes"})
        self.assertEqual(
            observation_features[0].parameters, {"x": 1.0, "y": 2.0, "z": 3.0}
        )
        self.assertEqual(
            observation_features[1].parameters, {"x": 3.0, "y": 4.0, "z": 3.0}
        )
        self.assertTrue(np.array_equal(weights, np.array([1.0, 2.0])))

        # Test with multiple objectives.
        oc2 = OptimizationConfig(
            objective=ScalarizedObjective(
                metrics=[Metric(name="a"), Metric(name="b")], minimize=True
            )
        )
        observation_features, weights, best_obsf, _ = ma._gen(
            n=3,
            search_space=self.search_space,
            optimization_config=oc2,
            pending_observations=self.pending_observations,
            fixed_features=ObservationFeatures({"z": 3.0}),
            model_gen_options=self.model_gen_options,
        )
        gen_args = mock_gen.mock_calls[1][2]
        self.assertEqual(gen_args["bounds"], [(0.0, 1.0), (1.0, 2.0), (0.0, 5.0)])
        self.assertIsNone(gen_args["outcome_constraints"])
        self.assertTrue(
            np.array_equal(gen_args["objective_weights"], np.array([-1.0, -1.0]))
        )

        # Test with MultiObjective (unweighted multiple objectives)
        oc3 = OptimizationConfig(
            objective=MultiObjective(
                metrics=[Metric(name="a"), Metric(name="b", lower_is_better=True)],
                minimize=True,
            )
        )
        search_space = SearchSpace(self.parameters)  # Unconstrained
        observation_features, weights, best_obsf, _ = ma._gen(
            n=3,
            search_space=search_space,
            optimization_config=oc3,
            pending_observations=self.pending_observations,
            fixed_features=ObservationFeatures({"z": 3.0}),
            model_gen_options=self.model_gen_options,
        )
        gen_args = mock_gen.mock_calls[2][2]
        self.assertEqual(gen_args["bounds"], [(0.0, 1.0), (1.0, 2.0), (0.0, 5.0)])
        self.assertIsNone(gen_args["outcome_constraints"])
        self.assertTrue(
            np.array_equal(gen_args["objective_weights"], np.array([1.0, -1.0]))
        )

        # Test with no constraints, no fixed feature, no pending observations
        search_space = SearchSpace(self.parameters[:2])
        optimization_config.outcome_constraints = []
        ma.parameters = ["x", "y"]
        ma._gen(3, search_space, {}, ObservationFeatures({}), None, optimization_config)
        gen_args = mock_gen.mock_calls[3][2]
        self.assertEqual(gen_args["bounds"], [(0.0, 1.0), (1.0, 2.0)])
        self.assertIsNone(gen_args["outcome_constraints"])
        self.assertIsNone(gen_args["linear_constraints"])
        self.assertIsNone(gen_args["fixed_features"])
        self.assertIsNone(gen_args["pending_observations"])

        # Test validation
        optimization_config = OptimizationConfig(
            objective=Objective(Metric("a"), minimize=False),
            outcome_constraints=[
                OutcomeConstraint(Metric("b"), ComparisonOp.GEQ, 2, False)
            ],
        )
        with self.assertRaises(ValueError):
            ma._gen(
                n=3,
                search_space=self.search_space,
                optimization_config=optimization_config,
                pending_observations={},
                fixed_features=ObservationFeatures({}),
            )
        optimization_config.objective.minimize = True
        optimization_config.outcome_constraints[0].relative = True
        with self.assertRaises(ValueError):
            ma._gen(
                n=3,
                search_space=self.search_space,
                optimization_config=optimization_config,
                pending_observations={},
                fixed_features=ObservationFeatures({}),
            )

    @mock.patch(
        "ax.modelbridge.numpy.NumpyModelBridge._model_cross_validate",
        return_value=(
            np.array([[1.0, -1], [2.0, -2]]),
            np.stack(
                (np.array([[1.0, 4.0], [4.0, 6]]), np.array([[2.0, 5.0], [5.0, 7]]))
            ),
        ),
        autospec=True,
    )
    @mock.patch("ax.modelbridge.numpy.NumpyModelBridge.__init__", return_value=None)
    def testCrossValidate(self, mock_init, mock_cv):
        ma = NumpyModelBridge()
        ma.parameters = ["x", "y", "z"]
        ma.outcomes = ["a", "b"]
        observation_data = ma._cross_validate(
            self.observation_features, self.observation_data, self.observation_features
        )
        Xs = [
            np.array([[0.2, 1.2, 3.0], [0.4, 1.4, 3.0], [0.6, 1.6, 3]]),
            np.array([[0.2, 1.2, 3.0], [0.4, 1.4, 3.0]]),
        ]
        Ys = [np.array([[1.0], [2.0], [3.0]]), np.array([[-1.0], [-2.0]])]
        Yvars = [np.array([[1.0], [2.0], [3.0]]), np.array([[6.0], [7.0]])]
        Xtest = np.array([[0.2, 1.2, 3.0], [0.4, 1.4, 3.0], [0.6, 1.6, 3]])
        # Transform to arrays:
        model_cv_args = mock_cv.mock_calls[0][2]
        for i, x in enumerate(model_cv_args["Xs_train"]):
            self.assertTrue(np.array_equal(x, Xs[i]))
        for i, y in enumerate(model_cv_args["Ys_train"]):
            self.assertTrue(np.array_equal(y, Ys[i]))
        for i, v in enumerate(model_cv_args["Yvars_train"]):
            self.assertTrue(np.array_equal(v, Yvars[i]))
        self.assertTrue(np.array_equal(model_cv_args["X_test"], Xtest))
        # Transform from arrays:
        for i, od in enumerate(observation_data):
            self.assertEqual(od, self.observation_data[i])

    def testGetBoundsAndTask(self):
        bounds, task_features, target_fidelities = get_bounds_and_task(
            self.search_space, ["x", "y", "z"]
        )
        self.assertEqual(bounds, [(0.0, 1.0), (1.0, 2.0), (0.0, 5.0)])
        self.assertEqual(task_features, [])
        self.assertEqual(target_fidelities, {1: 2.0})
        bounds, task_features, target_fidelities = get_bounds_and_task(
            self.search_space, ["x", "z"]
        )
        self.assertEqual(target_fidelities, {})
        # Test that Int param is treated as task feature
        search_space = SearchSpace(self.parameters)
        search_space._parameters["x"] = RangeParameter(
            "x", ParameterType.INT, lower=1, upper=4
        )
        bounds, task_features, target_fidelities = get_bounds_and_task(
            search_space, ["x", "y", "z"]
        )
        self.assertEqual(task_features, [0])
        # Test validation
        search_space._parameters["x"] = ChoiceParameter(
            "x", ParameterType.FLOAT, [0.1, 0.4]
        )
        with self.assertRaises(ValueError):
            get_bounds_and_task(search_space, ["x", "y", "z"])
        search_space._parameters["x"] = RangeParameter(
            "x", ParameterType.FLOAT, lower=1.0, upper=4.0, log_scale=True
        )
        with self.assertRaises(ValueError):
            get_bounds_and_task(search_space, ["x", "y", "z"])
