#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest import mock
from unittest.mock import Mock

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
from ax.exceptions.core import UserInputError
from ax.modelbridge.discrete import _get_parameter_values, DiscreteAdapter
from ax.models.discrete_base import DiscreteGenerator
from ax.utils.common.testutils import TestCase


class DiscreteAdapterTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
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

    @mock.patch("ax.modelbridge.discrete.DiscreteAdapter.__init__", return_value=None)
    def test_fit(self, _: Mock) -> None:
        # pyre-fixme[20]: Argument `model` expected.
        adapter = DiscreteAdapter()
        adapter._training_data = self.observations
        model = mock.create_autospec(DiscreteGenerator, instance=True)
        adapter.model = model
        adapter._fit(self.search_space, self.observations)
        self.assertEqual(adapter.parameters, ["x", "y", "z"])
        self.assertEqual(sorted(adapter.outcomes), ["a", "b"])
        Xs = {
            "a": [[0, "foo", True], [1, "foo", True], [1, "bar", True]],
            "b": [[0, "foo", True], [1, "foo", True]],
        }
        Ys = {"a": [[1.0], [2.0], [3.0]], "b": [[-1.0], [-2.0]]}
        Yvars = {"a": [[1.0], [2.0], [3.0]], "b": [[6.0], [7.0]]}
        parameter_values = [[0.0, 1.0], ["foo", "bar"], [True]]
        model_fit_args = model.fit.mock_calls[0][2]
        for i, x in enumerate(model_fit_args["Xs"]):
            self.assertEqual(x, Xs[adapter.outcomes[i]])
        for i, y in enumerate(model_fit_args["Ys"]):
            self.assertEqual(y, Ys[adapter.outcomes[i]])
        for i, v in enumerate(model_fit_args["Yvars"]):
            self.assertEqual(v, Yvars[adapter.outcomes[i]])
        self.assertEqual(model_fit_args["parameter_values"], parameter_values)

        sq_obs = Observation(
            features=ObservationFeatures({}), data=self.observation_data[0]
        )
        with self.assertRaises(ValueError):
            adapter._fit(self.search_space, self.observations + [sq_obs])

    @mock.patch("ax.modelbridge.discrete.DiscreteAdapter.__init__", return_value=None)
    def test_predict(self, mock_init: Mock) -> None:
        # pyre-fixme[20]: Argument `model` expected.
        adapter = DiscreteAdapter()
        model = mock.MagicMock(DiscreteGenerator, autospec=True, instance=True)
        model.predict.return_value = (
            np.array([[1.0, -1], [2.0, -2]]),
            np.stack(
                (np.array([[1.0, 4.0], [4.0, 6]]), np.array([[2.0, 5.0], [5.0, 7]]))
            ),
        )
        adapter.model = model
        adapter.parameters = ["x", "y", "z"]
        adapter.outcomes = ["a", "b"]
        observation_data = adapter._predict(self.observation_features)
        X = [[0, "foo", True], [1, "foo", True], [1, "bar", True]]
        self.assertTrue(model.predict.mock_calls[0][2]["X"], X)
        for i, od in enumerate(observation_data):
            self.assertEqual(od, self.observation_data[i])

    @mock.patch("ax.modelbridge.discrete.DiscreteAdapter.__init__", return_value=None)
    def test_gen(self, mock_init: Mock) -> None:
        # Test with constraints
        optimization_config = OptimizationConfig(
            objective=Objective(Metric("a"), minimize=True),
            outcome_constraints=[
                OutcomeConstraint(Metric("b"), ComparisonOp.GEQ, 2, False)
            ],
        )
        # pyre-fixme[20]: Argument `model` expected.
        adapter = DiscreteAdapter()
        # Test validation.
        with self.assertRaisesRegex(UserInputError, "positive integer or -1."):
            adapter._validate_gen_inputs(n=0)
        adapter._validate_gen_inputs(n=-1)
        # Test rest of gen.
        model = mock.MagicMock(DiscreteGenerator, autospec=True, instance=True)
        best_x = [0.0, 2.0, 1.0]
        model.gen.return_value = (
            [[0.0, 2.0, 3.0], [1.0, 1.0, 3.0]],
            [1.0, 2.0],
            {"best_x": best_x},
        )
        adapter.model = model
        adapter.parameters = ["x", "y", "z"]
        adapter.outcomes = ["a", "b"]
        gen_results = adapter._gen(
            n=3,
            search_space=self.search_space,
            optimization_config=optimization_config,
            pending_observations=self.pending_observations,
            fixed_features=ObservationFeatures({}),
            # pyre-fixme[6]: For 6th param expected `Optional[Dict[str, Union[None,
            #  Dict[str, typing.Any], OptimizationConfig, AcquisitionFunction, float,
            #  int, str]]]` but got `Dict[str, str]`.
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
            gen_results.observation_features[0].parameters,
            {"x": 0.0, "y": 2.0, "z": 3.0},
        )
        self.assertEqual(
            gen_results.observation_features[1].parameters,
            {"x": 1.0, "y": 1.0, "z": 3.0},
        )
        self.assertEqual(gen_results.weights, [1.0, 2.0])
        self.assertEqual(
            gen_results.best_observation_features,
            ObservationFeatures(parameters=dict(zip(adapter.parameters, best_x))),
        )

        # Test with no constraints, no fixed feature, no pending observations
        search_space = SearchSpace(self.parameters[:2])
        optimization_config.outcome_constraints = []
        adapter.parameters = ["x", "y"]
        adapter._gen(
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
            adapter._gen(
                n=3,
                search_space=search_space,
                optimization_config=optimization_config,
                pending_observations={},
                fixed_features=ObservationFeatures({}),
                model_gen_options={},
            )

    @mock.patch("ax.modelbridge.discrete.DiscreteAdapter.__init__", return_value=None)
    def test_cross_validate(self, mock_init: Mock) -> None:
        # pyre-fixme[20]: Argument `model` expected.
        adapter = DiscreteAdapter()
        model = mock.MagicMock(DiscreteGenerator, autospec=True, instance=True)
        model.cross_validate.return_value = (
            np.array([[1.0, -1], [2.0, -2]]),
            np.stack(
                (np.array([[1.0, 4.0], [4.0, 6]]), np.array([[2.0, 5.0], [5.0, 7]]))
            ),
        )
        adapter.model = model
        adapter.parameters = ["x", "y", "z"]
        adapter.outcomes = ["a", "b"]
        observation_data = adapter._cross_validate(
            search_space=self.search_space,
            cv_training_data=self.observations,
            cv_test_points=self.observation_features,
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

    def test_get_parameter_values(self) -> None:
        parameter_values = _get_parameter_values(self.search_space, ["x", "y", "z"])
        self.assertEqual(parameter_values, [[0.0, 1.0], ["foo", "bar"], [True]])
        search_space = SearchSpace(self.parameters)
        search_space._parameters["x"] = RangeParameter(
            "x", ParameterType.FLOAT, 0.1, 0.4
        )
        with self.assertRaises(ValueError):
            _get_parameter_values(search_space, ["x", "y", "z"])
