#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest import mock

import numpy as np
import pandas as pd
from ax.adapter.discrete import _get_parameter_values, DiscreteAdapter
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.observation import (
    ObservationFeatures,
    observations_from_data,
    separate_observations,
)
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.core.types import TParameterization
from ax.exceptions.core import UserInputError
from ax.generators.discrete_base import DiscreteGenerator
from ax.generators.types import TConfig
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations


class DiscreteAdapterTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.parameters = [
            ChoiceParameter("x", ParameterType.FLOAT, values=[0, 1]),
            ChoiceParameter("y", ParameterType.STRING, values=["foo", "bar"]),
            FixedParameter("z", ParameterType.BOOL, value=True),
        ]
        self.search_space = SearchSpace(parameters=self.parameters)
        self.parameterizations: list[TParameterization] = [
            {"x": 0.0, "y": "foo", "z": True},
            {"x": 1.0, "y": "foo", "z": True},
            {"x": 1.0, "y": "bar", "z": True},
        ]
        # Construct the experiment with complete trials.
        self.experiment = get_experiment_with_observations(
            observations=[[1.0, -1.0], [2.0, -2.0]],
            search_space=self.search_space,
            sems=[[1.0, 2.0], [1.5, 3.0]],
            parameterizations=self.parameterizations[:2],
        )
        # Add the partial trial with data for only one of the metrics.
        t = (
            self.experiment.new_trial()
            .add_arm(Arm(parameters=self.parameterizations[2]))
            .mark_running(no_runner_required=True)
        )
        data = Data(
            df=pd.DataFrame.from_records(
                [
                    {
                        "arm_name": t.arms[0].name,
                        "metric_name": "m1",
                        "mean": 3.0,
                        "sem": 1.2,
                        "trial_index": t.index,
                        "metric_signature": "m1",
                    }
                ]
            )
        )
        self.experiment.attach_data(data)
        self.observation_features, self.observation_data = separate_observations(
            observations_from_data(
                experiment=self.experiment, data=self.experiment.lookup_data()
            )
        )
        self.pending_observations = {
            "m2": [ObservationFeatures(parameters={"x": 0, "y": "foo", "z": True})]
        }
        self.model_gen_options: TConfig = {"option": "yes"}

    def test_fit(self) -> None:
        with mock.patch(
            "ax.generators.discrete_base.DiscreteGenerator.fit"
        ) as mock_fit:
            adapter = DiscreteAdapter(
                experiment=self.experiment,
                generator=DiscreteGenerator(),
            )
        self.assertEqual(adapter.parameters, ["x", "y", "z"])
        self.assertEqual(adapter.outcomes, ["m1", "m2"])
        Xs_array = [
            [[0.0, "foo", True], [1.0, "foo", True], [1.0, "bar", True]],  # m1
            [[0.0, "foo", True], [1.0, "foo", True]],  # m2
        ]
        Ys_array = [
            [1.0, 2.0, 3.0],  # m1
            [-1.0, -2.0],  # m2
        ]
        Yvars_array = [
            [1.0, 2.25, 1.44],  # m1
            [4.0, 9.0],  # m2
        ]
        parameter_values = [[0.0, 1.0], ["foo", "bar"], [True]]
        model_fit_args = mock_fit.call_args.kwargs
        self.assertEqual(model_fit_args["Xs"], Xs_array)
        self.assertEqual(model_fit_args["Ys"], Ys_array)
        self.assertEqual(model_fit_args["Yvars"], Yvars_array)
        self.assertEqual(model_fit_args["parameter_values"], parameter_values)

    def test_predict(self) -> None:
        with mock.patch("ax.generators.discrete_base.DiscreteGenerator.fit"):
            adapter = DiscreteAdapter(
                experiment=self.experiment,
                generator=DiscreteGenerator(),
            )
        with mock.patch.object(
            adapter.generator,
            "predict",
            return_value=(  # Matches the training data for first 2 trials.
                np.array([[1.0, -1], [2.0, -2]]),
                np.stack(
                    (
                        np.array([[1.0, 0.0], [0.0, 4.0]]),
                        np.array([[2.25, 0.0], [0.0, 9.0]]),
                    )
                ),
            ),
        ) as mock_predict:
            observation_data = adapter._predict(self.observation_features)
        X = [[0, "foo", True], [1, "foo", True], [1, "bar", True]]
        self.assertTrue(mock_predict.call_args.kwargs["X"], X)
        for i, od in enumerate(observation_data):
            self.assertEqual(od, self.observation_data[i])

    def test_gen(self) -> None:
        # Test with constraints
        optimization_config = OptimizationConfig(
            objective=Objective(Metric("m1"), minimize=True),
            outcome_constraints=[
                OutcomeConstraint(Metric("m2"), ComparisonOp.GEQ, 2, False)
            ],
        )
        with mock.patch("ax.generators.discrete_base.DiscreteGenerator.fit"):
            adapter = DiscreteAdapter(
                experiment=self.experiment,
                generator=DiscreteGenerator(),
                optimization_config=optimization_config,
            )
        # Test validation.
        with self.assertRaisesRegex(UserInputError, "positive integer or -1."):
            adapter._validate_gen_inputs(n=0)
        adapter._validate_gen_inputs(n=-1)
        # Test rest of gen.
        best_x = [0.0, 2.0, 1.0]
        return_value = (
            [[0.0, 2.0, 3.0], [1.0, 1.0, 3.0]],
            [1.0, 2.0],
            {"best_x": best_x},
        )
        with mock.patch.object(
            adapter.generator, "gen", return_value=return_value
        ) as mock_gen:
            gen_results = adapter._gen(
                n=3,
                search_space=self.search_space,
                optimization_config=optimization_config,
                pending_observations=self.pending_observations,
                fixed_features=ObservationFeatures({}),
                model_gen_options=self.model_gen_options,
            )
        gen_args = mock_gen.call_args.kwargs
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
        with mock.patch.object(
            adapter.generator, "gen", return_value=return_value
        ) as mock_gen:
            adapter._gen(
                n=3,
                search_space=search_space,
                optimization_config=optimization_config,
                pending_observations={},
                fixed_features=ObservationFeatures({}),
                model_gen_options={},
            )
        gen_args = mock_gen.call_args.kwargs
        self.assertEqual(gen_args["parameter_values"], [[0.0, 1.0], ["foo", "bar"]])
        self.assertIsNone(gen_args["outcome_constraints"])
        self.assertIsNone(gen_args["pending_observations"])

        # Test validation
        optimization_config = OptimizationConfig(
            objective=Objective(Metric("m1"), minimize=False),
            outcome_constraints=[
                OutcomeConstraint(Metric("m2"), ComparisonOp.GEQ, 2, True)
            ],
        )
        with self.assertRaisesRegex(ValueError, "relative constraint"):
            adapter._gen(
                n=3,
                search_space=search_space,
                optimization_config=optimization_config,
                pending_observations={},
                fixed_features=ObservationFeatures({}),
                model_gen_options={},
            )

    def test_cross_validate(self) -> None:
        with mock.patch("ax.generators.discrete_base.DiscreteGenerator.fit"):
            adapter = DiscreteAdapter(
                experiment=self.experiment,
                generator=DiscreteGenerator(),
            )
        return_value = (  # Matches the training data for first 2 trials.
            np.array([[1.0, -1], [2.0, -2]]),
            np.stack(
                (
                    np.array([[1.0, 0.0], [0.0, 4.0]]),
                    np.array([[2.25, 0.0], [0.0, 9.0]]),
                )
            ),
        )
        with mock.patch.object(
            adapter.generator, "cross_validate", return_value=return_value
        ) as mock_cv:
            observation_data = adapter._cross_validate(
                search_space=self.search_space,
                cv_training_data=adapter.get_training_data(),
                cv_test_points=self.observation_features,
            )

        Xs_array = [
            [[0.0, "foo", True], [1.0, "foo", True], [1.0, "bar", True]],  # m1
            [[0.0, "foo", True], [1.0, "foo", True]],  # m2
        ]
        Ys_array = [
            [1.0, 2.0, 3.0],  # m1
            [-1.0, -2.0],  # m2
        ]
        Yvars_array = [
            [1.0, 2.25, 1.44],  # m1
            [4.0, 9.0],  # m2
        ]
        Xtest = [[0, "foo", True], [1, "foo", True], [1, "bar", True]]
        model_cv_args = mock_cv.call_args.kwargs
        self.assertEqual(model_cv_args["Xs_train"], Xs_array)
        self.assertEqual(model_cv_args["Ys_train"], Ys_array)
        self.assertEqual(model_cv_args["Yvars_train"], Yvars_array)
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
