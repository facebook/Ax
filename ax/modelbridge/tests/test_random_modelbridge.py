#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from unittest import mock

import numpy as np
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.parameter_constraint import OrderConstraint, SumConstraint
from ax.core.search_space import SearchSpace
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.registry import Cont_X_trans
from ax.models.random.base import RandomModel
from ax.models.random.sobol import SobolGenerator
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_discrete_search_space


class RandomModelBridgeTest(TestCase):
    def setUp(self):
        x = RangeParameter("x", ParameterType.FLOAT, lower=0, upper=1)
        y = RangeParameter("y", ParameterType.FLOAT, lower=1, upper=2)
        z = RangeParameter("z", ParameterType.FLOAT, lower=0, upper=5)
        self.parameters = [x, y, z]
        parameter_constraints = [
            OrderConstraint(x, y),
            SumConstraint([x, z], False, 3.5),
        ]

        self.search_space = SearchSpace(self.parameters, parameter_constraints)

        self.model_gen_options = {"option": "yes"}

    @mock.patch("ax.modelbridge.random.RandomModelBridge.__init__", return_value=None)
    def testFit(self, mock_init):
        modelbridge = RandomModelBridge()
        model = mock.create_autospec(RandomModel, instance=True)
        modelbridge._fit(model, self.search_space, None, None)
        self.assertEqual(modelbridge.parameters, ["x", "y", "z"])
        self.assertTrue(isinstance(modelbridge.model, RandomModel))

    @mock.patch("ax.modelbridge.random.RandomModelBridge.__init__", return_value=None)
    def testPredict(self, mock_init):
        modelbridge = RandomModelBridge()
        modelbridge.transforms = OrderedDict()
        modelbridge.parameters = ["x", "y", "z"]
        with self.assertRaises(NotImplementedError):
            modelbridge._predict([])

    @mock.patch("ax.modelbridge.random.RandomModelBridge.__init__", return_value=None)
    def testCrossValidate(self, mock_init):
        modelbridge = RandomModelBridge()
        modelbridge.transforms = OrderedDict()
        modelbridge.parameters = ["x", "y", "z"]
        with self.assertRaises(NotImplementedError):
            modelbridge._cross_validate([], [], [])

    @mock.patch(
        "ax.models.random.base.RandomModel.gen",
        autospec=True,
        return_value=(
            np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 3.0]]),
            np.array([1.0, 2.0]),
        ),
    )
    @mock.patch("ax.modelbridge.random.RandomModelBridge.__init__", return_value=None)
    def testGen(self, mock_init, mock_gen):
        # Test with constraints
        modelbridge = RandomModelBridge()
        modelbridge.parameters = ["x", "y", "z"]
        modelbridge.transforms = OrderedDict()
        modelbridge.model = RandomModel()
        observation_features, weights, best_obsf, _ = modelbridge._gen(
            n=3,
            search_space=self.search_space,
            pending_observations={},
            fixed_features=ObservationFeatures({"z": 3.0}),
            optimization_config=None,
            model_gen_options=self.model_gen_options,
        )
        gen_args = mock_gen.mock_calls[0][2]
        self.assertEqual(gen_args["n"], 3)
        self.assertEqual(gen_args["bounds"], [(0.0, 1.0), (1.0, 2.0), (0.0, 5.0)])
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
        self.assertEqual(gen_args["model_gen_options"], {"option": "yes"})
        self.assertEqual(
            observation_features[0].parameters, {"x": 1.0, "y": 2.0, "z": 3.0}
        )
        self.assertEqual(
            observation_features[1].parameters, {"x": 3.0, "y": 4.0, "z": 3.0}
        )
        self.assertTrue(np.array_equal(weights, np.array([1.0, 2.0])))

        # Test with no constraints, no fixed feature, no pending observations
        search_space = SearchSpace(self.parameters[:2])
        modelbridge.parameters = ["x", "y"]
        modelbridge._gen(
            n=3,
            search_space=search_space,
            pending_observations={},
            fixed_features=ObservationFeatures({}),
            optimization_config=None,
            model_gen_options=self.model_gen_options,
        )
        gen_args = mock_gen.mock_calls[1][2]
        self.assertEqual(gen_args["bounds"], [(0.0, 1.0), (1.0, 2.0)])
        self.assertIsNone(gen_args["linear_constraints"])
        self.assertIsNone(gen_args["fixed_features"])

    def test_deduplicate(self):
        sobol = RandomModelBridge(
            search_space=get_discrete_search_space(),
            model=SobolGenerator(deduplicate=True),
            transforms=Cont_X_trans,
        )
        for _ in range(24):  # Search space is {[0, 3], [5, 7], {"red", "panda"}}
            self.assertEqual(len(sobol.gen(1).arms), 1)
        with self.assertRaises(ValueError):
            sobol.gen(1)
