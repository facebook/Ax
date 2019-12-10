#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import numpy as np
from ax.core.arm import Arm
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.modelbridge.array import ArrayModelBridge
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.models.numpy_base import NumpyModel
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_search_space_for_range_value
from ax.utils.testing.modeling_stubs import get_observation1


# Prepare mock transforms
class t1(Transform):
    def transform_search_space(self, ss):
        new_ss = ss.clone()
        new_ss.parameters["x"]._lower += 1.0
        new_ss.parameters["x"]._upper += 1.0
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
        new_ss.parameters["x"]._lower *= 2
        new_ss.parameters["x"]._upper *= 2
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


class ArrayModelBridgeTest(TestCase):
    @patch(
        f"{ModelBridge.__module__}.observations_from_data",
        autospec=True,
        return_value=([get_observation1()]),
    )
    @patch(
        f"{ModelBridge.__module__}.unwrap_observation_data",
        autospec=True,
        return_value=(2, 2),
    )
    @patch(
        f"{ModelBridge.__module__}.gen_arms",
        autospec=True,
        return_value=[Arm(parameters={})],
    )
    @patch(
        f"{ModelBridge.__module__}.ModelBridge.predict",
        autospec=True,
        return_value=({"m": [1.0]}, {"m": {"m": [2.0]}}),
    )
    @patch(f"{ModelBridge.__module__}.ModelBridge._fit", autospec=True)
    @patch(
        f"{NumpyModel.__module__}.NumpyModel.best_point",
        return_value=(np.array([1, 2])),
        autospec=True,
    )
    @patch(
        f"{NumpyModel.__module__}.NumpyModel.gen",
        return_value=(np.array([[1, 2]]), np.array([1]), {}),
        autospec=True,
    )
    def test_best_point(
        self,
        _mock_gen,
        _mock_best_point,
        _mock_fit,
        _mock_predict,
        _mock_gen_arms,
        _mock_unwrap,
        _mock_obs_from_data,
    ):
        exp = Experiment(get_search_space_for_range_value(), "test")
        modelbridge = ArrayModelBridge(
            get_search_space_for_range_value(), NumpyModel(), [t1, t2], exp, 0
        )
        self.assertEqual(list(modelbridge.transforms.keys()), ["t1", "t2"])
        # _fit is mocked, which typically sets this.
        modelbridge.outcomes = ["a"]
        run = modelbridge.gen(
            n=1,
            optimization_config=OptimizationConfig(
                objective=Objective(metric=Metric("a"), minimize=False),
                outcome_constraints=[],
            ),
        )
        arm, predictions = run.best_arm_predictions
        self.assertEqual(arm.parameters, {})
        self.assertEqual(predictions[0], {"m": 1.0})
        self.assertEqual(predictions[1], {"m": {"m": 2.0}})
        # test check that optimization config is required
        with self.assertRaises(ValueError):
            run = modelbridge.gen(n=1, optimization_config=None)

    @patch(
        f"{ModelBridge.__module__}.observations_from_data",
        autospec=True,
        return_value=([get_observation1()]),
    )
    @patch(
        f"{ModelBridge.__module__}.unwrap_observation_data",
        autospec=True,
        return_value=(2, 2),
    )
    @patch(
        f"{ModelBridge.__module__}.gen_arms",
        autospec=True,
        return_value=[Arm(parameters={})],
    )
    @patch(
        f"{ModelBridge.__module__}.ModelBridge.predict",
        autospec=True,
        return_value=({"m": [1.0]}, {"m": {"m": [2.0]}}),
    )
    @patch(f"{ModelBridge.__module__}.ModelBridge._fit", autospec=True)
    @patch(
        f"{NumpyModel.__module__}.NumpyModel.feature_importances",
        return_value=np.array([[[1.0]], [[2.0]]]),
        autospec=True,
    )
    def test_importances(
        self,
        _mock_feature_importances,
        _mock_fit,
        _mock_predict,
        _mock_gen_arms,
        _mock_unwrap,
        _mock_obs_from_data,
    ):
        exp = Experiment(get_search_space_for_range_value(), "test")
        modelbridge = ArrayModelBridge(
            get_search_space_for_range_value(), NumpyModel(), [t1, t2], exp, 0
        )
        modelbridge.outcomes = ["a", "b"]
        self.assertEqual(modelbridge.feature_importances("a"), {"x": [1.0]})
        self.assertEqual(modelbridge.feature_importances("b"), {"x": [2.0]})
