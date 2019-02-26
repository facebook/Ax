#!/usr/bin/env python3

from unittest.mock import patch

import numpy as np
from ae.lazarus.ae.core.arm import Arm
from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.core.metric import Metric
from ae.lazarus.ae.core.objective import Objective
from ae.lazarus.ae.core.optimization_config import OptimizationConfig
from ae.lazarus.ae.generator.array import ArrayGenerator
from ae.lazarus.ae.generator.base import Generator
from ae.lazarus.ae.generator.tests.test_base_generator import (
    observation1,
    search_space_for_range_value,
)
from ae.lazarus.ae.generator.transforms.base import Transform
from ae.lazarus.ae.models.numpy_base import NumpyModel
from ae.lazarus.ae.utils.common.testutils import TestCase


# Prepare mock transforms
class t1(Transform):
    def transform_search_space(self, ss):
        new_ss = ss.clone()
        new_ss.parameters["x"]._lower += 1.0
        new_ss.parameters["x"]._upper += 1.0
        return new_ss

    def transform_optimization_config(
        self, optimization_config, generator, fixed_features
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
        self, optimization_config, generator, fixed_features
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


class ArrayGeneratorTest(TestCase):
    @patch(
        f"{Generator.__module__}.observations_from_data",
        autospec=True,
        return_value=([observation1()]),
    )
    @patch(
        f"{Generator.__module__}.unwrap_observation_data",
        autospec=True,
        return_value=(2, 2),
    )
    @patch(
        f"{Generator.__module__}.gen_arms", autospec=True, return_value=[Arm(params={})]
    )
    @patch(
        f"{Generator.__module__}.Generator.predict",
        autospec=True,
        return_value=({"m": [1.0]}, {"m": {"m": [2.0]}}),
    )
    @patch(f"{Generator.__module__}.Generator._fit", autospec=True)
    @patch(
        f"{NumpyModel.__module__}.NumpyModel.best_point",
        return_value=(np.array([1, 2])),
        autospec=True,
    )
    @patch(
        f"{NumpyModel.__module__}.NumpyModel.gen",
        return_value=(np.array([[1, 2]]), np.array([1])),
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
        exp = Experiment("test", search_space_for_range_value())
        generator = ArrayGenerator(
            search_space_for_range_value(), NumpyModel(), [t1, t2], exp, 0
        )
        self.assertEqual(list(generator.transforms.keys()), ["t1", "t2"])
        run = generator.gen(
            n=1,
            optimization_config=OptimizationConfig(
                objective=Objective(metric=Metric("a"), minimize=False),
                outcome_constraints=[],
            ),
        )
        arm, predictions = run.best_arm_predictions
        self.assertEqual(arm.params, {})
        self.assertEqual(predictions[0], {"m": 1.0})
        self.assertEqual(predictions[1], {"m": {"m": 2.0}})
