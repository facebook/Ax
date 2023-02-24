#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import List, Sized

import numpy as np
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.modelbridge.transforms.choice_encode import ChoiceEncode, OrderedChoiceEncode
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast
from ax.utils.testing.core_stubs import get_robust_search_space


class ChoiceEncodeTransformTest(TestCase):
    t_class = ChoiceEncode

    def setUp(self) -> None:
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x", lower=1, upper=3, parameter_type=ParameterType.FLOAT
                ),
                RangeParameter("a", lower=1, upper=2, parameter_type=ParameterType.INT),
                ChoiceParameter(
                    "b",
                    parameter_type=ParameterType.FLOAT,
                    values=[1.0, 10.0, 100.0],
                    is_ordered=True,
                ),
                ChoiceParameter(
                    "c",
                    parameter_type=ParameterType.FLOAT,
                    values=[10.0, 100.0, 1000.0],
                    is_ordered=True,
                    sort_values=False,
                ),
                ChoiceParameter(
                    "d",
                    parameter_type=ParameterType.STRING,
                    values=["r", "q", "z"],
                    sort_values=True,
                ),
                ChoiceParameter(
                    "e",
                    parameter_type=ParameterType.STRING,
                    values=["r", "q", "z"],
                    is_ordered=False,
                    sort_values=False,
                ),
            ],
            parameter_constraints=[
                ParameterConstraint(constraint_dict={"x": -0.5, "a": 1}, bound=0.5)
            ],
        )
        self.t = self.t_class(
            search_space=self.search_space,
            observations=[],
        )
        self.observation_features = [
            ObservationFeatures(
                parameters={"x": 2.2, "a": 2, "b": 10.0, "c": 10.0, "d": "r"}
            )
        ]
        # expected parameters after transform
        self.expected_transformed_params = {
            "x": 2.2,
            "a": 2,
            # ordered float choice originally; transformed normalized value
            "b": normalize_values([1.0, 10.0, 100.0])[1],
            # ordered float choice originally; transformed normalized value
            "c": normalize_values([10.0, 100.0, 1000.0])[0],
            # string choice originally; transformed to int index.
            "d": 1,
        }

    def testInit(self) -> None:
        self.assertEqual(list(self.t.encoded_parameters.keys()), ["b", "c", "d", "e"])

    def testTransformObservationFeatures(self) -> None:
        observation_features = self.observation_features
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2,
            [ObservationFeatures(parameters=self.expected_transformed_params)],
        )
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)
        # Test transform on partial features
        obs_ft3 = [ObservationFeatures(parameters={"x": 2.2, "b": 10.0})]
        obs_ft3 = self.t.transform_observation_features(obs_ft3)
        self.assertEqual(
            obs_ft3[0],
            ObservationFeatures(
                parameters={"x": 2.2, "b": self.expected_transformed_params["b"]}
            ),
        )
        obs_ft5 = self.t.transform_observation_features([ObservationFeatures({})])
        self.assertEqual(obs_ft5[0], ObservationFeatures({}))

    def testItPreservesChoiceParameterArgs(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)
        for p in ("d", "e"):
            with self.subTest(p):
                tranformed_param = checked_cast(ChoiceParameter, ss2.parameters[p])
                original_param = checked_cast(
                    ChoiceParameter, self.search_space.parameters[p]
                )
                self.assertEqual(tranformed_param.is_ordered, original_param.is_ordered)
                self.assertEqual(
                    tranformed_param.sort_values,
                    original_param.sort_values,
                )
                if self.t_class == ChoiceEncode:
                    self.assertEqual(
                        tranformed_param.values,
                        [i for i, _ in enumerate(original_param.values)],
                    )
                else:
                    self.assertEqual(
                        tranformed_param.values,
                        original_param.values,
                    )

    def testTransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)

        for p in ("x", "a"):
            self.assertIsInstance(ss2.parameters[p], RangeParameter)
        for p in ("b", "c", "d"):
            self.assertIsInstance(ss2.parameters[p], ChoiceParameter)
        for p in ("x", "b", "c"):
            self.assertEqual(ss2.parameters[p].parameter_type, ParameterType.FLOAT)
        for p in ("a", "d"):
            self.assertEqual(ss2.parameters[p].parameter_type, ParameterType.INT)

        self.assertEqual(
            ss2.parameters["b"].values, normalize_values([1.0, 10.0, 100.0])
        )
        self.assertEqual(
            ss2.parameters["c"].values, normalize_values([10.0, 100.0, 1000.0])
        )
        self.assertEqual(ss2.parameters["d"].values, [0, 1, 2])

        # Ensure we error if we try to transform a fidelity parameter
        ss3 = SearchSpace(
            parameters=[
                ChoiceParameter(
                    "b",
                    parameter_type=ParameterType.FLOAT,
                    values=[1.0, 10.0, 100.0],
                    is_ordered=True,
                    is_fidelity=True,
                    target_value=100.0,
                )
            ]
        )
        t = OrderedChoiceEncode(search_space=ss3, observations=[])
        with self.assertRaises(ValueError):
            t.transform_search_space(ss3)

    def test_w_parameter_distributions(self) -> None:
        rss = get_robust_search_space()
        # pyre-fixme[16]: `Parameter` has no attribute `_is_ordered`.
        rss.parameters["c"]._is_ordered = True
        # Transform a non-distributional parameter.
        t = self.t_class(
            search_space=rss,
            observations=[],
        )
        rss_new = t.transform_search_space(rss)
        # Make sure that the return value is still a RobustSearchSpace.
        self.assertIsInstance(rss_new, RobustSearchSpace)
        self.assertEqual(set(rss.parameters.keys()), set(rss_new.parameters.keys()))
        # pyre-fixme[16]: `SearchSpace` has no attribute `parameter_distributions`.
        self.assertEqual(rss.parameter_distributions, rss_new.parameter_distributions)
        # pyre-fixme[16]: Optional type has no attribute `parameter_type`.
        self.assertEqual(rss_new.parameters.get("c").parameter_type, ParameterType.INT)
        # Test with environmental variables.
        all_params = list(rss.parameters.values())
        rss = RobustSearchSpace(
            parameters=all_params[2:],
            parameter_distributions=rss.parameter_distributions,
            num_samples=rss.num_samples,
            environmental_variables=all_params[:2],
        )
        t = self.t_class(
            search_space=rss,
            observations=[],
        )
        rss_new = t.transform_search_space(rss)
        self.assertIsInstance(rss_new, RobustSearchSpace)
        self.assertEqual(set(rss.parameters.keys()), set(rss_new.parameters.keys()))
        self.assertEqual(rss.parameter_distributions, rss_new.parameter_distributions)
        # pyre-fixme[16]: `SearchSpace` has no attribute `_environmental_variables`.
        self.assertEqual(rss._environmental_variables, rss_new._environmental_variables)
        self.assertEqual(rss_new.parameters.get("c").parameter_type, ParameterType.INT)


class OrderedChoiceEncodeTransformTest(ChoiceEncodeTransformTest):
    t_class = OrderedChoiceEncode

    def setUp(self) -> None:
        super().setUp()
        # expected parameters after transform
        self.expected_transformed_params = {
            "x": 2.2,
            "a": 2,
            # float choice originally; transformed to int index.
            "b": 1,
            # float choice originally; transformed to int index.
            "c": 0,
            "d": "r",
        }

    def testInit(self) -> None:
        self.assertEqual(list(self.t.encoded_parameters.keys()), ["b", "c"])

    def testTransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)

        for p in ("x", "a", "b", "c"):
            self.assertIsInstance(ss2.parameters[p], RangeParameter)
        self.assertIsInstance(ss2.parameters["d"], ChoiceParameter)
        self.assertEqual(ss2.parameters["x"].parameter_type, ParameterType.FLOAT)
        for p in ("a", "b", "c"):
            self.assertEqual(ss2.parameters[p].parameter_type, ParameterType.INT)
        self.assertEqual(ss2.parameters["d"].parameter_type, ParameterType.STRING)

        self.assertEqual(ss2.parameters["b"].lower, 0)
        self.assertEqual(ss2.parameters["b"].upper, 2)
        self.assertEqual(ss2.parameters["c"].lower, 0)
        self.assertEqual(ss2.parameters["c"].upper, 2)
        self.assertEqual(ss2.parameters["d"].values, ["q", "r", "z"])

        # Ensure we error if we try to transform a fidelity parameter
        ss3 = SearchSpace(
            parameters=[
                ChoiceParameter(
                    "b",
                    parameter_type=ParameterType.FLOAT,
                    values=[1.0, 10.0, 100.0],
                    is_ordered=True,
                    is_fidelity=True,
                    target_value=100.0,
                )
            ]
        )
        t = OrderedChoiceEncode(search_space=ss3, observations=[])
        with self.assertRaises(ValueError):
            t.transform_search_space(ss3)


def normalize_values(values: Sized) -> List[float]:
    values = np.array(values, dtype=float)
    vmin, vmax = values.min(), values.max()
    if len(values) > 1:
        values = (values - vmin) / (vmax - vmin)
    return values.tolist()
