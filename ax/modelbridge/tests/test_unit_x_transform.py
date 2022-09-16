#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import numpy as np
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.modelbridge.transforms.unit_x import UnitX
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_robust_search_space


class UnitXTransformTest(TestCase):

    transform_class = UnitX
    # pyre-fixme[4]: Attribute must be annotated.
    expected_c_dicts = [{"x": -1.0, "y": 1.0}, {"x": -1.0, "a": 1.0}]
    expected_c_bounds = [0.0, 1.0]

    def setUp(self) -> None:
        self.target_lb = self.transform_class.target_lb
        self.target_range = self.transform_class.target_range
        self.target_ub = self.target_lb + self.target_range
        self.search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    "x", lower=1, upper=3, parameter_type=ParameterType.FLOAT
                ),
                RangeParameter(
                    "y", lower=1, upper=2, parameter_type=ParameterType.FLOAT
                ),
                RangeParameter(
                    "z",
                    lower=1,
                    upper=2,
                    parameter_type=ParameterType.FLOAT,
                    log_scale=True,
                ),
                RangeParameter("a", lower=1, upper=2, parameter_type=ParameterType.INT),
                ChoiceParameter(
                    "b", parameter_type=ParameterType.STRING, values=["a", "b", "c"]
                ),
            ],
            parameter_constraints=[
                ParameterConstraint(constraint_dict={"x": -0.5, "y": 1}, bound=0.5),
                ParameterConstraint(constraint_dict={"x": -0.5, "a": 1}, bound=0.5),
            ],
        )
        self.t = self.transform_class(
            search_space=self.search_space,
            observations=[],
        )
        self.search_space_with_target = SearchSpace(
            parameters=[
                RangeParameter(
                    "x",
                    lower=1,
                    upper=3,
                    parameter_type=ParameterType.FLOAT,
                    is_fidelity=True,
                    target_value=3,
                )
            ]
        )

    def testInit(self) -> None:
        self.assertEqual(self.t.bounds, {"x": (1.0, 3.0), "y": (1.0, 2.0)})

    def testTransformObservationFeatures(self) -> None:
        observation_features = [
            ObservationFeatures(parameters={"x": 2, "y": 2, "z": 2, "a": 2, "b": "b"})
        ]
        obs_ft2 = deepcopy(observation_features)
        obs_ft2 = self.t.transform_observation_features(obs_ft2)
        self.assertEqual(
            obs_ft2,
            [
                ObservationFeatures(
                    parameters={
                        "x": self.target_lb + self.target_range / 2.0,
                        "y": 1.0,
                        "z": 2,
                        "a": 2,
                        "b": "b",
                    }
                )
            ],
        )
        obs_ft2 = self.t.untransform_observation_features(obs_ft2)
        self.assertEqual(obs_ft2, observation_features)
        # Test transform partial observation
        obs_ft3 = [ObservationFeatures(parameters={"x": 3.0, "z": 2})]
        obs_ft3 = self.t.transform_observation_features(obs_ft3)
        self.assertEqual(
            obs_ft3[0],
            ObservationFeatures(parameters={"x": self.target_ub, "z": 2}),
        )
        obs_ft5 = self.t.transform_observation_features([ObservationFeatures({})])
        self.assertEqual(obs_ft5[0], ObservationFeatures({}))

    def testTransformSearchSpace(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)

        # Parameters transformed
        true_bounds = {
            "x": (self.target_lb, 1.0),
            "y": (self.target_lb, 1.0),
            "z": (1.0, 2.0),
            "a": (1.0, 2.0),
        }
        for p_name, (l, u) in true_bounds.items():
            self.assertEqual(ss2.parameters[p_name].lower, l)
            self.assertEqual(ss2.parameters[p_name].upper, u)
        self.assertEqual(ss2.parameters["b"].values, ["a", "b", "c"])
        self.assertEqual(len(ss2.parameters), 5)
        # Constraints transformed
        self.assertEqual(
            ss2.parameter_constraints[0].constraint_dict, self.expected_c_dicts[0]
        )
        self.assertEqual(ss2.parameter_constraints[0].bound, self.expected_c_bounds[0])
        self.assertEqual(
            ss2.parameter_constraints[1].constraint_dict, self.expected_c_dicts[1]
        )
        self.assertEqual(ss2.parameter_constraints[1].bound, self.expected_c_bounds[1])

        # Test transform of target value
        t = self.transform_class(
            search_space=self.search_space_with_target,
            observations=[],
        )
        t.transform_search_space(self.search_space_with_target)
        self.assertEqual(
            self.search_space_with_target.parameters["x"].target_value, 1.0
        )

    def test_w_robust_search_space_univariate(self) -> None:
        # Check that if no transforms are needed, it is untouched.
        for multivariate in (True, False):
            rss = get_robust_search_space(
                multivariate=multivariate,
                lb=self.target_lb,
                ub=self.target_ub,
            )
            expected = str(rss)
            t = self.transform_class(
                search_space=rss,
                observations=[],
            )
            self.assertEqual(expected, str(t.transform_search_space(rss)))
        # Error if distribution is multiplicative.
        rss = get_robust_search_space()
        rss.parameter_distributions[0].multiplicative = True
        t = self.transform_class(
            search_space=rss,
            observations=[],
        )
        with self.assertRaisesRegex(NotImplementedError, "multiplicative"):
            t.transform_search_space(rss)
        # Correctly transform univariate additive distributions.
        rss = get_robust_search_space(lb=5.0, ub=10.0)
        t = self.transform_class(
            search_space=rss,
            observations=[],
        )
        t.transform_search_space(rss)
        dists = rss.parameter_distributions
        self.assertEqual(
            dists[0].distribution_parameters["loc"], 0.2 * self.target_range
        )
        self.assertEqual(dists[0].distribution_parameters["scale"], self.target_range)
        self.assertEqual(dists[1].distribution_parameters["loc"], 0.0)
        self.assertEqual(
            dists[1].distribution_parameters["scale"], 0.2 * self.target_range
        )
        # Correctly transform environmental distributions.
        rss = get_robust_search_space(lb=5.0, ub=10.0)
        all_parameters = list(rss.parameters.values())
        rss = RobustSearchSpace(
            parameters=all_parameters[1:],
            parameter_distributions=rss.parameter_distributions[:1],
            num_samples=rss.num_samples,
            environmental_variables=all_parameters[:1],
        )
        t.transform_search_space(rss)
        dist = rss.parameter_distributions[0]
        self.assertEqual(
            dist.distribution_parameters["loc"],
            t._normalize_value(1.0, (5.0, 10.0)),
        )
        self.assertEqual(dist.distribution_parameters["scale"], self.target_range)
        # Error if transform via loc / scale is not supported.
        rss = get_robust_search_space(use_discrete=True)
        rss.parameters["z"]._parameter_type = ParameterType.FLOAT
        t = self.transform_class(
            search_space=rss,
            observations=[],
        )
        with self.assertRaisesRegex(UnsupportedError, "`loc` and `scale`"):
            t.transform_search_space(rss)

    def test_w_robust_search_space_multivariate(self) -> None:
        # Error if trying to transform non-normal multivariate distributions.
        rss = get_robust_search_space(multivariate=True)
        rss.parameter_distributions[0].distribution_class = "multivariate_t"
        t = self.transform_class(
            search_space=rss,
            observations=[],
        )
        with self.assertRaisesRegex(UnsupportedError, "multivariate"):
            t.transform_search_space(rss)
        # Transform multivariate normal.
        rss = get_robust_search_space(multivariate=True)
        old_params = deepcopy(rss.parameter_distributions[0].distribution_parameters)
        t = self.transform_class(
            search_space=rss,
            observations=[],
        )
        t.transform_search_space(rss)
        new_params = rss.parameter_distributions[0].distribution_parameters
        self.assertIsInstance(new_params["mean"], np.ndarray)
        self.assertIsInstance(new_params["cov"], np.ndarray)
        self.assertTrue(
            np.allclose(
                new_params["mean"],
                np.asarray(old_params["mean"]) / 5.0 * self.target_range,
            )
        )
        self.assertTrue(
            np.allclose(
                new_params["cov"],
                np.asarray(old_params["cov"]) / ((5.0 / self.target_range) ** 2),
            )
        )
        # Transform multivariate normal environmental distribution.
        rss = get_robust_search_space(multivariate=True)
        rss_params = list(rss.parameters.values())
        rss = RobustSearchSpace(
            parameters=rss_params[2:],
            parameter_distributions=rss.parameter_distributions,
            num_samples=rss.num_samples,
            environmental_variables=rss_params[:2],
        )
        t = self.transform_class(
            search_space=rss,
            observations=[],
        )
        t.transform_search_space(rss)
        new_params = rss.parameter_distributions[0].distribution_parameters
        self.assertTrue(
            np.allclose(
                new_params["mean"],
                np.asarray(old_params["mean"]) / 5.0 * self.target_range
                + self.target_lb,
            )
        )
        # Errors if mean / cov are of wrong shape.
        rss.parameter_distributions[0].distribution_parameters["mean"] = [1.0]
        with self.assertRaisesRegex(UserInputError, "mean"):
            t.transform_search_space(rss)
        rss.parameter_distributions[0].distribution_parameters["mean"] = [1.0, 1.0]
        rss.parameter_distributions[0].distribution_parameters["cov"] = [1.0]
        with self.assertRaisesRegex(UserInputError, "cov"):
            t.transform_search_space(rss)
