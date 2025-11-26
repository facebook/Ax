#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy

from ax.adapter.base import DataLoaderConfig
from ax.adapter.data_utils import extract_experiment_data
from ax.adapter.transforms.choice_encode import (
    ChoiceToNumericChoice,
    OrderedChoiceToIntegerRange,
)
from ax.core.observation import ObservationFeatures
from ax.core.parameter import (
    ChoiceParameter,
    FixedParameter,
    ParameterType,
    RangeParameter,
)
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.core.types import TParameterization
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_experiment_with_observations
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from pyre_extensions import assert_is_instance


class ChoiceToNumericChoiceTransformTest(TestCase):
    t_class = ChoiceToNumericChoice

    def setUp(self) -> None:
        super().setUp()
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
        self.t = self.t_class(search_space=self.search_space)
        input_params: TParameterization = {
            "x": 2.2,
            "a": 2,
            "b": 10.0,
            "c": 10.0,
            "d": "r",
        }
        self.observation_features = [ObservationFeatures(parameters=input_params)]
        # expected parameters after transform
        self.expected_transformed_params: TParameterization = {
            **input_params,
            # string choice originally; transformed to int index.
            **{"d": 1},
        }

    def test_init(self) -> None:
        self.assertEqual(list(self.t.encoded_parameters.keys()), ["d", "e"])

    def test_transform_observation_features(self) -> None:
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

    def test_parameter_attributes_are_preserved(self) -> None:
        ss2 = deepcopy(self.search_space)
        ss2 = self.t.transform_search_space(ss2)
        for p in ("d", "e"):
            with self.subTest(p):
                transformed_param = assert_is_instance(
                    ss2.parameters[p], ChoiceParameter
                )
                original_param = assert_is_instance(
                    self.search_space.parameters[p], ChoiceParameter
                )
                self.assertEqual(
                    transformed_param.is_ordered, original_param.is_ordered
                )
                # Transformed param is numeric, so it is sorted if it is ordered.
                self.assertEqual(
                    transformed_param.sort_values,
                    transformed_param.is_ordered or original_param.sort_values,
                )
                if self.t_class == ChoiceToNumericChoice:
                    self.assertEqual(
                        transformed_param.values,
                        [i for i, _ in enumerate(original_param.values)],
                    )
                else:
                    self.assertEqual(
                        transformed_param.values,
                        original_param.values,
                    )

    def test_transform_search_space(self) -> None:
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

        for param_name in ["b", "c"]:
            self.assertEqual(
                ss2.parameters[param_name].values,
                assert_is_instance(
                    self.search_space[param_name], ChoiceParameter
                ).values,
            )
        self.assertEqual(ss2.parameters["d"].values, [0, 1, 2])

        # Fidelity parameter is transformed correctly.
        ss3 = SearchSpace(
            parameters=[
                ChoiceParameter(
                    "b",
                    parameter_type=ParameterType.STRING,
                    values=["a", "b", "c"],
                    is_ordered=True,
                    is_fidelity=True,
                    sort_values=False,
                    target_value="c",
                )
            ]
        )
        t = ChoiceToNumericChoice(search_space=ss3)
        self.assertEqual(
            t.transform_search_space(ss3.clone()),
            SearchSpace(
                parameters=[
                    ChoiceParameter(
                        "b",
                        parameter_type=ParameterType.INT,
                        values=[0, 1, 2],
                        is_ordered=True,
                        is_fidelity=True,
                        target_value=2,
                    )
                ]
            ),
        )

    def test_hss_dependents_are_preserved(self) -> None:
        # x0
        # ├── x1
        # └── x2
        #     ├── (False) EMPTY
        #     └── (True) x3
        hss = SearchSpace(
            parameters=[
                FixedParameter(
                    "x0",
                    parameter_type=ParameterType.BOOL,
                    value=True,
                    dependents={True: ["x1", "x2"]},
                ),
                RangeParameter(
                    "x1",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=1.0,
                ),
                ChoiceParameter(
                    "x2",
                    parameter_type=ParameterType.STRING,
                    values=["NO", "YES"],
                    is_ordered=True,
                    sort_values=True,
                    dependents={"NO": [], "YES": ["x3"]},
                ),
                RangeParameter(
                    "x3",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=1.0,
                ),
            ]
        )
        hss = self.t_class(search_space=hss).transform_search_space(hss)

        # x0 should be untouched because it's a fixed parameter.
        self.assertIsInstance(hss.parameters["x0"], FixedParameter)
        self.assertEqual(hss.parameters["x0"].parameter_type, ParameterType.BOOL)
        # pyre-ignore[16] # Pyre doesn't understand fixed parameters have `.value`
        self.assertEqual(hss.parameters["x0"].value, True)
        self.assertEqual(hss.parameters["x0"].dependents, {True: ["x1", "x2"]})

        self.assertFalse(hss.parameters["x1"].is_hierarchical)
        self.assertFalse(hss.parameters["x3"].is_hierarchical)

        self.assertTrue(hss.parameters["x2"].is_hierarchical)
        self.assertEqual(hss.parameters["x2"].parameter_type, ParameterType.INT)
        self.assertEqual(hss.parameters["x2"].dependents, {0: [], 1: ["x3"]})

    def test_transform_experiment_data(self) -> None:
        parameterizations = [
            {"x": 2.2, "a": 2, "b": 10.0, "c": 10.0, "d": "r", "e": "q"},
            {"x": 1.0, "a": 1, "b": 1.0, "c": 100.0, "d": "q", "e": "z"},
            {"x": 1.2, "a": 2, "b": 100.0, "c": 1000.0, "d": "z", "e": "r"},
        ]
        experiment = get_experiment_with_observations(
            observations=[[1.0], [2.0], [3.0]],
            search_space=self.search_space,
            parameterizations=parameterizations,
        )
        experiment_data = extract_experiment_data(
            experiment=experiment, data_loader_config=DataLoaderConfig()
        )
        transformed_data = self.t.transform_experiment_data(
            experiment_data=deepcopy(experiment_data)
        )

        # Check that values in arm_data are transformed as expected.
        if self.t_class is ChoiceToNumericChoice:
            expected_values = zip(
                [2.2, 1.0, 1.2],
                [2, 1, 2],
                [10.0, 1.0, 100.0],
                [10.0, 100.0, 1000.0],
                [1, 0, 2],
                [1, 2, 0],
            )
        elif self.t_class is OrderedChoiceToIntegerRange:
            expected_values = zip(
                [2.2, 1.0, 1.2],
                [2, 1, 2],
                [1, 0, 2],
                [0, 1, 2],
                ["r", "q", "z"],
                ["q", "z", "r"],
            )
        else:
            raise NotImplementedError
        expected_arm_data = DataFrame(
            [
                {"x": x, "a": a, "b": b, "c": c, "d": d, "e": e}
                for x, a, b, c, d, e in expected_values
            ],
            index=experiment_data.arm_data.index,
        )
        assert_frame_equal(
            transformed_data.arm_data.drop(columns="metadata"), expected_arm_data
        )

        # Check that observation data is unchanged.
        assert_frame_equal(
            transformed_data.observation_data, experiment_data.observation_data
        )

        # Test with no parameters transformed.
        # Setting `encoded_parameters` directly to simplify testing.
        self.t.encoded_parameters = {}
        copy_experiment_data = deepcopy(experiment_data)
        transformed_data = self.t.transform_experiment_data(
            experiment_data=copy_experiment_data
        )
        # Arm data is same as before but it is not the same object.
        assert_frame_equal(transformed_data.arm_data, experiment_data.arm_data)
        self.assertIsNot(transformed_data.arm_data, copy_experiment_data.arm_data)
        # Observation data is the same object.
        assert_frame_equal(
            transformed_data.observation_data, experiment_data.observation_data
        )
        self.assertIs(
            transformed_data.observation_data, copy_experiment_data.observation_data
        )


class OrderedChoiceToIntegerRangeTransformTest(ChoiceToNumericChoiceTransformTest):
    t_class = OrderedChoiceToIntegerRange

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

    def test_init(self) -> None:
        self.assertEqual(list(self.t.encoded_parameters.keys()), ["b", "c"])

    def test_transform_search_space(self) -> None:
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

    def test_transform_search_space_fidelity(self) -> None:
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
        t = OrderedChoiceToIntegerRange(search_space=ss3)
        with self.assertRaises(ValueError):
            t.transform_search_space(ss3)

    def test_transform_search_space_with_different_values(self) -> None:
        # Parameter with unseen values.
        ss = SearchSpace(
            parameters=[
                ChoiceParameter(
                    name="b", parameter_type=ParameterType.FLOAT, values=[5.0, 10.0]
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "contains values that are not present"):
            self.t.transform_search_space(ss)
        # Parameter that maps to a non-contiguous range.
        ss = SearchSpace(
            parameters=[
                ChoiceParameter(
                    name="b", parameter_type=ParameterType.FLOAT, values=[1.0, 100.0]
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "does not span a contiguous range"):
            self.t.transform_search_space(ss)
        # Parameter that maps to a contiguous range not starting at 0.
        ss = SearchSpace(
            parameters=[
                ChoiceParameter(
                    name="b", parameter_type=ParameterType.FLOAT, values=[10.0, 100.0]
                )
            ]
        )
        t_ss = self.t.transform_search_space(ss)
        self.assertEqual(t_ss.parameters["b"].lower, 1)
        self.assertEqual(t_ss.parameters["b"].upper, 2)

    def test_hss_dependents_are_preserved(self) -> None:
        """
        Skip the HSS test. `OrderedChoiceToIntegerRange` cannot support hierarchical
        search spaces, because range parameters cannot have dependents.
        """
