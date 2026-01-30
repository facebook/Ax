#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.parameter_constraint import (
    ParameterConstraint,
    validate_constraint_parameters,
)
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase


class ParameterConstraintTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.constraint = ParameterConstraint(inequality="2 * x - 3 * y <= 6.0")
        self.constraint_repr = "ParameterConstraint(2.0*x + -3.0*y <= 6.0)"

    def test_constraint_dict_and_bounds(self) -> None:
        constraint = ParameterConstraint(inequality="x1 + x2 <= 1")

        self.assertEqual(
            constraint.constraint_dict,
            {"x1": 1, "x2": 1},
        )
        self.assertEqual(constraint.bound, 1.0)

        with_coefficients = ParameterConstraint(inequality="2 * x1 + 3 * x2 <= 1")
        self.assertEqual(
            with_coefficients.constraint_dict,
            {"x1": 2, "x2": 3},
        )
        self.assertEqual(with_coefficients.bound, 1.0)

        flipped_sign = ParameterConstraint(inequality="x1 + x2 >= 1")
        self.assertEqual(
            flipped_sign.constraint_dict,
            {"x1": -1, "x2": -1},
        )
        self.assertEqual(flipped_sign.bound, -1.0)

        weird = ParameterConstraint(inequality="x1 + x2 <= 1.5 * x3 + 2")
        self.assertEqual(
            weird.constraint_dict,
            {"x1": 1, "x2": 1, "x3": -1.5},
        )
        self.assertEqual(weird.bound, 2.0)

        with self.assertRaisesRegex(UserInputError, "Only linear"):
            ParameterConstraint(inequality="x1 * x2 <= 1")

        # test with sanitization
        constraint = ParameterConstraint(inequality="foo.bar + foo.baz <= 1")
        self.assertEqual(
            constraint.constraint_dict,
            {"foo.bar": 1, "foo.baz": 1},
        )
        self.assertEqual(constraint.bound, 1.0)

    def test_Eq(self) -> None:
        constraint1 = ParameterConstraint(
            inequality="2 * x - 3 * y <= 6.0",
        )
        constraint2 = ParameterConstraint(inequality="2 * x - 3 * y <= 6.0")
        self.assertEqual(constraint1, constraint2)

        constraint3 = ParameterConstraint(
            inequality="2 * x - 5 * y <= 6.0",
        )
        self.assertNotEqual(constraint1, constraint3)

    def test_Properties(self) -> None:
        self.assertEqual(self.constraint.constraint_dict["x"], 2.0)
        self.assertEqual(self.constraint.bound, 6.0)

    def test_Repr(self) -> None:
        self.assertEqual(str(self.constraint), self.constraint_repr)

    def test_Validate(self) -> None:
        parameters = {"x": 4, "z": 3}
        with self.assertRaises(ValueError):
            # pyre-fixme[6]: For 1st param expected `Dict[str, Union[float, int]]`
            #  but got `Dict[str, int]`.
            self.constraint.check(parameters)

        # check slack constraint
        parameters = {"x": 4, "y": 1}
        # pyre-fixme[6]: For 1st param expected `Dict[str, Union[float, int]]` but
        #  got `Dict[str, int]`.
        self.assertTrue(self.constraint.check(parameters))

        # check tight constraint (within numerical tolerance)
        parameters = {"x": 4, "y": (2 - 0.5e-8) / 3}
        self.assertTrue(self.constraint.check(parameters))

        # check violated constraint
        parameters = {"x": 4, "y": (2 - 0.5e-6) / 3}
        self.assertFalse(self.constraint.check(parameters))

    def test_Clone(self) -> None:
        constraint_clone = self.constraint.clone()
        self.assertEqual(self.constraint.bound, constraint_clone.bound)

        constraint_clone._bound = 7.0
        self.assertNotEqual(self.constraint.bound, constraint_clone.bound)

    def test_CloneWithTransformedParameters(self) -> None:
        constraint_clone = self.constraint.clone_with_transformed_parameters(
            transformed_parameters={}
        )
        self.assertEqual(self.constraint.bound, constraint_clone.bound)

        constraint_clone._bound = 7.0
        self.assertNotEqual(self.constraint.bound, constraint_clone.bound)

    def test_Sortable(self) -> None:
        constraint1 = ParameterConstraint(
            inequality="2 * x - 3 * y <= 1.0",
        )
        constraint2 = ParameterConstraint(
            inequality="2 * x - 3 * y <= 6.0",
        )
        self.assertTrue(constraint1 < constraint2)


class ValidateConstraintParametersTest(TestCase):
    def test_range_parameter_allowed(self) -> None:
        """Test that RangeParameter is allowed in constraints."""
        param = RangeParameter(
            name="x",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=10.0,
        )
        # Should not raise
        validate_constraint_parameters(parameters=[param])

    def test_range_parameter_log_scale_rejected(self) -> None:
        """Test that log-scale RangeParameter is rejected."""
        param = RangeParameter(
            name="x",
            parameter_type=ParameterType.FLOAT,
            lower=0.1,
            upper=10.0,
            log_scale=True,
        )
        with self.assertRaisesRegex(ValueError, "log scale"):
            validate_constraint_parameters(parameters=[param])

    def test_numerical_ordered_choice_parameter_allowed(self) -> None:
        """Test that numerical ordered ChoiceParameter is allowed in constraints."""
        param = ChoiceParameter(
            name="x",
            parameter_type=ParameterType.INT,
            values=[8, 16, 32],
            is_ordered=True,
            log_scale=False,
        )
        # Should not raise
        validate_constraint_parameters(parameters=[param])

    def test_numerical_ordered_float_choice_parameter_allowed(self) -> None:
        """Test that numerical ordered float ChoiceParameter is allowed."""
        param = ChoiceParameter(
            name="x",
            parameter_type=ParameterType.FLOAT,
            values=[0.1, 0.5, 1.0],
            is_ordered=True,
        )
        # Should not raise
        validate_constraint_parameters(parameters=[param])

    def test_non_numerical_choice_parameter_rejected(self) -> None:
        """Test that non-numerical ChoiceParameter is rejected."""
        param = ChoiceParameter(
            name="x",
            parameter_type=ParameterType.STRING,
            values=["a", "b", "c"],
            is_ordered=True,
        )
        with self.assertRaisesRegex(ValueError, "numerical ChoiceParameters"):
            validate_constraint_parameters(parameters=[param])

    def test_unordered_choice_parameter_rejected(self) -> None:
        """Test that unordered numerical ChoiceParameter is rejected."""
        param = ChoiceParameter(
            name="x",
            parameter_type=ParameterType.INT,
            values=[8, 16, 32],
            is_ordered=False,
        )
        with self.assertRaisesRegex(ValueError, "ordered ChoiceParameters"):
            validate_constraint_parameters(parameters=[param])

    def test_log_scale_choice_parameter_rejected(self) -> None:
        """Test that log-scale ChoiceParameter is rejected."""
        param = ChoiceParameter(
            name="x",
            parameter_type=ParameterType.INT,
            values=[1, 10, 100, 1000],
            is_ordered=True,
            log_scale=True,
        )
        with self.assertRaisesRegex(ValueError, "log scale"):
            validate_constraint_parameters(parameters=[param])

    def test_mixed_range_and_choice_parameters_allowed(self) -> None:
        """Test that mixed RangeParameter and ChoiceParameter constraints work."""
        range_param = RangeParameter(
            name="x",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=10.0,
        )
        choice_param = ChoiceParameter(
            name="y",
            parameter_type=ParameterType.INT,
            values=[8, 16, 32],
            is_ordered=True,
            log_scale=False,
        )
        # Should not raise
        validate_constraint_parameters(parameters=[range_param, choice_param])

    def test_duplicate_parameters_rejected(self) -> None:
        """Test that duplicate parameters are rejected."""
        param = RangeParameter(
            name="x",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=10.0,
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_constraint_parameters(parameters=[param, param])
