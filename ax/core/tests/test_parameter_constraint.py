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
        self.eq_constraint = ParameterConstraint(equality="x + y == 1.0")
        self.eq_constraint_repr = "ParameterConstraint(1.0*x + 1.0*y == 1.0)"

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
        parameters: dict[str, float | int] = {"x": 4, "z": 3}
        with self.assertRaises(ValueError):
            self.constraint.check(parameters)

        # check slack constraint
        parameters = {"x": 4, "y": 1}
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

    def test_equality_constraint_init(self) -> None:
        cases = [
            ("x + y == 1.0", {"x": 1.0, "y": 1.0}, 1.0),
            ("2 * x + 3 * y == 5.0", {"x": 2.0, "y": 3.0}, 5.0),
            ("-x + y == 1.0", {"x": -1.0, "y": 1.0}, 1.0),
            ("- x + y == 1.0", {"x": -1.0, "y": 1.0}, 1.0),
        ]
        for expr, expected_dict, expected_bound in cases:
            with self.subTest(expr=expr):
                c = ParameterConstraint(equality=expr)
                self.assertTrue(c.is_equality)
                self.assertEqual(c.constraint_dict, expected_dict)
                self.assertEqual(c.bound, expected_bound)

        c_ineq = ParameterConstraint(inequality="x + y <= 1.0")
        self.assertFalse(c_ineq.is_equality)

    def test_equality_constraint_must_provide_one(self) -> None:
        # Neither provided
        with self.assertRaisesRegex(UserInputError, "Exactly one"):
            ParameterConstraint()

        # Both provided
        with self.assertRaisesRegex(UserInputError, "Exactly one"):
            ParameterConstraint(inequality="x <= 1", equality="x == 1")

    def test_equality_constraint_check(self) -> None:
        c = ParameterConstraint(equality="x + y == 1.0")

        cases = [
            ({"x": 0.5, "y": 0.5}, True, "exact"),
            ({"x": 0.5, "y": 0.5 + 0.5e-9}, True, "within tolerance"),
            ({"x": 0.5, "y": 0.6}, False, "violated above"),
            ({"x": 0.5, "y": 0.4}, False, "violated below"),
        ]
        for params, expected, label in cases:
            with self.subTest(label=label):
                self.assertEqual(c.check(params), expected)

    def test_equality_constraint_repr(self) -> None:
        self.assertEqual(str(self.eq_constraint), self.eq_constraint_repr)

    def test_equality_constraint_clone(self) -> None:
        clone = self.eq_constraint.clone()
        self.assertTrue(clone.is_equality)
        self.assertEqual(clone.constraint_dict, self.eq_constraint.constraint_dict)
        self.assertEqual(clone.bound, self.eq_constraint.bound)

        # Mutation of clone doesn't affect original
        clone._bound = 99.0
        self.assertNotEqual(self.eq_constraint.bound, clone.bound)

    def test_equality_constraint_clone_with_transformed_parameters(self) -> None:
        clone = self.eq_constraint.clone_with_transformed_parameters(
            transformed_parameters={}
        )
        self.assertTrue(clone.is_equality)
        self.assertEqual(clone.bound, self.eq_constraint.bound)

    def test_equality_constraint_eq(self) -> None:
        c1 = ParameterConstraint(equality="x + y == 1.0")
        c2 = ParameterConstraint(equality="x + y == 1.0")
        self.assertEqual(c1, c2)

        # Different from inequality with same coefficients
        c3 = ParameterConstraint(inequality="x + y <= 1.0")
        self.assertNotEqual(c1, c3)


class ValidateConstraintParametersTest(TestCase):
    def test_validate_constraint_parameters(self) -> None:
        """Test validation of parameters used in constraints."""
        # --- Allowed parameter types ---
        allowed_cases = [
            (
                "range_parameter",
                RangeParameter(
                    name="x",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.0,
                    upper=10.0,
                ),
            ),
            (
                "numerical_ordered_int_choice",
                ChoiceParameter(
                    name="x",
                    parameter_type=ParameterType.INT,
                    values=[8, 16, 32],
                    is_ordered=True,
                    log_scale=False,
                ),
            ),
            (
                "numerical_ordered_float_choice",
                ChoiceParameter(
                    name="x",
                    parameter_type=ParameterType.FLOAT,
                    values=[0.1, 0.5, 1.0],
                    is_ordered=True,
                ),
            ),
        ]

        for name, param in allowed_cases:
            with self.subTest(name=name):
                # Should not raise
                validate_constraint_parameters(parameters=[param])

        # --- Rejected parameter types ---
        rejected_cases = [
            (
                "range_log_scale",
                RangeParameter(
                    name="x",
                    parameter_type=ParameterType.FLOAT,
                    lower=0.1,
                    upper=10.0,
                    log_scale=True,
                ),
                "log scale",
            ),
            (
                "non_numerical_choice",
                ChoiceParameter(
                    name="x",
                    parameter_type=ParameterType.STRING,
                    values=["a", "b", "c"],
                    is_ordered=True,
                ),
                "numerical ChoiceParameters",
            ),
            (
                "unordered_choice",
                ChoiceParameter(
                    name="x",
                    parameter_type=ParameterType.INT,
                    values=[8, 16, 32],
                    is_ordered=False,
                ),
                "ordered ChoiceParameters",
            ),
            (
                "choice_log_scale",
                ChoiceParameter(
                    name="x",
                    parameter_type=ParameterType.INT,
                    values=[1, 10, 100, 1000],
                    is_ordered=True,
                    log_scale=True,
                ),
                "log scale",
            ),
        ]

        for name, param, expected_error in rejected_cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, expected_error):
                    validate_constraint_parameters(parameters=[param])

        # --- Mixed parameter types ---
        with self.subTest(name="mixed_range_and_choice"):
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

        # --- Duplicate parameters ---
        with self.subTest(name="duplicate_parameters"):
            param = RangeParameter(
                name="x",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=10.0,
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                validate_constraint_parameters(parameters=[param, param])
