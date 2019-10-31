#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from ax.core.parameter import ParameterType, RangeParameter
from ax.service.utils.instantiation import (
    _get_parameter_type,
    constraint_from_str,
    outcome_constraint_from_str,
)
from ax.utils.common.testutils import TestCase


class TestInstantiationtUtils(TestCase):
    """Testing the instantiation utilities functionality that is not tested in
    main `AxClient` testing suite (`TestServiceAPI`)."""

    def test_parameter_type_validation(self):
        with self.assertRaisesRegex(ValueError, "No AE parameter type"):
            _get_parameter_type(list)

    def test_constraint_from_str(self):
        with self.assertRaisesRegex(ValueError, "Bound for the constraint"):
            constraint_from_str(
                "x1 + x2 <= not_numerical_bound", {"x1": None, "x2": None}
            )
        with self.assertRaisesRegex(ValueError, "Outcome constraint bound"):
            outcome_constraint_from_str("m1 <= not_numerical_bound")
        three_val_constaint = constraint_from_str(
            "x1 + x2 + x3 <= 3",
            {
                "x1": RangeParameter(
                    name="x1", parameter_type=ParameterType.FLOAT, lower=0.1, upper=2.0
                ),
                "x2": RangeParameter(
                    name="x2", parameter_type=ParameterType.FLOAT, lower=0.1, upper=2.0
                ),
                "x3": RangeParameter(
                    name="x3", parameter_type=ParameterType.FLOAT, lower=0.1, upper=2.0
                ),
            },
        )

        self.assertEqual(three_val_constaint.bound, 3.0)
        with self.assertRaisesRegex(ValueError, "Parameter constraint should"):
            constraint_from_str("x1 + x2 + <= 3", {"x1": None, "x2": None, "x3": None})
        with self.assertRaisesRegex(ValueError, "Parameter constraint should"):
            constraint_from_str(
                "x1 + x2 + x3 = 3", {"x1": None, "x2": None, "x3": None}
            )
        three_val_constaint = constraint_from_str(
            "x1 - x2 - x3 >= 3",
            {
                "x1": RangeParameter(
                    name="x1", parameter_type=ParameterType.FLOAT, lower=0.1, upper=4.0
                ),
                "x2": RangeParameter(
                    name="x2", parameter_type=ParameterType.FLOAT, lower=0.1, upper=4.0
                ),
                "x3": RangeParameter(
                    name="x3", parameter_type=ParameterType.FLOAT, lower=0.1, upper=4.0
                ),
            },
        )

        self.assertEqual(three_val_constaint.bound, -3.0)
        self.assertEqual(
            three_val_constaint.constraint_dict, {"x1": -1.0, "x2": 1.0, "x3": 1.0}
        )
