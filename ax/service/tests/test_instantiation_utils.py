#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from ax.service.utils.instantiation import (
    _get_parameter_type,
    constraint_from_str,
    outcome_constraint_from_str,
)
from ax.utils.common.testutils import TestCase


class TestInstantiationtUtils(TestCase):
    """Testing the instantiation utilities functionality that is not tested in
    main `AxClient` testing suite (`TestSErviceAPI`)."""

    def test_parameter_type_validation(self):
        with self.assertRaisesRegex(ValueError, "No AE parameter type"):
            _get_parameter_type(list)

    def test_constraint_from_str(self):
        with self.assertRaisesRegex(ValueError, "Bound for sum constraint"):
            constraint_from_str(
                "x1 + x2 <= not_numerical_bound", {"x1": None, "x2": None}
            )
        with self.assertRaisesRegex(ValueError, "Outcome constraint bound"):
            outcome_constraint_from_str("m1 <= not_numerical_bound")
