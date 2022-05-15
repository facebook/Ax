#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.risk_measures import RISK_MEASURE_NAME_TO_CLASS, RiskMeasure, VaR
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase


class TestRiskMeasure(TestCase):
    def test_risk_measure(self):
        rm = RiskMeasure(
            risk_measure="VaR",
            options={"alpha": 0.8, "n_w": 5},
        )
        self.assertEqual(rm.risk_measure, "VaR")
        self.assertEqual(rm.options, {"alpha": 0.8, "n_w": 5})
        rm_module = rm.module
        self.assertIsInstance(rm_module, VaR)
        self.assertEqual(rm_module.alpha, 0.8)
        self.assertEqual(rm_module.n_w, 5)
        self.assertFalse(rm.is_multi_output)

        # Test repr.
        expected_repr = (
            "RiskMeasure(risk_measure=VaR, options={'alpha': 0.8, 'n_w': 5})"
        )
        self.assertEqual(str(rm), expected_repr)

        # Test clone.
        rm_clone = rm.clone()
        self.assertEqual(str(rm), str(rm_clone))

        # Test unknown risk measure.
        with self.assertRaisesRegex(UserInputError, "constructing"):
            RiskMeasure(
                risk_measure="VVar",
                options={},
            )
        # Test invalid options.
        with self.assertRaisesRegex(UserInputError, "constructing"):
            RiskMeasure(
                risk_measure="VaR",
                options={"alpha": 5, "n_w": 5},
            )

    def test_custom_risk_measure(self):
        # Test using user-defined risk measures.

        class CustomRM(VaR):
            pass

        RISK_MEASURE_NAME_TO_CLASS["custom"] = CustomRM

        rm = RiskMeasure(
            risk_measure="custom",
            options={"alpha": 0.8, "n_w": 5},
        )
        self.assertEqual(rm.risk_measure, "custom")
        self.assertIsInstance(rm.module, CustomRM)
