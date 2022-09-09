#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.core.risk_measures import RiskMeasure
from ax.utils.common.testutils import TestCase


class TestRiskMeasure(TestCase):
    def test_risk_measure(self) -> None:
        rm = RiskMeasure(
            risk_measure="VaR",
            options={"alpha": 0.8, "n_w": 5},
        )
        self.assertEqual(rm.risk_measure, "VaR")
        self.assertEqual(rm.options, {"alpha": 0.8, "n_w": 5})

        # Test repr.
        expected_repr = (
            "RiskMeasure(risk_measure=VaR, options={'alpha': 0.8, 'n_w': 5})"
        )
        self.assertEqual(str(rm), expected_repr)

        # Test clone.
        rm_clone = rm.clone()
        self.assertEqual(str(rm), str(rm_clone))
