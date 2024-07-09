#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import Mock

from ax.exceptions.core import UserInputError
from ax.modelbridge.best_model_selector import (
    ReductionCriterion,
    SingleDiagnosticBestModelSelector,
)
from ax.modelbridge.model_spec import ModelSpec
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase


class TestBestModelSelector(TestCase):
    def setUp(self) -> None:
        super().setUp()

        # Construct a series of model specs with dummy CV diagnostics.
        self.model_specs = []
        for diagnostics in [
            {"Fisher exact test p": {"y_a": 0.0, "y_b": 0.4}},
            {"Fisher exact test p": {"y_a": 0.1, "y_b": 0.1}},
            {"Fisher exact test p": {"y_a": 0.5, "y_b": 0.6}},
        ]:
            ms = ModelSpec(model_enum=Models.BOTORCH_MODULAR)
            ms._cv_results = Mock()
            ms._diagnostics = diagnostics
            ms._last_cv_kwargs = {}
            self.model_specs.append(ms)

    def test_user_input_error(self) -> None:
        with self.assertRaisesRegex(UserInputError, "ReductionCriterion"):
            SingleDiagnosticBestModelSelector(
                "Fisher exact test p", metric_aggregation=min, criterion=max
            )
        with self.assertRaisesRegex(UserInputError, "use MIN or MAX"):
            SingleDiagnosticBestModelSelector(
                "Fisher exact test p",
                metric_aggregation=ReductionCriterion.MEAN,
                criterion=ReductionCriterion.MEAN,
            )

    def test_SingleDiagnosticBestModelSelector_min_mean(self) -> None:
        s = SingleDiagnosticBestModelSelector(
            diagnostic="Fisher exact test p",
            criterion=ReductionCriterion.MIN,
            metric_aggregation=ReductionCriterion.MEAN,
        )
        self.assertEqual(s.best_model(model_specs=self.model_specs), 1)

    def test_SingleDiagnosticBestModelSelector_min_min(self) -> None:
        s = SingleDiagnosticBestModelSelector(
            diagnostic="Fisher exact test p",
            criterion=ReductionCriterion.MIN,
            metric_aggregation=ReductionCriterion.MIN,
        )
        self.assertEqual(s.best_model(model_specs=self.model_specs), 0)

    def test_SingleDiagnosticBestModelSelector_max_mean(self) -> None:
        s = SingleDiagnosticBestModelSelector(
            diagnostic="Fisher exact test p",
            criterion=ReductionCriterion.MAX,
            metric_aggregation=ReductionCriterion.MEAN,
        )
        self.assertEqual(s.best_model(model_specs=self.model_specs), 2)
