#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest.mock import Mock, patch

from ax.adapter.registry import Generators

from ax.exceptions.core import UserInputError
from ax.generation_strategy.best_model_selector import (
    ReductionCriterion,
    SingleDiagnosticBestModelSelector,
)
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.utils.common.testutils import TestCase


class TestBestModelSelector(TestCase):
    def setUp(self) -> None:
        super().setUp()

        # Construct a series of model specs with dummy CV diagnostics.
        self.generator_specs = []
        self.diagnostics = [
            {"Fisher exact test p": {"y_a": 0.0, "y_b": 0.4}},
            {"Fisher exact test p": {"y_a": 0.1, "y_b": 0.1}},
            {"Fisher exact test p": {"y_a": 0.5, "y_b": 0.6}},
        ]
        for diagnostics in self.diagnostics:
            ms = GeneratorSpec(generator_enum=Generators.BOTORCH_MODULAR)
            ms._cv_results = Mock()
            ms._diagnostics = diagnostics
            ms._last_cv_kwargs = {}
            self.generator_specs.append(ms)

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
        # Min/mean will pick index 1 since it has the lowest mean (0.1 vs 0.2 & 0.55).
        self.assertIs(
            s.best_model(generator_specs=self.generator_specs), self.generator_specs[1]
        )

    def test_SingleDiagnosticBestModelSelector_min_min(self) -> None:
        s = SingleDiagnosticBestModelSelector(
            diagnostic="Fisher exact test p",
            criterion=ReductionCriterion.MIN,
            metric_aggregation=ReductionCriterion.MIN,
        )
        # Min/min will pick index 0 since it has the lowest min (0.0 vs 0.1 & 0.5).
        self.assertIs(
            s.best_model(generator_specs=self.generator_specs), self.generator_specs[0]
        )

    def test_SingleDiagnosticBestModelSelector_max_mean(self) -> None:
        s = SingleDiagnosticBestModelSelector(
            diagnostic="Fisher exact test p",
            criterion=ReductionCriterion.MAX,
            metric_aggregation=ReductionCriterion.MEAN,
        )
        # Max/mean will pick index 2 since it has the largest mean (0.55 vs 0.1 & 0.2).
        self.assertIs(
            s.best_model(generator_specs=self.generator_specs), self.generator_specs[2]
        )

    def test_SingleDiagnosticBestModelSelector_model_cv_kwargs(self) -> None:
        s = SingleDiagnosticBestModelSelector(
            diagnostic="Fisher exact test p",
            criterion=ReductionCriterion.MAX,
            metric_aggregation=ReductionCriterion.MEAN,
            model_cv_kwargs={"test": "a"},
        )
        for ms in self.generator_specs:
            ms._fitted_adapter = Mock()
        with patch(
            "ax.generation_strategy.generator_spec.cross_validate",
            return_value=Mock(),
        ) as mock_cv, patch(
            "ax.generation_strategy.generator_spec.compute_diagnostics",
            side_effect=self.diagnostics,
        ):
            # Max/mean picks index 2 since it has the largest mean (0.55 vs 0.1 & 0.2).
            self.assertIs(
                s.best_model(generator_specs=self.generator_specs),
                self.generator_specs[2],
            )
        self.assertEqual(mock_cv.call_count, 3)
        for call in mock_cv.call_args_list:
            self.assertEqual(call.kwargs["test"], "a")
