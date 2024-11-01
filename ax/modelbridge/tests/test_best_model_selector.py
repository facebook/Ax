#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import inspect
from unittest.mock import Mock, patch

import numpy as np

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
        self.diagnostics = [
            {"Fisher exact test p": {"y_a": 0.0, "y_b": 0.4}},
            {"Fisher exact test p": {"y_a": 0.1, "y_b": 0.1}},
            {"Fisher exact test p": {"y_a": 0.5, "y_b": 0.6}},
        ]
        for diagnostics in self.diagnostics:
            ms = ModelSpec(model_enum=Models.BOTORCH_MODULAR)
            ms._cv_results = Mock()
            ms._diagnostics = diagnostics
            ms._last_cv_kwargs = {}
            self.model_specs.append(ms)

    def test_member_typing(self) -> None:
        for reduction_criterion in ReductionCriterion:
            signature = inspect.signature(reduction_criterion._get_function_for_value())
            self.assertEqual(signature.return_annotation, "np.ndarray")

    # pyre-fixme [56]: Pyre was not able to infer the type of argument
    # `numpy` to decorator factory `unittest.mock.patch`
    @patch(f"{ReductionCriterion.__module__}.np", wraps=np)
    def test_ReductionCriterion(self, mock_np: Mock) -> None:
        untested_reduction_criteria = set(ReductionCriterion)
        # Check MEAN (should just fall through to `np.mean`)
        array = np.array([1, 2, 3])  # and then use this var all the way down
        self.assertEqual(ReductionCriterion.MEAN(array), np.mean(array))
        mock_np.mean.assert_called_once()
        untested_reduction_criteria.remove(ReductionCriterion.MEAN)
        # Check MIN (should just fall through to `np.min`)
        self.assertEqual(ReductionCriterion.MIN(np.array([1, 2, 3])), 1.0)
        mock_np.min.assert_called_once()
        untested_reduction_criteria.remove(ReductionCriterion.MIN)
        # Check MAX (should just fall through to `np.max`)
        self.assertEqual(ReductionCriterion.MAX(np.array([1, 2, 3])), 3.0)
        mock_np.max.assert_called_once()
        untested_reduction_criteria.remove(ReductionCriterion.MAX)
        # There should be no untested reduction criteria left
        self.assertEqual(len(untested_reduction_criteria), 0)

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
        self.assertIs(s.best_model(model_specs=self.model_specs), self.model_specs[1])

    def test_SingleDiagnosticBestModelSelector_min_min(self) -> None:
        s = SingleDiagnosticBestModelSelector(
            diagnostic="Fisher exact test p",
            criterion=ReductionCriterion.MIN,
            metric_aggregation=ReductionCriterion.MIN,
        )
        # Min/min will pick index 0 since it has the lowest min (0.0 vs 0.1 & 0.5).
        self.assertIs(s.best_model(model_specs=self.model_specs), self.model_specs[0])

    def test_SingleDiagnosticBestModelSelector_max_mean(self) -> None:
        s = SingleDiagnosticBestModelSelector(
            diagnostic="Fisher exact test p",
            criterion=ReductionCriterion.MAX,
            metric_aggregation=ReductionCriterion.MEAN,
        )
        # Max/mean will pick index 2 since it has the largest mean (0.55 vs 0.1 & 0.2).
        self.assertIs(s.best_model(model_specs=self.model_specs), self.model_specs[2])

    def test_SingleDiagnosticBestModelSelector_model_cv_kwargs(self) -> None:
        s = SingleDiagnosticBestModelSelector(
            diagnostic="Fisher exact test p",
            criterion=ReductionCriterion.MAX,
            metric_aggregation=ReductionCriterion.MEAN,
            model_cv_kwargs={"test": "a"},
        )
        for ms in self.model_specs:
            ms._fitted_model = Mock()
        with patch(
            "ax.modelbridge.model_spec.cross_validate",
            return_value=Mock(),
        ) as mock_cv, patch(
            "ax.modelbridge.model_spec.compute_diagnostics",
            side_effect=self.diagnostics,
        ):
            # Max/mean picks index 2 since it has the largest mean (0.55 vs 0.1 & 0.2).
            self.assertIs(
                s.best_model(model_specs=self.model_specs), self.model_specs[2]
            )
        self.assertEqual(mock_cv.call_count, 3)
        for call in mock_cv.call_args_list:
            self.assertEqual(call.kwargs["test"], "a")
