#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core.experiment_status import ExperimentStatus
from ax.utils.common.testutils import TestCase


class TestExperimentStatus(TestCase):
    """Tests for the ExperimentStatus enum."""

    def test_status_values(self) -> None:
        """Test that status enum values are correctly defined."""
        self.assertEqual(ExperimentStatus.DRAFT.value, 0)
        self.assertEqual(ExperimentStatus.INITIALIZATION.value, 1)
        self.assertEqual(ExperimentStatus.OPTIMIZATION.value, 2)
        self.assertEqual(ExperimentStatus.COMPLETED.value, 4)

    def test_status_boolean_properties(self) -> None:
        """Test is_active and individual status check properties."""
        cases = [
            (ExperimentStatus.INITIALIZATION, "is_active", True),
            (ExperimentStatus.OPTIMIZATION, "is_active", True),
            (ExperimentStatus.DRAFT, "is_active", False),
            (ExperimentStatus.COMPLETED, "is_active", False),
            (ExperimentStatus.DRAFT, "is_draft", True),
            (ExperimentStatus.INITIALIZATION, "is_draft", False),
            (ExperimentStatus.INITIALIZATION, "is_initialization", True),
            (ExperimentStatus.OPTIMIZATION, "is_initialization", False),
            (ExperimentStatus.OPTIMIZATION, "is_optimization", True),
            (ExperimentStatus.COMPLETED, "is_optimization", False),
            (ExperimentStatus.COMPLETED, "is_completed", True),
            (ExperimentStatus.DRAFT, "is_completed", False),
        ]
        for status, prop, expected in cases:
            with self.subTest(status=status.name, prop=prop):
                self.assertEqual(getattr(status, prop), expected)

    def test_format_and_repr(self) -> None:
        """Test __format__ and __repr__ methods."""
        cases = [
            (ExperimentStatus.DRAFT, "ExperimentStatus.DRAFT"),
            (ExperimentStatus.OPTIMIZATION, "ExperimentStatus.OPTIMIZATION"),
        ]
        for status, expected_str in cases:
            with self.subTest(status=status.name):
                self.assertEqual(f"{status}", expected_str)
                self.assertEqual(repr(status), expected_str)
