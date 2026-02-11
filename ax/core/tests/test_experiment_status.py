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

    def test_is_active(self) -> None:
        """Test the is_active property."""
        # Active statuses
        self.assertTrue(ExperimentStatus.INITIALIZATION.is_active)
        self.assertTrue(ExperimentStatus.OPTIMIZATION.is_active)

        # Inactive statuses
        self.assertFalse(ExperimentStatus.DRAFT.is_active)
        self.assertFalse(ExperimentStatus.COMPLETED.is_active)

    def test_individual_status_checks(self) -> None:
        """Test individual status check properties."""
        self.assertTrue(ExperimentStatus.DRAFT.is_draft)
        self.assertFalse(ExperimentStatus.INITIALIZATION.is_draft)

        self.assertTrue(ExperimentStatus.INITIALIZATION.is_initialization)
        self.assertFalse(ExperimentStatus.OPTIMIZATION.is_initialization)

        self.assertTrue(ExperimentStatus.OPTIMIZATION.is_optimization)
        self.assertFalse(ExperimentStatus.COMPLETED.is_optimization)

        self.assertTrue(ExperimentStatus.COMPLETED.is_completed)
        self.assertFalse(ExperimentStatus.DRAFT.is_completed)

    def test_format_and_repr(self) -> None:
        """Test __format__ and __repr__ methods."""
        status = ExperimentStatus.DRAFT
        self.assertEqual(f"{status}", "ExperimentStatus.DRAFT")
        self.assertEqual(repr(status), "ExperimentStatus.DRAFT")

        status = ExperimentStatus.OPTIMIZATION
        self.assertEqual(f"{status}", "ExperimentStatus.OPTIMIZATION")
        self.assertEqual(repr(status), "ExperimentStatus.OPTIMIZATION")
