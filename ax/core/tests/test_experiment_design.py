#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.core import Experiment
from ax.core.experiment import ExperimentDesign
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_search_space


class ExperimentDesignTest(TestCase):
    """Tests for ExperimentDesign and related logic."""

    def test_experiment_design_defaults(self) -> None:
        """Test that ExperimentDesign has expected defaults."""
        design = ExperimentDesign()
        self.assertIsNone(design.concurrency_limit)

    def test_experiment_design_with_concurrency_limit(self) -> None:
        """Test ExperimentDesign with concurrency_limit set."""
        design = ExperimentDesign(concurrency_limit=10)
        self.assertEqual(design.concurrency_limit, 10)

        design_zero = ExperimentDesign(concurrency_limit=0)
        self.assertEqual(design_zero.concurrency_limit, 0)

    def test_experiment_design_property(self) -> None:
        """Test that Experiment.design property returns ExperimentDesign instance."""
        experiment = Experiment(
            name="test",
            search_space=get_branin_search_space(),
        )
        self.assertIsInstance(experiment.design, ExperimentDesign)
        self.assertIsNone(experiment.design.concurrency_limit)

    def test_experiment_design_modification(self) -> None:
        """Test that ExperimentDesign can be modified after experiment creation."""
        experiment = Experiment(
            name="test",
            search_space=get_branin_search_space(),
        )
        # Modify concurrency_limit
        experiment.design.concurrency_limit = 5
        self.assertEqual(experiment.design.concurrency_limit, 5)

        # Set to None
        experiment.design.concurrency_limit = None
        self.assertIsNone(experiment.design.concurrency_limit)

    def test_experiment_design_restore_from_properties(self) -> None:
        """Test that ExperimentDesign is restored from properties during init."""
        # Simulate deserializing an experiment with design stored in properties
        experiment = Experiment(
            name="test",
            search_space=get_branin_search_space(),
            properties={"design": {"concurrency_limit": 25}},
        )
        # The design should be restored and the "design" key should be removed
        # from properties
        self.assertEqual(experiment.design.concurrency_limit, 25)
        self.assertNotIn("design", experiment._properties)

    def test_experiment_design_restore_from_properties_with_none(self) -> None:
        """Test that ExperimentDesign handles None concurrency_limit in properties."""
        experiment = Experiment(
            name="test",
            search_space=get_branin_search_space(),
            properties={"design": {"concurrency_limit": None}},
        )
        self.assertIsNone(experiment.design.concurrency_limit)
        # The "design" key should be removed from properties once it's consumed
        # to recreate `ExperimentDesign`.
        self.assertNotIn("design", experiment._properties)

    def test_experiment_design_restore_from_properties_empty_dict(self) -> None:
        """Test that ExperimentDesign handles empty design dict in properties."""
        experiment = Experiment(
            name="test",
            search_space=get_branin_search_space(),
            properties={"design": {}},
        )
        self.assertIsNone(experiment.design.concurrency_limit)
        # The "design" key should be removed from properties once it's consumed
        # to recreate `ExperimentDesign`.
        self.assertNotIn("design", experiment._properties)

    def test_experiment_design_not_affecting_other_properties(self) -> None:
        """Test that ExperimentDesign restoration doesn't affect other properties."""
        experiment = Experiment(
            name="test",
            search_space=get_branin_search_space(),
            properties={
                "design": {"concurrency_limit": 15},
                "custom_property": "custom_value",
                "another_property": 42,
            },
        )
        self.assertEqual(experiment.design.concurrency_limit, 15)
        # The "design" key should be removed from properties once it's consumed
        # to recreate `ExperimentDesign`.
        self.assertNotIn("design", experiment._properties)
        self.assertEqual(experiment._properties["custom_property"], "custom_value")
        self.assertEqual(experiment._properties["another_property"], 42)
