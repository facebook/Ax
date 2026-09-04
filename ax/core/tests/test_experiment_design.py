#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any

from ax.core import Experiment
from ax.core.experiment_design import (
    AutomationSettings,
    EXPERIMENT_DESIGN_KEY,
    ExperimentDesign,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_search_space


class ExperimentDesignTest(TestCase):
    """Tests covering ExperimentDesign, AutomationSettings, and their usage
    in ax Experiment.
    """

    def setUp(self) -> None:
        super().setUp()
        self.experiment = Experiment(
            name="test",
            search_space=get_branin_search_space(),
        )

    def test_default_design_and_setting_automation_settings(self) -> None:
        """Default ExperimentDesign has no automation_settings; setting design
        with AutomationSettings stores them correctly.
        """
        self.assertIsInstance(self.experiment.design, ExperimentDesign)
        self.assertEqual(self.experiment.design.automation_settings, {})
        self.assertIsNone(self.experiment.get_concurrency_limit())

        settings = AutomationSettings(
            concurrency_limit=5,
            budget=100,
        )
        design = ExperimentDesign(
            automation_settings={None: settings},
            analysis_frequency_seconds=3600,
        )
        self.experiment.design = design

        self.assertEqual(
            self.experiment.design.automation_settings[None].concurrency_limit, 5
        )
        self.assertEqual(self.experiment.design.automation_settings[None].budget, 100)
        self.assertEqual(self.experiment.design.analysis_frequency_seconds, 3600)
        self.assertEqual(self.experiment.get_concurrency_limit(), 5)

    def test_design_setter_validates_trial_types(self) -> None:
        """Setting design with unsupported trial type raises ValueError."""
        design = ExperimentDesign(
            automation_settings={
                "unsupported_type": AutomationSettings(concurrency_limit=3),
            },
        )
        with self.assertRaisesRegex(ValueError, "unsupported_type"):
            self.experiment.design = design

    def test_concurrency_limit_convenience_methods(self) -> None:
        """set_concurrency_limit creates an entry; get_concurrency_limit
        retrieves it; missing trial type with other entries present raises.
        """
        self.experiment.set_concurrency_limit(concurrency_limit=10)
        self.assertEqual(self.experiment.get_concurrency_limit(), 10)
        self.assertIn(None, self.experiment.design.automation_settings)

        with self.assertRaisesRegex(ValueError, "No AutomationSettings"):
            self.experiment.get_concurrency_limit(trial_type="nonexistent")

        # set_concurrency_limit validates trial types via the design setter.
        with self.assertRaisesRegex(ValueError, "unsupported_type"):
            self.experiment.set_concurrency_limit(
                concurrency_limit=5, trial_type="unsupported_type"
            )

    def test_serialization_roundtrip(self) -> None:
        """ExperimentDesign survives to_json / from_json and deserialization
        from experiment properties.
        """
        with self.subTest("to_json / from_json roundtrip"):
            design = ExperimentDesign(
                automation_settings={
                    None: AutomationSettings(concurrency_limit=42, budget=100),
                },
                analysis_frequency_seconds=3600,
                generation_frequency_seconds=7200,
            )
            json_dict = design.to_json()
            restored = ExperimentDesign.from_json(json_dict)
            self.assertEqual(restored.automation_settings[None].concurrency_limit, 42)
            self.assertEqual(restored.automation_settings[None].budget, 100)
            self.assertEqual(restored.analysis_frequency_seconds, 3600)
            self.assertEqual(restored.generation_frequency_seconds, 7200)

        with self.subTest("deserialization from properties"):
            properties: dict[str, Any] = {
                EXPERIMENT_DESIGN_KEY: {
                    "automation_settings": {
                        "null": {
                            "concurrency_limit": 42,
                            "generation_lookahead": None,
                            "budget": 100,
                            "stage_after_seconds": None,
                            "run_after_seconds": None,
                        },
                    },
                    "analysis_frequency_seconds": 3600,
                    "generation_frequency_seconds": None,
                },
            }
            exp = Experiment(
                name="test_new",
                search_space=get_branin_search_space(),
                properties=properties,
            )
            self.assertEqual(exp.get_concurrency_limit(), 42)
            self.assertEqual(exp.design.automation_settings[None].budget, 100)
            self.assertEqual(exp.design.analysis_frequency_seconds, 3600)
            self.assertIsNone(exp.design.generation_frequency_seconds)
