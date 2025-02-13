# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.analysis import AnalysisCardLevel
from ax.analysis.plotly.progression import ProgressionPlot
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_test_map_data_experiment


class TestProgression(TestCase):
    def test_compute(self) -> None:
        analysis = ProgressionPlot(metric_name="branin_map")

        with self.assertRaisesRegex(UserInputError, "requires an Experiment"):
            analysis.compute()

        experiment = get_test_map_data_experiment(
            num_trials=2, num_fetches=5, num_complete=2
        )

        card = analysis.compute(experiment=experiment)

        self.assertEqual(card.name, "ProgressionPlot")
        self.assertEqual(card.title, "branin_map by progression")
        self.assertEqual(
            card.subtitle,
            "Observe how the metric changes as each trial progresses",
        )
        self.assertEqual(card.level, AnalysisCardLevel.MID)
        self.assertEqual(
            {*card.df.columns}, {"trial_index", "arm_name", "branin_map", "progression"}
        )

        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, "plotly")
