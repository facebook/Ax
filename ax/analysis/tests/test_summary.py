# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.analysis.analysis import AnalysisCardLevel
from ax.analysis.summary import Summary
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment_with_multi_objective


class TestSummary(TestCase):
    def test_compute(self) -> None:
        analysis = Summary()
        experiment = get_branin_experiment_with_multi_objective(
            with_completed_trial=True
        )

        with self.assertRaisesRegex(UserInputError, "requires an `Experiment`"):
            analysis.compute()

        card = analysis.compute(experiment=experiment)

        # Test metadata
        self.assertEqual(card.name, "Summary")
        self.assertEqual(card.title, "Summary for branin_test_experiment")
        self.assertEqual(
            card.subtitle,
            "High-level summary of the `Trial`-s in this `Experiment`",
        )
        self.assertEqual(card.level, AnalysisCardLevel.MID)
        self.assertIsNotNone(card.blob)
        self.assertEqual(card.blob_annotation, "dataframe")

        # Test dataframe for accuracy
        self.assertEqual(
            {*card.df.columns},
            {
                "trial_index",
                "arm_name",
                "generation_method",
                "status",
                "x1",
                "x2",
                "branin_a",
                "branin_b",
            },
        )
        self.assertEqual(len(card.df), len(experiment.arms_by_name))
        self.assertEqual(card.df.head()["trial_index"].item(), 0)
        self.assertEqual(card.df.head()["arm_name"].item(), "0_0")
        self.assertEqual(card.df.head()["generation_method"].item(), "Sobol")
        self.assertEqual(card.df.head()["status"].item(), "COMPLETED")
        self.assertEqual(
            card.df.head()["x1"].item(), experiment.arms_by_name["0_0"].parameters["x1"]
        )
        self.assertEqual(
            card.df.head()["x2"].item(), experiment.arms_by_name["0_0"].parameters["x2"]
        )
        self.assertEqual(card.df.head()["branin_a"].item(), 5.0)
        self.assertEqual(card.df.head()["branin_b"].item(), 5.0)

        # Test without omitting empty columns
        analysis_no_omit = Summary(omit_empty_columns=False)
        card_no_omit = analysis_no_omit.compute(experiment=experiment)
        self.assertEqual(
            {*card_no_omit.df.columns},
            {
                "trial_index",
                "arm_name",
                "generation_method",
                "generation_node",
                "status",
                "fail_reason",
                "x1",
                "x2",
                "branin_a",
                "branin_b",
            },
        )
        self.assertEqual(len(card_no_omit.df), len(experiment.arms_by_name))
