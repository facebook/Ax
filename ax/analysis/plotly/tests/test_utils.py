# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.analysis.plotly.utils import get_arm_tooltip, trial_index_to_color
from ax.core.trial_status import TrialStatus
from ax.utils.common.testutils import TestCase


class TestUtils(TestCase):
    def test_trial_index_to_color(self) -> None:
        trials_list = [0, 1, 11]
        test_df = pd.DataFrame(
            {
                "trial_index": [0, 1, 11, 15],  # Trial 15 is a candidate trial
                "trial_status": [
                    TrialStatus.COMPLETED.name,
                    TrialStatus.COMPLETED.name,
                    TrialStatus.COMPLETED.name,
                    TrialStatus.CANDIDATE.name,
                ],
            }
        )
        # Test last completed trial is Botorch Blue
        botorch_blue_no_transparency = trial_index_to_color(
            trial_df=test_df.iloc[[2]],
            trials_list=trials_list,
            trial_index=11,
            transparent=False,
        )
        expected_color = (
            "rgba(76, 110, 243, 1)"  # RGB for Botorch Blue with no transparency
        )
        self.assertEqual(botorch_blue_no_transparency, expected_color)

        # Test last completed trial is Botorch Blue with transparency
        botorch_blue_with_transparency = trial_index_to_color(
            trial_df=test_df.iloc[[2]],
            trials_list=trials_list,
            trial_index=11,
            transparent=True,
        )
        expected_color_with_transparency = (
            "rgba(76, 110, 243, 0.5)"  # RGB for #4C6EF3 with transparency
        )
        self.assertEqual(
            botorch_blue_with_transparency, expected_color_with_transparency
        )

        # Test candidate trial is LIGHT_AX_BLUE with transparency
        candidate_light_ax_blue_with_transparency = trial_index_to_color(
            trial_df=test_df.iloc[[3]],
            trials_list=trials_list,
            trial_index=15,
            transparent=True,
        )
        expected_color_for_candidate = (
            "rgba(173, 192, 253, 0.5)"  # RGB for LIGHT_AX_BLUE with transparency
        )
        self.assertEqual(
            candidate_light_ax_blue_with_transparency, expected_color_for_candidate
        )

    def test_get_arm_tooltip(self) -> None:
        row = pd.Series(
            {
                "trial_index": 5,
                "trial_status": "COMPLETED",
                "arm_name": "5_0",
                "foo_mean": 0.450508,
                "bar_mean": -0.222466,
                "foo_sem": 0.15232,
                "bar_sem": 0.638425,
                "generation_node": "MBM",
                "p_feasible_mean": 0.1,
            }
        )

        foo_tooltip = get_arm_tooltip(row=row, metric_names=["foo"])
        self.assertEqual(
            foo_tooltip,
            "Trial: 5<br />Arm: 5_0<br />Status: COMPLETED<br />"
            "Generation Node: MBM<br />foo: 0.45051±0.29855<br />",
        )

        bar_tooltip = get_arm_tooltip(row=row, metric_names=["bar"])
        self.assertEqual(
            bar_tooltip,
            "Trial: 5<br />Arm: 5_0<br />Status: COMPLETED<br />"
            "Generation Node: MBM<br />bar: -0.22247±1.25131<br />",
        )
