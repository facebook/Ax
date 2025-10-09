# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import pandas as pd

from ax.adapter.factory import get_sobol
from ax.analysis.summary import Summary
from ax.api.client import Client
from ax.api.configs import RangeParameterConfig
from ax.core.base_trial import TrialStatus
from ax.core.trial import Trial
from ax.exceptions.core import UserInputError
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_data_batch,
    get_branin_experiment,
    get_offline_experiments,
    get_online_experiments,
)
from pyre_extensions import assert_is_instance, none_throws


class TestSummary(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.client = Client()
        self.client.configure_experiment(
            name="test_experiment",
            parameters=[
                RangeParameterConfig(
                    name="x1",
                    parameter_type="float",
                    bounds=(0, 1),
                ),
                RangeParameterConfig(
                    name="x2",
                    parameter_type="float",
                    bounds=(0, 1),
                ),
            ],
        )
        self.client.configure_optimization(objective="foo, bar")

    def test_compute(self) -> None:
        client = self.client

        # Get two trials and fail one, giving us a ragged structure
        client.get_next_trials(max_trials=2)
        client.complete_trial(trial_index=0, raw_data={"foo": 1.0, "bar": 2.0})
        client.mark_trial_failed(trial_index=1)

        analysis = Summary()

        with self.assertRaisesRegex(UserInputError, "requires an `Experiment`"):
            analysis.compute()

        experiment = client._experiment
        card = analysis.compute(experiment=experiment)

        # Test metadata
        self.assertEqual(card.name, "Summary")
        self.assertEqual(card.title, "Summary for test_experiment")
        self.assertEqual(
            card.subtitle,
            "High-level summary of the `Trial`-s in this `Experiment`",
        )
        self.assertIsNotNone(card.blob)

        # Test dataframe for accuracy
        self.assertEqual(
            {*card.df.columns},
            {
                "trial_index",
                "arm_name",
                "trial_status",
                "generation_node",
                "foo",
                "bar",
                "x1",
                "x2",
            },
        )

        trial_0_parameters = none_throws(
            assert_is_instance(experiment.trials[0], Trial).arm
        ).parameters
        trial_1_parameters = none_throws(
            assert_is_instance(experiment.trials[1], Trial).arm
        ).parameters
        expected = pd.DataFrame(
            {
                "trial_index": {0: 0, 1: 1},
                "arm_name": {0: "0_0", 1: "1_0"},
                "trial_status": {0: "COMPLETED", 1: "FAILED"},
                "generation_node": {0: "CenterOfSearchSpace", 1: "Sobol"},
                "foo": {0: 1.0, 1: np.nan},  # NaN because trial 1 failed
                "bar": {0: 2.0, 1: np.nan},
                "x1": {
                    0: trial_0_parameters["x1"],
                    1: trial_1_parameters["x1"],
                },
                "x2": {
                    0: trial_0_parameters["x2"],
                    1: trial_1_parameters["x2"],
                },
            }
        )
        pd.testing.assert_frame_equal(card.df, expected)

        # Test without omitting empty columns
        analysis_no_omit = Summary(omit_empty_columns=False)
        card_no_omit = analysis_no_omit.compute(experiment=experiment)
        self.assertEqual(
            {*card_no_omit.df.columns},
            {
                "trial_index",
                "arm_name",
                "trial_status",
                "fail_reason",
                "generation_node",
                "foo",
                "bar",
                "x1",
                "x2",
            },
        )
        self.assertEqual(len(card_no_omit.df), len(experiment.arms_by_name))

    def test_online(self) -> None:
        # Test MetricSummary can be computed for a variety of experiments which
        # resemble those we see in an online setting.

        for omit_empty_columns in [True, False]:
            analysis = Summary(omit_empty_columns=omit_empty_columns)

            for experiment in get_online_experiments():
                _ = analysis.compute(experiment=experiment)

    def test_offline(self) -> None:
        # Test MetricSummary can be computed for a variety of experiments which
        # resemble those we see in an offline setting.

        for omit_empty_columns in [True, False]:
            analysis = Summary(omit_empty_columns=omit_empty_columns)

            for experiment in get_offline_experiments():
                _ = analysis.compute(experiment=experiment)

    def test_trial_indices_filter(self) -> None:
        """Test that Client.summarize correctly uses Summary."""
        client = self.client

        # Get a trial
        client.get_next_trials(max_trials=1)
        client.complete_trial(trial_index=0, raw_data={"foo": 1.0})

        # Test summarize with trial_indices
        df_filtered = client.summarize(trial_indices=[0])
        self.assertEqual(len(df_filtered), 1)

        # Test that changes to the experiment are reflected in the summary
        client.get_next_trials(max_trials=1)
        client.complete_trial(trial_index=1, raw_data={"foo": 2.0})

    def test_trial_status_filter(self) -> None:
        """Test that Summary correctly filters by trial_status."""
        client = Client()
        client.configure_experiment(
            name="test_experiment",
            parameters=[
                RangeParameterConfig(
                    name="x1",
                    parameter_type="float",
                    bounds=(0, 1),
                ),
            ],
        )
        client.configure_optimization(objective="foo")

        # Create trials with different statuses
        client.get_next_trials(max_trials=1)
        client.complete_trial(trial_index=0, raw_data={"foo": 1.0})

        client.get_next_trials(max_trials=1)
        client.mark_trial_failed(trial_index=1)

        client.get_next_trials(max_trials=1)
        # Trial 2 remains in RUNNING state

        # Test filtering by completed status
        analysis = Summary(trial_statuses=[TrialStatus.COMPLETED])
        experiment = client._experiment
        card = analysis.compute(experiment=experiment)
        self.assertEqual(len(card.df), 1)
        self.assertEqual(card.df["trial_index"].iloc[0], 0)
        self.assertEqual(card.df["trial_status"].iloc[0], "COMPLETED")

        # Test filtering by failed status
        analysis = Summary(trial_statuses=[TrialStatus.FAILED])
        card = analysis.compute(experiment=experiment)
        self.assertEqual(len(card.df), 1)
        self.assertEqual(card.df["trial_index"].iloc[0], 1)
        self.assertEqual(card.df["trial_status"].iloc[0], "FAILED")

        # Test filtering by running status
        analysis = Summary(trial_statuses=[TrialStatus.RUNNING])
        card = analysis.compute(experiment=experiment)
        self.assertEqual(len(card.df), 1)
        self.assertEqual(card.df["trial_index"].iloc[0], 2)
        self.assertEqual(card.df["trial_status"].iloc[0], "RUNNING")

        # Test filtering by multiple statuses
        analysis = Summary(trial_statuses=[TrialStatus.COMPLETED, TrialStatus.FAILED])
        card = analysis.compute(experiment=experiment)
        self.assertEqual(len(card.df), 2)
        self.assertIn(0, card.df["trial_index"].values)
        self.assertIn(1, card.df["trial_index"].values)

    def test_default_excludes_stale_trials(self) -> None:
        """Test that Summary defaults to excluding STALE trials."""
        client = self.client

        # Create 3 trials with different statuses to test default filtering behavior
        client.get_next_trials(max_trials=3)

        # Mark trial 0 as STALE - this should be excluded from results
        stale_trial = client._experiment.trials[0]
        stale_trial.mark_stale(unsafe=True)

        # Trial 1 remains RUNNING - should be included

        # Mark trial 2 as COMPLETED - should be included
        completed_trial = client._experiment.trials[2]
        completed_trial.mark_completed()

        experiment = client._experiment

        # Compute Summary analysis with default trial_statuses setting
        analysis = Summary()
        card = analysis.compute(experiment=experiment)

        # Verify only 2 trials are included (RUNNING and COMPLETED, excluding STALE)
        self.assertEqual(len(card.df), 2)

        # Verify first result is the RUNNING trial (index 1)
        self.assertEqual(card.df["trial_index"].iloc[0], 1)
        self.assertEqual(card.df["trial_status"].iloc[0], "RUNNING")

        # Verify second result is the COMPLETED trial (index 2)
        self.assertEqual(card.df["trial_index"].iloc[1], 2)
        self.assertEqual(card.df["trial_status"].iloc[1], "COMPLETED")

        # Verify that no trials in the output have STALE status
        stale_statuses = card.df[card.df["trial_status"] == "STALE"]
        self.assertEqual(len(stale_statuses), 0)

    def test_metrics_relativized_with_status_quo(self) -> None:
        """Test that Summary relativizes metrics by default when status
        quos are present."""
        experiment = get_branin_experiment(with_status_quo=True, named=True)
        experiment.name = "test_experiment_relativize"

        # Create batch trials with status quo
        for _ in range(2):
            sobol_generator = get_sobol(search_space=experiment.search_space)
            trial = experiment.new_batch_trial(should_add_status_quo_arm=True)
            trial.add_generator_run(sobol_generator.gen(n=1))
            trial.mark_running(no_runner_required=True)
            experiment.attach_data(
                get_branin_data_batch(batch=trial, metrics=[*experiment.metrics.keys()])
            )
            trial.mark_completed()

        analysis = Summary()
        card = analysis.compute(experiment=experiment)

        # Verify subtitle indicates relativization
        self.assertIn("relativized", card.subtitle.lower())

        # Verify metric values are formatted as percentages
        metric_values = card.df["branin"].dropna()
        self.assertGreater(len(metric_values), 0)
        for val in metric_values:
            self.assertIsInstance(val, str)
            self.assertTrue(val.endswith("%"))

        # Verify relativization calculation is correct for one treatment arm
        raw_data = experiment.lookup_data().df
        sq_name = none_throws(experiment.status_quo).name
        trial_0_data = raw_data[raw_data["trial_index"] == 0]
        treatment_arm = [a for a in experiment.trials[0].arms if a.name != sq_name][0]

        sq_val = trial_0_data[trial_0_data["arm_name"] == sq_name]["mean"].values[0]
        arm_val = trial_0_data[trial_0_data["arm_name"] == treatment_arm.name][
            "mean"
        ].values[0]
        expected = ((arm_val - sq_val) / sq_val) * 100

        actual = float(
            card.df[card.df["arm_name"] == treatment_arm.name]["branin"]
            .values[0]
            .rstrip("%")
        )
        self.assertAlmostEqual(actual, expected, places=1)
