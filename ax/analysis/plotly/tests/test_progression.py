# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import pandas as pd
from ax.analysis.plotly.progression import (
    _calculate_wallclock_timeseries,
    ProgressionPlot,
)
from ax.core.data import Data, MAP_KEY
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_test_map_data_experiment,
)
from pyre_extensions import none_throws


class TestProgression(TestCase):
    def test_validate_applicable_state(self) -> None:
        self.assertIn(
            "Requires an Experiment",
            none_throws(
                ProgressionPlot(metric_name="branin_map").validate_applicable_state()
            ),
        )

        with self.subTest("No 'step' data"):
            experiment = get_branin_experiment(
                with_trial=True, with_completed_trial=True
            )
            plot = ProgressionPlot(metric_name="branin")
            state = plot.validate_applicable_state(experiment=experiment)
            self.assertEqual(state, "Requires data to have a column 'step.'")

        with self.subTest("All step values are NaN"):
            # Create a new experiment with map data where all MAP_KEY values are NaN
            experiment = get_test_map_data_experiment(
                num_trials=2, num_fetches=3, num_complete=2
            )

            # Replace all MAP_KEY values with NaN in the fetched data
            original_data = experiment.fetch_data()
            modified_df = original_data.full_df.copy()
            modified_df[MAP_KEY] = np.nan

            # Create a new Data object and attach it
            nan_data = Data(df=modified_df)
            experiment.data = nan_data

            # Validate that progression plot is not applicable
            plot = ProgressionPlot(metric_name="branin_map")
            state = plot.validate_applicable_state(experiment=experiment)
            self.assertEqual(
                state, "All progression values for metric 'branin_map' are NaN."
            )

    def test_compute(self) -> None:
        analysis = ProgressionPlot(metric_name="branin_map")

        experiment = get_test_map_data_experiment(
            num_trials=2, num_fetches=5, num_complete=2
        )

        card = analysis.compute(experiment=experiment)

        self.assertEqual(card.name, "ProgressionPlot")
        self.assertEqual(card.title, "branin_map by progression")
        self.assertEqual(
            card.subtitle,
            (
                "The progression plot tracks the evolution of each metric "
                "over the course of the experiment. This visualization is typically "
                "used to monitor the improvement of metrics over Trial iterations, "
                "but can also be useful in informing decisions about early stopping "
                "for Trials."
            ),
        )
        self.assertEqual(
            {*card.df.columns},
            {"trial_index", "arm_name", "branin_map", "progression", "wallclock_time"},
        )

        self.assertIsNotNone(card.blob)

    def test_calculate_wallclock_timeseries(self) -> None:
        experiment = get_test_map_data_experiment(
            num_trials=2, num_fetches=5, num_complete=2
        )
        wallclock_timeseries = _calculate_wallclock_timeseries(
            experiment=experiment, metric_name="branin_map"
        )

        self.assertEqual(len(wallclock_timeseries), 2)
        self.assertTrue(
            all(len(timeseries) == 5 for timeseries in wallclock_timeseries.values())
        )

        for timeseries in wallclock_timeseries.values():
            self.assertTrue(pd.Series(timeseries).is_monotonic_increasing)
