# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest

import plotly.graph_objects as go
from ax.analysis.predicted_outcomes_dot_plot import PredictedOutcomesDotPlot
from ax.exceptions.core import UnsupportedPlotError
from ax.modelbridge.registry import Models
from ax.utils.testing.core_stubs import get_branin_experiment


class TestPredictedOutcomesDotPlot(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.exp = get_branin_experiment(with_batch=True, with_status_quo=True)
        self.exp.trials[0].run()
        self.model = Models.BOTORCH_MODULAR(
            # Model bridge kwargs
            experiment=self.exp,
            data=self.exp.fetch_data(),
        )

    def test_predicted_outcomes_dot_plot_no_status_quo(self) -> None:
        exp = get_branin_experiment(with_batch=True, with_status_quo=False)
        exp.trials[0].run()
        model = Models.BOTORCH_MODULAR(
            # Model bridge kwargs
            experiment=exp,
            data=exp.fetch_data(),
        )

        with self.assertRaisesRegex(
            UnsupportedPlotError,
            "status quo must be specified for PredictedOutcomesDotPlot",
        ):
            _ = PredictedOutcomesDotPlot(
                experiment=exp,
                model=model,
            )

    def test_predicted_outcomes_dot_plot(self) -> None:
        predicted_outcomes_dot_plot = PredictedOutcomesDotPlot(
            experiment=self.exp,
            model=self.model,
        )

        _ = predicted_outcomes_dot_plot.get_df()

        fig = predicted_outcomes_dot_plot.get_fig()
        self.assertIsInstance(fig, go.Figure)
