#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy

import plotly.graph_objects as go
from ax.analysis.cross_validation_plot import CrossValidationPlot
from ax.analysis.helpers.cross_validation_helpers import (
    error_scatter_data_from_cv_results,
)
from ax.analysis.helpers.scatter_helpers import error_scatter_trace_from_df
from ax.modelbridge.cross_validation import cross_validate
from ax.modelbridge.registry import Models
from ax.plot.base import PlotMetric
from ax.plot.diagnostic import (
    _get_cv_plot_data as PLOT_get_cv_plot_data,
    interact_cross_validation_plotly as PLOT_interact_cross_validation_plotly,
)
from ax.plot.scatter import (
    _error_scatter_data as PLOT_error_scatter_data,
    _error_scatter_trace as PLOT_error_scatter_trace,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import fast_botorch_optimize


class TestCVConsistencyCheck(TestCase):
    @fast_botorch_optimize
    def setUp(self) -> None:
        super().setUp()
        self.exp = get_branin_experiment(with_batch=True)
        self.exp.trials[0].run()
        self.model = Models.BOTORCH_MODULAR(
            # Model bridge kwargs
            experiment=self.exp,
            data=self.exp.fetch_data(),
        )

        self.exp_status_quo = get_branin_experiment(
            with_batch=True, with_status_quo=True
        )
        self.exp_status_quo.trials[0].run()
        self.model_status_quo = Models.BOTORCH_MODULAR(
            # Model bridge kwargs
            experiment=self.exp,
            data=self.exp.fetch_data(),
        )

    def test_error_scatter_data_branin(self) -> None:
        cv_results = cross_validate(self.model)
        cv_results_plot = copy.deepcopy(cv_results)

        result_analysis = error_scatter_data_from_cv_results(
            cv_results=cv_results,
            metric_name="branin",
        )

        data = PLOT_get_cv_plot_data(cv_results_plot, label_dict={})
        result_plot = PLOT_error_scatter_data(
            list(data.in_sample.values()),
            y_axis_var=PlotMetric("branin", pred=True, rel=False),
            x_axis_var=PlotMetric("branin", pred=False, rel=False),
        )
        self.assertEqual(result_analysis, result_plot)

    def test_error_scatter_trace_branin(self) -> None:
        cv_results = cross_validate(self.model)
        cv_results_plot = copy.deepcopy(cv_results)

        cross_validation_plot = CrossValidationPlot(
            experiment=self.exp, model=self.model
        )
        df = cross_validation_plot.get_df()

        metric_filtered_df = df.loc[df["metric_name"] == "branin"]
        result_analysis = error_scatter_trace_from_df(
            df=metric_filtered_df,
            show_CI=True,
            visible=True,
            x_axis_label="Actual Outcome",
            y_axis_label="Predicted Outcome",
        )

        data = data = PLOT_get_cv_plot_data(cv_results_plot, label_dict={})

        result_plot = PLOT_error_scatter_trace(
            arms=list(data.in_sample.values()),
            hoverinfo="text",
            show_arm_details_on_hover=True,
            show_CI=True,
            show_context=False,
            status_quo_arm=None,
            visible=True,
            y_axis_var=PlotMetric("branin", pred=True, rel=False),
            x_axis_var=PlotMetric("branin", pred=False, rel=False),
            x_axis_label="Actual Outcome",
            y_axis_label="Predicted Outcome",
        )

        print(str(result_analysis))
        print(str(result_plot))

        self.assertEqual(result_analysis, result_plot)

    def test_obs_vs_pred_dropdown_plot_branin(self) -> None:

        label_dict = {"branin": "BrAnIn"}

        cross_validation_plot = CrossValidationPlot(
            experiment=self.exp, model=self.model, label_dict=label_dict
        )
        fig = cross_validation_plot.get_fig()

        self.assertIsInstance(fig, go.Figure)

        cv_results_plot = cross_validate(self.model)

        fig_PLOT = PLOT_interact_cross_validation_plotly(
            cv_results_plot,
            show_context=False,
            label_dict=label_dict,
            caption=CrossValidationPlot.CROSS_VALIDATION_CAPTION,
        )
        self.assertEqual(fig, fig_PLOT)
