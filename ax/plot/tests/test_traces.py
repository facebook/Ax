#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import plotly.graph_objects as go
from ax.modelbridge.registry import Models
from ax.plot.base import AxPlotConfig
from ax.plot.trace import (
    optimization_trace_single_method,
    optimization_trace_single_method_plotly,
    plot_objective_value_vs_trial_index,
)
from ax.service.utils.report_utils import exp_to_df
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import mock_botorch_optimize


class TracesTest(TestCase):
    @mock_botorch_optimize
    def setUp(self) -> None:
        super().setUp()
        self.exp = get_branin_experiment(with_batch=True)
        self.exp.trials[0].run()
        self.model = Models.BOTORCH_MODULAR(
            # Model bridge kwargs
            experiment=self.exp,
            data=self.exp.fetch_data(),
        )

    def test_Traces(self) -> None:
        # Assert that each type of plot can be constructed successfully
        plot = optimization_trace_single_method_plotly(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            list(self.model.metric_names)[0],
            optimization_direction="minimize",
            autoset_axis_limits=False,
        )
        self.assertIsInstance(plot, go.Figure)
        plot = optimization_trace_single_method(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            list(self.model.metric_names)[0],
            optimization_direction="minimize",
            autoset_axis_limits=False,
        )
        self.assertIsInstance(plot, AxPlotConfig)

    def test_TracesAutoAxes(self) -> None:
        for optimization_direction in ["minimize", "maximize", "passthrough"]:
            plot = optimization_trace_single_method_plotly(
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                list(self.model.metric_names)[0],
                optimization_direction=optimization_direction,
                autoset_axis_limits=True,
            )
            self.assertIsNone(plot.layout.xaxis.range)
            if optimization_direction == "minimize":
                self.assertAlmostEqual(plot.layout.yaxis.range[0], 0.525)
                self.assertAlmostEqual(plot.layout.yaxis.range[1], 6.225)
            elif optimization_direction == "maximize":
                self.assertAlmostEqual(plot.layout.yaxis.range[0], 0.775)
                self.assertAlmostEqual(plot.layout.yaxis.range[1], 6.475)
            else:
                self.assertIsNone(plot.layout.yaxis.range)

    @mock_botorch_optimize
    def test_plot_objective_value_vs_trial_index(self) -> None:
        # Generate some trials with different model types, including batch trial.
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].mark_completed(unsafe=True)
        sobol = Models.SOBOL(search_space=exp.search_space)
        for _ in range(2):
            t = exp.new_trial(sobol.gen(1)).run()
            t.mark_completed()
        model = Models.BOTORCH_MODULAR(
            experiment=exp,
            data=exp.fetch_data(),
        )
        for _ in range(2):
            t = exp.new_trial(model.gen(1)).run()
            t.mark_completed()
        exp.fetch_data()

        # Create exp_df and add feasibility column.
        exp_df = exp_to_df(exp)
        exp_df["is_feasible"] = True
        exp_df.loc[0, "is_feasible"] = False

        # Assert that plot can be constructed successfully.
        plot = plot_objective_value_vs_trial_index(
            exp_df=exp_df,
            metric_colname=list(self.model.metric_names)[0],
            minimize=True,
            hover_data_colnames=["trial_index"],
        )
        self.assertIsInstance(plot, go.Figure)

        # Assert that plot can be constructed successfully
        # without feasibility and generation method columns.
        del exp_df["is_feasible"]
        del exp_df["generation_method"]
        plot = plot_objective_value_vs_trial_index(
            exp_df=exp_df,
            metric_colname=list(self.model.metric_names)[0],
            minimize=True,
            hover_data_colnames=["trial_index"],
        )
        self.assertIsInstance(plot, go.Figure)
