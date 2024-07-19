# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

from ax.analysis.old.base_analysis import BaseAnalysis
from ax.analysis.old.base_plotly_visualization import BasePlotlyVisualization

from ax.modelbridge.registry import Models

from ax.utils.common.testutils import TestCase

from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import fast_botorch_optimize


class TestBaseClasses(TestCase):
    class TestAnalysis(BaseAnalysis):
        get_df_call_count: int = 0

        def get_df(self) -> pd.DataFrame:
            self.get_df_call_count = self.get_df_call_count + 1
            return pd.DataFrame()

    class TestPlotlyVisualization(BasePlotlyVisualization):
        get_df_call_count: int = 0
        get_fig_call_count: int = 0

        def get_df(self) -> pd.DataFrame:
            self.get_df_call_count = self.get_df_call_count + 1
            return pd.DataFrame()

        def get_fig(self) -> go.Figure:
            self.get_fig_call_count = self.get_fig_call_count + 1
            return go.Figure()

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

        self.test_analysis = self.TestAnalysis(experiment=self.exp)
        self.test_plotly_visualization = self.TestPlotlyVisualization(
            experiment=self.exp
        )

    def test_base_analysis_df_property(self) -> None:
        test_analysis = self.TestAnalysis(experiment=self.exp)

        self.assertEqual(test_analysis.get_df_call_count, 0)

        # accessing the df property calls get_df
        _ = test_analysis.df
        self.assertEqual(test_analysis.get_df_call_count, 1)

        # once saved, get_df is not called again.
        _ = test_analysis.df
        self.assertEqual(test_analysis.get_df_call_count, 1)

    def test_base_analysis_pass_df_in(self) -> None:
        existing_df = pd.DataFrame([1])
        test_analysis = self.TestAnalysis(experiment=self.exp, df_input=existing_df)

        self.assertEqual(test_analysis.get_df_call_count, 0)
        saved_df = test_analysis.df
        self.assertTrue(existing_df.equals(saved_df))

        # when df is passed in directly, get_df is never called.
        self.assertEqual(test_analysis.get_df_call_count, 0)

    def test_base_plotly_visualization_fig_property(self) -> None:
        test_analysis = self.TestPlotlyVisualization(experiment=self.exp)

        self.assertEqual(test_analysis.get_fig_call_count, 0)

        # accessing the fig property calls get_fig
        _ = test_analysis.fig
        self.assertEqual(test_analysis.get_fig_call_count, 1)

        # once saved, get_fig is not called again.
        _ = test_analysis.fig
        self.assertEqual(test_analysis.get_fig_call_count, 1)

    def test_base_plotly_visualization_pass_fig_in(self) -> None:
        fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
        test_analysis = self.TestPlotlyVisualization(
            experiment=self.exp,
            fig_input=fig,
        )

        self.assertEqual(test_analysis.get_fig_call_count, 0)
        saved_fig = test_analysis.fig
        self.assertEqual(fig, saved_fig)

        # when fig is passed in directly, get_fig is never called.
        self.assertEqual(test_analysis.get_fig_call_count, 0)

    def test_base_plotly_visualization_pass_df_and_fig(self) -> None:
        df = pd.DataFrame([1])
        fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

        test_analysis = self.TestPlotlyVisualization(
            experiment=self.exp,
            df_input=df,
            fig_input=fig,
        )

        self.assertEqual(test_analysis.get_df_call_count, 0)
        saved_df = test_analysis.df
        self.assertTrue(df.equals(saved_df))
        # when df is passed in directly, get_df is never called.
        self.assertEqual(test_analysis.get_df_call_count, 0)

        self.assertEqual(test_analysis.get_fig_call_count, 0)
        saved_fig = test_analysis.fig
        self.assertEqual(fig, saved_fig)
        # when fig is passed in directly, get_fig is never called.
        self.assertEqual(test_analysis.get_fig_call_count, 0)

    def test_instantiate_base_classes(self) -> None:
        test_analysis = BaseAnalysis(experiment=self.exp)
        with self.assertRaises(NotImplementedError):
            _ = test_analysis.df

        test_fig = BasePlotlyVisualization(experiment=self.exp)
        with self.assertRaises(NotImplementedError):
            _ = test_fig.fig
