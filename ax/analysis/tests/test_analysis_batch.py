# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
import plotly.graph_objects as go

from ax.analysis.analysis_batch import AnalysisBatch

from ax.analysis.base_analysis import BaseAnalysis
from ax.analysis.base_plotly_visualization import BasePlotlyVisualization

from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import fast_botorch_optimize


class TestCrossValidationPlot(TestCase):

    class TestAnalysis(BaseAnalysis):
        def get_df(self) -> pd.DataFrame:
            return pd.DataFrame()

    class TestPlotlyVisualization(BasePlotlyVisualization):
        def get_df(self) -> pd.DataFrame:
            return pd.DataFrame()

        def get_fig(self) -> go.Figure:
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

    def test_init_analysis_batch(self) -> None:
        analysis_batch = AnalysisBatch(experiment=self.exp)

        analysis_batch.add_analysis(self.test_analysis)
        analysis_batch.add_analysis(self.test_plotly_visualization)
        self.assertEqual(len(analysis_batch.analyses), 2)

        # check that dataframe is created for all analyses
        self.assertIsInstance(analysis_batch.analyses[0], BaseAnalysis)

        # check that figure is accessed only for plotly visualizations
        self.assertIsInstance(analysis_batch.analyses[1], BasePlotlyVisualization)

        self.assertIsNone(analysis_batch.time_started)
        self.assertIsNone(analysis_batch.time_completed)
        self.assertIsNone(analysis_batch.analysis_output)

    def test_execute_analysis_batch(self) -> None:
        analysis_batch = AnalysisBatch(experiment=self.exp)

        analysis_batch.add_analysis(self.test_analysis)
        analysis_batch.add_analysis(self.test_plotly_visualization)
        self.assertEqual(len(analysis_batch.analyses), 2)

        results = analysis_batch.run_analysis_batch()
        self.assertEqual(len(results), 2)

        self.assertIsNotNone(analysis_batch.time_started)
        self.assertIsNotNone(analysis_batch.time_completed)
        self.assertIsNotNone(analysis_batch.analysis_output)

        self.assertIsInstance(results[0][1], pd.DataFrame)
        self.assertIsNone(results[0][2])

        self.assertIsInstance(results[1][1], pd.DataFrame)
        self.assertIsInstance(results[1][2], go.Figure)

    def test_analysis_batch_execute_gives_saved_results(self) -> None:
        analysis_batch = AnalysisBatch(experiment=self.exp)

        analysis_batch.add_analysis(self.test_analysis)
        analysis_batch.add_analysis(self.test_plotly_visualization)
        self.assertEqual(len(analysis_batch.analyses), 2)

        _ = analysis_batch.run_analysis_batch()

        saved_start = analysis_batch.time_started
        self.assertIsNotNone(saved_start)

        # ensure analyses are not re-ran
        _ = analysis_batch.run_analysis_batch()
        self.assertEqual(saved_start, analysis_batch.time_started)

        # trying to add new analysis after batch raises exception
        with self.assertRaises(ValueError):
            analysis_batch.add_analysis(self.test_analysis)
