# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
import plotly.graph_objects as go

from ax.analysis.old.analysis_report import AnalysisReport

from ax.analysis.old.base_analysis import BaseAnalysis
from ax.analysis.old.base_plotly_visualization import BasePlotlyVisualization

from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase

from ax.utils.common.timeutils import current_timestamp_in_millis
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

    def test_init_analysis_report(self) -> None:
        analysis_report = AnalysisReport(
            experiment=self.exp,
            analyses=[self.test_analysis, self.test_plotly_visualization],
        )

        self.assertEqual(len(analysis_report.analyses), 2)

        self.assertIsInstance(analysis_report.analyses[0], BaseAnalysis)
        self.assertIsInstance(analysis_report.analyses[1], BasePlotlyVisualization)

        self.assertIsNone(analysis_report.time_started)
        self.assertIsNone(analysis_report.time_completed)
        self.assertFalse(analysis_report.report_completed)

    def test_execute_analysis_report(self) -> None:
        analysis_report = AnalysisReport(
            experiment=self.exp,
            analyses=[self.test_analysis, self.test_plotly_visualization],
        )

        self.assertEqual(len(analysis_report.analyses), 2)

        results = analysis_report.run_analysis_report()
        self.assertEqual(len(results), 2)

        self.assertIsNotNone(analysis_report.time_started)
        self.assertIsNotNone(analysis_report.time_completed)
        self.assertTrue(analysis_report.report_completed)

        # assert no plot is returned for BaseAnalysis
        self.assertIsInstance(results[0][1], pd.DataFrame)
        self.assertIsNone(results[0][2])

        # assert plot is returned for BasePlotlyVisualization
        self.assertIsInstance(results[1][1], pd.DataFrame)
        self.assertIsInstance(results[1][2], go.Figure)

    def test_analysis_report_repeated_execute(self) -> None:
        analysis_report = AnalysisReport(
            experiment=self.exp,
            analyses=[self.test_analysis, self.test_plotly_visualization],
        )

        self.assertEqual(len(analysis_report.analyses), 2)

        _ = analysis_report.run_analysis_report()

        saved_start = analysis_report.time_started
        self.assertIsNotNone(saved_start)

        # ensure analyses are not re-ran
        _ = analysis_report.run_analysis_report()
        self.assertEqual(saved_start, analysis_report.time_started)

    def test_no_analysis_report(self) -> None:
        analysis_report = AnalysisReport(
            experiment=self.exp,
            analyses=[],
        )

        self.assertEqual(len(analysis_report.analyses), 0)

        results = analysis_report.run_analysis_report()
        self.assertEqual(len(results), 0)

    def test_singleton_analysis_report(self) -> None:
        analysis_report = AnalysisReport(
            experiment=self.exp,
            analyses=[self.test_plotly_visualization],
        )

        self.assertEqual(len(analysis_report.analyses), 1)

        results = analysis_report.run_analysis_report()
        self.assertEqual(len(results), 1)

        # assert plot is returned for BasePlotlyVisualization
        self.assertIsInstance(results[0][1], pd.DataFrame)
        self.assertIsInstance(results[0][2], go.Figure)

    def test_create_report_as_completed(self) -> None:
        analysis_report = AnalysisReport(
            experiment=self.exp,
            analyses=[self.test_analysis, self.test_plotly_visualization],
            time_started=current_timestamp_in_millis(),
            time_completed=current_timestamp_in_millis(),
        )

        self.assertIsNotNone(analysis_report.time_started)
        self.assertIsNotNone(analysis_report.time_completed)
        self.assertTrue(analysis_report.report_completed)
