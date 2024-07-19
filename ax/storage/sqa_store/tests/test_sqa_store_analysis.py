#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger

import pandas as pd
import plotly.graph_objects as go

from ax.analysis.old.base_analysis import BaseAnalysis
from ax.analysis.old.base_plotly_visualization import BasePlotlyVisualization

from ax.analysis.old.cross_validation_plot import CrossValidationPlot

from ax.modelbridge.registry import Models

from ax.storage.sqa_store.db import init_test_engine_and_session_factory
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.load import (
    _get_generation_strategy_sqa_immutable_opt_config_and_search_space,
)

from ax.storage.sqa_store.sqa_config import SQAConfig

from ax.storage.utils import AnalysisType

from ax.utils.common.logger import get_logger

from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_experiment_with_batch_trial,
    get_range_parameter,
    get_range_parameter2,
)

from pandas.testing import assert_frame_equal

logger: Logger = get_logger(__name__)

GET_GS_SQA_IMM_FUNC = _get_generation_strategy_sqa_immutable_opt_config_and_search_space


class SQAStoreTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        init_test_engine_and_session_factory(force_init=True)
        self.config = SQAConfig()
        self.encoder = Encoder(config=self.config)
        self.decoder = Decoder(config=self.config)
        self.experiment = get_experiment_with_batch_trial()
        self.dummy_parameters = [
            get_range_parameter(),  # w
            get_range_parameter2(),  # x
        ]

        self.exp = get_branin_experiment(with_batch=True)
        self.exp.trials[0].run()
        self.model = Models.BOTORCH_MODULAR(
            # Model bridge kwargs
            experiment=self.exp,
            data=self.exp.fetch_data(),
        )

    def test_EncodeCrossValidationPlot(self) -> None:
        plot = CrossValidationPlot(experiment=self.exp, model=self.model)

        sqa_analysis = self.encoder.analysis_to_sqa(analysis=plot)

        self.assertIn("CrossValidationPlot", sqa_analysis.analysis_class_name)
        self.assertEqual(
            AnalysisType.PLOTLY_VISUALIZATION, sqa_analysis.experiment_analysis_type
        )

    def test_EncodeBaseAnalysis(self) -> None:
        analysis = BaseAnalysis(experiment=self.exp, df_input=pd.DataFrame())

        sqa_analysis = self.encoder.analysis_to_sqa(analysis=analysis)

        self.assertIn("BaseAnalysis", sqa_analysis.analysis_class_name)
        self.assertEqual(AnalysisType.ANALYSIS, sqa_analysis.experiment_analysis_type)

    def test_DecodeBaseAnalysis(self) -> None:
        df = pd.DataFrame()
        analysis = BaseAnalysis(experiment=self.exp, df_input=df)

        sqa_analysis = self.encoder.analysis_to_sqa(analysis=analysis)

        self.assertIn("BaseAnalysis", sqa_analysis.analysis_class_name)
        self.assertEqual(AnalysisType.ANALYSIS, sqa_analysis.experiment_analysis_type)

        decoded_analysis = self.decoder.analysis_from_sqa(
            experiment=self.exp, analysis_sqa=sqa_analysis
        )
        self.assertFalse(isinstance(decoded_analysis, BasePlotlyVisualization))

        # throws if not equal
        assert_frame_equal(df, decoded_analysis.df, check_dtype=False)

    def test_EncodeBasePlotlyVisualization(self) -> None:
        analysis = BasePlotlyVisualization(
            experiment=self.exp, df_input=pd.DataFrame(), fig_input=go.Figure()
        )

        sqa_analysis = self.encoder.analysis_to_sqa(analysis=analysis)

        self.assertIn("BasePlotlyVisualization", sqa_analysis.analysis_class_name)
        self.assertEqual(
            AnalysisType.PLOTLY_VISUALIZATION, sqa_analysis.experiment_analysis_type
        )

    def test_DecodeCrossValidationPlot(self) -> None:
        plot = CrossValidationPlot(experiment=self.exp, model=self.model)

        df = plot.get_df()
        fig = plot.get_fig()

        sqa_analysis = self.encoder.analysis_to_sqa(analysis=plot)

        self.assertIn("CrossValidationPlot", sqa_analysis.analysis_class_name)
        self.assertEqual(
            AnalysisType.PLOTLY_VISUALIZATION, sqa_analysis.experiment_analysis_type
        )

        decoded_plot = self.decoder.analysis_from_sqa(
            experiment=self.exp, analysis_sqa=sqa_analysis
        )
        self.assertTrue(isinstance(decoded_plot, BasePlotlyVisualization))
        decoded_fig = checked_cast(BasePlotlyVisualization, decoded_plot)
        # throws if not equal
        assert_frame_equal(df, decoded_fig.df, check_dtype=False)
        self.assertEqual(fig, decoded_fig.fig)
        # add the equal check of the plot
