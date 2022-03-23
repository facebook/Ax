#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.modelbridge.base import ModelBridge
from ax.modelbridge.registry import Models
from ax.plot.base import AxPlotConfig
from ax.plot.feature_importances import (
    plot_feature_importance_by_feature,
    plot_feature_importance_by_feature_plotly,
    plot_feature_importance_by_metric,
    plot_feature_importance_by_metric_plotly,
    plot_relative_feature_importance,
    plot_relative_feature_importance_plotly,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import fast_modeling
from plotly import graph_objects as go

DUMMY_CAPTION = "test_caption"


def get_modelbridge() -> ModelBridge:
    exp = get_branin_experiment(with_batch=True)
    exp.trials[0].run()
    return Models.BOTORCH(
        # Model bridge kwargs
        experiment=exp,
        data=exp.fetch_data(),
    )


class FeatureImportancesTest(TestCase):
    @fast_modeling
    def testFeatureImportances(self):
        model = get_modelbridge()
        # Assert that each type of plot can be constructed successfully
        plot = plot_feature_importance_by_feature_plotly(model=model)
        self.assertIsInstance(plot, go.Figure)
        plot = plot_feature_importance_by_feature_plotly(
            model=model, caption=DUMMY_CAPTION
        )
        self.assertIsInstance(plot, go.Figure)
        self.assertEqual(len(plot.layout.annotations), 1)
        self.assertEqual(plot.layout.annotations[0].text, DUMMY_CAPTION)
        plot = plot_feature_importance_by_feature(model=model)
        self.assertIsInstance(plot, AxPlotConfig)
        plot = plot_feature_importance_by_metric_plotly(model=model)
        self.assertIsInstance(plot, go.Figure)
        plot = plot_feature_importance_by_metric(model=model)
        self.assertIsInstance(plot, AxPlotConfig)
        plot = plot_relative_feature_importance_plotly(model=model)
        self.assertIsInstance(plot, go.Figure)
        plot = plot_relative_feature_importance(model=model)
        self.assertIsInstance(plot, AxPlotConfig)
