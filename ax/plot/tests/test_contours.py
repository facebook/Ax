#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import plotly.graph_objects as go
from ax.modelbridge.registry import Models
from ax.plot.base import AxPlotConfig
from ax.plot.contour import (
    plot_contour_plotly,
    interact_contour_plotly,
    plot_contour,
    interact_contour,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class ContoursTest(TestCase):
    def testContours(self):
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run()
        model = Models.BOTORCH(
            # Model bridge kwargs
            experiment=exp,
            data=exp.fetch_data(),
        )
        # Assert that each type of plot can be constructed successfully
        plot = plot_contour_plotly(
            model, model.parameters[0], model.parameters[1], list(model.metric_names)[0]
        )
        self.assertIsInstance(plot, go.Figure)
        plot = interact_contour_plotly(model, list(model.metric_names)[0])
        self.assertIsInstance(plot, go.Figure)
        plot = plot = plot_contour(
            model, model.parameters[0], model.parameters[1], list(model.metric_names)[0]
        )
        self.assertIsInstance(plot, AxPlotConfig)
        plot = interact_contour(model, list(model.metric_names)[0])
        self.assertIsInstance(plot, AxPlotConfig)
