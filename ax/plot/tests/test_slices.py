#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import plotly.graph_objects as go
from ax.modelbridge.registry import Models
from ax.plot.base import AxPlotConfig
from ax.plot.slice import (
    plot_slice_plotly,
    interact_slice_plotly,
    plot_slice,
    interact_slice,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment


class SlicesTest(TestCase):
    def testSlices(self):
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run()
        model = Models.BOTORCH(
            # Model bridge kwargs
            experiment=exp,
            data=exp.fetch_data(),
        )
        # Assert that each type of plot can be constructed successfully
        plot = plot_slice_plotly(
            model, model.parameters[0], list(model.metric_names)[0]
        )
        self.assertIsInstance(plot, go.Figure)
        plot = interact_slice_plotly(
            model, model.parameters[0], list(model.metric_names)[0]
        )
        self.assertIsInstance(plot, go.Figure)
        plot = plot = plot_slice(
            model, model.parameters[0], list(model.metric_names)[0]
        )
        self.assertIsInstance(plot, AxPlotConfig)
        plot = interact_slice(model, model.parameters[0], list(model.metric_names)[0])
        self.assertIsInstance(plot, AxPlotConfig)
