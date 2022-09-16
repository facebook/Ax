#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import plotly.graph_objects as go
from ax.modelbridge.registry import Models
from ax.plot.base import AxPlotConfig
from ax.plot.contour import (
    interact_contour,
    interact_contour_plotly,
    plot_contour,
    plot_contour_plotly,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import fast_botorch_optimize


class ContoursTest(TestCase):
    @fast_botorch_optimize
    def testContours(self) -> None:
        exp = get_branin_experiment(with_str_choice_param=True, with_batch=True)
        exp.trials[0].run()
        model = Models.BOTORCH(
            # Model bridge kwargs
            experiment=exp,
            data=exp.fetch_data(),
        )
        # Assert that each type of plot can be constructed successfully
        plot = plot_contour_plotly(
            model,
            # pyre-fixme[16]: `ModelBridge` has no attribute `parameters`.
            model.parameters[0],
            model.parameters[1],
            list(model.metric_names)[0],
        )
        self.assertIsInstance(plot, go.Figure)
        plot = interact_contour_plotly(model, list(model.metric_names)[0])
        self.assertIsInstance(plot, go.Figure)
        plot = interact_contour(model, list(model.metric_names)[0])
        self.assertIsInstance(plot, AxPlotConfig)
        plot = plot = plot_contour(
            model, model.parameters[0], model.parameters[1], list(model.metric_names)[0]
        )
        self.assertIsInstance(plot, AxPlotConfig)

        # Make sure all parameters and metrics are displayed in tooltips
        tooltips = list(exp.parameters.keys()) + list(exp.metrics.keys())
        for d in plot.data["data"]:
            # Only check scatter plots hoverovers
            if d["type"] != "scatter":
                continue
            for text in d["text"]:
                for tt in tooltips:
                    self.assertTrue(tt in text)
