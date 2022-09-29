#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import plotly.graph_objects as go
from ax.modelbridge.registry import Models
from ax.plot.base import AxPlotConfig
from ax.plot.trace import (
    optimization_trace_single_method,
    optimization_trace_single_method_plotly,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import fast_botorch_optimize


class TracesTest(TestCase):
    @fast_botorch_optimize
    def setUp(self) -> None:
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run()
        self.model = Models.BOTORCH(
            # Model bridge kwargs
            experiment=exp,
            data=exp.fetch_data(),
        )

    def testTraces(self) -> None:
        # Assert that each type of plot can be constructed successfully
        plot = optimization_trace_single_method_plotly(
            np.array([[1, 2, 3], [4, 5, 6]]),
            list(self.model.metric_names)[0],
            optimization_direction="minimize",
            autoset_axis_limits=False,
        )
        self.assertIsInstance(plot, go.Figure)
        plot = optimization_trace_single_method(
            np.array([[1, 2, 3], [4, 5, 6]]),
            list(self.model.metric_names)[0],
            optimization_direction="minimize",
            autoset_axis_limits=False,
        )
        self.assertIsInstance(plot, AxPlotConfig)

    def testTracesAutoAxes(self) -> None:
        for optimization_direction in ["minimize", "maximize", "passthrough"]:
            plot = optimization_trace_single_method_plotly(
                np.array([[1, 2, 3], [4, 5, 6]]),
                list(self.model.metric_names)[0],
                optimization_direction=optimization_direction,
                autoset_axis_limits=True,
            )
            self.assertIsNone(plot.layout.xaxis.range)  # pyre-ignore
            if optimization_direction == "minimize":
                self.assertAlmostEqual(plot.layout.yaxis.range[0], 0.525)
                self.assertAlmostEqual(plot.layout.yaxis.range[1], 6.225)
            elif optimization_direction == "maximize":
                self.assertAlmostEqual(plot.layout.yaxis.range[0], 0.775)
                self.assertAlmostEqual(plot.layout.yaxis.range[1], 6.475)
            else:
                self.assertIsNone(plot.layout.yaxis.range)
