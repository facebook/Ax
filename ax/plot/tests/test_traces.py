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
    def testTraces(self) -> None:
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run()
        model = Models.BOTORCH(
            # Model bridge kwargs
            experiment=exp,
            data=exp.fetch_data(),
        )
        # Assert that each type of plot can be constructed successfully
        plot = optimization_trace_single_method_plotly(
            np.array([[1, 2, 3], [4, 5, 6]]),
            # pyre-fixme[6]: For 2nd param expected `Optional[float]` but got `str`.
            list(model.metric_names)[0],
            optimization_direction="minimize",
        )
        self.assertIsInstance(plot, go.Figure)
        plot = optimization_trace_single_method(
            np.array([[1, 2, 3], [4, 5, 6]]),
            # pyre-fixme[6]: For 2nd param expected `Optional[float]` but got `str`.
            list(model.metric_names)[0],
            optimization_direction="minimize",
        )
        self.assertIsInstance(plot, AxPlotConfig)
