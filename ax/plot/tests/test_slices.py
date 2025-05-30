#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import plotly.graph_objects as go
from ax.adapter.registry import Generators
from ax.plot.base import AxPlotConfig
from ax.plot.slice import (
    interact_slice,
    interact_slice_plotly,
    plot_slice,
    plot_slice_plotly,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import mock_botorch_optimize


class SlicesTest(TestCase):
    @mock_botorch_optimize
    def test_Slices(self) -> None:
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run()
        model = Generators.BOTORCH_MODULAR(
            # Adapter kwargs
            experiment=exp,
            data=exp.fetch_data(),
        )
        # Assert that each type of plot can be constructed successfully
        plot = plot_slice_plotly(
            model,
            # pyre-fixme[16]: `Adapter` has no attribute `parameters`.
            model.parameters[0],
            list(model.metric_names)[0],
        )
        self.assertIsInstance(plot, go.Figure)
        plot = interact_slice_plotly(model)
        self.assertIsInstance(plot, go.Figure)
        plot = plot_slice(model, model.parameters[0], list(model.metric_names)[0])
        self.assertIsInstance(plot, AxPlotConfig)
        plot = interact_slice(model)
        self.assertIsInstance(plot, AxPlotConfig)
