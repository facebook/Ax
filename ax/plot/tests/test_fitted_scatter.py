#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import plotly.graph_objects as go
from ax.modelbridge.registry import Models
from ax.plot.base import AxPlotConfig
from ax.plot.scatter import interact_fitted, interact_fitted_plotly
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import fast_botorch_optimize


class FittedScatterTest(TestCase):
    @fast_botorch_optimize
    def test_fitted_scatter(self) -> None:
        exp = get_branin_experiment(with_str_choice_param=True, with_batch=True)
        exp.trials[0].run()
        model = Models.BOTORCH(
            # Model bridge kwargs
            experiment=exp,
            data=exp.fetch_data(),
        )
        # Assert that each type of plot can be constructed successfully
        plot = interact_fitted_plotly(model=model, rel=False)
        self.assertIsInstance(plot, go.Figure)
        plot = interact_fitted(model=model, rel=False)
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
