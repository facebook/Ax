# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import plotly.graph_objects as go
from ax.analysis.cross_validation_plot import CrossValidationPlot
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import fast_botorch_optimize


class TestCrossValidationPlot(TestCase):
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

    def test_cross_validation_plot(self) -> None:
        plot = CrossValidationPlot(experiment=self.exp, model=self.model).get_fig()
        # pyre-ignore [16]
        x_range = plot.layout.updatemenus[0].buttons[0].args[1]["xaxis.range"]
        y_range = plot.layout.updatemenus[0].buttons[0].args[1]["yaxis.range"]
        self.assertTrue((len(x_range) == 2) and (x_range[0] < x_range[1]))
        self.assertTrue((len(y_range) == 2) and (y_range[0] < y_range[1]))

        self.assertIsInstance(plot, go.Figure)

    def test_cross_validation_plot_get_df(self) -> None:
        plot = CrossValidationPlot(experiment=self.exp, model=self.model)
        _ = plot.get_df()
