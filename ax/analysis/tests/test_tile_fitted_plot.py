# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import plotly.graph_objects as go

from ax.analysis.tile_fitted_plot import TileFittedPlot

from ax.modelbridge.registry import Models

from ax.utils.testing.core_stubs import get_branin_experiment


class TestTileFittedPlot(unittest.TestCase):
    def setUp(self) -> None:
        self.exp = get_branin_experiment(with_batch=True)
        self.exp.trials[0].run()
        self.model = Models.BOTORCH_MODULAR(
            # Model bridge kwargs
            experiment=self.exp,
            data=self.exp.fetch_data(),
        )

        super().setUp()

    def test_tile_fitted_plot(self) -> None:
        tile_fitted_plot = TileFittedPlot(
            experiment=self.exp,
            model=self.model,
        )

        df = tile_fitted_plot.get_df()
        print(df)

        fig = tile_fitted_plot.get_fig()
        self.assertIsInstance(fig, go.Figure)
