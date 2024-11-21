#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from copy import deepcopy

import plotly.graph_objects as go
from ax.core.data import Data
from ax.modelbridge.registry import Models
from ax.plot.base import AxPlotConfig
from ax.plot.scatter import interact_fitted, interact_fitted_plotly
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment, get_branin_metric
from ax.utils.testing.mock import mock_botorch_optimize


class FittedScatterTest(TestCase):
    @mock_botorch_optimize
    def test_fitted_scatter(self) -> None:
        exp = get_branin_experiment(with_str_choice_param=True, with_batch=True)
        exp.trials[0].run()
        # dup branin
        data = exp.fetch_data()
        df = deepcopy(data.df)
        df["metric_name"] = "branin_dup"
        exp.add_tracking_metric(get_branin_metric(name="branin_dup"))

        model = Models.BOTORCH_MODULAR(
            # Model bridge kwargs
            experiment=exp,
            data=Data.from_multiple_data([data, Data(df)]),
        )
        # Assert that each type of plot can be constructed successfully
        scalarized_metric_config = [
            {"name": "branin:agg", "weight": {"branin": 0.5, "branin_dup": 0.5}}
        ]
        plot = interact_fitted_plotly(
            model=model, rel=False, scalarized_metric_config=scalarized_metric_config
        )
        self.assertIsInstance(plot, go.Figure)
        plot = interact_fitted(
            model=model, rel=False, scalarized_metric_config=scalarized_metric_config
        )
        self.assertIsInstance(plot, AxPlotConfig)

        # Make sure all parameters and metrics are displayed in tooltips
        metric_names = ["branin", "branin_dup", "branin:agg"]
        tooltips = [list(exp.parameters.keys()) + [m_name] for m_name in metric_names]
        for idata, d in enumerate(plot.data["data"]):
            # Only check scatter plots hoverovers
            if d["type"] != "scatter":
                continue
            for text in d["text"]:
                for tt in tooltips[idata]:
                    self.assertTrue(tt in text)
