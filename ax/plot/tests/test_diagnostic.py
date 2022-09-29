#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import plotly.graph_objects as go
from ax.modelbridge.cross_validation import cross_validate
from ax.modelbridge.registry import Models
from ax.plot.base import AxPlotConfig
from ax.plot.diagnostic import (
    interact_cross_validation,
    interact_cross_validation_plotly,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import fast_botorch_optimize


class DiagnosticTest(TestCase):
    @fast_botorch_optimize
    def setUp(self) -> None:
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run()
        self.model = Models.BOTORCH(
            # Model bridge kwargs
            experiment=exp,
            data=exp.fetch_data(),
        )

    def test_cross_validation(self) -> None:
        for autoset_axis_limits in [False, True]:
            cv = cross_validate(self.model)
            # Assert that each type of plot can be constructed successfully
            label_dict = {"branin": "BrAnIn"}
            plot = interact_cross_validation_plotly(
                cv, label_dict=label_dict, autoset_axis_limits=autoset_axis_limits
            )
            # pyre-ignore [16]
            x_range = plot.layout.updatemenus[0].buttons[0].args[1]["xaxis.range"]
            y_range = plot.layout.updatemenus[0].buttons[0].args[1]["yaxis.range"]
            if autoset_axis_limits:
                self.assertTrue((len(x_range) == 2) and (x_range[0] < x_range[1]))
                self.assertTrue((len(y_range) == 2) and (y_range[0] < y_range[1]))
            else:
                self.assertIsNone(x_range)
                self.assertIsNone(y_range)

            self.assertIsInstance(plot, go.Figure)
            plot = interact_cross_validation(
                cv, label_dict=label_dict, autoset_axis_limits=autoset_axis_limits
            )
            self.assertIsInstance(plot, AxPlotConfig)
