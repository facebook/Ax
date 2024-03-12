#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import plotly.graph_objects as go

from ax.analysis.cross_validation_plot import CrossValidationPlot

from ax.analysis.helpers.constants import Z

from ax.analysis.helpers.cross_validation_helpers import get_min_max_with_errors

from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import fast_botorch_optimize


class TestCrossValidationHelpers(TestCase):
    @fast_botorch_optimize
    def setUp(self) -> None:
        self.exp = get_branin_experiment(with_batch=True)
        self.exp.trials[0].run()
        self.model = Models.BOTORCH_MODULAR(
            # Model bridge kwargs
            experiment=self.exp,
            data=self.exp.fetch_data(),
        )

        self.exp_status_quo = get_branin_experiment(
            with_batch=True, with_status_quo=True
        )
        self.exp_status_quo.trials[0].run()
        self.model_status_quo = Models.BOTORCH_MODULAR(
            # Model bridge kwargs
            experiment=self.exp,
            data=self.exp.fetch_data(),
        )

    def test_get_min_max_with_errors(self) -> None:
        # Test with sample data
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        sd_x = [0.1, 0.2, 0.3]
        sd_y = [0.1, 0.2, 0.3]
        min_, max_ = get_min_max_with_errors(x, y, sd_x, sd_y)

        expected_min = 1.0 - 0.1 * Z
        expected_max = 6.0 + 0.3 * Z
        # Check that the returned values are correct
        print(f"min: {min_} {expected_min=}")
        print(f"max: {max_} {expected_max=}")
        self.assertAlmostEqual(min_, expected_min, delta=1e-4)
        self.assertAlmostEqual(max_, expected_max, delta=1e-4)

    def test_obs_vs_pred_dropdown_plot(self) -> None:

        cross_validation_plot = CrossValidationPlot(
            experiment=self.exp, model=self.model
        )
        fig = cross_validation_plot.get_fig()

        self.assertIsInstance(fig, go.Figure)
