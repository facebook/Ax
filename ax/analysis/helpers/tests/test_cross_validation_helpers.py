#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import plotly.graph_objects as go

from ax.analysis.helpers.constants import Z

from ax.analysis.helpers.cross_validation_helpers import (
    get_cv_plot_data,
    get_min_max_with_errors,
    obs_vs_pred_dropdown_plot,
)

from ax.modelbridge.cross_validation import cross_validate
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import fast_botorch_optimize


class TestCrossValidationHelpers(TestCase):
    @fast_botorch_optimize
    def setUp(self) -> None:
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run()
        self.model = Models.BOTORCH_MODULAR(
            # Model bridge kwargs
            experiment=exp,
            data=exp.fetch_data(),
        )

        self.exp_status_quo = get_branin_experiment(
            with_batch=True, with_status_quo=True
        )
        self.exp_status_quo.trials[0].run()
        self.model_status_quo = Models.BOTORCH_MODULAR(
            # Model bridge kwargs
            experiment=exp,
            data=exp.fetch_data(),
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
        cv_results = cross_validate(self.model)

        label_dict = {"branin": "BrAnIn"}

        data = get_cv_plot_data(cv_results, label_dict=label_dict)
        fig = obs_vs_pred_dropdown_plot(
            data=data,
        )

        self.assertIsInstance(fig, go.Figure)
