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
    # pyre-fixme[3]: Return type must be annotated.
    def test_cross_validation(self):
        exp = get_branin_experiment(with_batch=True)
        exp.trials[0].run()
        model = Models.BOTORCH(
            # Model bridge kwargs
            experiment=exp,
            data=exp.fetch_data(),
        )
        cv = cross_validate(model)
        # Assert that each type of plot can be constructed successfully
        label_dict = {"branin": "BrAnIn"}
        plot = interact_cross_validation_plotly(cv, label_dict=label_dict)
        self.assertIsInstance(plot, go.Figure)
        plot = interact_cross_validation(cv, label_dict=label_dict)
        self.assertIsInstance(plot, AxPlotConfig)
