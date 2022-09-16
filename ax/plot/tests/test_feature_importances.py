#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.registry import Models
from ax.plot.base import AxPlotConfig
from ax.plot.feature_importances import (
    plot_feature_importance_by_feature,
    plot_feature_importance_by_feature_plotly,
    plot_feature_importance_by_metric,
    plot_feature_importance_by_metric_plotly,
    plot_relative_feature_importance,
    plot_relative_feature_importance_plotly,
)
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_experiment
from ax.utils.testing.mock import fast_botorch_optimize
from plotly import graph_objects as go

DUMMY_CAPTION = "test_caption"


def get_modelbridge() -> ModelBridge:
    exp = get_branin_experiment(with_batch=True)
    exp.trials[0].run()
    return Models.BOTORCH(
        # Model bridge kwargs
        experiment=exp,
        data=exp.fetch_data(),
    )


# pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use `typing.Dict`
#  to avoid runtime subscripting errors.
def get_sensitivity_values(ax_model: ModelBridge) -> Dict:
    """
    Compute lengscale sensitivity value for on an ax model.

    Returns map {'metric_name': {'parameter_name': sensitivity_value}}
    """
    ls = ax_model.model.model.covar_module.base_kernel.lengthscale.squeeze()
    if len(ls.shape) > 1:
        ls = ls.mean(dim=0)
    # pyre-fixme[16]: `float` has no attribute `detach`.
    importances_tensor = torch.stack([(1 / ls).detach().cpu()])
    # pyre-fixme[16]: `ModelBridge` has no attribute `outcomes`.
    importances_dict = dict(zip(ax_model.outcomes, importances_tensor))
    res = {}
    for metric_name in ax_model.outcomes:
        importances_arr = importances_dict[metric_name].numpy()
        # pyre-fixme[16]: `ModelBridge` has no attribute `parameters`.
        res[metric_name] = dict(zip(ax_model.parameters, importances_arr))
    return res


class FeatureImportancesTest(TestCase):
    @fast_botorch_optimize
    def testFeatureImportances(self) -> None:
        model = get_modelbridge()
        # Assert that each type of plot can be constructed successfully
        plot = plot_feature_importance_by_feature_plotly(model=model)
        self.assertIsInstance(plot, go.Figure)
        plot = plot_feature_importance_by_feature_plotly(
            model=model, caption=DUMMY_CAPTION
        )
        self.assertIsInstance(plot, go.Figure)
        # pyre-fixme[16]: `Figure` has no attribute `layout`.
        self.assertEqual(len(plot.layout.annotations), 1)
        self.assertEqual(plot.layout.annotations[0].text, DUMMY_CAPTION)
        plot = plot_feature_importance_by_feature(model=model)
        self.assertIsInstance(plot, AxPlotConfig)
        plot = plot_feature_importance_by_metric_plotly(model=model)
        self.assertIsInstance(plot, go.Figure)
        plot = plot_feature_importance_by_metric(model=model)
        self.assertIsInstance(plot, AxPlotConfig)
        plot = plot_relative_feature_importance_plotly(model=model)
        self.assertIsInstance(plot, go.Figure)
        plot = plot_relative_feature_importance(model=model)
        self.assertIsInstance(plot, AxPlotConfig)

        lengthscale_sensitivity_values = get_sensitivity_values(model)
        plot = plot_feature_importance_by_feature_plotly(
            sensitivity_values=lengthscale_sensitivity_values
        )
        self.assertIsInstance(plot, go.Figure)
        plot = plot_feature_importance_by_feature_plotly(
            sensitivity_values=lengthscale_sensitivity_values, caption=DUMMY_CAPTION
        )
        self.assertIsInstance(plot, go.Figure)
        self.assertEqual(len(plot.layout.annotations), 1)
        self.assertEqual(plot.layout.annotations[0].text, DUMMY_CAPTION)
        plot = plot_feature_importance_by_feature_plotly(
            sensitivity_values=lengthscale_sensitivity_values,
            importance_measure="Lengthscales",
        )
        self.assertIsInstance(plot, go.Figure)
        plot = plot_feature_importance_by_feature(
            sensitivity_values=lengthscale_sensitivity_values
        )
        self.assertIsInstance(plot, AxPlotConfig)
