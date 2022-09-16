#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from unittest import mock

from ax.core.arm import Arm
from ax.core.metric import Metric
from ax.core.search_space import SearchSpace
from ax.modelbridge.base import ModelBridge
from ax.models.discrete.full_factorial import FullFactorialGenerator
from ax.plot.scatter import tile_fitted, tile_observations
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_data,
    get_experiment,
    get_experiment_with_data,
    get_search_space,
)
from ax.utils.testing.modeling_stubs import get_observation


@mock.patch(
    "ax.modelbridge.base.observations_from_data",
    autospec=True,
    return_value=([get_observation()]),
)
@mock.patch(
    "ax.modelbridge.base.gen_arms", autospec=True, return_value=[Arm(parameters={})]
)
def get_modelbridge(
    # pyre-fixme[2]: Parameter must be annotated.
    mock_gen_arms,
    # pyre-fixme[2]: Parameter must be annotated.
    mock_observations_from_data,
    status_quo_name: Optional[str] = None,
) -> ModelBridge:
    exp = get_experiment()
    modelbridge = ModelBridge(
        search_space=get_search_space(),
        model=FullFactorialGenerator(),
        experiment=exp,
        data=get_data(),
        status_quo_name=status_quo_name,
    )
    modelbridge._predict = mock.MagicMock(
        "ax.modelbridge.base.ModelBridge._predict",
        autospec=True,
        return_value=[get_observation().data],
    )
    return modelbridge


class TileFittedTest(TestCase):
    def testTileFitted(self) -> None:
        model = get_modelbridge(status_quo_name=None)

        # Should throw if `status_quo_arm` is None and rel=True
        with self.assertRaises(ValueError):
            tile_fitted(model, rel=True)
        tile_fitted(model, rel=False)

        model = get_modelbridge(status_quo_name="1_1")
        config = tile_fitted(model, rel=True)
        self.assertIsNotNone(config)

        for key in ["layout", "data"]:
            self.assertIn(key, config.data)

        # Layout
        for key in [
            "annotations",
            "margin",
            "hovermode",
            "updatemenus",
            "font",
            "width",
            "height",
            "legend",
        ]:
            self.assertIn(key, config.data["layout"])

        # Data
        for i in range(2):
            self.assertEqual(config.data["data"][i]["x"], ["1_1"])
            self.assertEqual(config.data["data"][i]["y"], [0.0])
            self.assertEqual(config.data["data"][i]["type"], "scatter")
            self.assertEqual(
                config.data["data"][i]["error_y"]["array"], [138.59292911256333]
            )
            self.assertIn("Arm 1_1", config.data["data"][i]["text"][0])
            self.assertIn("[-138.593%, 138.593%]", config.data["data"][i]["text"][0])
            self.assertIn("0.0%", config.data["data"][i]["text"][0])

            for key in [
                "type",
                "x",
                "y",
                "marker",
                "mode",
                "name",
                "text",
                "hoverinfo",
                "error_y",
                "visible",
                "legendgroup",
                "showlegend",
                "xaxis",
                "yaxis",
            ]:
                self.assertIn(key, config.data["data"][i])


class TileObservationsTest(TestCase):
    def testTileObservations(self) -> None:
        exp = get_experiment_with_data()
        exp.trials[0].run()
        exp.trials[0].mark_completed()
        exp.add_tracking_metric(Metric("ax_test_metric"))
        # pyre-fixme[6]: For 1st param expected `List[Parameter]` but got
        #  `dict_values[str, Parameter]`.
        exp.search_space = SearchSpace(parameters=exp.search_space.parameters.values())
        config = tile_observations(experiment=exp, arm_names=["0_1", "0_2"], rel=False)

        for key in ["layout", "data"]:
            self.assertIn(key, config.data)

        # Layout
        for key in [
            "annotations",
            "margin",
            "hovermode",
            "updatemenus",
            "font",
            "width",
            "height",
            "legend",
        ]:
            self.assertIn(key, config.data["layout"])

        # Data
        self.assertEqual(config.data["data"][0]["x"], ["0_1", "0_2"])
        self.assertEqual(config.data["data"][0]["y"], [2.0, 2.25])
        self.assertEqual(config.data["data"][0]["type"], "scatter")
        self.assertIn("Arm 0_1", config.data["data"][0]["text"][0])
