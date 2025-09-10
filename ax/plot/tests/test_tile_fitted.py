#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest import mock

import numpy as np
from ax.adapter.base import Adapter
from ax.adapter.registry import Generators
from ax.core import Experiment
from ax.core.arm import Arm
from ax.core.metric import Metric
from ax.core.observation import Observation, ObservationData
from ax.core.search_space import SearchSpace
from ax.generators.base import Generator
from ax.metrics.branin import BraninMetric
from ax.plot.scatter import tile_fitted, tile_observations
from ax.runners.synthetic import SyntheticRunner
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_arms,
    get_branin_experiment,
    get_branin_search_space,
    get_data,
    get_experiment_with_data,
)


def get_adapter(with_status_quo: bool) -> Adapter:
    experiment = get_branin_experiment(
        with_completed_batch=True, with_status_quo=with_status_quo
    )
    adapter = Adapter(experiment=experiment, generator=Generator())
    adapter._predict = mock.MagicMock(
        "ax.adapter.base.Adapter._predict",
        autospec=True,
        return_value=[
            ObservationData(
                metric_names=["branin"],
                means=np.array([1.0]),
                covariance=np.array([[1.0]]),
            )
        ],
    )
    return adapter


class TileFittedTest(TestCase):
    def test_TileFitted(self) -> None:
        adapter = get_adapter(with_status_quo=False)

        # Should throw if `status_quo_arm` is None and rel=True
        with self.assertRaises(ValueError):
            tile_fitted(adapter, rel=True)
        tile_fitted(adapter, rel=False)

        adapter = get_adapter(with_status_quo=True)
        config = tile_fitted(adapter, rel=True)
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
        data = config.data["data"][0]
        arm_names = [arm.name for arm in adapter._experiment.trials[0].arms]
        # Bring SQ to the front.
        arm_names.insert(0, arm_names.pop(arm_names.index("status_quo")))
        self.assertEqual(data["x"], arm_names)
        self.assertEqual(data["y"], [0.0] * len(arm_names))
        self.assertEqual(data["type"], "scatter")
        for text, arm in zip(data["text"], arm_names):
            self.assertIn(f"Arm {arm}", text)
            self.assertIn("[-277.186%, 277.186%]", text)
            self.assertIn("0.0%", text)

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
            self.assertIn(key, data)

    def test_TileFittedOutOfDesignDataSelector(self) -> None:
        exp = Experiment(
            name="test3",
            search_space=get_branin_search_space(),
            tracking_metrics=[BraninMetric(name="b", param_names=["x1", "x2"])],
            runner=SyntheticRunner(),
        )
        batch = exp.new_batch_trial()
        arms = get_branin_arms(n=4, seed=0)
        arms.append(Arm(name="status_quo", parameters={"x1": None, "x2": None}))
        batch.add_arms_and_weights(arms=arms)
        batch.run()
        # batch.mark_running()
        batch.mark_completed()
        batch = exp.new_batch_trial()
        arms = get_branin_arms(n=4, seed=1)
        arms.append(Arm(name="status_quo", parameters={"x1": None, "x2": None}))
        batch.add_arms_and_weights(arms=arms)
        batch.run()
        batch.mark_completed()
        data0 = get_data(
            metric_name="b",
            trial_index=0,
            num_non_sq_arms=4,
        )
        data1 = get_data(
            metric_name="b",
            trial_index=1,
            num_non_sq_arms=4,
        )
        exp.attach_data(data0)
        exp.attach_data(data1)

        adapter = Generators.THOMPSON(experiment=exp, data=exp.lookup_data())

        def data_selector(obs: Observation) -> bool:
            return obs.features.trial_index != 0 or obs.arm_name == "status_quo"

        tile_fitted(adapter, rel=False)
        tile_fitted(
            adapter,
            rel=False,
            data_selector=data_selector,
        )


class TileObservationsTest(TestCase):
    def test_TileObservations(self) -> None:
        exp = get_experiment_with_data()
        exp.trials[0].run()
        exp.trials[0].mark_completed()
        exp.add_tracking_metric(Metric("ax_test_metric"))
        exp.search_space = SearchSpace(
            parameters=list(exp.search_space.parameters.values())
        )
        config = tile_observations(experiment=exp, arm_names=["0_0", "0_1"], rel=False)

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

        self.assertEqual(
            config.data["layout"]["annotations"][0]["text"], "ax_test_metric"
        )

        # Data
        self.assertEqual(config.data["data"][0]["x"], ["0_0"])
        self.assertEqual(config.data["data"][0]["y"], [3.0])
        self.assertEqual(config.data["data"][0]["type"], "scatter")
        self.assertIn("Arm 0_0", config.data["data"][0]["text"][0])

        label_dict = {"ax_test_metric": "mapped_name"}
        config = tile_observations(
            experiment=exp, arm_names=["0_0"], rel=False, label_dict=label_dict
        )
        self.assertEqual(config.data["layout"]["annotations"][0]["text"], "mapped_name")
