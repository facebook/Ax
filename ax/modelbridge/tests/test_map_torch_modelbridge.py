#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import numpy as np

import torch

from ax.core.base_trial import TrialStatus
from ax.core.observation import (
    ObservationData,
    ObservationFeatures,
    recombine_observations,
)
from ax.modelbridge.map_torch import MapTorchModelBridge
from ax.models.torch_base import TorchGenResults, TorchModel
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_arms,
    get_branin_experiment_with_timestamp_map_metric,
)


class MapTorchModelBridgeTest(TestCase):
    def testTorchModelBridge(self) -> None:
        experiment = get_branin_experiment_with_timestamp_map_metric(rate=0.5)
        for i in range(3):
            trial = experiment.new_trial().add_arm(arm=get_branin_arms(n=1, seed=i)[0])
            trial.run()

        for _ in range(2):
            # each time we call fetch, we grab another timestamp
            experiment.fetch_data()

        for i in range(3):
            experiment.trials[i].mark_as(status=TrialStatus.COMPLETED)

        experiment.attach_data(data=experiment.fetch_data())
        modelbridge = MapTorchModelBridge(
            experiment=experiment,
            search_space=experiment.search_space,
            data=experiment.lookup_data(),
            model=TorchModel(),
            transforms=[],
            fit_out_of_design=True,
            default_model_gen_options={"target_map_values": {"timestamp": 4.0}},
        )
        # Check map data is converted to observations, that we get one Observation
        # per row of MapData
        # pyre-fixme[16]: `Data` has no attribute `map_df`.
        map_df = experiment.lookup_data().map_df
        objective_df = map_df[map_df["metric_name"] == "branin_map"]
        self.assertEqual(len(modelbridge.get_training_data()), len(objective_df))

        # Test _gen
        model = mock.MagicMock(TorchModel, autospec=True, instance=True)
        model.gen.return_value = TorchGenResults(
            points=torch.tensor([[0.0, 0.0]]),
            weights=torch.tensor([1.0]),
            gen_metadata={},
        )
        model.predict.return_value = (
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        )
        modelbridge.model = model
        gen_results = modelbridge._gen(
            n=1,
            search_space=experiment.search_space,
            optimization_config=experiment.optimization_config,
            pending_observations={},
            fixed_features=ObservationFeatures({}),
        )
        gen_args = model.gen.mock_calls[0][2]
        self.assertEqual(
            gen_args["torch_opt_config"].model_gen_options[Keys.ACQF_KWARGS],
            {"map_dim_to_target": {2: 4.0}},
        )
        self.assertEqual(
            gen_results.observation_features[0].parameters, {"x1": 0.0, "x2": 0.0}
        )

        # Test _predict
        model = mock.MagicMock(TorchModel, autospec=True, instance=True)
        model.predict.return_value = (
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        )
        modelbridge.model = model
        features = [
            ObservationFeatures(parameters={"x1": 0.0, "x2": 0.0}),
            ObservationFeatures(parameters={"x1": 1.0, "x2": 1.0}),
        ]
        modelbridge._predict(features)
        predict_args = model.predict.mock_calls[0][2]
        # check that the target map value is inserted before prediction
        self.assertTrue(
            torch.allclose(
                predict_args["X"],
                torch.tensor([[0.0, 0.0, 4.0], [1.0, 1.0, 4.0]]).to(predict_args["X"]),
            )
        )

        # Test _cross_validate
        features = [
            ObservationFeatures(parameters={"x1": 0.0, "x2": 0.0, "timestamp": 0.0}),
            ObservationFeatures(parameters={"x1": 1.0, "x2": 1.0, "timestamp": 2.0}),
        ]
        data = [
            ObservationData(
                metric_names=["branin_map"],
                means=np.array([0.0]),
                covariance=np.array([[1.0]]),
            ),
            ObservationData(
                metric_names=["branin_map"],
                means=np.array([2.0]),
                covariance=np.array([[1.0]]),
            ),
        ]
        test_features = [
            ObservationFeatures(parameters={"x1": 0.0, "x2": 0.0, "timestamp": 1.0}),
            ObservationFeatures(parameters={"x1": 1.0, "x2": 1.0, "timestamp": 3.0}),
        ]
        test_data = [
            ObservationData(
                metric_names=["branin_map"],
                means=np.array([1.0]),
                covariance=np.array([[1.0]]),
            ),
            ObservationData(
                metric_names=["branin_map", "branin"],
                means=np.array([3.0, 2.0]),
                covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
            ),
        ]
        cv_training_data = recombine_observations(features, data)
        with mock.patch(
            "ax.modelbridge.torch.TorchModelBridge._cross_validate",
            return_value=test_data,
        ):
            cv_obs_data = modelbridge._cross_validate(
                search_space=experiment.search_space,
                cv_training_data=cv_training_data,
                cv_test_points=test_features,
            )
            # check that the out-of-design metric is deleted
            self.assertEqual(cv_obs_data[1].metric_names, ["branin"])
