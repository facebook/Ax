#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import numpy as np
from ax.core.observation import ObservationFeatures
from ax.modelbridge.prediction_utils import predict_at_point, predict_by_features
from ax.service.ax_client import AxClient
from ax.utils.common.testutils import TestCase


class TestPredictionUtils(TestCase):
    """Tests prediction utilities."""

    # pyre-fixme[3]: Return type must be annotated.
    def test_predict_at_point(self):
        ax_client = _set_up_client_for_get_model_predictions_no_next_trial()
        _attach_completed_trials(ax_client)
        ax_client.get_model_predictions()  # Ensures model is instantiated

        observation_features = ObservationFeatures(parameters={"x1": 0.3, "x2": 0.5})
        y_hat, se_hat = predict_at_point(
            model=ax_client.generation_strategy.model,
            obsf=observation_features,
            # pyre-fixme[6]: For 3rd param expected `Set[str]` but got `List[str]`.
            metric_names=["test_metric1"],
        )

        self.assertEqual(len(y_hat), 1)
        self.assertEqual(len(se_hat), 1)

    # pyre-fixme[3]: Return type must be annotated.
    def test_predict_by_features(self):
        ax_client = _set_up_client_for_get_model_predictions_no_next_trial()
        _attach_completed_trials(ax_client)
        ax_client.get_model_predictions()  # Ensures model is instantiated

        observation_features_dict = {
            18: ObservationFeatures(parameters={"x1": 0.3, "x2": 0.5}),
            19: ObservationFeatures(parameters={"x1": 0.4, "x2": 0.5}),
            20: ObservationFeatures(parameters={"x1": 0.8, "x2": 0.5}),
        }
        predictions_map = predict_by_features(
            model=ax_client.generation_strategy.model,
            label_to_feature_dict=observation_features_dict,
            # pyre-fixme[6]: For 3rd param expected `Set[str]` but got `List[str]`.
            metric_names=["test_metric1"],
        )
        self.assertEqual(len(predictions_map), 3)

    @mock.patch("ax.modelbridge.random.RandomModelBridge.predict")
    @mock.patch("ax.modelbridge.random.RandomModelBridge")
    # pyre-fixme[3]: Return type must be annotated.
    def test_predict_by_features_with_non_predicting_model(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        model_bridge_mock,
        # pyre-fixme[2]: Parameter must be annotated.
        predict_mock,
    ):
        ax_client = _set_up_client_for_get_model_predictions_no_next_trial()
        _attach_completed_trials(ax_client)

        # Do not call get_next_trial or get_model_predictions.
        # This test is for handling the use case where no model
        # is instantiated.

        observation_features_dict = {
            18: ObservationFeatures(parameters={"x1": 0.3, "x2": 0.5}),
            19: ObservationFeatures(parameters={"x1": 0.4, "x2": 0.5}),
            20: ObservationFeatures(parameters={"x1": 0.8, "x2": 0.5}),
        }

        predict_mock.side_effect = NotImplementedError()
        self.assertRaises(
            NotImplementedError,
            predict_by_features,
            **{
                "model": model_bridge_mock,
                "label_to_feature_dict": observation_features_dict,
                "metric_names": ["test_metric1"],
            },
        )


# Utility functions for testing get_model_predictions without calling
# get_next_trial. Create Ax Client with an experiment where
# num_initial_trials kwarg is zero. Note that this kwarg is
# needed to be able to instantiate the model for the first time
# without calling get_next_trial().
# pyre-fixme[3]: Return type must be annotated.
def _set_up_client_for_get_model_predictions_no_next_trial():
    ax_client = AxClient()
    ax_client.create_experiment(
        name="test_experiment",
        choose_generation_strategy_kwargs={"num_initialization_trials": 0},
        # pyre-fixme[6]: For 3rd param expected `List[Dict[str, Union[None,
        #  Dict[str, List[str]], List[Union[None, bool, float, int, str]], bool, float,
        #  int, str]]]` but got `List[Dict[str, Union[List[float], str]]]`.
        parameters=[
            {
                "name": "x1",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
            {
                "name": "x2",
                "type": "range",
                "bounds": [0.1, 1.0],
            },
        ],
        objective_name="test_metric1",
        outcome_constraints=["test_metric2 <= 1.5"],
    )

    return ax_client


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _attach_completed_trials(ax_client):
    # Attach completed trials
    trial1 = {"x1": 0.1, "x2": 0.1}
    parameters, trial_index = ax_client.attach_trial(trial1)
    ax_client.complete_trial(
        trial_index=trial_index, raw_data=_evaluate_test_metrics(parameters)
    )

    trial2 = {"x1": 0.2, "x2": 0.1}
    parameters, trial_index = ax_client.attach_trial(trial2)
    ax_client.complete_trial(
        trial_index=trial_index, raw_data=_evaluate_test_metrics(parameters)
    )


# Test metric evaluation method
# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _evaluate_test_metrics(parameters):
    x = np.array([parameters.get(f"x{i+1}") for i in range(2)])
    return {"test_metric1": (x[0] / x[1], 0.0), "test_metric2": (x[0] + x[1], 0.0)}
