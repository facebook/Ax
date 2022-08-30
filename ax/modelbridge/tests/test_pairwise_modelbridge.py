#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import numpy as np
import torch
from ax.core.observation import ObservationData, ObservationFeatures
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.pairwise import (
    _binary_pref_to_comp_pair,
    _consolidate_comparisons,
    PairwiseModelBridge,
)
from ax.utils.common.testutils import TestCase
from botorch.utils.datasets import RankingDataset


class PairwiseModelBridgeTest(TestCase):
    @mock.patch(
        f"{ModelBridge.__module__}.ModelBridge.__init__",
        autospec=True,
        return_value=None,
    )
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def testPairwiseModelBridge(self, mock_init):
        # Test _convert_observations
        pmb = PairwiseModelBridge(
            # pyre-fixme[6]: For 1st param expected `Experiment` but got `None`.
            experiment=None,
            # pyre-fixme[6]: For 2nd param expected `SearchSpace` but got `None`.
            search_space=None,
            # pyre-fixme[6]: For 3rd param expected `Data` but got `None`.
            data=None,
            # pyre-fixme[6]: For 4th param expected `TorchModel` but got `None`.
            model=None,
            transforms=[],
            torch_dtype=None,
            torch_device=None,
        )

        observation_data = [
            ObservationData(
                metric_names=["pairwise_pref_query"],
                means=np.array([0]),
                covariance=np.array([[np.nan]]),
            ),
            ObservationData(
                metric_names=["pairwise_pref_query"],
                means=np.array([1]),
                covariance=np.array([[np.nan]]),
            ),
        ]
        observation_features = [
            # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
            ObservationFeatures(parameters={"y1": 0.1, "y2": 0.2}, trial_index=0),
            # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
            ObservationFeatures(parameters={"y1": 0.3, "y2": 0.4}, trial_index=0),
        ]
        observation_features_with_metadata = [
            # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
            ObservationFeatures(parameters={"y1": 0.1, "y2": 0.2}, trial_index=0),
            ObservationFeatures(
                parameters={"y1": 0.3, "y2": 0.4},
                # pyre-fixme[6]: For 2nd param expected `Optional[int64]` but got `int`.
                trial_index=0,
                metadata={"metadata_key": "metadata_val"},
            ),
        ]
        parameters = ["y1", "y2"]
        outcomes = ["pairwise_pref_query"]

        datasets, candidate_metadata = pmb._convert_observations(
            observation_data=observation_data,
            observation_features=observation_features,
            outcomes=outcomes,
            parameters=parameters,
        )
        self.assertTrue(len(datasets) == 1)
        self.assertTrue(isinstance(datasets[0], RankingDataset))
        self.assertTrue(candidate_metadata is None)

        datasets, candidate_metadata = pmb._convert_observations(
            observation_data=observation_data,
            observation_features=observation_features_with_metadata,
            outcomes=outcomes,
            parameters=parameters,
        )
        self.assertTrue(len(datasets) == 1)
        self.assertTrue(isinstance(datasets[0], RankingDataset))
        self.assertTrue(candidate_metadata is not None)

        # Test individual helper methods
        X = torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [1.0, 2.0, 3.0], [2.1, 3.1, 4.1]]
        )
        Y = torch.tensor([[1, 0, 0, 1]])
        expected_X = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [2.1, 3.1, 4.1]])
        ordered_Y = torch.tensor([[0, 1], [3, 2]])
        expected_Y = torch.tensor([[0, 1], [2, 0]])

        # `_binary_pref_to_comp_pair`.
        comp_pair_Y = _binary_pref_to_comp_pair(Y=Y)
        self.assertTrue(torch.equal(comp_pair_Y, ordered_Y))

        # test `_binary_pref_to_comp_pair` with invalid data
        bad_Y = torch.tensor([[1, 1, 0, 0]])
        with self.assertRaises(ValueError):
            _binary_pref_to_comp_pair(Y=bad_Y)

        # `_consolidate_comparisons`.
        consolidated_X, consolidated_Y = _consolidate_comparisons(X=X, Y=comp_pair_Y)
        self.assertTrue(torch.equal(consolidated_X, expected_X))
        self.assertTrue(torch.equal(consolidated_Y, expected_Y))
