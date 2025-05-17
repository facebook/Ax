#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import numpy as np
import torch
from ax.adapter.adapter_utils import prep_pairwise_data
from ax.adapter.torch import TorchAdapter
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata
from ax.utils.common.constants import Keys
from botorch.utils.containers import DenseContainer
from botorch.utils.datasets import SupervisedDataset
from pyre_extensions import none_throws


class PairwiseAdapter(TorchAdapter):
    def _convert_observations(
        self,
        observation_data: list[ObservationData],
        observation_features: list[ObservationFeatures],
        outcomes: list[str],
        parameters: list[str],
        search_space_digest: SearchSpaceDigest | None,
    ) -> tuple[
        list[SupervisedDataset], list[str], list[list[TCandidateMetadata]] | None
    ]:
        """Converts observations to a dictionary of `Dataset` containers and (optional)
        candidate metadata.
        """
        if len(observation_features) != len(observation_data):
            raise ValueError("Observation features and data must have the same length!")
        # pyre-fixme[6]: For 1st argument expected `Union[_SupportsArray[dtype[typing...
        ordered_idx = np.argsort([od.trial_index for od in observation_features])
        observation_features = [observation_features[i] for i in ordered_idx]
        observation_data = [observation_data[i] for i in ordered_idx]

        (
            Xs,
            Ys,
            _,  # Yvars is not used here.
            candidate_metadata_dict,
            any_candidate_metadata_is_not_none,
            trial_indices,
        ) = self._extract_observation_data(
            observation_data, observation_features, parameters
        )

        datasets: list[SupervisedDataset] = []
        candidate_metadata = []
        for outcome in outcomes:
            X = torch.stack(Xs[outcome], dim=0)
            Y = torch.tensor(Ys[outcome], dtype=torch.long).unsqueeze(-1)
            if outcome == Keys.PAIRWISE_PREFERENCE_QUERY.value:
                dataset = prep_pairwise_data(
                    X=X,
                    Y=Y,
                    group_indices=torch.tensor(none_throws(trial_indices)[outcome]),
                    outcome=outcome,
                    parameters=parameters,
                )
            else:  # pragma: no cover
                event_shape = torch.Size([X.shape[-1]])
                dataset_X = DenseContainer(X, event_shape=event_shape)
                dataset = SupervisedDataset(
                    X=dataset_X,
                    Y=Y,
                    feature_names=parameters,
                    outcome_names=[outcome],
                    group_indices=torch.tensor(trial_indices[outcome])
                    if trial_indices is not None
                    else None,
                )

            datasets.append(dataset)
            candidate_metadata.append(candidate_metadata_dict[outcome])

        if not any_candidate_metadata_is_not_none:
            return datasets, outcomes, None

        return datasets, outcomes, candidate_metadata

    def _predict(
        self,
        observation_features: list[ObservationFeatures],
        use_posterior_predictive: bool = False,
    ) -> list[ObservationData]:
        # TODO: Implement `_predict` to enable examining predicted effects
        raise NotImplementedError
