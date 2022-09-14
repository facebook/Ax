#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.types import TCandidateMetadata
from ax.modelbridge.modelbridge_utils import detect_duplicates
from ax.modelbridge.torch import TorchModelBridge
from botorch.utils.containers import SliceContainer
from botorch.utils.datasets import RankingDataset, SupervisedDataset
from torch import Tensor


class PairwiseModelBridge(TorchModelBridge):
    def _convert_observations(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
        outcomes: List[str],
        parameters: List[str],
    ) -> Tuple[
        List[Optional[SupervisedDataset]], Optional[List[List[TCandidateMetadata]]]
    ]:
        """Converts observations to a dictionary of `Dataset` containers and (optional)
        candidate metadata.
        """
        if len(observation_features) != len(observation_data):
            raise ValueError(  # pragma: no cover
                "Observation features and data must have the same length!"
            )
        ordered_idx = np.argsort([od.trial_index for od in observation_features])
        observation_features = [observation_features[i] for i in ordered_idx]
        observation_data = [observation_data[i] for i in ordered_idx]

        (
            Xs,
            Ys,
            Yvars,
            candidate_metadata_dict,
            any_candidate_metadata_is_not_none,
        ) = self._extract_observation_data(
            observation_data, observation_features, parameters
        )

        datasets: List[Optional[SupervisedDataset]] = []
        candidate_metadata = []
        for outcome in outcomes:
            X = torch.stack(Xs[outcome], dim=0)
            Y = torch.tensor(Ys[outcome], dtype=torch.long).unsqueeze(-1)

            # Update Xs and Ys shapes for PairwiseGP
            Y = _binary_pref_to_comp_pair(Y=Y)
            X, Y = _consolidate_comparisons(X=X, Y=Y)

            datapoints, comparisons = X, Y.long()
            event_shape = torch.Size([2 * datapoints.shape[-1]])
            # pyre-fixme[6]: For 2nd param expected `LongTensor` but got `Tensor`.
            dataset_X = SliceContainer(datapoints, comparisons, event_shape=event_shape)
            dataset_Y = torch.tensor([[0, 1]]).expand(comparisons.shape)
            dataset = RankingDataset(X=dataset_X, Y=dataset_Y)

            datasets.append(dataset)
            candidate_metadata.append(candidate_metadata_dict[outcome])

        if not any_candidate_metadata_is_not_none:
            return datasets, None

        return datasets, candidate_metadata


def _binary_pref_to_comp_pair(Y: Tensor) -> Tensor:
    """Convert Y from binary indicator pair to index pair comparisons

    Convert Y from binary indicator pair such as [[0, 1], [1, 0], ...]
    to index comparisons like [[1, 0], [2, 3], ...]
    """
    Y_shape = Y.shape[:-2] + (-1, 2)
    Y = Y.reshape(Y_shape)

    _validate_Y_values(Y)

    idx_shift = (torch.arange(0, Y.shape[-2]) * 2).unsqueeze(-1).expand_as(Y)
    comparison_pairs = idx_shift + (1 - Y)
    return comparison_pairs


def _consolidate_comparisons(X: Tensor, Y: Tensor) -> Tuple[Tensor, Tensor]:
    """Drop duplicated Xs and update the indices in Ys accordingly"""
    if len(X.shape) != 2:
        raise ValueError("X must have 2 dimensions.")  # pragma: no cover
    if len(Y.shape) != 2:
        raise ValueError("Y must have 2 dimensions.")  # pragma: no cover
    if Y.shape[-1] != 2:
        raise ValueError(  # pragma: no cover
            "The last dimension of Y must contain 2 elements "
            "representing the pairwise comparison."
        )

    n = X.shape[-2]
    dupplicates = list(detect_duplicates(X=X))
    if len(dupplicates) != 0:
        dup_indices, kept_indices = zip(*dupplicates)
        unique_indices = set(range(n)) - set(dup_indices)

        # After dropping the duplicates,
        # the kept ones' indices may also change by being shifted up
        new_idx_map = dict(zip(unique_indices, range(len(unique_indices))))
        new_indices_for_dup = (new_idx_map[idx] for idx in kept_indices)
        new_idx_map.update(dict(zip(dup_indices, new_indices_for_dup)))

        consolidated_X = X[list(unique_indices), :]
        consolidated_Y = torch.tensor(
            [(new_idx_map[y1.item()], new_idx_map[y2.item()]) for y1, y2 in Y],
            dtype=torch.long,
        )
        return consolidated_X, consolidated_Y
    else:
        return X, Y


def _validate_Y_values(Y: Tensor) -> None:
    """Check if Ys have valid values"""
    # Y must have even number of elements
    if Y.shape[-1] != 2:
        raise ValueError(  # pragma: no cover
            f"Trailing dimension of `Y` should be size 2 but is {Y.shape[-1]}"
        )

    # all adjacent pairs must have exactly a 0 and a 1
    if not (Y.min(dim=-1).values.eq(0).all() and Y.max(dim=-1).values.eq(1).all()):
        raise ValueError("`Y` values must be `{0, 1}.`")
