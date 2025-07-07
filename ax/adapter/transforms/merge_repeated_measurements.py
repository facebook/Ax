#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

import numpy as np
from ax.adapter.base import Adapter
from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.core.observation import Observation, ObservationData
from ax.core.search_space import SearchSpace
from ax.generators.types import TConfig
from pyre_extensions import none_throws


class MergeRepeatedMeasurements(Transform):
    """Merge repeated measurements for to obtain one observation per arm.

    Repeated measurements are merged via inverse variance weighting (e.g. over
    different trials). This intentionally ignores the trial index and assumes
    stationarity.

    TODO: Support inverse variance weighting correlated outcomes (full covariance).

    Note: this is not reversible.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        if observations is None:
            raise RuntimeError("MergeRepeatedMeasurements requires observations")
        super().__init__(
            search_space=search_space,
            observations=observations,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        # create a mapping of arm_key -> {metric_name: {means: [], vars: []}}
        arm_to_multi_obs: defaultdict[
            str, defaultdict[str, defaultdict[str, list[float]]]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for obs in observations:
            if (arm_name := obs.arm_name) is None:
                # Since the transform will be initialized with Adapter training data,
                # all observations will have arm names.
                raise NotImplementedError("All observations must have arm names.")
            # TODO: support inverse variance weighting for multivariate distributions
            # (full covariance)
            obsd = obs.data
            diag = np.diag(np.diag(obsd.covariance))
            if np.any(np.isnan(obsd.covariance)):
                raise NotImplementedError("All metrics must have noise observations.")
            elif ~np.all(obsd.covariance == diag):
                raise NotImplementedError(
                    "Only independent metrics are currently supported."
                )
            for i, m in enumerate(obsd.metric_names):
                arm_to_multi_obs[arm_name][m]["means"].append(obsd.means[i])
                arm_to_multi_obs[arm_name][m]["vars"].append(obsd.covariance[i, i])

        self.arm_to_merged: defaultdict[str, dict[str, dict[str, float]]] = defaultdict(
            dict
        )
        for k, metric_dict in arm_to_multi_obs.items():
            for m, v in metric_dict.items():
                # inverse variance weighting
                var = np.array(v["vars"])
                means = np.array(v["means"])
                noiseless = var == 0
                if np.any(noiseless):
                    noiseless_means = means[noiseless]
                    if (noiseless_means.shape[0] > 1) and (
                        not np.all(noiseless_means[1:] == noiseless_means[0])
                    ):
                        raise ValueError(
                            "All repeated arms with noiseless measurements "
                            "must have the same means."
                        )
                    self.arm_to_merged[k][m] = {
                        "mean": noiseless_means[0],
                        "var": 0.0,
                    }
                else:
                    inv_var = 1 / np.array(var)
                    inv_sum_inv_var = 1 / np.sum(inv_var)
                    weights = inv_var * inv_sum_inv_var
                    self.arm_to_merged[k][m] = {
                        "mean": np.sum(means * weights),
                        "var": inv_sum_inv_var,
                    }

    def transform_observations(
        self,
        observations: list[Observation],
    ) -> list[Observation]:
        new_observations = []
        arm_to_merged = deepcopy(self.arm_to_merged)
        for obs in observations:
            arm_name = obs.arm_name
            if arm_name not in arm_to_merged:
                continue
            arm_name = none_throws(arm_name)
            metric_dict = arm_to_merged.pop(arm_name)
            obsd = obs.data
            merged_means = np.zeros(len(obsd.metric_names))
            merged_covariance = np.zeros(
                (len(obsd.metric_names), len(obsd.metric_names))
            )
            for i, m in enumerate(obsd.metric_names):
                merged_metric = metric_dict[m]
                merged_means[i] = merged_metric["mean"]
                merged_covariance[i, i] = merged_metric["var"]
            new_obsd = ObservationData(
                metric_names=obsd.metric_names,
                means=merged_means,
                covariance=merged_covariance,
            )
            new_obs = Observation(
                features=obs.features,
                data=new_obsd,
                arm_name=arm_name,
            )
            new_observations.append(new_obs)
        return new_observations
