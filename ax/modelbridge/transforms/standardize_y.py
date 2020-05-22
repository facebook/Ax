#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import TYPE_CHECKING, DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig, TParamValue
from ax.modelbridge.transforms.base import Transform
from ax.utils.common.logger import get_logger


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax.modelbridge import base as base_modelbridge  # noqa F401  # pragma: no cover


logger = get_logger("StandardizeY")


class StandardizeY(Transform):
    """Standardize Y, separately for each metric.

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
        config: Optional[TConfig] = None,
    ) -> None:
        if len(observation_data) == 0:
            raise ValueError(
                "StandardizeY transform requires non-empty observation data."
            )
        # Compute means and SDs
        Ys: DefaultDict[str, List[float]] = defaultdict(list)
        for obsd in observation_data:
            for i, m in enumerate(obsd.metric_names):
                Ys[m].append(obsd.means[i])
        # Expected `DefaultDict[Union[str, typing.Tuple[str, Optional[Union[bool, float,
        # str]]]], List[float]]` for 1st anonymous parameter to call
        # `ax.modelbridge.transforms.standardize_y.compute_standardization_parameters`
        # but got `DefaultDict[str, List[float]]`.
        # pyre-fixme[6]: Expected `DefaultDict[Union[str, Tuple[str, Optional[Union[b...
        self.Ymean, self.Ystd = compute_standardization_parameters(Ys)

    def transform_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
    ) -> List[ObservationData]:
        # Transform observation data
        for obsd in observation_data:
            means = np.array([self.Ymean[m] for m in obsd.metric_names])
            stds = np.array([self.Ystd[m] for m in obsd.metric_names])
            obsd.means = (obsd.means - means) / stds
            obsd.covariance /= np.dot(stds[:, None], stds[:, None].transpose())
        return observation_data

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        modelbridge: Optional["base_modelbridge.ModelBridge"],
        fixed_features: ObservationFeatures,
    ) -> OptimizationConfig:
        for c in optimization_config.outcome_constraints:
            if c.relative:
                raise ValueError(
                    f"StandardizeY transform does not support relative constraint {c}"
                )
            c.bound = float(
                (c.bound - self.Ymean[c.metric.name]) / self.Ystd[c.metric.name]
            )
        return optimization_config

    def untransform_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
    ) -> List[ObservationData]:
        for obsd in observation_data:
            means = np.array([self.Ymean[m] for m in obsd.metric_names])
            stds = np.array([self.Ystd[m] for m in obsd.metric_names])
            obsd.means = obsd.means * stds + means
            obsd.covariance *= np.dot(stds[:, None], stds[:, None].transpose())
        return observation_data


def compute_standardization_parameters(
    Ys: DefaultDict[Union[str, Tuple[str, TParamValue]], List[float]]
) -> Tuple[
    Dict[Union[str, Tuple[str, str]], float], Dict[Union[str, Tuple[str, str]], float]
]:
    """Compute mean and std. dev of Ys."""
    Ymean = {k: np.mean(y) for k, y in Ys.items()}
    Ystd = {k: np.std(y) for k, y in Ys.items()}
    for k, s in Ystd.items():
        # Don't standardize if variance is too small.
        if s < 1e-8:
            Ystd[k] = 1.0
            logger.info(f"Outcome {k} is constant, within tolerance.")
    # pyre-fixme[7]: Expected `Tuple[Dict[Union[Tuple[str, str], str], float],
    #  Dict[Union[Tuple[str, str], str], float]]` but got `Tuple[Dict[Union[Tuple[str,
    #  Union[None, bool, float, int, str]], str], typing.Any], Dict[Union[Tuple[str,
    #  Union[None, bool, float, int, str]], str], typing.Any]]`.
    return Ymean, Ystd
