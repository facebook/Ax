#!/usr/bin/env python3

from collections import defaultdict
from typing import TYPE_CHECKING, DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np
from ae.lazarus.ae.core.observation import ObservationData, ObservationFeatures
from ae.lazarus.ae.core.optimization_config import OptimizationConfig
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.core.types.types import TConfig, TParamValue
from ae.lazarus.ae.modelbridge.transforms.base import Transform
from ae.lazarus.ae.utils.common.logger import get_logger


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ae.lazarus.ae.modelbridge import (  # noqa F401  # pragma: no cover
        base as base_modelbridge,
    )


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
        # pyre-fixme[9]: Ys has type `DefaultDict[str, List[float]]`; used as `Defaul...
        Ys: DefaultDict[str, List[float]] = defaultdict(list)
        for obsd in observation_data:
            for i, m in enumerate(obsd.metric_names):
                Ys[m].append(obsd.means[i])
        # Expected `DefaultDict[Union[str, typing.Tuple[str, Optional[Union[bool, float,
        # str]]]], List[float]]` for 1st anonymous parameter to call
        # `ae.lazarus.ae.modelbridge.transforms.standardize_y.compute_standardization_params`
        # but got `DefaultDict[str, List[float]]`.
        # pyre-fixme[6]: Expected `DefaultDict[Union[str, Tuple[str, Optional[Union[b...
        self.Ymean, self.Ystd = compute_standardization_params(Ys)

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
            c.bound = (c.bound - self.Ymean[c.metric.name]) / self.Ystd[c.metric.name]
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


def compute_standardization_params(
    Ys: DefaultDict[Union[str, Tuple[str, TParamValue]], List[float]]
) -> Tuple[
    Dict[Union[str, Tuple[str, str]], float], Dict[Union[str, Tuple[str, str]], float]
]:
    Ymean = {k: np.mean(y) for k, y in Ys.items()}
    Ystd = {k: np.std(y) for k, y in Ys.items()}
    for k, s in Ystd.items():
        # Don't standardize if variance is too small.
        if s < 1e-8:
            Ystd[k] = 1.0
            logger.info(f"Outcome {k} is constant, within tolerance.")
    return Ymean, Ystd
