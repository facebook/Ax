#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger
from typing import DefaultDict, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint, ScalarizedOutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.core.types import TParamValue
from ax.exceptions.core import DataRequiredError
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.utils import get_data
from ax.models.types import TConfig
from ax.utils.common.logger import get_logger


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax.modelbridge import base as base_modelbridge  # noqa F401  # pragma: no cover


logger: Logger = get_logger(__name__)


class StandardizeY(Transform):
    """Standardize Y, separately for each metric.

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        observations: Optional[List[Observation]] = None,
        modelbridge: Optional["base_modelbridge.ModelBridge"] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        if observations is None or len(observations) == 0:
            raise DataRequiredError("`StandardizeY` transform requires non-empty data.")
        observation_data = [obs.data for obs in observations]
        Ys = get_data(observation_data=observation_data)
        # Compute means and SDs
        # pyre-fixme[6]: Expected `DefaultDict[Union[str, Tuple[str, Optional[Union[b...
        # pyre-fixme[4]: Attribute must be annotated.
        self.Ymean, self.Ystd = compute_standardization_parameters(Ys)

    def _transform_observation_data(
        self,
        observation_data: List[ObservationData],
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
        modelbridge: Optional["base_modelbridge.ModelBridge"] = None,
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> OptimizationConfig:
        for c in optimization_config.all_constraints:
            if c.relative:
                raise ValueError(
                    f"StandardizeY transform does not support relative constraint {c}"
                )
            if isinstance(c, ScalarizedOutcomeConstraint):
                # transform \sum (wi * yi) <= C to
                # \sum (wi * si * zi) <= C - \sum (wi * mu_i) that zi = (yi - mu_i) / si

                # update bound C to new c = C.bound - sum_i (wi * mu_i)
                agg_mean = np.sum(
                    [
                        c.weights[i] * self.Ymean[metric.name]
                        for i, metric in enumerate(c.metrics)
                    ]
                )
                c.bound = float(c.bound - agg_mean)

                # update the weights in the scalarized constraint
                # new wi = wi * si
                new_weight = [
                    c.weights[i] * self.Ystd[metric.name]
                    for i, metric in enumerate(c.metrics)
                ]
                c.weights = new_weight
            else:
                c.bound = float(
                    (c.bound - self.Ymean[c.metric.name]) / self.Ystd[c.metric.name]
                )
        return optimization_config

    def _untransform_observation_data(
        self,
        observation_data: List[ObservationData],
    ) -> List[ObservationData]:
        for obsd in observation_data:
            means = np.array([self.Ymean[m] for m in obsd.metric_names])
            stds = np.array([self.Ystd[m] for m in obsd.metric_names])
            obsd.means = obsd.means * stds + means
            obsd.covariance *= np.dot(stds[:, None], stds[:, None].transpose())
        return observation_data

    def untransform_outcome_constraints(
        self,
        outcome_constraints: List[OutcomeConstraint],
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> List[OutcomeConstraint]:
        for c in outcome_constraints:
            if c.relative:
                raise ValueError(  # pragma: no cover
                    f"StandardizeY transform does not support relative constraint {c}"
                )
            if isinstance(c, ScalarizedOutcomeConstraint):
                raise ValueError(  # pragma: no cover
                    "ScalarizedOutcomeConstraint not supported"
                )
            c.bound = float(
                c.bound * self.Ystd[c.metric.name] + self.Ymean[c.metric.name]
            )
        return outcome_constraints


def compute_standardization_parameters(
    Ys: DefaultDict[Union[str, Tuple[str, TParamValue]], List[float]]
) -> Tuple[
    Dict[Union[str, Tuple[str, str]], float], Dict[Union[str, Tuple[str, str]], float]
]:
    """Compute mean and std. dev of Ys."""
    Ymean = {k: np.mean(y) for k, y in Ys.items()}
    # We use the Bessel correction term (divide by N-1) here in order to
    # be consistent with the default behavior of torch.std that is used to
    # validate input data standardization in BoTorch.
    Ystd = {k: np.std(y, ddof=1) if len(y) > 1 else 0.0 for k, y in Ys.items()}
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
