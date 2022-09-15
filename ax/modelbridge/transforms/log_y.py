#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from logging import Logger

from typing import Callable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.base import Transform
from ax.models.types import TConfig
from ax.utils.common.logger import get_logger
from scipy.stats import norm


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax.modelbridge import base as base_modelbridge  # noqa F401  # pragma: no cover


logger: Logger = get_logger(__name__)


class LogY(Transform):
    """Apply (natural) log-transform to Y.

    This essentially means that we are model the observations as log-normally
    distributed. If `config` specifies `match_ci_width=True`, use a matching
    procedure based on the width of the CIs, otherwise (the default), use the
    delta method,

    Transform is applied only for the metrics specified in the transform config.
    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        observations: Optional[List[Observation]] = None,
        modelbridge: Optional["base_modelbridge.ModelBridge"] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        if config is None:
            raise ValueError("LogY requires a config.")
        # pyre-fixme[6]: Expected `Iterable[Variable[_T]]` for 1st param but got
        #  `Union[List[Variable[_T]],
        #  botorch.acquisition.acquisition.AcquisitionFunction, float, int, str]`.
        metric_names = list(config.get("metrics", []))
        if len(metric_names) == 0:
            raise ValueError("Must specify at least one metric in the config.")
        super().__init__(
            search_space=search_space,
            observations=observations,
            config=config,
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self.metric_names = metric_names
        if config.get("match_ci_width", False):
            # perform moment-matching to compute variance that results in a CI
            # of same width as the when transforming the moments
            # pyre-fixme[4]: Attribute must be annotated.
            self._transform = lambda m, v: match_ci_width(m, v, np.log)
            # pyre-fixme[4]: Attribute must be annotated.
            self._untransform = lambda m, v: match_ci_width(m, v, np.exp)
        else:
            self._transform = lognorm_to_norm
            self._untransform = norm_to_lognorm

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        modelbridge: Optional[base_modelbridge.ModelBridge] = None,
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> OptimizationConfig:
        for c in optimization_config.all_constraints:
            if c.metric.name in self.metric_names:
                base_str = f"LogY transform cannot be applied to metric {c.metric.name}"
                if c.relative:
                    raise ValueError(
                        f"{base_str} since it is subject to a relative constraint."
                    )
                elif c.bound <= 0:
                    raise ValueError(
                        f"{base_str} since the bound isn't positive, got: {c.bound}."
                    )
                else:
                    c.bound = np.log(c.bound)
        return optimization_config

    def _tf_obs_data(
        self,
        observation_data: List[ObservationData],
        transform: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    ) -> List[ObservationData]:
        for obsd in observation_data:
            cov = obsd.covariance
            idcs = [
                i for i, m in enumerate(obsd.metric_names) if m in self.metric_names
            ]
            if len(idcs) != len(obsd.metric_names):
                # TODO: Support covariances for a subset of observations
                diff = cov - np.diag(np.diag(cov))
                if not np.all(np.isnan(diff) | (diff == 0)):
                    raise NotImplementedError(
                        "LogY transform for a subset of metrics not supported for "
                        " correlated observations"
                    )
                for i, m in enumerate(obsd.metric_names):
                    if m in self.metric_names:
                        mu, cov = transform(
                            np.array(obsd.means[i], ndmin=1),
                            np.array(obsd.covariance[i, i], ndmin=1),
                        )
                        obsd.means[i] = mu
                        obsd.covariance[i, i] = cov
            else:
                mu, cov = transform(obsd.means, obsd.covariance)
                obsd.means = mu
                obsd.covariance = cov
        return observation_data

    def _transform_observation_data(
        self,
        observation_data: List[ObservationData],
    ) -> List[ObservationData]:
        return self._tf_obs_data(observation_data, self._transform)

    def _untransform_observation_data(
        self,
        observation_data: List[ObservationData],
    ) -> List[ObservationData]:
        return self._tf_obs_data(observation_data, self._untransform)

    def untransform_outcome_constraints(
        self,
        outcome_constraints: List[OutcomeConstraint],
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> List[OutcomeConstraint]:
        for c in outcome_constraints:  # pragma: no cover
            if c.metric.name in self.metric_names:  # pragma: no cover
                if c.relative:  # pragma: no cover
                    raise ValueError(  # pragma: no cover
                        "Unexpected relative transform."
                    )
                c.bound = np.exp(c.bound)  # pragma: no cover
        return outcome_constraints  # pragma: no cover


def match_ci_width(
    mean: np.ndarray,
    variance: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
    level: float = 0.95,
) -> np.ndarray:
    fac = norm.ppf(1 - (1 - level) / 2)
    d = fac * np.sqrt(variance)
    width_asym = transform(mean + d) - transform(mean - d)
    new_mean = transform(mean)
    new_variance = (width_asym / 2 / fac) ** 2
    # pyre-fixme[7]: Expected `ndarray` but got `Tuple[ndarray, float]`.
    return new_mean, new_variance


def lognorm_to_norm(
    mu_ln: np.ndarray, Cov_ln: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of a MVN from those of the associated log-MVN

    If `Y` is log-normal with mean mu_ln and covariance Cov_ln, then
    `X ~ N(mu_n, Cov_n)` with

        Cov_n_{ij} = log(1 + Cov_ln_{ij} / (mu_ln_{i} * mu_n_{j}))
        mu_n_{i} = log(mu_ln_{i}) - 0.5 * log(1 + Cov_ln_{ii} / mu_ln_{i}**2)
    """
    Cov_n = np.log(1 + Cov_ln / np.outer(mu_ln, mu_ln))
    mu_n = np.log(mu_ln) - 0.5 * np.diag(Cov_n)
    return mu_n, Cov_n


def norm_to_lognorm(
    mu_n: np.ndarray, Cov_n: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of a log-MVN from its MVN sufficient statistics

    If `X ~ N(mu_n, Cov_n)` and `Y = exp(X)`, then `Y` is log-normal with

        mu_ln_{i} = exp(mu_n_{i}) + 0.5 * Cov_n_{ii}
        Cov_ln_{ij} = exp(mu_n_{i} + mu_n_{j} + 0.5 * (Cov_n_{ii} + Cov_n_{jj})) *
        (exp(Cov_n_{ij}) - 1)
    """
    diag_n = np.diag(Cov_n)
    b = mu_n + 0.5 * diag_n
    mu_ln = np.exp(b)
    Cov_ln = (np.exp(Cov_n) - 1) * np.exp(b.reshape(-1, 1) + b.reshape(1, -1))
    return mu_ln, Cov_ln
