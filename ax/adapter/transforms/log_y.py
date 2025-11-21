#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Callable, Iterable
from logging import Logger
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.utils import match_ci_width, T_MATCH_CI_WIDTH
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.generators.types import TConfig
from ax.utils.common.logger import get_logger
from pyre_extensions import assert_is_instance


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax.adapter import base as base_adapter  # noqa F401


logger: Logger = get_logger(__name__)


class LogY(Transform):
    """Apply (natural) log-transform to Y.

    This essentially means that we are model the observations as log-normally
    distributed. If `config` specifies `match_ci_width=True`, use a matching
    procedure based on the width of the CIs, otherwise (the default), use the
    delta method,

    Transform is applied only for the metrics specified in the transform config.
    Transform is done in-place.

    NOTE: If the observation noise is not provided, we simply log-transform the
    mean as if the observation noise was zero. This can be inaccurate when the
    unknown observation noise is large.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: base_adapter.Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        config = config or {}
        metric_signatures = [
            assert_is_instance(m, str)
            for m in list(assert_is_instance(config.get("metrics", []), Iterable))
        ]
        if len(metric_signatures) == 0:
            raise ValueError("Must specify at least one metric in the config.")
        super().__init__(
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        self.metric_signatures: list[str] = metric_signatures
        if config.get("match_ci_width", False):
            # perform moment-matching to compute variance that results in a CI
            # of same width as the when transforming the moments
            self._transform: T_MATCH_CI_WIDTH = lambda m, v: match_ci_width(
                mean=m, sem=None, variance=v, transform=np.log
            )
            self._untransform: T_MATCH_CI_WIDTH = lambda m, v: match_ci_width(
                mean=m, sem=None, variance=v, transform=np.exp
            )
        else:
            self._transform = lognorm_to_norm
            self._untransform = norm_to_lognorm

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        adapter: base_adapter.Adapter | None = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> OptimizationConfig:
        for c in optimization_config.all_constraints:
            if c.metric.signature in self.metric_signatures:
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
        observation_data: list[ObservationData],
        transform: Callable[
            [npt.NDArray, npt.NDArray], tuple[npt.NDArray, npt.NDArray]
        ],
    ) -> list[ObservationData]:
        for obsd in observation_data:
            cov = obsd.covariance
            idcs = [
                i
                for i, m in enumerate(obsd.metric_signatures)
                if m in self.metric_signatures
            ]
            if len(idcs) != len(obsd.metric_signatures):
                # TODO: Support covariances for a subset of observations
                diff = cov - np.diag(np.diag(cov))
                if not np.all(np.isnan(diff) | (diff == 0)):
                    raise NotImplementedError(
                        "LogY transform for a subset of metrics not supported for "
                        " correlated observations"
                    )
                for i, m in enumerate(obsd.metric_signatures):
                    if m in self.metric_signatures:
                        obsd.means[i], obsd.covariance[i, i] = transform(
                            np.array(obsd.means[i], ndmin=1),
                            np.array(obsd.covariance[i, i], ndmin=1),
                        )
            else:
                obsd.means, obsd.covariance = transform(obsd.means, obsd.covariance)
        return observation_data

    def _transform_observation_data(
        self,
        observation_data: list[ObservationData],
    ) -> list[ObservationData]:
        return self._tf_obs_data(observation_data, self._transform)

    def _untransform_observation_data(
        self,
        observation_data: list[ObservationData],
    ) -> list[ObservationData]:
        return self._tf_obs_data(observation_data, self._untransform)

    def untransform_outcome_constraints(
        self,
        outcome_constraints: list[OutcomeConstraint],
        fixed_features: ObservationFeatures | None = None,
    ) -> list[OutcomeConstraint]:
        for c in outcome_constraints:
            if c.metric.signature in self.metric_signatures:
                if c.relative:
                    raise ValueError("Unexpected relative transform.")
                c.bound = np.exp(c.bound)
        return outcome_constraints


def lognorm_to_norm(
    mu_ln: npt.NDArray,
    Cov_ln: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute mean and covariance of a MVN from those of the associated log-MVN.

    If `Y` is log-normal with mean mu_ln and covariance Cov_ln, then
    `X ~ N(mu_n, Cov_n)` with

        Cov_n_{ij} = log(1 + Cov_ln_{ij} / (mu_ln_{i} * mu_n_{j}))
        mu_n_{i} = log(mu_ln_{i}) - 0.5 * log(1 + Cov_ln_{ii} / mu_ln_{i}**2)

    NOTE: If the observation noise is not provided, we simply log-transform the
    mean as if the observation noise was zero. This can be inaccurate when the
    unknown observation noise is large.
    """
    Cov_n = np.log(1 + Cov_ln / np.outer(mu_ln, mu_ln))
    mu_n = np.log(mu_ln) - 0.5 * np.nan_to_num(np.diag(Cov_n), nan=0.0)
    return mu_n, Cov_n


def norm_to_lognorm(
    mu_n: npt.NDArray,
    Cov_n: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute mean and covariance of a log-MVN from its MVN sufficient statistics.

    If `X ~ N(mu_n, Cov_n)` and `Y = exp(X)`, then `Y` is log-normal with

        mu_ln_{i} = exp(mu_n_{i}) + 0.5 * Cov_n_{ii}
        Cov_ln_{ij} = exp(mu_n_{i} + mu_n_{j} + 0.5 * (Cov_n_{ii} + Cov_n_{jj})) *
        (exp(Cov_n_{ij}) - 1)


    NOTE: If the observation noise is not provided, we simply take the exponent of the
    mean as if the observation noise was zero. This can be inaccurate when the
    unknown observation noise is large.
    """
    diag_n = np.diag(Cov_n)
    b = mu_n + 0.5 * np.nan_to_num(diag_n, nan=0.0)
    mu_ln = np.exp(b)
    Cov_ln = (np.exp(Cov_n) - 1) * np.exp(b.reshape(-1, 1) + b.reshape(1, -1))
    return mu_ln, Cov_ln
