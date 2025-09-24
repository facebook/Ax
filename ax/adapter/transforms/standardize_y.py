#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import defaultdict
from logging import Logger
from typing import Optional, TYPE_CHECKING

import numpy as np
from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint, ScalarizedOutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.core.types import TParamValue
from ax.exceptions.core import DataRequiredError
from ax.generators.types import TConfig
from ax.utils.common.logger import get_logger
from pyre_extensions import none_throws


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax.adapter import base as base_adapter  # noqa F401


logger: Logger = get_logger(__name__)


class StandardizeY(Transform):
    """Standardize Y, separately for each metric.

    Transform is done in-place.
    """

    requires_data_for_initialization: bool = True

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: Optional["base_adapter.Adapter"] = None,
        config: TConfig | None = None,
    ) -> None:
        super().__init__(
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        means_df = none_throws(experiment_data).observation_data["mean"]
        # Dropping NaNs here since the DF will have NaN for missing values.
        Ys = {
            signature: column.dropna().values for signature, column in means_df.items()
        }
        # Compute means and SDs
        # pyre-fixme[4]: Attribute must be annotated.
        self.Ymean, self.Ystd = compute_standardization_parameters(Ys=Ys)

    def _transform_observation_data(
        self,
        observation_data: list[ObservationData],
    ) -> list[ObservationData]:
        # Transform observation data
        for obsd in observation_data:
            means = np.array([self.Ymean[m] for m in obsd.metric_signatures])
            stds = np.array([self.Ystd[m] for m in obsd.metric_signatures])
            obsd.means = (obsd.means - means) / stds
            obsd.covariance /= np.dot(stds[:, None], stds[:, None].transpose())
        return observation_data

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        adapter: Optional["base_adapter.Adapter"] = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> OptimizationConfig:
        for c in optimization_config.all_constraints:
            if c.relative:
                raise ValueError(
                    f"StandardizeY transform does not support relative constraint {c}"
                )
            # For required data checks, metrics must be available in Ymean and Ystd.
            available_metrics = set(self.Ymean).intersection(set(self.Ystd))
            if isinstance(c, ScalarizedOutcomeConstraint):
                # check metrics are present.
                constraint_metrics = {metric.signature for metric in c.metrics}
                if len(constraint_metrics - available_metrics) > 0:
                    raise DataRequiredError(
                        "`StandardizeY` transform requires constraint metric(s) "
                        f"{constraint_metrics} but received only {available_metrics}."
                    )

                # transform \sum (wi * yi) <= C to
                # \sum (wi * si * zi) <= C - \sum (wi * mu_i) that zi = (yi - mu_i) / si

                # update bound C to new c = C.bound - sum_i (wi * mu_i)
                agg_mean = np.sum(
                    [
                        c.weights[i] * self.Ymean[metric.signature]
                        for i, metric in enumerate(c.metrics)
                    ]
                )
                c.bound = float(c.bound - agg_mean)

                # update the weights in the scalarized constraint
                # new wi = wi * si
                new_weight = [
                    c.weights[i] * self.Ystd[metric.signature]
                    for i, metric in enumerate(c.metrics)
                ]
                c.weights = new_weight
            else:
                if c.metric.signature not in available_metrics:
                    raise DataRequiredError(
                        "`StandardizeY` transform requires constraint metric(s) "
                        f"{c.metric.signature} but got {available_metrics}"
                    )
                c.bound = float(
                    (c.bound - self.Ymean[c.metric.signature])
                    / self.Ystd[c.metric.signature]
                )
        return optimization_config

    def _untransform_observation_data(
        self,
        observation_data: list[ObservationData],
    ) -> list[ObservationData]:
        for obsd in observation_data:
            means = np.array([self.Ymean[m] for m in obsd.metric_signatures])
            stds = np.array([self.Ystd[m] for m in obsd.metric_signatures])
            obsd.means = obsd.means * stds + means
            obsd.covariance *= np.dot(stds[:, None], stds[:, None].transpose())
        return observation_data

    def untransform_outcome_constraints(
        self,
        outcome_constraints: list[OutcomeConstraint],
        fixed_features: ObservationFeatures | None = None,
    ) -> list[OutcomeConstraint]:
        for c in outcome_constraints:
            if c.relative:
                raise ValueError(
                    f"StandardizeY transform does not support relative constraint {c}"
                )
            if isinstance(c, ScalarizedOutcomeConstraint):
                raise ValueError("ScalarizedOutcomeConstraint not supported")
            c.bound = float(
                c.bound * self.Ystd[c.metric.signature] + self.Ymean[c.metric.signature]
            )
        return outcome_constraints

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        obs_data = experiment_data.observation_data
        # Process metrics one by one.
        for metric in self.Ymean:
            mean_col = obs_data["mean", metric]
            obs_data["mean", metric] = (mean_col - self.Ymean[metric]) / self.Ystd[
                metric
            ]
            sem_col = obs_data["sem", metric]
            if sem_col.isnull().all():
                # If SEM is NaN, we don't need to transform it.
                continue
            obs_data["sem", metric] = sem_col / self.Ystd[metric]
        return ExperimentData(
            arm_data=experiment_data.arm_data, observation_data=obs_data
        )


def compute_standardization_parameters(
    Ys: defaultdict[str | tuple[str, TParamValue], list[float]],
) -> tuple[dict[str | tuple[str, str], float], dict[str | tuple[str, str], float]]:
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
            logger.warning(f"Outcome {k} is constant, within tolerance.")
    # pyre-fixme[7]: Expected `Tuple[Dict[Union[Tuple[str, str], str], float],
    #  Dict[Union[Tuple[str, str], str], float]]` but got `Tuple[Dict[Union[Tuple[str,
    #  Union[None, bool, float, int, str]], str], typing.Any], Dict[Union[Tuple[str,
    #  Union[None, bool, float, int, str]], str], typing.Any]]`.
    return Ymean, Ystd
