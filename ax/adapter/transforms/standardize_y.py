#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections import defaultdict
from logging import Logger
from typing import TYPE_CHECKING

import numpy as np
from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.core.metric import Metric
from ax.core.objective import ScalarizedObjective
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
        adapter: base_adapter.Adapter | None = None,
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

    def _check_metrics_available(self, metrics: list[Metric], context: str) -> set[str]:
        """Check that all metrics are available and return the set of metrics."""
        available_metrics = set(self.Ymean).intersection(set(self.Ystd))
        required_metrics = {metric.signature for metric in metrics}
        if len(required_metrics - available_metrics) > 0:
            raise DataRequiredError(
                f"`StandardizeY` transform requires {context} metric(s) "
                f"{required_metrics} but received only {available_metrics}."
            )
        return available_metrics

    def _transform_scalarized_weights(
        self, metrics: list[Metric], weights: list[float]
    ) -> list[float]:
        """Transform weights for scalarized objectives/constraints.

        When standardizing yi to zi = (yi - μi) / σi, the scalarized term
        Σ(wi * yi) transforms to Σ(wi * σi * zi) + Σ(wi * μi).
        This method returns the new weights: new_wi = wi * σi.

        Args:
            metrics: List of metrics in the scalarized term.
            weights: Original weights for each metric.

        Returns:
            Transformed weights scaled by standard deviations.
        """
        return [
            weights[i] * float(self.Ystd[metric.signature])
            for i, metric in enumerate(metrics)
        ]

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        adapter: base_adapter.Adapter | None = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> OptimizationConfig:
        # Handle ScalarizedObjective
        if isinstance(optimization_config.objective, ScalarizedObjective):
            objective = optimization_config.objective
            self._check_metrics_available(objective.metrics, context="objective")
            objective.weights = self._transform_scalarized_weights(
                objective.metrics, objective.weights
            )

        for c in optimization_config.all_constraints:
            if c.relative:
                raise ValueError(
                    f"StandardizeY transform does not support relative constraint {c}"
                )
            if isinstance(c, ScalarizedOutcomeConstraint):
                self._check_metrics_available(c.metrics, context="constraint")

                # Transform Σ(wi * yi) <= C to Σ(wi * σi * zi) <= C - Σ(wi * μi)
                # where zi = (yi - μi) / σi
                agg_mean = np.sum(
                    [
                        c.weights[i] * float(self.Ymean[metric.signature])
                        for i, metric in enumerate(c.metrics)
                    ]
                )
                c.bound = float(c.bound - agg_mean)
                c.weights = self._transform_scalarized_weights(c.metrics, c.weights)
            else:
                self._check_metrics_available([c.metric], context="constraint")
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
