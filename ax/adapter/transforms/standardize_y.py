#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections import defaultdict
from logging import Logger
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.core.objective import Objective
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp, TParamValue
from ax.exceptions.core import DataRequiredError
from ax.generators.types import TConfig
from ax.utils.common.logger import get_logger
from ax.utils.common.sympy import build_constraint_expression_str
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
        self.Ymean: dict[str, float]
        self.Ystd: dict[str, float]
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

    def _check_metrics_available(
        self, metric_signatures: list[str], context: str
    ) -> set[str]:
        """Check that all metrics are available and return the set of metrics."""
        available_metrics = set(self.Ymean).intersection(set(self.Ystd))
        required_metrics = set(metric_signatures)
        if len(required_metrics - available_metrics) > 0:
            raise DataRequiredError(
                f"`StandardizeY` transform requires {context} metric(s) "
                f"{required_metrics} but received only {available_metrics}."
            )
        return available_metrics

    def _transform_scalarized_weights(
        self, metric_signatures: list[str], weights: list[float]
    ) -> list[float]:
        """Transform weights for scalarized objectives/constraints.

        When standardizing yi to zi = (yi - μi) / σi, the scalarized term
        Σ(wi * yi) transforms to Σ(wi * σi * zi) + Σ(wi * μi).
        This method returns the new weights: new_wi = wi * σi.

        Args:
            metric_signatures: List of metric signatures in the scalarized term.
            weights: Original weights for each metric.

        Returns:
            Transformed weights scaled by standard deviations.
        """
        return [
            weights[i] * float(self.Ystd[sig])
            for i, sig in enumerate(metric_signatures)
        ]

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        adapter: base_adapter.Adapter | None = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> OptimizationConfig:
        # Handle scalarized objective (linear combination of metrics).
        if optimization_config.objective.is_scalarized_objective:
            objective = optimization_config.objective
            obj_sigs = [
                self._get_metric_signature(n, adapter) for n in objective.metric_names
            ]
            self._check_metrics_available(obj_sigs, context="objective")
            old_weights = [w for _, w in objective.metric_weights]
            new_weights = self._transform_scalarized_weights(obj_sigs, old_weights)
            new_metric_weights = [
                (name, new_w)
                for (name, _), new_w in zip(objective.metric_weights, new_weights)
            ]
            optimization_config.objective = _build_objective_from_metric_weights(
                new_metric_weights
            )

        new_constraints = self._transform_constraints(
            optimization_config.outcome_constraints, adapter
        )
        optimization_config.outcome_constraints = new_constraints

        # For MOO configs, transform objective thresholds separately.
        if isinstance(optimization_config, MultiObjectiveOptimizationConfig):
            new_thresholds = self._transform_constraints(
                optimization_config.objective_thresholds, adapter
            )
            optimization_config.objective_thresholds = new_thresholds

        return optimization_config

    def _transform_constraints(
        self,
        constraints: list[OutcomeConstraint],
        adapter: base_adapter.Adapter | None = None,
    ) -> list[OutcomeConstraint]:
        """Transform a list of constraints by standardizing bounds."""
        new_constraints = []
        for c in constraints:
            if c.relative:
                raise ValueError(
                    f"StandardizeY transform does not support relative constraint {c}"
                )
            if len(c.metric_names) > 1:
                c_sigs = [
                    self._get_metric_signature(n, adapter) for n in c.metric_names
                ]
                self._check_metrics_available(c_sigs, context="constraint")

                # Transform Σ(wi * yi) <= C to Σ(wi * σi * zi) <= C - Σ(wi * μi)
                # where zi = (yi - μi) / σi
                old_weights = [w for _, w in c.metric_weights]
                agg_mean = np.sum(
                    [
                        old_weights[i] * float(self.Ymean[sig])
                        for i, sig in enumerate(c_sigs)
                    ]
                )
                new_bound = float(c.bound - agg_mean)
                new_weights = self._transform_scalarized_weights(c_sigs, old_weights)
                new_metric_weights = [
                    (name, new_w)
                    for (name, _), new_w in zip(c.metric_weights, new_weights)
                ]
                op_str = ">=" if c.op == ComparisonOp.GEQ else "<="
                new_constraints.append(
                    OutcomeConstraint(
                        expression=build_constraint_expression_str(
                            metric_weights=new_metric_weights,
                            op=op_str,
                            bound=new_bound,
                            relative=False,
                        )
                    )
                )
            else:
                c_sig = self._get_metric_signature(c.metric_names[0], adapter)
                self._check_metrics_available([c_sig], context="constraint")
                new_bound = float((c.bound - self.Ymean[c_sig]) / self.Ystd[c_sig])
                op_str = ">=" if c.op == ComparisonOp.GEQ else "<="
                new_constraints.append(
                    OutcomeConstraint(
                        expression=build_constraint_expression_str(
                            metric_weights=c.metric_weights,
                            op=op_str,
                            bound=new_bound,
                            relative=c.relative,
                        )
                    )
                )
        return new_constraints

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
        new_constraints = []
        for c in outcome_constraints:
            if c.relative:
                raise ValueError(
                    f"StandardizeY transform does not support relative constraint {c}"
                )
            if len(c.metric_names) > 1:
                raise ValueError("ScalarizedOutcomeConstraint not supported")
            c_sig = self._get_metric_signature(c.metric_names[0])
            new_bound = float(c.bound * self.Ystd[c_sig] + self.Ymean[c_sig])
            op_str = ">=" if c.op == ComparisonOp.GEQ else "<="
            new_constraints.append(
                OutcomeConstraint(
                    expression=build_constraint_expression_str(
                        metric_weights=c.metric_weights,
                        op=op_str,
                        bound=new_bound,
                        relative=c.relative,
                    )
                )
            )
        return new_constraints

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


def _build_objective_from_metric_weights(
    metric_weights: list[tuple[str, float]],
) -> Objective:
    """Build a new Objective from (metric_name, weight) pairs.

    Args:
        metric_weights: List of (metric_name, weight) tuples.

    Returns:
        A new Objective with the corresponding expression.
    """
    parts: list[str] = []
    for name, w in metric_weights:
        if w == 1.0:
            parts.append(name)
        elif w == -1.0:
            parts.append(f"-{name}")
        else:
            parts.append(f"{w}*{name}")
    expr_str = " + ".join(parts).replace(" + -", " - ")
    return Objective(expression=expr_str)


_TYKey = TypeVar("_TYKey", bound=str | tuple[str, TParamValue])


def compute_standardization_parameters(
    Ys: defaultdict[_TYKey, list[float]] | dict[_TYKey, list[float]],
) -> tuple[dict[_TYKey, float], dict[_TYKey, float]]:
    """Compute mean and std. dev of Ys."""
    Ymean: dict[_TYKey, float] = {k: float(np.mean(y)) for k, y in Ys.items()}
    # We use the Bessel correction term (divide by N-1) here in order to
    # be consistent with the default behavior of torch.std that is used to
    # validate input data standardization in BoTorch.
    Ystd: dict[_TYKey, float] = {
        k: float(np.std(y, ddof=1)) if len(y) > 1 else 0.0 for k, y in Ys.items()
    }
    for k, s in Ystd.items():
        # Don't standardize if variance is too small.
        if s < 1e-8:
            Ystd[k] = 1.0
            logger.warning(f"Outcome {k} is constant, within tolerance.")
    return Ymean, Ystd
