#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.utils import match_ci_width
from ax.core.objective import ScalarizedObjective
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint, ScalarizedOutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.generators.types import TConfig
from ax.utils.common.typeutils import assert_is_instance_list
from pyre_extensions import assert_is_instance, none_throws
from sklearn.preprocessing import PowerTransformer


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


class PowerTransformY(Transform):
    """Transform the values to look as normally distributed as possible.

    This fits a power transform to the data with the goal of making the transformed
    values look as normally distributed as possible. We use Yeo-Johnson
    (https://www.stat.umn.edu/arc/yjpower.pdf), which can handle both positive and
    negative values.

    While the transform seems to be quite robust, it probably makes sense to apply a
    bit of winsorization and also standardize the inputs before applying the power
    transform. The power transform will automatically standardize the data so the
    data will remain standardized.

    The transform can't be inverted for all values, so we apply clipping to move
    values to the image of the transform. This behavior can be controlled via the
    `clip_mean` setting.
    """

    requires_data_for_initialization: bool = True

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: adapter_module.base.Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        """Initialize the ``PowerTransformY`` transform.

        Args:
            search_space: The search space of the experiment.
            experiment_data: A container for the parameterizations, metadata and
                observations for the trials in the experiment.
                Constructed using ``extract_experiment_data``.
            adapter: Adapter for referencing experiment, status quo, etc.
            config: A dictionary of options to control the behavior of the transform.
                Can contain the following keys:
                - "metrics": A list of metric names to apply the transform to. If
                    omitted, all metrics are transformed.
                - "clip_mean": Whether to clip the mean to the image of the transform.
                    Defaults to True.
        """
        super().__init__(
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        # pyre-fixme[9]: Can't annotate config["metrics"] properly.
        metric_signatures: list[str] | None = self.config.get("metrics", None)
        self.clip_mean: bool = assert_is_instance(
            self.config.get("clip_mean", True), bool
        )
        means_df = none_throws(experiment_data).observation_data["mean"]
        # Dropping NaNs here since the DF will have NaN for missing values.
        Ys = {
            name: column.dropna().values
            for name, column in means_df.items()
            if metric_signatures is None or name in metric_signatures
        }
        self.metric_signatures: list[str] = list(Ys.keys())
        self.power_transforms: dict[str, PowerTransformer] = _compute_power_transforms(
            Ys=Ys
        )
        self.inv_bounds: dict[str, tuple[float, float]] = _compute_inverse_bounds(
            self.power_transforms, tol=1e-10
        )

    def _transform_observation_data(
        self,
        observation_data: list[ObservationData],
    ) -> list[ObservationData]:
        """Winsorize observation data in place."""
        for obsd in observation_data:
            for i, m in enumerate(obsd.metric_signatures):
                if m in self.metric_signatures:
                    transform = self.power_transforms[m].transform
                    mean, cov = match_ci_width(
                        mean=obsd.means[i],
                        sem=None,
                        variance=obsd.covariance[i, i],
                        transform=lambda y, t=transform: t(np.array(y, ndmin=2)),
                    )
                    obsd.means[i] = mean.item()
                    obsd.covariance[i, i] = cov.item()
        return observation_data

    def _untransform_observation_data(
        self,
        observation_data: list[ObservationData],
    ) -> list[ObservationData]:
        """Winsorize observation data in place."""
        for obsd in observation_data:
            for i, m in enumerate(obsd.metric_signatures):
                if m in self.metric_signatures:
                    l, u = self.inv_bounds[m]
                    transform = self.power_transforms[m].inverse_transform
                    if not self.clip_mean and (obsd.means[i] < l or obsd.means[i] > u):
                        raise ValueError(
                            "Can't untransform mean outside the bounds without clipping"
                        )
                    mean, cov = match_ci_width(
                        mean=obsd.means[i],
                        sem=None,
                        variance=obsd.covariance[i, i],
                        transform=lambda y, t=transform: t(np.array(y, ndmin=2)),
                        lower_bound=l + 1e-3,
                        upper_bound=u - 1e-3,
                    )
                    obsd.means[i] = mean.item()
                    obsd.covariance[i, i] = cov.item()
        return observation_data

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        adapter: adapter_module.base.Adapter | None = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> OptimizationConfig:
        if isinstance(optimization_config.objective, ScalarizedObjective):
            objective_metric_signatures = [
                metric.signature for metric in optimization_config.objective.metrics
            ]
            intersection = set(objective_metric_signatures) & set(
                self.metric_signatures
            )
            if intersection:
                raise NotImplementedError(
                    "PowerTransformY cannot be used for metric(s) "
                    f"{intersection} that are part of a ScalarizedObjective. "
                    "The power transform is a non-linear transformation and cannot "
                    "preserve the linear scalarization of the objective."
                )

        for c in optimization_config.all_constraints:
            if isinstance(c, ScalarizedOutcomeConstraint):
                c_metric_signatures = [metric.signature for metric in c.metrics]
                intersection = set(c_metric_signatures) & set(self.metric_signatures)
                if intersection:
                    raise NotImplementedError(
                        "PowerTransformY cannot be used for metric(s) "
                        f"{intersection} that are part of a "
                        "ScalarizedOutcomeConstraint."
                    )
            elif c.metric.signature in self.metric_signatures:
                if c.relative:
                    raise ValueError(
                        "PowerTransformY cannot be applied to metric "
                        f"{c.metric.signature} since it is subject to "
                        "a relative constraint."
                    )
                else:
                    transform = self.power_transforms[c.metric.signature].transform
                    c.bound = transform(np.array(c.bound, ndmin=2)).item()
        return optimization_config

    def untransform_outcome_constraints(
        self,
        outcome_constraints: list[OutcomeConstraint],
        fixed_features: ObservationFeatures | None = None,
    ) -> list[OutcomeConstraint]:
        for c in outcome_constraints:
            if isinstance(c, ScalarizedOutcomeConstraint):
                raise ValueError("ScalarizedOutcomeConstraint not supported here")
            elif c.metric.signature in self.metric_signatures:
                if c.relative:
                    raise ValueError("Relative constraints not supported here.")
                else:
                    transform = self.power_transforms[
                        c.metric.signature
                    ].inverse_transform
                    c.bound = transform(np.array(c.bound, ndmin=2)).item()
        return outcome_constraints

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        obs_data = experiment_data.observation_data
        metrics_in_data = experiment_data.metric_signatures
        for metric in self.metric_signatures:
            if metric not in metrics_in_data:
                continue
            new_mean, new_sem = match_ci_width(
                mean=obs_data[("mean", metric)].to_numpy().reshape(-1, 1),
                sem=obs_data[("sem", metric)].to_numpy().reshape(-1, 1),
                variance=None,
                transform=self.power_transforms[metric].transform,
            )
            obs_data[("mean", metric)] = new_mean.flatten()
            obs_data[("sem", metric)] = new_sem.flatten()
        return ExperimentData(
            arm_data=experiment_data.arm_data, observation_data=obs_data
        )


def _compute_power_transforms(
    Ys: dict[str, list[float]],
) -> dict[str, PowerTransformer]:
    """Compute power transforms."""
    power_transforms = {}
    for k, ys in Ys.items():
        y = np.array(ys)[:, None]  # Need to unsqueeze the last dimension
        pt = PowerTransformer(method="yeo-johnson").fit(y)
        power_transforms[k] = pt
    return power_transforms


def _compute_inverse_bounds(
    power_transforms: dict[str, PowerTransformer], tol: float = 1e-10
) -> dict[str, tuple[float, float]]:
    """Computes the image of the transform so we can clip when we untransform.

    The inverse of the Yeo-Johnson transform is given by:
    if X >= 0 and lambda == 0:
        X = exp(X_trans) - 1
    elif X >= 0 and lambda != 0:
        X = (X_trans * lambda + 1) ** (1 / lambda) - 1
    elif X < 0 and lambda != 2:
        X = 1 - (-(2 - lambda) * X_trans + 1) ** (1 / (2 - lambda))
    elif X < 0 and lambda == 2:
        X = 1 - exp(-X_trans)

    We can break this down into three cases:
    lambda < 0:        X < -1 / lambda
    0 <= lambda <= 2:  X is unbounded
    lambda > 2:        X > 1 / (2 - lambda)

    Sklearn standardizes the transformed values to have mean zero and standard
    deviation 1, so we also need to account for this when we compute the bounds.
    """
    inv_bounds = defaultdict()
    for k, pt in power_transforms.items():
        bounds = [-np.inf, np.inf]
        mu, sigma = pt._scaler.mean_.item(), pt._scaler.scale_.item()  # pyre-ignore
        lambda_ = pt.lambdas_.item()  # pyre-ignore
        if lambda_ < -1 * tol:
            bounds[1] = (-1.0 / lambda_ - mu) / sigma
        elif lambda_ > 2.0 + tol:
            bounds[0] = (1.0 / (2.0 - lambda_) - mu) / sigma
        inv_bounds[k] = tuple(assert_is_instance_list(bounds, float))
    return inv_bounds
