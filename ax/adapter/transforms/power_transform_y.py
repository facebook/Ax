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
from ax.adapter.transforms.utils import get_data, match_ci_width_truncated
from ax.core.observation import Observation, ObservationData, ObservationFeatures
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
        observations: list[Observation] | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: adapter_module.base.Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        """Initialize the ``PowerTransformY`` transform.

        Args:
            search_space: The search space of the experiment.
            observations: A list of observations from the experiment.
            experiment_data: A container for the parameterizations, metadata and
                observations for the trials in the experiment.
                Constructed using ``extract_experiment_data``.
            adapter: Adapter for referencing experiment, status quo, etc.
            config: A dictionary of options to control the behavior of the transform.
                Can contain the following keys:
                - "metrics": A list of metric names to apply the transform to. If
                    omitted, all metrics found in `observations` are transformed.
                - "clip_mean": Whether to clip the mean to the image of the transform.
                    Defaults to True.
        """
        super().__init__(
            search_space=search_space,
            observations=observations,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        # pyre-fixme[9]: Can't annotate config["metrics"] properly.
        metric_names: list[str] | None = config.get("metrics", None) if config else None
        self.clip_mean: bool = (
            assert_is_instance(config.get("clip_mean", True), bool) if config else True
        )
        if experiment_data is not None:
            means_df = experiment_data.observation_data["mean"]
            # Dropping NaNs here since the DF will have NaN for missing values.
            Ys = {name: column.dropna().values for name, column in means_df.items()}
        else:
            observation_data = [obs.data for obs in none_throws(observations)]
            Ys = get_data(observation_data=observation_data, metric_names=metric_names)
        self.metric_names: list[str] = list(Ys.keys())
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
            for i, m in enumerate(obsd.metric_names):
                if m in self.metric_names:
                    transform = self.power_transforms[m].transform
                    obsd.means[i], obsd.covariance[i, i] = match_ci_width_truncated(
                        mean=obsd.means[i],
                        variance=obsd.covariance[i, i],
                        transform=lambda y: transform(np.array(y, ndmin=2)),
                        lower_bound=-np.inf,
                        upper_bound=np.inf,
                    )
        return observation_data

    def _untransform_observation_data(
        self,
        observation_data: list[ObservationData],
    ) -> list[ObservationData]:
        """Winsorize observation data in place."""
        for obsd in observation_data:
            for i, m in enumerate(obsd.metric_names):
                if m in self.metric_names:
                    l, u = self.inv_bounds[m]
                    transform = self.power_transforms[m].inverse_transform
                    if not self.clip_mean and (obsd.means[i] < l or obsd.means[i] > u):
                        raise ValueError(
                            "Can't untransform mean outside the bounds without clipping"
                        )
                    obsd.means[i], obsd.covariance[i, i] = match_ci_width_truncated(
                        mean=obsd.means[i],
                        variance=obsd.covariance[i, i],
                        transform=lambda y: transform(np.array(y, ndmin=2)),
                        lower_bound=l,
                        upper_bound=u,
                        clip_mean=True,
                    )
        return observation_data

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        adapter: adapter_module.base.Adapter | None = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> OptimizationConfig:
        for c in optimization_config.all_constraints:
            if isinstance(c, ScalarizedOutcomeConstraint):
                c_metric_names = [metric.name for metric in c.metrics]
                intersection = set(c_metric_names) & set(self.metric_names)
                if intersection:
                    raise NotImplementedError(
                        f"PowerTransformY cannot be used for metric(s) {intersection} "
                        "that are part of a ScalarizedOutcomeConstraint."
                    )
            elif c.metric.name in self.metric_names:
                if c.relative:
                    raise ValueError(
                        f"PowerTransformY cannot be applied to metric {c.metric.name} "
                        "since it is subject to a relative constraint."
                    )
                else:
                    transform = self.power_transforms[c.metric.name].transform
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
            elif c.metric.name in self.metric_names:
                if c.relative:
                    raise ValueError("Relative constraints not supported here.")
                else:
                    transform = self.power_transforms[c.metric.name].inverse_transform
                    c.bound = transform(np.array(c.bound, ndmin=2)).item()
        return outcome_constraints


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
