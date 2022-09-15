#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
from logging import Logger
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint, ScalarizedOutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.utils import get_data, match_ci_width_truncated
from ax.models.types import TConfig
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast_list
from sklearn.preprocessing import PowerTransformer

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401  # pragma: no cover


logger: Logger = get_logger(__name__)


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

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        observations: Optional[List[Observation]] = None,
        modelbridge: Optional[modelbridge_module.base.ModelBridge] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        assert observations is not None, "PowerTransformY requires observations"
        if config is None:
            raise ValueError("PowerTransform requires a config.")
        # pyre-fixme[6]: Same issue as for LogY
        metric_names = list(config.get("metrics", []))
        if len(metric_names) == 0:
            raise ValueError("Must specify at least one metric in the config.")
        # pyre-fixme[4]: Attribute must be annotated.
        self.clip_mean = config.get("clip_mean", True)
        # pyre-fixme[4]: Attribute must be annotated.
        self.metric_names = metric_names
        observation_data = [obs.data for obs in observations]
        Ys = get_data(observation_data=observation_data, metric_names=metric_names)
        # pyre-fixme[4]: Attribute must be annotated.
        self.power_transforms = _compute_power_transforms(Ys=Ys)
        # pyre-fixme[4]: Attribute must be annotated.
        self.inv_bounds = _compute_inverse_bounds(self.power_transforms, tol=1e-10)

    def _transform_observation_data(
        self,
        observation_data: List[ObservationData],
    ) -> List[ObservationData]:
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
        observation_data: List[ObservationData],
    ) -> List[ObservationData]:
        """Winsorize observation data in place."""
        for obsd in observation_data:
            for i, m in enumerate(obsd.metric_names):
                if m in self.metric_names:
                    l, u = self.inv_bounds[m]
                    transform = self.power_transforms[m].inverse_transform
                    if not self.clip_mean and (obsd.means[i] < l or obsd.means[i] > u):
                        raise ValueError(  # pragma: no cover
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
        modelbridge: Optional[modelbridge_module.base.ModelBridge] = None,
        fixed_features: Optional[ObservationFeatures] = None,
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
        outcome_constraints: List[OutcomeConstraint],
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> List[OutcomeConstraint]:
        for c in outcome_constraints:
            if isinstance(c, ScalarizedOutcomeConstraint):
                raise ValueError(  # pragma: no cover
                    "ScalarizedOutcomeConstraint not supported here"
                )
            elif c.metric.name in self.metric_names:
                if c.relative:
                    raise ValueError("Relative constraints not supported here.")
                else:
                    transform = self.power_transforms[c.metric.name].inverse_transform
                    c.bound = transform(np.array(c.bound, ndmin=2)).item()
        return outcome_constraints


def _compute_power_transforms(
    Ys: Dict[str, List[float]]
) -> Dict[str, PowerTransformer]:
    """Compute power transforms."""
    power_transforms = {}
    for k, ys in Ys.items():
        y = np.array(ys)[:, None]  # Need to unsqueeze the last dimension
        pt = PowerTransformer(method="yeo-johnson").fit(y)
        power_transforms[k] = pt
    return power_transforms


def _compute_inverse_bounds(
    power_transforms: Dict[str, PowerTransformer], tol: float = 1e-10
) -> Dict[str, Tuple[float, float]]:
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
        inv_bounds[k] = tuple(checked_cast_list(float, bounds))
    return inv_bounds
