#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from ax.core.observation import Observation, ObservationData
from ax.core.search_space import SearchSpace
from ax.exceptions.core import DataRequiredError
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.log_y import match_ci_width
from ax.models.types import TConfig

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401


class BilogY(Transform):
    """Apply a bilog-transform to the outcome constraint metrics.

    The bilog transform was introduced in
    https://proceedings.mlr.press/v130/eriksson21a/eriksson21a.pdf, with the key idea
    being to magnify the region around the outcome constraint constraint boundary,
    thus making it easier to model that part of the metric space. We use a matching
    procedure based on the width of the CIs to transform thevariance observations.
    This transform is applied in-place.

    We define the transform as: f(y) = bound + sign(y - bound) * log(|y - bound| + 1)
    where bound is the outcome constraint bound. This has the properties:
        1. It magnifies the region around y=bound, which makes it easier to
            model the constraint boundary.
        2. f(bound) = bound, so we don't have to modify the optimization config.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        modelbridge: modelbridge_module.base.Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        """Initialize the ``BilogY`` transform.

        Args:
            search_space: The search space of the experiment. Unused.
            observations: A list of observations from the experiment.
            modelbridge: The `Adapter` within which the transform is used.
        """
        if observations is None or len(observations) == 0:
            raise DataRequiredError("BilogY requires observations.")
        if modelbridge is not None and modelbridge._optimization_config is not None:
            # TODO @deriksson: Add support for relative outcome constraints
            self.metric_to_bound: dict[str, float] = {
                oc.metric.name: oc.bound
                for oc in modelbridge._optimization_config.outcome_constraints
                if not oc.relative
            }
        else:
            self.metric_to_bound = {}

    def _transform_observation_data(
        self,
        observation_data: list[ObservationData],
    ) -> list[ObservationData]:
        """Apply bilog to observation data in place."""
        return self._reusable_transform(
            observation_data=observation_data,
            transform=bilog_transform,
        )

    def _untransform_observation_data(
        self,
        observation_data: list[ObservationData],
    ) -> list[ObservationData]:
        """Apply inverse bilog to observation data in place."""
        return self._reusable_transform(
            observation_data=observation_data,
            transform=inv_bilog_transform,
        )

    def _reusable_transform(
        self,
        observation_data: list[ObservationData],
        # pyre-fixme[11]: Annotation `npt.NDarray` is not defined as a type.
        transform: Callable[[npt.NDArray, npt.NDarray], npt.NDarray],
    ) -> list[ObservationData]:
        for obsd in observation_data:
            for i, m in enumerate(obsd.metric_names):
                if m in self.metric_to_bound.keys():
                    bound = self.metric_to_bound[m]
                    obsd.means[i], obsd.covariance[i, i] = match_ci_width(
                        mean=obsd.means[i],
                        variance=obsd.covariance[i, i],
                        transform=lambda y, bound=bound: transform(y, bound=bound),
                    )
        return observation_data


def bilog_transform(y: npt.NDarray, bound: npt.NDarray) -> npt.NDarray:
    """Bilog transform: f(y) = bound + sign(y - bound) * log(|y - bound| + 1)"""
    return bound + np.sign(y - bound) * np.log(np.abs(y - bound) + 1)


def inv_bilog_transform(y: npt.NDarray, bound: npt.NDarray) -> npt.NDarray:
    """Inverse bilog transform: f(y) = bound + sign(y - bound) * expm1(|y - bound|)"""
    return bound + np.sign(y - bound) * np.expm1((y - bound) * np.sign(y - bound))
