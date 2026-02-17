#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.log_y import match_ci_width
from ax.core.observation import ObservationData
from ax.core.search_space import SearchSpace
from ax.generators.types import TConfig

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


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
        experiment_data: ExperimentData | None = None,
        adapter: adapter_module.base.Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        """Initialize the ``BilogY`` transform.

        Args:
            search_space: The search space of the experiment.
            experiment_data: A container for the parameterizations, metadata and
                observations for the trials in the experiment.
                Constructed using ``extract_experiment_data``.
            adapter: Adapter for referencing experiment, status quo, etc.
            config: A dictionary of options specific to each transform.
        """
        super().__init__(
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        if adapter is not None and adapter._optimization_config is not None:
            # TODO @deriksson: Add support for relative outcome constraints
            self.metric_to_bound: dict[str, float] = {
                oc.metric.signature: oc.bound
                for oc in adapter._optimization_config.outcome_constraints
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
        transform: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
    ) -> list[ObservationData]:
        for obsd in observation_data:
            for i, m in enumerate(obsd.metric_signatures):
                if m in self.metric_to_bound.keys():
                    bound = self.metric_to_bound[m]
                    obsd.means[i], obsd.covariance[i, i] = match_ci_width(
                        mean=obsd.means[i],
                        sem=None,
                        variance=obsd.covariance[i, i],
                        transform=lambda y, bound=bound: transform(y, bound),
                    )
        return observation_data

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        obs_data = experiment_data.observation_data
        metric_signatures = experiment_data.metric_signatures
        for metric, bound in self.metric_to_bound.items():
            if metric not in metric_signatures:
                continue
            obs_data[("mean", metric)], obs_data[("sem", metric)] = match_ci_width(
                mean=obs_data[("mean", metric)].to_numpy(),
                sem=obs_data[("sem", metric)].to_numpy(),
                variance=None,
                transform=lambda y, bound=bound: bilog_transform(y, bound),
            )
        return ExperimentData(
            arm_data=experiment_data.arm_data, observation_data=obs_data
        )


def bilog_transform(y: npt.NDArray, bound: npt.NDArray | float) -> npt.NDArray:
    """Bilog transform: f(y) = bound + sign(y - bound) * log(|y - bound| + 1)"""
    diff = y - bound
    return bound + np.sign(diff) * np.log(np.abs(diff) + 1)


def inv_bilog_transform(y: npt.NDArray, bound: npt.NDArray | float) -> npt.NDArray:
    """Inverse bilog transform: f(y) = bound + sign(y - bound) * expm1(|y - bound|)"""
    diff = y - bound
    sign = np.sign(diff)
    return bound + sign * np.expm1(diff * sign)
