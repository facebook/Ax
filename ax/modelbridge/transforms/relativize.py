#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import warnings
from math import sqrt
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.base import Transform
from ax.models.types import TConfig
from ax.utils.common.typeutils import not_none
from ax.utils.stats.statstools import relativize


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401  # pragma: no cover


class Relativize(Transform):
    """
    Change the relative flag of the given relative optimization configuration
    to False. This is needed in order for the new opt config to pass ModelBridge
    that requires non-relativized opt config.

    Also transforms absolute data and opt configs to relative.

    Requires a modelbridge with a status quo set to work.
    """

    MISSING_STATUS_QUO_ERROR = "Cannot relativize data without status quo data"

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        observations: Optional[List[Observation]] = None,
        modelbridge: Optional[modelbridge_module.base.ModelBridge] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        assert observations is not None, "Relativize requires observations"
        super().__init__(
            search_space=search_space,
            observations=observations,
            modelbridge=modelbridge,
            config=config,
        )
        # self.modelbridge should NOT be modified
        # pyre-fixme[4]: Attribute must be annotated.
        self.modelbridge = not_none(
            modelbridge, "Relativize transform requires a modelbridge"
        )
        # pyre-fixme[4]: Attribute must be annotated.
        self.status_quo_by_trial = self._get_status_quo_by_trial(
            observations=observations,
            status_quo_feature=not_none(
                self.modelbridge.status_quo, self.MISSING_STATUS_QUO_ERROR
            ).features,
        )

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        modelbridge: Optional[modelbridge_module.base.ModelBridge] = None,
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> OptimizationConfig:
        r"""
        Change the relative flag of the given relative optimization configuration
        to False. This is needed in order for the new opt config to pass ModelBridge
        that requires non-relativized opt config.

        Args:
            opt_config: Optimization configuaration relative to status quo.

        Returns:
            Optimization configuration relative to status quo with relative flag
            equal to false.

        """
        # Getting constraints
        constraints = [
            constraint.clone() for constraint in optimization_config.outcome_constraints
        ]
        if not all(
            constraint.relative
            for constraint in optimization_config.outcome_constraints
        ):
            raise ValueError(
                "All constraints must be relative to use the Relativize transform."
            )
        for constraint in constraints:
            constraint.relative = False

        if isinstance(optimization_config, MultiObjectiveOptimizationConfig):
            # Getting objective thresholds
            obj_thresholds = [
                obj_threshold.clone()
                for obj_threshold in optimization_config.objective_thresholds
            ]
            for obj_threshold in obj_thresholds:
                if not obj_threshold.relative:
                    raise ValueError(
                        "All objective thresholds must be relative to use "
                        "the Relativize transform."
                    )
                obj_threshold.relative = False

            new_optimization_config = MultiObjectiveOptimizationConfig(
                objective=optimization_config.objective,
                outcome_constraints=constraints,
                objective_thresholds=obj_thresholds,
            )
        else:
            new_optimization_config = OptimizationConfig(
                objective=optimization_config.objective,
                outcome_constraints=constraints,
            )

        return new_optimization_config

    def untransform_outcome_constraints(
        self,
        outcome_constraints: List[OutcomeConstraint],
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> List[OutcomeConstraint]:
        for c in outcome_constraints:
            c.relative = True
        return outcome_constraints

    def transform_observations(
        self,
        observations: List[Observation],
    ) -> List[Observation]:
        return [
            Observation(
                features=obs.features,
                data=self._get_relative_data(
                    data=obs.data,
                    status_quo_data=not_none(
                        self.status_quo_by_trial.get(obs.features.trial_index, None),
                        self.MISSING_STATUS_QUO_ERROR,
                    ),
                ),
                arm_name=obs.arm_name,
            )
            for obs in observations
        ]

    def _untransform_observation_data(
        self,
        observation_data: List[ObservationData],
    ) -> List[ObservationData]:
        warnings.warn(
            "`Relativize._untransform_observation_data()` not yet implemented. "
            "Returning relative data."
        )
        return observation_data

    @staticmethod
    def _get_relative_data(
        data: ObservationData, status_quo_data: ObservationData
    ) -> ObservationData:
        L = len(data.metric_names)
        result = ObservationData(
            metric_names=data.metric_names,
            # zeros are just to create the shape so values can be set by index
            means=np.zeros(L),
            covariance=np.zeros((L, L)),
        )
        for i, metric in enumerate(data.metric_names):
            try:
                j = next(
                    k for k in range(L) if status_quo_data.metric_names[k] == metric
                )
            except (IndexError, StopIteration):  # pragma: no cover
                raise ValueError(  # pragma: no cover
                    "Relativization cannot be performed because "
                    "ObservationData for status quo is missing metrics"
                )

            means_t = data.means[i]
            sems_t = sqrt(data.covariance[i][i])
            mean_c = status_quo_data.means[j]
            sem_c = sqrt(status_quo_data.covariance[j][j])

            # if the is the status quo
            if means_t == mean_c and sems_t == sem_c:
                means_rel, sems_rel = 0, 0
            else:
                means_rel, sems_rel = relativize(
                    means_t=means_t,
                    sems_t=sems_t,
                    mean_c=mean_c,
                    sem_c=sem_c,
                    as_percent=True,
                )
            result.means[i] = means_rel
            result.covariance[i][i] = sems_rel**2
        return result

    @staticmethod
    def _get_status_quo_by_trial(
        observations: List[Observation],
        status_quo_feature: ObservationFeatures,
    ) -> Dict[int, ObservationData]:
        status_quo_signature = json.dumps(status_quo_feature.parameters, sort_keys=True)
        return {
            int(not_none(obs.features.trial_index)): obs.data
            for obs in observations
            if json.dumps(obs.features.parameters, sort_keys=True)
            == status_quo_signature
        }
