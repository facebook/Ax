#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Callable
from logging import Logger

from math import sqrt
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.core.utils import get_target_trial_index
from ax.modelbridge.transforms.relativize import BaseRelativize, get_metric_index
from ax.models.types import TConfig
from ax.utils.common.logger import get_logger
from ax.utils.stats.statstools import relativize, unrelativize
from pyre_extensions import assert_is_instance, none_throws

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401
logger: Logger = get_logger(__name__)


class TransformToNewSQ(BaseRelativize):
    """Map relative values of one batch to SQ of another.

    Will compute the relative metrics for each arm in each batch, and will then turn
    those back into raw metrics but using the status quo values set on the Modelbridge.

    This is useful if batches are comparable on a relative scale, but
    have offset in their status quo. This is often approximately true for online
    experiments run in separate batches.

    Note that relativization is done using the delta method, so it will not
    simply be the ratio of the means."""

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        modelbridge: modelbridge_module.base.ModelBridge | None = None,
        config: TConfig | None = None,
    ) -> None:
        super().__init__(
            search_space=search_space,
            observations=observations,
            modelbridge=modelbridge,
            config=config,
        )
        self._status_quo_name: str = none_throws(
            none_throws(modelbridge).status_quo_name
        )

        target_trial_index = None

        if config is not None:
            target_trial_index = config.get("target_trial_index", None)

        if (
            target_trial_index is None
            and modelbridge is not None
            and modelbridge._experiment is not None
        ):
            target_trial_index = get_target_trial_index(
                experiment=modelbridge._experiment
            )
            trials_indices_with_sq_data = self.status_quo_data_by_trial.keys()
            if target_trial_index not in trials_indices_with_sq_data:
                target_trial_index = max(trials_indices_with_sq_data)
                logger.info(
                    "No SQ data for target trial. Failing back to "
                    f"{target_trial_index}."
                )

        if target_trial_index is not None:
            self.default_trial_idx: int = assert_is_instance(
                target_trial_index,
                int,
            )

    @property
    def control_as_constant(self) -> bool:
        """Whether or not the control is treated as a constant in the model."""
        return True

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        modelbridge: modelbridge_module.base.ModelBridge | None = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> OptimizationConfig:
        return optimization_config

    def untransform_outcome_constraints(
        self,
        outcome_constraints: list[OutcomeConstraint],
        fixed_features: ObservationFeatures | None = None,
    ) -> list[OutcomeConstraint]:
        return outcome_constraints

    def _get_relative_data_from_obs(
        self,
        obs: Observation,
        rel_op: Callable[..., tuple[npt.NDArray, npt.NDArray]],
    ) -> ObservationData:
        idx = (
            int(obs.features.trial_index)
            if obs.features.trial_index is not None
            else self.default_trial_idx
        )
        if idx == self.default_trial_idx:
            # don't transform data from target batch
            return obs.data
        return super()._get_relative_data_from_obs(
            obs=obs,
            rel_op=rel_op,
        )

    def _rel_op_on_observations(
        self,
        observations: list[Observation],
        rel_op: Callable[..., tuple[npt.NDArray, npt.NDArray]],
    ) -> list[Observation]:
        rel_observations = super()._rel_op_on_observations(
            observations=observations,
            rel_op=rel_op,
        )
        return [
            obs
            for obs in rel_observations
            # drop SQ observations
            if (
                obs.arm_name != self._status_quo_name
                or obs.features.trial_index == self.default_trial_idx
            )
        ]

    def _get_relative_data(
        self,
        data: ObservationData,
        status_quo_data: ObservationData,
        rel_op: Callable[..., tuple[npt.NDArray, npt.NDArray]],
    ) -> ObservationData:
        r"""
        Transform or untransform `data` based on `status_quo_data` based on `rel_op`.

        Args:
            data: ObservationData object to relativize
            status_quo_data: The status quo data associated with the specific trial
                that `data` belongs to.
            rel_op: relativize or unrelativize operator.
            control_as_constant: if treating the control metric as constant

        Returns:
            (un)transformed ObservationData
        """
        L = len(data.metric_names)
        result = ObservationData(
            metric_names=data.metric_names,
            # zeros are just to create the shape so values can be set by index
            means=np.zeros(L),
            covariance=np.zeros((L, L)),
        )
        for i, metric in enumerate(data.metric_names):
            j = get_metric_index(data=status_quo_data, metric_name=metric)
            means_t = data.means[i]
            sems_t = sqrt(data.covariance[i][i])
            mean_c = status_quo_data.means[j]
            sem_c = sqrt(status_quo_data.covariance[j][j])
            means_rel, sems_rel = self._get_rel_mean_sem(
                means_t=means_t,
                sems_t=sems_t,
                mean_c=mean_c,
                sem_c=sem_c,
                metric=metric,
                rel_op=rel_op,
            )
            result.means[i] = means_rel
            result.covariance[i][i] = sems_rel**2
        return result

    def _get_rel_mean_sem(
        self,
        means_t: float,
        sems_t: float,
        mean_c: float,
        sem_c: float,
        metric: str,
        rel_op: Callable[..., tuple[npt.NDArray, npt.NDArray]],
    ) -> tuple[float, float]:
        """Compute (un)transformed mean and sem for a single metric."""
        target_status_quo_data = self.status_quo_data_by_trial[self.default_trial_idx]
        j = get_metric_index(data=target_status_quo_data, metric_name=metric)
        target_mean_c = target_status_quo_data.means[j]
        abs_target_mean_c = np.abs(target_mean_c)
        if rel_op == unrelativize:
            means_t = (means_t - target_mean_c) / abs_target_mean_c
            sems_t = sems_t / abs_target_mean_c
        means_rel, sems_rel = rel_op(
            means_t=means_t,
            sems_t=sems_t,
            mean_c=mean_c,
            sem_c=sem_c,
            as_percent=False,
            control_as_constant=self.control_as_constant,
        )
        if rel_op == relativize:
            means_rel = means_rel * abs_target_mean_c + target_mean_c
            sems_rel = sems_rel * abs_target_mean_c
        # pyre-fixme[7]: Expected `Tuple[float, float]` but got
        #  `Tuple[ndarray[typing.Any, typing.Any], ndarray[typing.Any, typing.Any]]`.
        return means_rel, sems_rel
