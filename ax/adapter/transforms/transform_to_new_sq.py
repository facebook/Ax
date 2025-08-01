#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Callable
from logging import Logger
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.relativize import BaseRelativize, get_metric_index
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.core.utils import get_target_trial_index
from ax.generators.types import TConfig
from ax.utils.common.logger import get_logger
from ax.utils.stats.statstools import relativize, unrelativize
from pyre_extensions import assert_is_instance, none_throws

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401
logger: Logger = get_logger(__name__)


class TransformToNewSQ(BaseRelativize):
    """Map relative values of one batch to SQ of another.

    Will compute the relative metrics for each arm in each batch, and will then turn
    those back into raw metrics but using the status quo values set on the Adapter.

    This is useful if batches are comparable on a relative scale, but
    have offset in their status quo. This is often approximately true for online
    experiments run in separate batches.

    Note that relativization is done using the delta method, so it will not
    simply be the ratio of the means.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        observations: list[Observation] | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: adapter_module.base.Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        """Initialize the transform.

        Args:
            config: Can be used to specify the target trial index. The SQ data from
                the target trial will be used to relativize the data from all other
                trials. If not specified, the target trial will be inferred using
                `get_target_trial_index`.
        """
        super().__init__(
            search_space=search_space,
            observations=observations,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        self._status_quo_name: str = none_throws(none_throws(adapter).status_quo_name)

        target_trial_index = None

        if config is not None:
            target_trial_index = config.get("target_trial_index", None)

        if (
            target_trial_index is None
            and adapter is not None
            and adapter._experiment is not None
        ):
            target_trial_index = get_target_trial_index(experiment=adapter._experiment)
            trials_indices_with_sq_data = self.status_quo_data_by_trial.keys()
            if target_trial_index not in trials_indices_with_sq_data:
                target_trial_index = max(trials_indices_with_sq_data)
                logger.warning(
                    "No status quo data for target trial. Failing back to "
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
        adapter: adapter_module.base.Adapter | None = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> OptimizationConfig:
        return optimization_config

    def untransform_outcome_constraints(
        self,
        outcome_constraints: list[OutcomeConstraint],
        fixed_features: ObservationFeatures | None = None,
    ) -> list[OutcomeConstraint]:
        return outcome_constraints

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        observation_data = experiment_data.observation_data.copy(deep=True)
        all_trial_indices = observation_data.index.get_level_values(
            "trial_index"
        ).unique()
        if not all_trial_indices.isin(self.status_quo_data_by_trial.keys()).all():
            raise ValueError(
                f"{self.__class__.__name__} requires status quo data for all "
                f"trials in the experiment data. Found trial indices "
                f"{all_trial_indices} but status quo data is only available for "
                f"trials {list(self.status_quo_data_by_trial.keys())}."
            )

        trial_indices = observation_data.index.get_level_values("trial_index")
        transform_mask = trial_indices != self.default_trial_idx
        # Get the target trial's status quo data
        target_sq_data = self.status_quo_data_by_trial[self.default_trial_idx]

        metrics = experiment_data.metric_names
        if not transform_mask.any():
            # Nothing to transform, set metrics to empty list to skip the loop.
            # We still need to drop SQ after.
            metrics = []

        for metric in metrics:
            # Create arrays of control values for each row based on trial_index.
            mean_c, sem_c = [], []
            for idx in trial_indices:
                sq_data = self.status_quo_data_by_trial[idx]
                j = get_metric_index(data=sq_data, metric_name=metric)
                mean_c.append(sq_data.means[j])
                sem_c.append(sq_data.covariance[j, j] ** 0.5)
            mean_c = np.array(mean_c)
            sem_c = np.array(sem_c)

            # Only transform rows that are not from the target trial.
            means_t = observation_data.loc[transform_mask, ("mean", metric)]
            sems_t = observation_data.loc[transform_mask, ("sem", metric)]

            # Relativize with respect to original trial's status quo.
            means_rel, sems_rel = relativize(
                means_t=means_t,
                sems_t=sems_t,
                mean_c=mean_c[transform_mask],
                sem_c=sem_c[transform_mask],
                as_percent=False,
                control_as_constant=self.control_as_constant,
            )

            # Unrelativize with respect to target trial's status quo.
            target_j = get_metric_index(data=target_sq_data, metric_name=metric)
            target_mean_c = target_sq_data.means[target_j]
            abs_target_mean_c = np.abs(target_mean_c)
            observation_data.loc[transform_mask, ("mean", metric)] = (
                means_rel * abs_target_mean_c + target_mean_c
            )
            observation_data.loc[transform_mask, ("sem", metric)] = (
                sems_rel * abs_target_mean_c
            )

        # Drop SQ observations from the data -- except for the target trial.
        # Keep rows where arm_name != SQ or trial_index == target_trial_idx.
        observation_data = observation_data[
            (
                observation_data.index.get_level_values("arm_name")
                != self._status_quo_name
            )
            | (
                observation_data.index.get_level_values("trial_index")
                == self.default_trial_idx
            )
        ]
        arm_data = experiment_data.arm_data
        arm_data = arm_data[
            (arm_data.index.get_level_values("arm_name") != self._status_quo_name)
            | (arm_data.index.get_level_values("trial_index") == self.default_trial_idx)
        ]
        return ExperimentData(
            arm_data=arm_data,
            observation_data=observation_data,
        )

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
