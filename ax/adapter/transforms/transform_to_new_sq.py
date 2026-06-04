#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging
from collections.abc import Callable
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
from ax.utils.stats.math_utils import MEAN_CONTROL_EPSILON, relativize, unrelativize
from pyre_extensions import assert_is_instance, none_throws

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


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
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )

        target_trial_index = self.config.get("target_trial_index", None)
        if (
            target_trial_index is None
            and adapter is not None
            and adapter._experiment is not None
        ):
            target_trial_index = get_target_trial_index(
                experiment=none_throws(adapter)._experiment
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
        trial_indices = observation_data.index.get_level_values("trial_index")

        # Trials to skip: target trial, non-relativizable trials (e.g.,
        # LILO labeling), and trials without SQ data.
        all_unique = set(trial_indices.unique().tolist())
        trials_without_sq = all_unique - set(self.status_quo_data_by_trial.keys())
        skip_trials = (
            {self.default_trial_idx}
            | self._non_relativizable_trial_indices
            | trials_without_sq
        )
        transform_mask = ~trial_indices.isin(skip_trials)

        target_sq_data = self.status_quo_data_by_trial[self.default_trial_idx]
        metrics = experiment_data.metric_signatures
        if not transform_mask.any():
            metrics = []

        for metric in metrics:
            if metric not in target_sq_data.metric_signatures:
                continue

            # Check target SQ mean first -- if near-zero, relativization is
            # undefined (unrelativization would collapse all values to zero).
            target_j = get_metric_index(data=target_sq_data, metric_signature=metric)
            target_mean_c = target_sq_data.means[target_j]
            if np.abs(target_mean_c) < MEAN_CONTROL_EPSILON:
                logger.warning(
                    f"Skipping TransformToNewSQ for metric '{metric}': "
                    f"target trial status quo mean is near-zero "
                    f"({target_mean_c}). This can happen when the metric "
                    f"is already relativized (e.g., ExpressionDerivedMetric)."
                )
                continue

            # Build per-row control arrays from each trial's SQ data.
            mean_c, sem_c = [], []
            for idx in trial_indices[transform_mask]:
                sq_data = self.status_quo_data_by_trial[idx]
                j = get_metric_index(data=sq_data, metric_signature=metric)
                mean_c.append(sq_data.means[j])
                sem_c.append(sq_data.covariance[j, j] ** 0.5)

            mean_c_arr = np.array(mean_c)
            if np.any(np.abs(mean_c_arr) < MEAN_CONTROL_EPSILON):
                logger.warning(
                    f"Skipping TransformToNewSQ for metric '{metric}': "
                    f"one or more trial status quo means are near-zero. "
                    f"This can happen when the metric is already relativized "
                    f"(e.g., ExpressionDerivedMetric)."
                )
                continue

            means_rel, sems_rel = relativize(
                means_t=observation_data.loc[transform_mask, ("mean", metric)],
                sems_t=observation_data.loc[transform_mask, ("sem", metric)],
                mean_c=mean_c_arr,
                sem_c=np.array(sem_c),
                as_percent=False,
                control_as_constant=self.control_as_constant,
            )

            # Unrelativize with respect to target trial's status quo.
            abs_target_mean_c = np.abs(target_mean_c)
            observation_data.loc[transform_mask, ("mean", metric)] = (
                means_rel * abs_target_mean_c + target_mean_c
            )
            observation_data.loc[transform_mask, ("sem", metric)] = (
                sems_rel * abs_target_mean_c
            )

        # Drop SQ observations from transformed trials (their SQ values
        # are now zero / redundant).  Non-transformed trials (e.g., LILO
        # labeling) keep all observations including SQ arms.
        transformed_trials = all_unique - skip_trials
        arm_data = experiment_data.arm_data
        if transformed_trials:
            obs_drop = trial_indices.isin(transformed_trials) & (
                observation_data.index.get_level_values("arm_name")
                == self.status_quo_name
            )
            observation_data = observation_data[~obs_drop]
            arm_trial_indices = arm_data.index.get_level_values("trial_index")
            arm_drop = arm_trial_indices.isin(transformed_trials) & (
                arm_data.index.get_level_values("arm_name") == self.status_quo_name
            )
            arm_data = arm_data[~arm_drop]
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
        if idx not in self.status_quo_data_by_trial:
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
        # Keep SQ observations from non-relativizable trials (e.g.,
        # LILO labeling) — dropping an arm from a pairwise comparison
        # would break PairwiseGP.
        return [
            obs
            for obs in rel_observations
            if (
                obs.arm_name != self.status_quo_name
                or obs.features.trial_index == self.default_trial_idx
                or obs.features.trial_index in self._non_relativizable_trial_indices
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
        j = get_metric_index(data=target_status_quo_data, metric_signature=metric)
        target_mean_c = target_status_quo_data.means[j]
        abs_target_mean_c = np.abs(target_mean_c)
        # Skip if control or target SQ mean is near-zero -- relativization
        # is undefined (division by zero).  The guard here is needed for
        # untransform symmetry: if transform_experiment_data skipped a
        # metric, the untransform path must also skip it.
        if abs_target_mean_c < MEAN_CONTROL_EPSILON or (
            np.abs(mean_c) < MEAN_CONTROL_EPSILON
        ):
            logger.warning(
                f"Skipping TransformToNewSQ for metric '{metric}': "
                f"status quo mean is near-zero (target={target_mean_c}, "
                f"control={mean_c})."
            )
            return means_t, sems_t
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
        return float(means_rel), float(sems_rel)
