#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from math import sqrt
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from ax.adapter import Adapter
from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.exceptions.core import DataRequiredError
from ax.generators.types import TConfig
from ax.utils.stats.math_utils import relativize, unrelativize
from pyre_extensions import none_throws

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


class BaseRelativize(Transform, ABC):
    """
    Change the relative flag of the given relative optimization configuration
    to False. This is needed in order for the new opt config to pass Adapter
    that requires non-relativized opt config.

    Also transforms absolute data and opt configs to relative.

    Requires an adapter with a status quo set to work.

    Abstract property control_as_constant is set to True/False in its subclasses
    Relativize and RelativizeWithConstantControl respectively to account for
    appropriate transform/untransform differently.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: adapter_module.base.Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        cls_name = self.__class__.__name__
        super().__init__(
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        # self.adapter should NOT be modified
        self.adapter: Adapter = none_throws(
            adapter, f"{cls_name} transform requires an adapter"
        )

        sq_data_by_trial = self.adapter.status_quo_data_by_trial
        if not sq_data_by_trial:  # None or empty dict.
            raise DataRequiredError(f"{cls_name} requires status quo data.")
        self.status_quo_data_by_trial: dict[int, ObservationData] = none_throws(
            sq_data_by_trial.copy()
        )
        # use latest index of latest observed trial by default
        # to handle pending trials, which may not have a trial_index
        # if TrialAsTask was not used to generate the trial.
        self.default_trial_idx: int = max(self.status_quo_data_by_trial.keys())

    @property
    @abstractmethod
    def control_as_constant(self) -> bool:
        """Whether or not the control is treated as a constant in the model."""

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        adapter: adapter_module.base.Adapter | None = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> OptimizationConfig:
        r"""
        Change the relative flag of the given relative optimization configuration
        to False. This is needed in order for the new opt config to pass Adapter
        that requires non-relativized opt config.

        Args:
            opt_config: Optimization configuration relative to status quo.

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
        outcome_constraints: list[OutcomeConstraint],
        fixed_features: ObservationFeatures | None = None,
    ) -> list[OutcomeConstraint]:
        for c in outcome_constraints:
            c.relative = True
        return outcome_constraints

    def transform_observations(
        self,
        observations: list[Observation],
    ) -> list[Observation]:
        return self._rel_op_on_observations(
            observations=observations, rel_op=relativize
        )

    def untransform_observations(
        self, observations: list[Observation]
    ) -> list[Observation]:
        """Unrelativize the data"""
        return self._rel_op_on_observations(
            observations=observations, rel_op=unrelativize
        )

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
        for metric in experiment_data.metric_signatures:
            # Create arrays of control values for each row based on trial_index.
            mean_c, sem_c = [], []
            for idx in trial_indices:
                sq_data = self.status_quo_data_by_trial[idx]
                j = get_metric_index(data=sq_data, metric_signature=metric)
                mean_c.append(sq_data.means[j])
                sem_c.append(sq_data.covariance[j, j] ** 0.5)

            # Relativize the whole column in one operation.
            observation_data["mean", metric], observation_data["sem", metric] = (
                relativize(
                    means_t=observation_data["mean", metric],
                    sems_t=observation_data["sem", metric],
                    mean_c=np.array(mean_c),
                    sem_c=np.array(sem_c),
                    as_percent=True,
                    control_as_constant=self.control_as_constant,
                )
            )

        # Set the SQ values to 0.
        mask = (
            observation_data.index.get_level_values("arm_name")
            == self.adapter.status_quo_name
        )
        observation_data.loc[mask, "mean"] = 0
        observation_data.loc[mask, "sem"] = 0

        return ExperimentData(
            arm_data=experiment_data.arm_data,
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
        if idx not in self.status_quo_data_by_trial:
            raise ValueError(
                f"{self.__class__.__name__} requires status quo data for trial "
                f"index {idx}."
            )
        return self._get_relative_data(
            data=obs.data,
            status_quo_data=self.status_quo_data_by_trial[idx],
            rel_op=rel_op,
        )

    def _rel_op_on_observations(
        self,
        observations: list[Observation],
        rel_op: Callable[..., tuple[npt.NDArray, npt.NDArray]],
    ) -> list[Observation]:
        return [
            Observation(
                features=obs.features,
                data=self._get_relative_data_from_obs(obs, rel_op),
                arm_name=obs.arm_name,
            )
            for obs in observations
        ]

    def _get_relative_data(
        self,
        data: ObservationData,
        status_quo_data: ObservationData,
        rel_op: Callable[..., tuple[npt.NDArray, npt.NDArray]],
    ) -> ObservationData:
        r"""
        Relativize or unrelativize `data` based on `status_quo_data` based on `rel_op`

        Args:
            data: ObservationData object to relativize
            status_quo_data: The status quo data (un)relativization is based upon
            rel_op: relativize or unrelativize operator.
            control_as_constant: if treating the control metric as constant

        Returns:
            (un)relativized ObservationData
        """
        L = len(data.metric_signatures)
        result = ObservationData(
            metric_signatures=data.metric_signatures,
            # zeros are just to create the shape so values can be set by index
            means=np.zeros(L),
            covariance=np.zeros((L, L)),
        )
        for i, metric in enumerate(data.metric_signatures):
            j = get_metric_index(data=status_quo_data, metric_signature=metric)
            means_t = data.means[i]
            sems_t = sqrt(data.covariance[i, i])
            mean_c = status_quo_data.means[j]
            sem_c = sqrt(status_quo_data.covariance[j, j])

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
    ) -> tuple[float | npt.NDArray, float | npt.NDArray]:
        """Compute (un)relativized mean and sem for a single metric."""
        # if the is the status quo
        if means_t == mean_c and sems_t == sem_c:
            return 0, 0
        return rel_op(
            means_t=means_t,
            sems_t=sems_t,
            mean_c=mean_c,
            sem_c=sem_c,
            as_percent=True,
            control_as_constant=self.control_as_constant,
        )


def get_metric_index(data: ObservationData, metric_signature: str) -> int:
    """Get the index of a metric in the ObservationData."""
    try:
        return data.metric_signatures.index(metric_signature)
    except (IndexError, StopIteration):
        raise ValueError(
            "Relativization cannot be performed because "
            "ObservationData for status quo is missing metrics"
        )


class Relativize(BaseRelativize):
    """
    Relative transform that by applying delta method.

    Note that not all valid-valued relativized mean and
    standard error can be unrelativized when control_as_constant=True.
    See utils.stats.statstools.unrelativize for more details.
    """

    @property
    def control_as_constant(self) -> bool:
        return False


class RelativizeWithConstantControl(BaseRelativize):
    """
    Relative transform that treats the control metric as a constant when transforming
    and untransforming the data.
    """

    @property
    def control_as_constant(self) -> bool:
        return True
