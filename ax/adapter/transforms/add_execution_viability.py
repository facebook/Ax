#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from logging import Logger
from typing import TYPE_CHECKING

import pandas as pd
from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.metric import Metric
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.types import ComparisonOp
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401

logger: Logger = get_logger(__name__)

EXECUTION_VIABLE_METRIC_NAME = "execution_viable"


class AddExecutionViability(Transform):
    """Transform that adds failure-awareness capability to Ax optimization.

    This transform enables Ax to learn from deterministic trial failures (ABANDONED
    trials) and avoid sampling similar parameter configurations that are likely to
    fail. It achieves this by:

    1. Adding an "execution_viable" metric to experiment data based on trial status
       - ABANDONED trials get execution_viable value of 0.0 (not viable)
       - Other trials get execution_viable value of 1.0 (viable)

    2. Adding an execution viability constraint to the optimization config
       - The constraint enforces P(execution_viable) >= threshold
       - This guides the acquisition function to avoid regions likely to fail

    The transform only activates after observing a minimum number of ABANDONED trials
    to ensure there is sufficient data to model the failure region. Before reaching
    this threshold, the transform acts as a no-op.

    Config options:
        feasibility_threshold: float (default 0.8)
            Minimum probability of execution viability required for new candidates.
        min_abandoned_trials: int (default 3)
            Minimum number of ABANDONED trials required before the transform activates.
            If fewer than this many ABANDONED trials exist, the transform does nothing.

    Example usage:
        >>> transform = AddExecutionViability(
        ...     config={
        ...         "feasibility_threshold": 0.8,
        ...         "min_abandoned_trials": 3,
        ...     }
        ... )
        >>> # Transform adds execution viability constraint to optimization
        >>> new_opt_config = transform.transform_optimization_config(opt_config)
        >>> # Transform adds execution_viable metric to data
        >>> transformed_data = transform.transform_experiment_data(exp_data)
    """

    @property
    def min_abandoned_trials(self) -> int:
        """Minimum ABANDONED trials required before the transform activates."""
        raw_value = self.config.get("min_abandoned_trials", 3)
        return int(raw_value) if isinstance(raw_value, (int, float)) else 3

    def _should_activate(
        self, adapter: adapter_module.base.Adapter
    ) -> tuple[bool, int, list[BaseTrial]]:
        """Check if transform should activate based on abandoned trial count.

        Returns:
            A tuple of (should_activate, abandoned_count, abandoned_trials)
        """
        experiment = adapter._experiment
        abandoned_trials = experiment.trials_by_status.get(TrialStatus.ABANDONED, [])
        abandoned_count = len(abandoned_trials)
        should_activate = abandoned_count >= self.min_abandoned_trials
        return should_activate, abandoned_count, abandoned_trials

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        """Transform experiment data to add execution viability metrics.

        Only activates after observing at least min_abandoned_trials ABANDONED trials.
        Returns the original data unchanged if this threshold is not met.

        This method handles two types of ABANDONED trials:
        1. ABANDONED trials WITH data: These already exist in
           experiment_data and will get execution_viable = 0 added to their
           existing observations.
        2. ABANDONED trials WITHOUT data: These are missing from
           experiment_data (e.g., trials that failed due to metric errors).
           We add synthetic observations for these with execution_viable = 0 so
           the model can learn about regions likely to fail.
        """
        if self.adapter is None:
            raise ValueError(
                "Adapter must be provided for AddExecutionViability transform."
            )

        adapter = self.adapter
        should_activate, abandoned_count, abandoned_trials = self._should_activate(
            adapter
        )

        if not should_activate:
            logger.debug(
                f"AddExecutionViability transform inactive: "
                f"only {abandoned_count} ABANDONED trials observed "
                f"(need {self.min_abandoned_trials}). Returning original data."
            )
            return experiment_data

        # Proceed with adding execution viability metric
        obs_data = experiment_data.observation_data
        arm_data = experiment_data.arm_data

        # Add rows for abandoned trials without data
        trials_in_data = set(obs_data.index.get_level_values("trial_index").unique())
        has_step = "step" in obs_data.index.names
        new_arm_entries = []
        for trial in abandoned_trials:
            if trial.index not in trials_in_data:
                for arm in trial.arms:
                    idx = (
                        (trial.index, arm.name, 1.0)
                        if has_step
                        else (trial.index, arm.name)
                    )
                    obs_data.loc[idx, :] = float("nan")

                    arm_row_data = dict(arm.parameters)
                    metadata_raw = trial._get_candidate_metadata(arm.name)
                    metadata = metadata_raw if metadata_raw is not None else {}
                    if (
                        Keys.TRIAL_COMPLETION_TIMESTAMP not in metadata
                        and trial._time_completed is not None
                    ):
                        metadata[Keys.TRIAL_COMPLETION_TIMESTAMP] = (
                            trial._time_completed.timestamp()
                        )
                    arm_row_data["metadata"] = metadata  # pyre-ignore[6]
                    new_arm_entries.append(
                        {
                            "trial_index": trial.index,
                            "arm_name": arm.name,
                            **arm_row_data,
                        }
                    )

        if new_arm_entries:
            new_arm_df = pd.DataFrame(new_arm_entries).set_index(
                ["trial_index", "arm_name"]
            )
            arm_data = pd.concat([arm_data, new_arm_df])
            logger.debug(
                f"AddExecutionViability: Added synthetic observations for "
                f"{len(new_arm_entries)} arms from ABANDONED trials without data"
            )

        # Assign viability for ALL rows in one shot
        trial_indices = obs_data.index.get_level_values("trial_index")
        abandoned_set = {t.index for t in abandoned_trials}
        obs_data[("mean", EXECUTION_VIABLE_METRIC_NAME)] = [
            0.0 if idx in abandoned_set else 1.0 for idx in trial_indices
        ]
        obs_data[("sem", EXECUTION_VIABLE_METRIC_NAME)] = float("nan")

        logger.debug(
            f"AddExecutionViability transform active: "
            f"{abandoned_count} ABANDONED trials observed "
            f"(threshold: {self.min_abandoned_trials})"
        )

        return ExperimentData(
            arm_data=arm_data,
            observation_data=obs_data,
        )

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        adapter: adapter_module.base.Adapter | None = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> OptimizationConfig:
        """Transform optimization config to add execution viability constraint.

        Only activates after observing at least min_abandoned_trials ABANDONED trials.
        Returns the original config unchanged if this threshold is not met.
        """
        adapter = adapter or self.adapter
        if adapter is None:
            raise ValueError(
                "Adapter must be provided for AddExecutionViability transform."
            )

        should_activate, abandoned_count, _ = self._should_activate(adapter)

        if not should_activate:
            logger.debug(
                f"AddExecutionViability transform inactive: "
                f"only {abandoned_count} ABANDONED trials observed "
                f"(need {self.min_abandoned_trials}). Returning original config."
            )
            return optimization_config

        # Proceed with adding execution viability constraint
        viability_metric = Metric(
            name=EXECUTION_VIABLE_METRIC_NAME,
            lower_is_better=False,
        )
        viability_constraint = OutcomeConstraint(
            metric=viability_metric,
            op=ComparisonOp.GEQ,
            bound=self.config.get("feasibility_threshold", 0.8),  # pyre-ignore [6]
            relative=False,
        )

        # Create a new list with existing constraints plus the viability constraint
        new_outcome_constraints = list(optimization_config.outcome_constraints)
        new_outcome_constraints.append(viability_constraint)

        transformed_opt_config = optimization_config.clone_with_args(
            outcome_constraints=new_outcome_constraints,
        )

        # Add viability metric to outcomes if not already present
        if viability_metric.name not in adapter.outcomes:
            adapter.outcomes.append(viability_metric.name)

        logger.debug(
            f"AddExecutionViability constraint active: "
            f"{abandoned_count} ABANDONED trials observed "
            f"(threshold: {self.min_abandoned_trials})"
        )

        return transformed_opt_config
