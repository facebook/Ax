#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from logging import Logger
from typing import TYPE_CHECKING

from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.core.base_trial import TrialStatus
from ax.core.metric import Metric
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.generators.types import TConfig
from ax.utils.common.logger import get_logger

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401

logger: Logger = get_logger(__name__)

FEASIBILITY_METRIC_NAME = "is_feasible"


class AddFeasibility(Transform):
    """Transform that adds failure-awareness capability to Ax optimization.

    This transform enables Ax to learn from deterministic trial failures (ABANDONED
    trials) and avoid sampling similar parameter configurations that are likely to
    fail. It achieves this by:

    1. Adding a "is_feasible" metric to experiment data based on trial status
       - ABANDONED trials get feasibility value of 0.0 (infeasible)
       - Other trials get feasibility value of 1.0 (feasible)

    2. Adding a feasibility constraint to the optimization config
       - The constraint enforces P(is_feasible) >= threshold
       - This guides the acquisition function to avoid infeasible regions

    The transform only activates after observing a minimum number of ABANDONED trials
    to ensure there is sufficient data to model the failure region. Before reaching
    this threshold, the transform acts as a no-op.

    Config options:
        feasibility_threshold: float (default 0.0)
            Minimum probability of feasibility required for new candidates.
        min_abandoned_trials: int (default 3)
            Minimum number of ABANDONED trials required before the transform activates.
            If fewer than this many ABANDONED trials exist, the transform does nothing.

    Example usage:
        >>> transform = AddFeasibility(
        ...     config={
        ...         "feasibility_threshold": 0.8,
        ...         "min_abandoned_trials": 3,
        ...     }
        ... )
        >>> # Transform adds feasibility constraint to optimization
        >>> new_opt_config = transform.transform_optimization_config(opt_config)
        >>> # Transform adds feasibility metric to data
        >>> transformed_data = transform.transform_experiment_data(exp_data)
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: adapter_module.base.Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        super().__init__(
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        """Transform experiment data to add feasibility metrics.

        Only activates after observing at least min_abandoned_trials ABANDONED trials.
        Returns the original data unchanged if this threshold is not met.

        This method handles two types of ABANDONED trials:
        1. ABANDONED trials WITH data: These already exist in
           experiment_data and will get is_feasible = 0 added to their
           existing observations.
        2. ABANDONED trials WITHOUT data: These are missing from
           experiment_data (e.g., trials that failed due to metric errors).
           We add synthetic observations for these with is_feasible = 0 so
           the model can learn about infeasible regions.
        """
        if self.adapter is None:
            raise ValueError(
                "Adapter must be provided for using feasibility constraints."
            )

        adapter = self.adapter
        experiment = adapter._experiment

        # Count ABANDONED trials using trials_by_status
        abandoned_trials = experiment.trials_by_status.get(TrialStatus.ABANDONED, [])
        abandoned_count = len(abandoned_trials)

        # Check if we have enough ABANDONED trials to activate the transform
        raw_min_abandoned = self.config.get("min_abandoned_trials", 3)
        min_abandoned_trials = (
            int(raw_min_abandoned) if isinstance(raw_min_abandoned, (int, float)) else 3
        )
        if abandoned_count < min_abandoned_trials:
            logger.info(
                f"AddFeasibility transform inactive: only {abandoned_count} ABANDONED "
                f"trials observed (need {min_abandoned_trials}). Returning original data."
            )
            return experiment_data

        # Proceed with adding feasibility metric
        obs_data = experiment_data.observation_data.copy(deep=True)
        arm_data = experiment_data.arm_data.copy(deep=True)

        # Step 1: Add feasibility metric to existing observations
        trial_feasibilities = []
        for t_idx, _ in obs_data.index:
            trial_status = experiment.trials[t_idx].status
            is_feasible = float(trial_status != TrialStatus.ABANDONED)
            trial_feasibilities.append(is_feasible)

        obs_data[("mean", FEASIBILITY_METRIC_NAME)] = trial_feasibilities
        obs_data[("sem", FEASIBILITY_METRIC_NAME)] = float("nan")

        # Step 2: Identify ABANDONED trials that are NOT in the observation data
        trials_in_data = set(obs_data.index.get_level_values("trial_index").unique())
        abandoned_trials_without_data = [
            trial for trial in abandoned_trials if trial.index not in trials_in_data
        ]

        # Step 3: Add observations for ABANDONED trials without data
        if abandoned_trials_without_data:
            import pandas as pd

            new_rows = []
            new_arm_rows = []

            for trial in abandoned_trials_without_data:
                # Each trial can have multiple arms
                for arm in trial.arms:
                    trial_idx = trial.index
                    arm_name = arm.name

                    new_row_data = {
                        "trial_index": trial_idx,
                        "arm_name": arm_name,
                        ("mean", FEASIBILITY_METRIC_NAME): 0.0,
                        ("sem", FEASIBILITY_METRIC_NAME): float("nan"),
                    }

                    # Add NaN values for all other metrics that exist in obs_data
                    for col in obs_data.columns:
                        if col not in [
                            ("mean", FEASIBILITY_METRIC_NAME),
                            ("sem", FEASIBILITY_METRIC_NAME),
                        ]:
                            new_row_data[col] = float("nan")

                    new_rows.append(new_row_data)

                    # Also add to arm_data
                    arm_row_data = dict(arm.parameters)
                    metadata_raw = trial._get_candidate_metadata(arm.name)
                    metadata = metadata_raw if metadata_raw is not None else {}
                    if (
                        "trial_completion_timestamp" not in metadata
                        and trial._time_completed is not None
                    ):
                        metadata["trial_completion_timestamp"] = (
                            trial._time_completed.timestamp()
                        )
                    arm_row_data["metadata"] = metadata  # pyre-ignore[6]
                    new_arm_rows.append(
                        {"trial_index": trial_idx, "arm_name": arm_name, **arm_row_data}
                    )

            if new_rows:
                new_obs_df = pd.DataFrame(new_rows)
                new_obs_df = new_obs_df.set_index(["trial_index", "arm_name"])

                obs_data = pd.concat([obs_data, new_obs_df])

                new_arm_df = pd.DataFrame(new_arm_rows)
                new_arm_df = new_arm_df.set_index(["trial_index", "arm_name"])
                arm_data = pd.concat([arm_data, new_arm_df])

                logger.info(
                    f"AddFeasibility: Added synthetic observations for "
                    f"{len(abandoned_trials_without_data)} ABANDONED trials "
                    "without data"
                )

        logger.info(
            f"AddFeasibility transform active: {abandoned_count} ABANDONED trials "
            f"observed (threshold: {min_abandoned_trials})"
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
        """Transform optimization config to add feasibility constraint.

        Only activates after observing at least min_abandoned_trials ABANDONED trials.
        Returns the original config unchanged if this threshold is not met.
        """
        adapter = adapter or self.adapter
        if adapter is None:
            raise ValueError("Adapter must be provided for using feasibility.")

        experiment = adapter._experiment

        # Count ABANDONED trials using trials_by_status
        abandoned_trials = experiment.trials_by_status.get(TrialStatus.ABANDONED, [])
        abandoned_count = len(abandoned_trials)

        # Check if we have enough ABANDONED trials to activate the transform
        raw_min_abandoned = self.config.get("min_abandoned_trials", 3)
        min_abandoned_trials = (
            int(raw_min_abandoned) if isinstance(raw_min_abandoned, (int, float)) else 3
        )
        if abandoned_count < min_abandoned_trials:
            logger.info(
                f"AddFeasibility transform inactive: only {abandoned_count} ABANDONED "
                f"trials observed (need {min_abandoned_trials}). Returning original config."
            )
            return optimization_config

        # Proceed with adding feasibility constraint
        feasibility_metric = Metric(
            name=FEASIBILITY_METRIC_NAME,
            lower_is_better=False,
        )
        feasibility_constraint = OutcomeConstraint(
            metric=feasibility_metric,
            op=ComparisonOp.GEQ,
            bound=self.config.get("feasibility_threshold", 0.0),  # pyre-ignore [6]
            relative=False,
        )

        # Create a new list with existing constraints plus the feasibility constraint
        new_outcome_constraints = list(optimization_config.outcome_constraints)
        new_outcome_constraints.append(feasibility_constraint)

        transformed_opt_config = optimization_config.clone_with_args(
            outcome_constraints=new_outcome_constraints,
        )

        # Add feasibility metric to outcomes if not already present
        if feasibility_metric.name not in adapter.outcomes:
            adapter.outcomes.append(feasibility_metric.name)

        logger.info(
            f"AddFeasibility constraint active: {abandoned_count} ABANDONED trials "
            f"observed (threshold: {min_abandoned_trials})"
        )

        return transformed_opt_config
