#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import ObjectiveThreshold
from ax.core.trial import Trial
from ax.core.types import ComparisonOp
from ax.global_stopping.strategies.base import BaseGlobalStoppingStrategy
from ax.modelbridge.modelbridge_utils import observed_hypervolume
from ax.plot.pareto_utils import get_tensor_converter_model
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none


logger: Logger = get_logger(__name__)


class ImprovementGlobalStoppingStrategy(BaseGlobalStoppingStrategy):
    """
    A stopping strategy which stops the optimization if there is no significant
    improvement over the iterations. For single-objective optimizations, this
    strategy stops the loop if the feasible (mean) objective has not improved
    over the past "window_size" iterations. In MOO loops, it stops the optimization
    loop if the hyper-volume of the pareto front has not improved in the past
    "window_size" iterations.
    """

    def __init__(
        self,
        min_trials: int,
        window_size: int = 5,
        improvement_bar: float = 0.1,
        inactive_when_pending_trials: bool = True,
    ) -> None:
        """
        Initialize an improvement-based stopping strategy.

        Args:
            min_trials: Minimum number of trials before the stopping strategy kicks in.
            window_size: Number of recent trials to check the improvement in.
            improvement_bar: Threshold (in [0,1]) for considering relative improvement
                over the best point.
            inactive_when_pending_trials: If set, the optimization will not stopped as
                long as it has running trials.
        """
        super().__init__(
            min_trials=min_trials,
            inactive_when_pending_trials=inactive_when_pending_trials,
        )
        self.window_size = window_size
        self.improvement_bar = improvement_bar
        self.hv_by_trial: Dict[int, float] = {}

    def should_stop_optimization(
        self,
        experiment: Experiment,
        trial_to_check: Optional[int] = None,
        objective_thresholds: Optional[List[ObjectiveThreshold]] = None,
        **kwargs: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Check if the optimization has improved in the past "window_size" iterations.
        For single-objective optimization experiments, it will call
        _should_stop_single_objective() and for MOO experiments, it will call
        _should_stop_moo(). Before making either of these calls, this function carries
        out some sanity checks to handle obvious/invalid cases.

        Args:
            experiment: The experiment to apply the strategy on.
            trial_to_check: The trial in the experiment at which we want to check
                for stopping. If None, we check at the latest trial.
            objective_thresholds: Custom objective thresholds to use as reference pooint
                when computing hv of the pareto front against. This is used only in the
                MOO setting. If not specified, the objective thresholds on the
                experiment's optimization config will be used for the purpose.

        Returns:
            A Tuple with a boolean determining whether the optimization should stop,
            and a str declaring the reason for stopping.
        """
        if (
            self.inactive_when_pending_trials
            and len(experiment.trials_by_status[TrialStatus.RUNNING]) > 0
        ):
            message = "There are pending trials in the experiment."
            return False, message

        if len(experiment.trials_by_status[TrialStatus.COMPLETED]) == 0:
            message = "There are no completed trials yet."
            return False, message

        max_completed_trial = max(
            experiment.trial_indices_by_status[TrialStatus.COMPLETED]
        )

        if trial_to_check is None:
            trial_to_check = max_completed_trial
        elif trial_to_check > max_completed_trial:
            raise ValueError(
                "trial_to_check is larger than the total number of "
                f"trials (={max_completed_trial})."
            )

        # Only counting the trials up to trial_to_check.
        num_completed_trials = sum(
            index <= trial_to_check
            for index in experiment.trial_indices_by_status[TrialStatus.COMPLETED]
        )
        min_required_trials = max(self.min_trials, self.window_size)
        if num_completed_trials < min_required_trials:
            stop = False
            message = (
                "There are not enough completed trials to make a stopping decision "
                f"(completed: {num_completed_trials}, required: {min_required_trials})."
            )
            return stop, message

        if isinstance(experiment.optimization_config, MultiObjectiveOptimizationConfig):
            return self._should_stop_moo(
                experiment=experiment,
                trial_to_check=trial_to_check,
                objective_thresholds=objective_thresholds,
            )
        else:
            return self._should_stop_single_objective(
                experiment=experiment, trial_to_check=trial_to_check
            )

    def _should_stop_moo(
        self,
        experiment: Experiment,
        trial_to_check: int,
        objective_thresholds: Optional[List[ObjectiveThreshold]] = None,
    ) -> Tuple[bool, str]:
        """
        This is just the "should_stop_optimization" method of the class specialized to
        MOO experiments. It computes the (feasible) hypervolume of the pareto front at
        "trial_to_check" trial and "window_size" trials before, and suggest to stop the
        optimization if there is no significant improvement.

        Args:
            experiment: The experiment to apply the strategy on.
            trial_to_check: The trial in the experiment at which we want to check
                for stopping. If None, we check at the latest trial.
            objective_thresholds: Custom objective thresholds to use as reference pooint
                when computing hv of the pareto front against. This is used only in the
                MOO setting. If not specified, the objective thresholds on the
                experiment's optimization config will be used for the purpose.

        Returns:
            A Tuple with a boolean determining whether the optimization should stop,
                and a str declaring the reason for stopping.
        """
        reference_trial_index = trial_to_check - self.window_size + 1
        data_df = experiment.fetch_data().df
        data_df_reference = data_df[data_df["trial_index"] <= reference_trial_index]
        data_df = data_df[data_df["trial_index"] <= trial_to_check]

        reference_point = (
            objective_thresholds
            or checked_cast(
                MultiObjectiveOptimizationConfig, experiment.optimization_config
            ).objective_thresholds
        )

        # Computing or retrieving HV at "window_size" iteration before
        if reference_trial_index in self.hv_by_trial:
            hv_reference = self.hv_by_trial[reference_trial_index]
        else:
            mb_reference = get_tensor_converter_model(
                experiment=experiment, data=Data(data_df_reference)
            )
            hv_reference = observed_hypervolume(
                modelbridge=mb_reference, objective_thresholds=reference_point
            )
            self.hv_by_trial[reference_trial_index] = hv_reference

        if hv_reference == 0:
            message = "The reference hypervolume is 0. Continue the optimization."
            return False, message

        # Computing HV at current trial
        mb = get_tensor_converter_model(experiment=experiment, data=Data(data_df))
        hv = observed_hypervolume(mb, objective_thresholds=reference_point)
        self.hv_by_trial[trial_to_check] = hv

        hv_improvement = (hv - hv_reference) / hv_reference
        stop = hv_improvement < self.improvement_bar

        if stop:
            message = (
                f"The improvement in hypervolume in the past {self.window_size} "
                f"trials (={hv_improvement:.3f}) is less than {self.improvement_bar}."
            )
        else:
            message = ""
        return stop, message

    def _should_stop_single_objective(
        self, experiment: Experiment, trial_to_check: int
    ) -> Tuple[bool, str]:
        """
        This is the "should_stop_optimization" method of the class specialized to
        single-objective experiments. It computes the best feasible objective  at
        "trial_to_check" trial and "window_size" trials before, and suggest to stop
        the trial if there is no significant improvement.

        Args:
            experiment: The experiment to apply the strategy on.
            trial_to_check: The trial in the experiment at which we want to check
                for stopping. If None, we check at the latest trial.

        Returns:
            A Tuple with a boolean determining whether the optimization should stop,
            and a str declaring the reason for stopping.
        """
        objectives = []
        is_feasible = []
        for trial in experiment.trials_by_status[TrialStatus.COMPLETED]:
            if trial.index <= trial_to_check:
                tr = checked_cast(Trial, trial)
                objectives.append(tr.objective_mean)
                is_feasible.append(constraint_satisfaction(tr))

        if checked_cast(
            OptimizationConfig, experiment.optimization_config
        ).objective.minimize:
            selector, mask_val = np.minimum, np.inf
        else:
            selector, mask_val = np.maximum, -np.inf

        # Replace objective value at infeasible iterations with mask_val
        masked_obj = np.where(is_feasible, objectives, mask_val)
        running_optimum = selector.accumulate(masked_obj)

        # Computing the interquartile for scaling the difference
        feasible_objectives = np.array(objectives)[is_feasible]
        if len(feasible_objectives) <= 1:
            message = "There are not enough feasible arms tried yet."
            return False, message

        q3, q1 = np.percentile(feasible_objectives, [75, 25])
        iqr = q3 - q1

        relative_improvement = np.abs(
            (running_optimum[-1] - running_optimum[-self.window_size]) / iqr
        )
        stop = relative_improvement < self.improvement_bar

        if stop:
            message = (
                f"The improvement in best objective in the past {self.window_size} "
                f"trials (={relative_improvement:.3f}) is less than "
                f"{self.improvement_bar}."
            )
        else:
            message = ""

        return stop, message


def constraint_satisfaction(trial: BaseTrial) -> bool:
    """
    This function checks whether the outcome constraints of the
    optimization config of an experiment are satisfied in the
    given trial.

    Args:
        trial: A single-arm Trial at which we want to check the constraint.

    Returns:
        A boolean which is True iff all outcome constraints are satisifed.
    """
    outcome_constraints = not_none(
        trial.experiment.optimization_config
    ).outcome_constraints
    if len(outcome_constraints) == 0:
        return True

    df = trial.lookup_data().df
    for constraint in outcome_constraints:
        bound = constraint.bound
        metric_name = constraint.metric.name
        metric_data = df.loc[df["metric_name"] == metric_name]
        mean, sem = metric_data.iloc[0][["mean", "sem"]]
        if sem > 0.0:
            logger.warning(
                f"There is observation noise for metric {metric_name}. This may "
                "negatively affect the way we check constraint satisfaction."
            )
        if constraint.op is ComparisonOp.LEQ:
            satisfied = mean <= bound
        else:
            satisfied = mean >= bound
        if not satisfied:
            return False

    return True
