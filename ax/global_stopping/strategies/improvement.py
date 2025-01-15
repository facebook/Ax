#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger

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
from ax.exceptions.core import AxError
from ax.global_stopping.strategies.base import BaseGlobalStoppingStrategy
from ax.modelbridge.modelbridge_utils import observed_hypervolume
from ax.plot.pareto_utils import (
    get_tensor_converter_model,
    infer_reference_point_from_experiment,
)
from ax.utils.common.logger import get_logger
from pyre_extensions import assert_is_instance, none_throws


logger: Logger = get_logger(__name__)


class ImprovementGlobalStoppingStrategy(BaseGlobalStoppingStrategy):
    """
    A Global Stopping Strategy which recommends stopping optimization if there
    is no significant improvement over recent iterations.

    This stopping strategy recommends stopping if there is no significant improvement
    over the past `window_size` trials, among those that are feasible
    (satisfying constraints). The meaning of a "significant"
    improvement differs between single-objective and multi-objective optimizations.
    For single-objective optimizations, improvement is as a fraction of the
    interquartile range (IQR) of the objective values seen so far. For
    multi-objective optimizations (MOO), improvement is as a fraction of the hypervolume
    obtained `window_size` iterations ago.
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
            min_trials: Minimum number of trials before the stopping strategy
                kicks in.
            window_size: Number of recent trials to check the improvement in.
                The first trial that could be used for analysis is
                `min_trials - window_size`; the first trial for which stopping
                might be recommended is `min_trials`.
            improvement_bar: Threshold for considering improvement over the best
                point, relative to the interquartile range of values seen so
                far. Must be >= 0.
            inactive_when_pending_trials: If set, the optimization will not stopped as
                long as it has running trials.
        """
        if improvement_bar < 0:
            raise ValueError("improvement_bar must be >= 0.")
        super().__init__(
            min_trials=min_trials,
            inactive_when_pending_trials=inactive_when_pending_trials,
        )
        self.window_size = window_size
        self.improvement_bar = improvement_bar
        self.hv_by_trial: dict[int, float] = {}
        self._inferred_objective_thresholds: list[ObjectiveThreshold] | None = None

    def __repr__(self) -> str:
        return super().__repr__() + (
            f" min_trials={self.min_trials} "
            f"window_size={self.window_size} "
            f"improvement_bar={self.improvement_bar} "
            f"inactive_when_pending_trials={self.inactive_when_pending_trials}"
        )

    def _should_stop_optimization(
        self,
        experiment: Experiment,
        trial_to_check: int | None = None,
        objective_thresholds: list[ObjectiveThreshold] | None = None,
    ) -> tuple[bool, str]:
        """
        Check if the objective has improved significantly in the past
        "window_size" iterations.

        For single-objective optimization experiments, it will call
        _should_stop_single_objective() and for MOO experiments, it will call
        _should_stop_moo(). Before making either of these calls, this function carries
        out some sanity checks to handle obvious/invalid cases. For more detail
        on what it means to "significantly" improve, see the class docstring.

        Args:
            experiment: The experiment to apply the strategy on.
            trial_to_check: The trial in the experiment at which we want to check
                for stopping. If None, we check at the latest trial.
            objective_thresholds: Custom objective thresholds to use as reference pooint
                when computing hv of the pareto front against. This is used only in the
                MOO setting. If not specified, the objective thresholds on the
                experiment's optimization config will be used for the purpose.
                If no thresholds are provided, they are automatically inferred. They are
                only inferred once for each instance of the strategy (i.e. inferred
                thresholds don't update with additional data).

        Returns:
            A Tuple with a boolean determining whether the optimization should stop,
            and a str declaring the reason for stopping.
        """

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

        data = experiment.lookup_data()
        if data.df.empty:
            raise AxError(
                f"Experiment {experiment} does not have any data attached "
                f"to it, despite having {num_completed_trials} completed "
                f"trials. Data is required for {self}, so this is an invalid "
                "state of the experiment."
            )

        if isinstance(experiment.optimization_config, MultiObjectiveOptimizationConfig):
            if objective_thresholds is None:
                # self._inferred_objective_thresholds is cached and only computed once.
                if self._inferred_objective_thresholds is None:
                    # only infer reference point if there is data on the experiment.
                    if not data.df.empty:
                        # We infer the nadir reference point to be used by the GSS.
                        self._inferred_objective_thresholds = (
                            infer_reference_point_from_experiment(
                                experiment=experiment, data=data
                            )
                        )
                # TODO: move this out into a separate infer_objective_thresholds
                # instance method or property that handles the caching.
                objective_thresholds = self._inferred_objective_thresholds
            if not objective_thresholds:
                # TODO: This is headed to ax.modelbridge.modelbridge_utils.hypervolume,
                # where an empty list would lead to an opaque indexing error.
                # A list that is nonempty and of the wrong length could be worse,
                # since it might wind up running without error, but with thresholds for
                # the wrong metrics. We should validate correctness of the length of the
                # objective thresholds, ideally in hypervolume utils.
                raise AxError(
                    f"Objective thresholds were not specified and could not be inferred"
                    f". They are required for {self} when performing multi-objective "
                    "optimization, so this is an invalid state of the experiment."
                )
            return self._should_stop_moo(
                experiment=experiment,
                trial_to_check=trial_to_check,
                objective_thresholds=none_throws(objective_thresholds),
            )
        else:
            return self._should_stop_single_objective(
                experiment=experiment, trial_to_check=trial_to_check
            )

    def _should_stop_moo(
        self,
        experiment: Experiment,
        trial_to_check: int,
        objective_thresholds: list[ObjectiveThreshold],
    ) -> tuple[bool, str]:
        """
        This is the "should_stop_optimization" method of this class, specialized
        to MOO experiments.

        It computes the (feasible) hypervolume of the Pareto front at
        `trial_to_check` trial and `window_size` trials before, and suggest to stop the
        optimization if the improvment in hypervolume over the past
        `window_size` trials, as a fraction of the hypervolume at the start of
        the window, is less than `self.improvement_bar`. When the hypervolume is
        zero at the beginning of the window, stopping is never recommended.

        Becaues hypervolume computations are expensive, these are stored to
        increase the speed of future checks.

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
        data_df = experiment.lookup_data().df
        data_df_reference = data_df[data_df["trial_index"] <= reference_trial_index]
        data_df = data_df[data_df["trial_index"] <= trial_to_check]

        # Computing or retrieving HV at "window_size" iteration before
        if reference_trial_index in self.hv_by_trial:
            hv_reference = self.hv_by_trial[reference_trial_index]
        else:
            mb_reference = get_tensor_converter_model(
                experiment=experiment, data=Data(data_df_reference)
            )
            hv_reference = observed_hypervolume(
                modelbridge=mb_reference, objective_thresholds=objective_thresholds
            )
            self.hv_by_trial[reference_trial_index] = hv_reference

        if hv_reference == 0:
            message = "The reference hypervolume is 0. Continue the optimization."
            return False, message

        # Computing HV at current trial
        mb = get_tensor_converter_model(experiment=experiment, data=Data(data_df))
        hv = observed_hypervolume(mb, objective_thresholds=objective_thresholds)
        self.hv_by_trial[trial_to_check] = hv

        hv_improvement = (hv - hv_reference) / hv_reference
        stop = hv_improvement < self.improvement_bar

        if stop:
            message = (
                f"The improvement in hypervolume in the past {self.window_size} "
                f"trials (={hv_improvement:.3f}) is less than improvement_bar "
                f"(={self.improvement_bar}) times the hypervolume at the start "
                f"of the window (={hv_reference:.3f})."
            )
        else:
            message = ""
        return stop, message

    def _should_stop_single_objective(
        self, experiment: Experiment, trial_to_check: int
    ) -> tuple[bool, str]:
        """
        This is the `_should_stop_optimization` method of this class,
        specialized to single-objective experiments.

        It computes the interquartile range (IQR) of feasible objective values
        found so far, then computes the improvement in the best seen over the
        last `window_size` trials. If the recent improvement as a fraction of
        the IQR is less than `self.improvement_bar`, it recommends stopping.

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
                tr = assert_is_instance(trial, Trial)
                objectives.append(tr.objective_mean)
                is_feasible.append(constraint_satisfaction(tr))

        if assert_is_instance(
            experiment.optimization_config, OptimizationConfig
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
                f"{self.improvement_bar} times the interquartile range (IQR) of "
                f"objectives attained so far (IQR={iqr:.3f})."
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
        A boolean which is True iff all outcome constraints are satisfied.
    """
    outcome_constraints = none_throws(
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
