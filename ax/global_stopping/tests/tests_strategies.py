#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import numpy as np
import pandas as pd
from ax.core.arm import Arm
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import ObjectiveThreshold, OutcomeConstraint
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.trial import Trial
from ax.core.types import ComparisonOp
from ax.global_stopping.strategies.improvement import (
    constraint_satisfaction,
    ImprovementGlobalStoppingStrategy,
)
from ax.utils.common.testutils import TestCase
from ax.utils.common.typeutils import checked_cast, not_none
from ax.utils.testing.core_stubs import get_experiment, get_experiment_with_data


class TestImprovementGlobalStoppingStrategy(TestCase):
    def test_base_cases(self) -> None:
        exp = get_experiment_with_data()
        exp.trials[0].mark_running(no_runner_required=True)

        gss = ImprovementGlobalStoppingStrategy(min_trials=2, window_size=3)
        stop, message = gss.should_stop_optimization(experiment=exp)
        self.assertFalse(stop)
        self.assertEqual(message, "There are pending trials in the experiment.")

        gss_2 = ImprovementGlobalStoppingStrategy(
            min_trials=2, window_size=3, inactive_when_pending_trials=False
        )
        stop, message = gss_2.should_stop_optimization(experiment=exp)
        # This one should be fine with having pending trials, but is not
        # stopping due to lack of completed trials.
        self.assertFalse(stop)
        self.assertEqual(
            message,
            "There are no completed trials yet.",
        )

        # Insufficient trials to stop.
        exp.trials[0].mark_completed()
        stop, message = gss.should_stop_optimization(experiment=exp)
        self.assertFalse(stop)
        self.assertEqual(
            message,
            "There are not enough completed trials to "
            "make a stopping decision (completed: 1, required: 3).",
        )

        # Check that we properly count completed trials.
        for _ in range(4):
            checked_cast(BatchTrial, exp.trials[0]).clone()
        exp.trials[3].mark_running(no_runner_required=True).mark_completed()
        stop, message = gss.should_stop_optimization(experiment=exp)
        self.assertFalse(stop)
        self.assertEqual(
            message,
            "There are not enough completed trials to "
            "make a stopping decision (completed: 2, required: 3).",
        )

        # Should raise ValueError if trying to check an invalid trial
        with self.assertRaises(ValueError):
            stop, message = gss.should_stop_optimization(
                experiment=exp, trial_to_check=4
            )

    def _get_arm(self) -> Arm:
        """generates random arm in [0,1]^2."""
        return Arm(parameters={"x": np.random.rand(), "y": np.random.rand()})

    def _get_data_for_trial(
        self, trial: Trial, values: Tuple[float, float, float]
    ) -> Data:
        """
        Generates data for a given trial, from the provided values for
        the metrics (m1,m2,m3).

        Args:
            trial: Trial to generate the data for.
            values: A tuple (m1, m2, m3) representing the mean for each of
                the three metrics.

        Returns:
            A Data object to be attached to the experiment.
        """
        df_dicts = [
            {
                "trial_index": trial.index,
                "metric_name": "m1",
                "arm_name": not_none(trial.arm).name,
                "mean": values[0],
                "sem": 0.0,
            },
            {
                "trial_index": trial.index,
                "metric_name": "m2",
                "arm_name": not_none(trial.arm).name,
                "mean": values[1],
                "sem": 0.0,
            },
            {
                "trial_index": trial.index,
                "metric_name": "m3",
                "arm_name": not_none(trial.arm).name,
                "mean": values[2],
                "sem": 0.0,
            },
        ]
        return Data(df=pd.DataFrame.from_records(df_dicts))

    def _create_single_objective_experiment(
        self, metric_values: List[Tuple[float, float, float]]
    ) -> Experiment:
        """
        Creates a synthetic experiments with 2 parameters, one objective and two
        outcome constraints for testing different components. It also populates
        the experiment with trials and their data.

        Args:
            metric_values: A list of triples representing the mean values for three
                metrics in trials.

        Retruns:
            An experiment with len(metric_values) trials.
        """
        search_space = SearchSpace(
            [
                RangeParameter("x", ParameterType.FLOAT, 0.0, 1.0),
                RangeParameter("y", ParameterType.FLOAT, 0.0, 1.0),
            ]
        )
        objective = Objective(metric=Metric(name="m1"), minimize=False)
        outcome_constraints = [
            OutcomeConstraint(
                metric=Metric(name="m2"),
                op=ComparisonOp.GEQ,
                bound=0.25,
                relative=False,
            ),
            OutcomeConstraint(
                metric=Metric(name="m3"),
                op=ComparisonOp.LEQ,
                bound=0.25,
                relative=False,
            ),
        ]
        optimization_config = OptimizationConfig(
            objective=objective, outcome_constraints=outcome_constraints
        )
        exp = Experiment(
            name="test_experiment",
            search_space=search_space,
            optimization_config=optimization_config,
        )

        # Adding trials to the experiment
        for metric_value in metric_values:
            trial = exp.new_trial().add_arm(arm=self._get_arm())
            _ = trial.mark_running(no_runner_required=True)
            data = self._get_data_for_trial(trial=trial, values=metric_value)
            _ = exp.attach_data(data)
            trial.mark_completed()

        return exp

    def _create_multi_objective_experiment(
        self, metric_values: List[Tuple[float, float, float]]
    ) -> Experiment:
        """
        Creates a synthetic experiments with 2 parameters, two objectives and
        one outcome constraint for testing different components. It also
        populates the experiment with trials and their data.

        Args:
            metric_values: A list of triples representing the mean values for three
                metrics in trials.

        Retruns:
            An experiment with len(metric_values) trials.
        """
        search_space = SearchSpace(
            [
                RangeParameter("x", ParameterType.FLOAT, 0.0, 1.0),
                RangeParameter("y", ParameterType.FLOAT, 0.0, 1.0),
            ]
        )

        objectives = [
            Objective(metric=Metric(name="m1"), minimize=False),
            Objective(metric=Metric(name="m2"), minimize=False),
        ]
        objective_thresholds = [
            ObjectiveThreshold(
                metric=objectives[0].metric,
                bound=0.1,
                op=ComparisonOp.GEQ,
                relative=False,
            ),
            ObjectiveThreshold(
                metric=objectives[1].metric,
                bound=0.2,
                op=ComparisonOp.GEQ,
                relative=False,
            ),
        ]

        outcome_constraints = [
            OutcomeConstraint(
                metric=Metric(name="m3"),
                op=ComparisonOp.LEQ,
                bound=0.5,
                relative=False,
            ),
        ]
        optimization_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(objectives),
            outcome_constraints=outcome_constraints,
            objective_thresholds=objective_thresholds,
        )

        exp = Experiment(
            name="test_experiment",
            search_space=search_space,
            optimization_config=optimization_config,
        )

        # Adding trials to the experiment
        for metric_value in metric_values:
            trial = exp.new_trial().add_arm(arm=self._get_arm())
            _ = trial.mark_running(no_runner_required=True)
            data = self._get_data_for_trial(trial=trial, values=metric_value)
            _ = exp.attach_data(data)
            trial.mark_completed()

        return exp

    def test_multi_objective(self) -> None:

        metric_values = [
            (0.15, 0.6, 0.1),
            (0.25, 0.5, 0.2),
            (0.4, 0.5, 0.6),
            (0.3, 0.4, 0.0),
            (0.6, 0.1, 0.3),
            (0.9, 0.1, 0.1),
        ]
        exp = self._create_multi_objective_experiment(metric_values=metric_values)
        gss = ImprovementGlobalStoppingStrategy(
            min_trials=3, window_size=3, improvement_bar=0.1
        )
        stop, message = gss.should_stop_optimization(experiment=exp, trial_to_check=4)
        self.assertFalse(stop)
        self.assertEqual(message, "")

        stop, message = gss.should_stop_optimization(experiment=exp, trial_to_check=5)
        self.assertTrue(stop)
        self.assertEqual(
            message,
            "The improvement in hypervolume in the past 3 trials (=0.000) is less than "
            "0.1.",
        )

        # Now we select a very far custom reference point against which the pareto front
        # has not increased in hypervolume at trial 4. Hence, it should stop the
        # optimization at this trial.
        gss2 = ImprovementGlobalStoppingStrategy(
            min_trials=3, window_size=3, improvement_bar=0.1
        )
        objectives = exp.optimization_config.objective.objectives  # pyre-ignore
        custom_objective_thresholds = [
            ObjectiveThreshold(
                metric=objectives[0].metric,
                bound=-10,
                op=ComparisonOp.GEQ,
                relative=False,
            ),
            ObjectiveThreshold(
                metric=objectives[1].metric,
                bound=-10,
                op=ComparisonOp.GEQ,
                relative=False,
            ),
        ]
        stop, message = gss2.should_stop_optimization(
            experiment=exp,
            trial_to_check=4,
            objective_thresholds=custom_objective_thresholds,
        )
        self.assertTrue(stop)
        self.assertEqual(
            message,
            "The improvement in hypervolume in the past 3 trials (=0.033) is less than "
            "0.1.",
        )

    def test_single_objective(self) -> None:

        metric_values = [
            (0.1, 0.6, 0.1),  # feasible, best_objective_so_far = 0.1
            (0.2, 0.3, 0.2),  # feasible, best_objective_so_far = 0.2
            (0.4, 0.5, 0.6),  # infeasible, best_objective_so_far = 0.2
            (0.3, 0.4, 0.0),  # feasible, best_objective_so_far = 0.3
            (0.6, 0.6, 0.3),  # infeasible, best_objective_so_far = 0.3
            (0.9, 0.1, 0.1),  # infeasible, best_objective_so_far = 0.3
        ]
        exp = self._create_single_objective_experiment(metric_values=metric_values)
        gss = ImprovementGlobalStoppingStrategy(
            min_trials=2, window_size=3, improvement_bar=0.1
        )
        stop, message = gss.should_stop_optimization(experiment=exp, trial_to_check=2)
        self.assertFalse(stop)
        self.assertEqual(message, "")

        top, message = gss.should_stop_optimization(experiment=exp, trial_to_check=5)
        self.assertFalse(stop)
        self.assertEqual(
            message,
            "The improvement in best objective in the past 3 trials (=0.000) is "
            "less than 0.1.",
        )

    def test_safety_check(self) -> None:
        experiment = get_experiment()
        gss = ImprovementGlobalStoppingStrategy(min_trials=2, window_size=3)

        stop, message = gss.should_stop_optimization(experiment=experiment)
        self.assertFalse(stop)
        self.assertEqual(
            message,
            "There are no completed trials yet.",
        )

    def test_constraint_satisfaction(self) -> None:
        metric_values = [
            (0.1, 0.6, 0.1),  # feasible
            (0.2, 0.1, 0.2),  # infeasible
            (0.4, 0.5, 0.6),  # infeasible
            (0.5, 0.2, 0.4),  # infeasible
        ]
        exp = self._create_single_objective_experiment(metric_values=metric_values)
        self.assertTrue(constraint_satisfaction(exp.trials[0]))
        self.assertFalse(constraint_satisfaction(exp.trials[1]))
        self.assertFalse(constraint_satisfaction(exp.trials[2]))
        self.assertFalse(constraint_satisfaction(exp.trials[3]))
