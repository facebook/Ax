#!/usr/bin/env python3
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.core.trial import Trial
from ax.core.types import TEvaluationOutcome, TParameterization
from ax.runners.synthetic import SyntheticRunner


# Function that evaluates one parameter configuration.
TEvaluationFunction = Callable[[TParameterization, Optional[float]], TEvaluationOutcome]


def unimplemented_evaluation_function(
    parameters: TParameterization, weight: Optional[float] = None
) -> TEvaluationOutcome:
    """
    Default evaluation function used if none is provided during initialization.
    The evaluation function must be manually set before use.
    """
    raise Exception("The evaluation function has not been set yet.")


class SimpleExperiment(Experiment):
    """
    Simplified experiment class with defaults.

    Args:
        name: name of this experiment
        search_space: parameter space
        objective_name: which of the metrics computed by the evaluation
            function is the objective
        evaluation_function: function that evaluates
            mean and standard error for a parameter configuration. This
            function should accept a dictionary of parameter names to parameter
            values (TParametrization) and an optional weight, and return a tuple
            of a dictionary of metric names to means and a dictionary of metric
            names to standard errors (TEvaluationOutcome)
        minimize: whether the objective should be minimized,
            defaults to False
        outcome_constraints: constraints on the outcome,
            if any
        status_quo: Arm representing existing "control" arm
    """

    _evaluation_function: TEvaluationFunction

    def __init__(
        self,
        name: str,
        search_space: SearchSpace,
        objective_name: str,
        evaluation_function: TEvaluationFunction = unimplemented_evaluation_function,
        minimize: bool = False,
        outcome_constraints: Optional[List[OutcomeConstraint]] = None,
        status_quo: Optional[Arm] = None,
    ) -> None:
        optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric(name=objective_name), minimize=minimize),
            outcome_constraints=outcome_constraints,
        )
        super().__init__(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            runner=SyntheticRunner(),
            status_quo=status_quo,
        )
        self._evaluation_function = evaluation_function

    def eval_trial(self, trial: BaseTrial) -> Data:
        """
        Evaluate trial arms with the evaluation function of this
        experiment.

        Args:
            trial: trial, whose arms to evaluate.
        """
        cached_data = self.lookup_data_for_trial(trial.index)
        if not cached_data.df.empty:
            return cached_data

        evaluations = {}
        if isinstance(trial, Trial):
            if trial.arm is None:
                return Data()
            trial.run()
            evaluations[trial.arm.name] = self.evaluation_function(
                trial.arm.params, None
            )
        elif isinstance(trial, BatchTrial):
            trial.run()
            for arm, weight in trial.normalized_arm_weights().items():
                arm_params: TParameterization = arm.params
                evaluations[arm.name] = self.evaluation_function(arm_params, weight)

        data = self.data_from_evaluations(evaluations, trial.index)
        self.attach_data(data)
        return data

    @staticmethod
    def data_from_evaluations(
        evaluations: Dict[str, TEvaluationOutcome], trial_index: int
    ) -> Data:
        """
        Convert dict of evaluations to Ax data object.

        Args:
            evaluations: Map from condition name to metric outcomes.

        Returns:
            Ax Data object.
        """
        return Data(
            df=pd.DataFrame(
                [
                    {
                        "arm_name": name,
                        "metric_name": metric_name,
                        "mean": evaluation[metric_name][0],
                        "sem": evaluation[metric_name][1],
                        "trial_index": trial_index,
                    }
                    for name, evaluation in evaluations.items()
                    for metric_name in evaluation.keys()
                ]
            )
        )

    def eval(self) -> Data:
        """
        Evaluate all arms in the experiment with the evaluation
        function passed as argument to this SimpleExperiment.
        """

        return Data.from_multiple_data(
            [
                self.eval_trial(trial)
                for trial in self.trials.values()
                if trial.status != TrialStatus.FAILED
            ]
        )

    @property
    def evaluation_function(self) -> TEvaluationFunction:
        """
        Get the evaluation function.
        """
        return self._evaluation_function

    @evaluation_function.setter
    def evaluation_function(self, evaluation_function: TEvaluationFunction) -> None:
        """
        Set the evaluation function.
        """

        self._evaluation_function = evaluation_function

    def fetch_data(self, **kwargs: Any) -> Data:
        return self.eval()

    def fetch_trial_data(self, trial_index: int, **kwargs: Any) -> Data:
        return self.eval_trial(self.trials[trial_index])
