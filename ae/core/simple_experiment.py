#!/usr/bin/env python3
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from ae.lazarus.ae.core.base_trial import BaseTrial
from ae.lazarus.ae.core.batch_trial import BatchTrial
from ae.lazarus.ae.core.condition import Condition
from ae.lazarus.ae.core.data import Data
from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.core.metric import Metric
from ae.lazarus.ae.core.objective import Objective
from ae.lazarus.ae.core.optimization_config import OptimizationConfig
from ae.lazarus.ae.core.outcome_constraint import OutcomeConstraint
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.core.trial import Trial
from ae.lazarus.ae.core.types.types import TParameterization
from ae.lazarus.ae.runners.synthetic import SyntheticRunner


# Outcome of the evaluation function: {metric_name -> (mean, standard error)}
TEvaluationOutcome = Dict[str, Tuple[float, float]]
# Function that evaluates one parameter configuration.
TEvaluationFunction = Callable[[TParameterization, Optional[float]], TEvaluationOutcome]


class SimpleExperiment(Experiment):
    """
    Simplified experiment class with defaults.

    Args:
        name (str): name of this experiment
        search_space (SearchSpace): parameter space
        evaluation_function (TEvaluationFunction): function that evaluates
            mean and standard error for a parameter configuration. This
            function should accept a dictionary of parameter names to parameter
            values (TParametrization) and an optional weight, and return a tuple
            of a dictionary of metric names to means and a dictionary of metric
            names to standard errors (TEvaluationOutcome)
        objective_name (str): which of the metrics computed by the evaluation
            function is the objective
        minimize (bool, optional): whether the objective should be minimized,
            defaults to False
        outcome_constraints (List[OutcomeConstraint]): constraints on the outcome,
            if any
        status_quo: Condition representing existing "control" condition
    """

    evaluation_function: TEvaluationFunction

    def __init__(
        self,
        name: str,
        search_space: SearchSpace,
        evaluation_function: TEvaluationFunction,
        objective_name: str,
        minimize: bool = False,
        outcome_constraints: Optional[List[OutcomeConstraint]] = None,
        status_quo: Optional[Condition] = None,
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
        self.evaluation_function = evaluation_function

    def eval_trial(self, trial: BaseTrial) -> Data:
        """
        Evaluate trial conditions with the evaluation function of this
        experiment.

        Args:
            trial (BatchTrial): trial, whose conditions to evaluate.
        """
        cached_data = self.lookup_data_for_trial(trial.index)
        if not cached_data.df.empty:
            return cached_data

        evaluations = {}
        if isinstance(trial, Trial):
            if trial.condition is None:
                return Data()
            trial.run()
            evaluations[trial.condition.name] = self.evaluation_function(
                trial.condition.params, None
            )
        elif isinstance(trial, BatchTrial):
            trial.run()
            for condition, weight in trial.normalized_condition_weights().items():
                condition_params: TParameterization = condition.params
                evaluations[condition.name] = self.evaluation_function(
                    condition_params, weight
                )

        data = Data(
            df=pd.DataFrame(
                [
                    {
                        "condition_name": name,
                        "metric_name": metric_name,
                        "mean": evaluation[metric_name][0],
                        "sem": evaluation[metric_name][1],
                        "trial_index": trial.index,
                    }
                    for name, evaluation in evaluations.items()
                    for metric_name in evaluation.keys()
                ]
            )
        )

        self.attach_data(data)
        return data

    def eval(self) -> Data:
        """
        Evaluate all conditions in the experiment with the evaluation
        function passed as argument to this SimpleExperiment.
        """

        return Data.from_multiple_data(
            [self.eval_trial(trial) for trial in self.trials.values()]
        )

    def fetch_data(self, **kwargs: Any) -> Data:
        return self.eval()

    def fetch_trial_data(self, trial_index: int, **kwargs: Any) -> Data:
        return self.eval_trial(self.trials[trial_index])
