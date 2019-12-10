#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from typing import Any, Callable, List, Optional

import numpy as np
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
from ax.core.types import TEvaluationOutcome, TParameterization, TTrialEvaluation
from ax.utils.common.docutils import copy_doc
from ax.utils.common.typeutils import not_none, numpy_type_to_python_type


DEFAULT_OBJECTIVE_NAME = "objective"

# Function that evaluates one parameter configuration.
TEvaluationFunction = Callable[[TParameterization, Optional[float]], TEvaluationOutcome]


def unimplemented_evaluation_function(
    parameterization: TParameterization, weight: Optional[float] = None
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
        search_space: parameter space
        name: name of this experiment
        objective_name: which of the metrics computed by the evaluation
            function is the objective
        evaluation_function: function that evaluates
            mean and standard error for a parameter configuration. This
            function should accept a dictionary of parameter names to parameter
            values (TParametrization) and optionally a weight, and return a
            dictionary of metric names to a tuple of means and standard errors
            (TEvaluationOutcome). The function can also return a single tuple,
            in which case we assume the metric is the objective.
        minimize: whether the objective should be minimized,
            defaults to False
        outcome_constraints: constraints on the outcome,
            if any
        status_quo: Arm representing existing "control" arm
    """

    _evaluation_function: TEvaluationFunction

    def __init__(
        self,
        search_space: SearchSpace,
        name: Optional[str] = None,
        objective_name: Optional[str] = None,
        evaluation_function: TEvaluationFunction = unimplemented_evaluation_function,
        minimize: bool = False,
        outcome_constraints: Optional[List[OutcomeConstraint]] = None,
        status_quo: Optional[Arm] = None,
    ) -> None:
        optimization_config = OptimizationConfig(
            objective=Objective(
                metric=Metric(name=objective_name or DEFAULT_OBJECTIVE_NAME),
                minimize=minimize,
            ),
            outcome_constraints=outcome_constraints,
        )
        super().__init__(
            name=name,
            search_space=search_space,
            optimization_config=optimization_config,
            status_quo=status_quo,
        )
        self._evaluation_function = evaluation_function

    @copy_doc(Experiment.is_simple_experiment)
    @property
    def is_simple_experiment(self):
        return True

    def eval_trial(self, trial: BaseTrial) -> Data:
        """
        Evaluate trial arms with the evaluation function of this
        experiment.

        Args:
            trial: trial, whose arms to evaluate.
        """
        cached_data = self.lookup_data_for_trial(trial.index)[0]
        if not cached_data.df.empty:
            return cached_data

        evaluations = {}
        if not self.has_evaluation_function:
            raise ValueError(  # pragma: no cover
                f"Cannot evaluate trial {trial.index} as no attached data was "
                "found and no evaluation function is set on this `SimpleExperiment.`"
                "`SimpleExperiment` is geared to synchronous and sequential cases "
                "where each trial is evaluated before more trials are created. "
                "For all other cases, use `Experiment`."
            )
        if isinstance(trial, Trial):
            if not trial.arm:
                return Data()  # pragma: no cover
            trial.mark_running()
            evaluations[not_none(trial.arm).name] = self.evaluation_function_outer(
                not_none(trial.arm).parameters, None
            )
        elif isinstance(trial, BatchTrial):
            if not trial.arms:
                return Data()  # pragma: no cover
            trial.mark_running()
            for arm, weight in trial.normalized_arm_weights().items():
                arm_parameters: TParameterization = arm.parameters
                evaluations[arm.name] = self.evaluation_function_outer(
                    arm_parameters, weight
                )
        trial.mark_completed()
        data = Data.from_evaluations(evaluations, trial.index)
        self.attach_data(data)
        return data

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
    def has_evaluation_function(self) -> bool:
        """Whether this `SimpleExperiment` has a valid evaluation function
        attached."""
        return self._evaluation_function is not unimplemented_evaluation_function

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

    def evaluation_function_outer(
        self, parameterization: TParameterization, weight: Optional[float] = None
    ) -> TTrialEvaluation:
        signature = inspect.signature(self._evaluation_function)
        num_evaluation_function_params = len(signature.parameters.items())
        if num_evaluation_function_params == 1:
            # pyre-fixme[20]: Anonymous call expects argument `$1`.
            evaluation = self._evaluation_function(parameterization)
        elif num_evaluation_function_params == 2:
            evaluation = self._evaluation_function(parameterization, weight)
        else:
            raise ValueError(  # pragma: no cover
                "Evaluation function must take either one parameter "
                "(parameterization) or two parameters (parameterization and weight)."
            )

        if isinstance(evaluation, dict):
            return evaluation
        elif isinstance(evaluation, tuple):
            return {self.optimization_config.objective.metric.name: evaluation}
        elif isinstance(evaluation, (float, int)):
            return {self.optimization_config.objective.metric.name: (evaluation, 0.0)}
        elif isinstance(evaluation, (np.float32, np.float64, np.int32, np.int64)):
            return {
                self.optimization_config.objective.metric.name: (
                    numpy_type_to_python_type(evaluation),
                    0.0,
                )
            }
        raise Exception(  # pragma: no cover
            "Evaluation function returned an invalid type. The function must "
            "either return a dictionary of metric names to mean, sem tuples "
            "or a single mean, sem tuple, or a single mean."
        )

    @copy_doc(Experiment.fetch_data)
    def fetch_data(self, metrics: Optional[List[Metric]] = None, **kwargs: Any) -> Data:
        return self.eval()

    @copy_doc(Experiment._fetch_trial_data)
    def _fetch_trial_data(
        self, trial_index: int, metrics: Optional[List[Metric]] = None, **kwargs: Any
    ) -> Data:
        return self.eval_trial(self.trials[trial_index])

    @copy_doc(Experiment.add_tracking_metric)
    def add_tracking_metric(self, metric: Metric) -> "SimpleExperiment":
        raise NotImplementedError("SimpleExperiment does not support metric addition.")

    @copy_doc(Experiment.update_tracking_metric)
    def update_tracking_metric(self, metric: Metric) -> "SimpleExperiment":
        raise NotImplementedError("SimpleExperiment does not support metric updates.")
