#!/usr/bin/env python3
from typing import Any, Callable, List, Optional

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
from ax.utils.common.docutils import copy_doc
from ax.utils.common.typeutils import not_none


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
        cached_data = self.lookup_data_for_trial(trial.index)
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
            evaluations[not_none(trial.arm).name] = self.evaluation_function(
                not_none(trial.arm).params, None
            )
        elif isinstance(trial, BatchTrial):
            if not trial.arms:
                return Data()  # pragma: no cover
            trial.mark_running()
            for arm, weight in trial.normalized_arm_weights().items():
                arm_params: TParameterization = arm.params
                evaluations[arm.name] = self.evaluation_function(arm_params, weight)

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
        """Whether this `SimpleExperiment` has a valid evalutation function
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
