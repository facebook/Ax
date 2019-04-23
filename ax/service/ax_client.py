#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from typing import Dict, List, Optional, Tuple, Union

from ax.core.arm import Arm
from ax.core.base_trial import TrialStatus
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.trial import Trial
from ax.core.types import (
    TEvaluationOutcome,
    TModelPredictArm,
    TParameterization,
    TParamValue,
)
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.service.utils.dispatch import choose_generation_strategy
from ax.service.utils.instantiation import make_experiment
from ax.service.utils.storage import load_experiment, save_experiment
from ax.storage.sqa_store.structs import DBSettings
from ax.utils.common.typeutils import checked_cast, not_none


class AxClient:
    """
    Convenience handler for management of experimentation cycle through a
    service-like API. External system manages scheduling of the cycle and makes
    calls to this client to get next suggestion in the experiment and log back
    data from the evaluation of that suggestion.

    Note: `AxClient` expects to only propose 1 arm (suggestion) per trial; for
    use cases that require use of batches (multiple suggestions per trial), use
    `AxBatchClient`.

    Two custom types used in this class for convenience are `TParamValue` and
    `TParameterization`. Those are shortcuts for `Union[str, bool, float, int]`
    and `Dict[str, Union[str, bool, float, int]]`, respectively.
    """

    def __init__(
        self,
        generation_strategy: Optional[GenerationStrategy] = None,
        db_settings: Optional[DBSettings] = None,
    ) -> None:
        self.generation_strategy = generation_strategy
        self.db_settings = db_settings
        self._experiment: Optional[Experiment] = None
        # Trials, for which we received data since last `GenerationStrategy.gen`,
        # used to make sure that generation strategy is updated with new data.
        self._updated_trials: List[int] = []

    # ------------------------ Public API methods. ------------------------

    def create_experiment(
        self,
        parameters: List[Dict[str, Union[TParamValue, List[TParamValue]]]],
        name: Optional[str] = None,
        objective_name: Optional[str] = None,
        minimize: bool = False,
        parameter_constraints: Optional[List[str]] = None,
        outcome_constraints: Optional[List[str]] = None,
    ) -> None:
        """Create a new experiment and save it if DBSettings available.

        Args:
            parameters: List of dictionaries representing parameters in the
                experiment search space. Required elements in the dictionaries
                are: "name" (name of this parameter, string), "type" (type of the
                parameter: "range", "fixed", or "choice", string), and "bounds"
                for range parameters (list of two values, lower bound first),
                "values" for choice parameters (list of values), and "value" for
                fixed parameters (single value).
            objective: Name of the metric used as objective in this experiment.
                This metric must be present in `raw_data` argument to `log_data`.
            name: Name of the experiment to be created.
            minimize: Whether this experiment represents a minimization problem.
            parameter_constraints: List of string representation of parameter
                constraints, such as "x3 >= x4" or "x3 + x4 >= 2".
            outcome_constraints: List of string representation of outcome
                constraints of form "metric_name >= bound", like "m1 <= 3."
        """
        if self.db_settings and not name:
            raise ValueError(
                "Must give the experiment a name if `db_settings` is not None."
            )

        self._experiment = make_experiment(
            name=name,
            parameters=parameters,
            objective_name=objective_name,
            minimize=minimize,
            parameter_constraints=parameter_constraints,
            outcome_constraints=outcome_constraints,
        )
        if self.generation_strategy is None:
            self.generation_strategy = choose_generation_strategy(
                search_space=self._experiment.search_space
            )
        self._save_experiment_if_possible()

    def get_next_trial(self) -> Tuple[TParameterization, int]:
        """
        Generate trial with the next set of parameters to try in the iteration process.

        Use `get_next_batch_trial` to generate multiple points at once.

        Returns:
            Tuple of trial parameterization, trial index
        """
        # NOTE: Could move this into log_data to save latency on this call.
        trial = self._suggest_new_trial()
        self._updated_trials = []
        self._save_experiment_if_possible()
        return not_none(trial.arm).parameters, trial.index

    def complete_trial(
        self,
        trial_index: int,
        # acceptable `raw_data` argument formats:
        # 1) {metric_name -> (mean, standard error)}
        # 2) (mean, standard error) and we assume metric name == objective name
        # 3) only the mean, and we assume metric name == objective name and
        #    standard error == 0
        raw_data: TEvaluationOutcome,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Completes the trial with given metric values and adds optional metadata
        to it.

        Args:
            trial_index: Index of trial within the experiment.
            raw_data: Map from metric name to (mean, standard_error).
            metadata: Additional metadata to track about this run.
        """
        trial = self.experiment.trials[trial_index]
        if not isinstance(trial, Trial):
            raise ValueError("To log data for BatchTrial use `AxBatchClient`.")

        trial._status = TrialStatus.COMPLETED
        if metadata is not None:
            trial._run_metadata = metadata

        if isinstance(raw_data, dict):
            evaluations = {not_none(trial.arm).name: raw_data}
        elif isinstance(raw_data, tuple):
            evaluations = {
                not_none(trial.arm).name: {
                    self.experiment.optimization_config.objective.metric.name: raw_data
                }
            }
        elif isinstance(raw_data, float) or isinstance(raw_data, int):
            evaluations = {
                not_none(trial.arm).name: {
                    self.experiment.optimization_config.objective.metric.name: (
                        raw_data,
                        0.0,
                    )
                }
            }
        else:
            raise Exception(  # pragma: no cover
                "Raw_data has an invalid type. The data must either be in the form "
                "of a dictionary of metric names to mean, sem tuples, "
                "or a single mean, sem tuple, or a single mean."
            )

        data = Data.from_evaluations(evaluations, trial.index)
        self.experiment.attach_data(data)
        self._updated_trials.append(trial_index)
        self._save_experiment_if_possible()

    def log_trial_failure(
        self, trial_index: int, metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """Mark that the given trial has failed while running.

        Args:
            trial_index: Index of trial within the experiment.
            metadata: Additional metadata to track about this run.
        """
        trial = self.experiment.trials[trial_index]
        trial._status = TrialStatus.FAILED
        if metadata is not None:
            trial._run_metadata = metadata
        self._save_experiment_if_possible()

    def attach_trial(
        self, parameters: TParameterization
    ) -> Tuple[TParameterization, int]:
        """Attach a new trial with the given parameterization to the experiment.

        Args:
            parameters: Parameterization of the new trial.

        Returns:
            Tuple of parameterization and trial index from newly created trial.
        """
        trial = self.experiment.new_trial().add_arm(Arm(parameters=parameters))
        self._save_experiment_if_possible()
        return not_none(trial.arm).parameters, trial.index

    # TODO[T42389552]: this is currently only compatible with some models.
    def get_best_parameters(
        self
    ) -> Optional[Tuple[TParameterization, Optional[TModelPredictArm]]]:
        """
        Return the best set of parameters the experiment has knowledge of.

        If experiment is in the optimization phase, return the best point
        determined by the model used in the latest optimization round, otherwise
        return none.

        Custom type `TModelPredictArm` is defined as
        `Tuple[Dict[str, float], Dict[str, Dict[str, float]]]`, and stands
        for tuple of two mappings: metric name to its mean value and metric
        name to a mapping of other mapping name to covariance of the two metrics.

        Returns:
            Tuple of (best parameters, model predictions for best parameters).
            None if no data.
        """
        # Find latest trial which has a generator_run attached and get its predictions
        for _, trial in sorted(
            list(self.experiment.trials.items()), key=lambda x: x[0], reverse=True
        ):
            tr = checked_cast(Trial, trial)
            gr = tr.generator_run
            if gr is not None and gr.best_arm_predictions is not None:
                best_arm, best_arm_predictions = gr.best_arm_predictions
                return best_arm.parameters, best_arm_predictions
        return None

    def load_experiment(self, experiment_name: str) -> None:
        """Load an existing experiment.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            Experiment object.
        """
        if not self.db_settings:
            raise ValueError("Need to set db_settings on handler to load experiment.")
        self.experiment = load_experiment(experiment_name, self.db_settings)

    def get_report(self) -> str:
        """Returns HTML of a generated report containing vizualizations."""
        raise NotImplementedError("Report generation not supported for `AxClient` yet.")

    def should_stop_early(self, trial_index: int, data: TEvaluationOutcome) -> bool:
        """Whether to stop the given parameterization given early data."""
        raise NotImplementedError(
            "Early stopping of trials not supported for `AxClient` yet."
        )

    # ---------------------- Private helper methods. ---------------------

    @property
    def experiment(self) -> Experiment:
        """Returns the experiment set on this Ax client"""
        if self._experiment is None:
            raise ValueError(
                "Experiment not set on Ax client. Must first "
                "call load_experiment or create_experiment to use handler functions."
            )
        return self._experiment

    def _save_experiment_if_possible(self) -> bool:
        """Saves attached experiment if DB settings are set on this AxClient
        instance.

        Returns:
            bool: Whether the experiment was saved.
        """
        if self.db_settings and self._experiment:
            save_experiment(self._experiment, self.db_settings)
            return True
        return False

    def _suggest_new_trial(self) -> Trial:
        """
        Suggest new candidate for this experiment.

        Args:
            n: Number of candidates to generate.

        Returns:
            Trial with candidate.
        """
        new_data = Data.from_multiple_data(
            [self.experiment.lookup_data_for_trial(idx) for idx in self._updated_trials]
        )
        try:
            generator_run = not_none(self.generation_strategy).gen(
                experiment=self.experiment, new_data=new_data
            )
        except ValueError as err:
            raise ValueError(
                f"Error getting next trial: {err} Likely cause of the error is "
                "that more trials need to be completed with data."
            )
        return self.experiment.new_trial(generator_run=generator_run)
