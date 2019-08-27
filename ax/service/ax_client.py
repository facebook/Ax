#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import ax.service.utils.best_point as best_point_utils
import numpy as np
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.trial import Trial
from ax.core.types import (
    TEvaluationOutcome,
    TFidelityTrialEvaluation,
    TModelPredictArm,
    TParameterization,
    TParamValue,
    TTrialEvaluation,
)
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.modelbridge_utils import get_pending_observation_features
from ax.plot.base import AxPlotConfig
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.utils.dispatch import choose_generation_strategy
from ax.service.utils.instantiation import make_experiment, raw_data_to_evaluation
from ax.service.utils.storage import (
    load_experiment_and_generation_strategy,
    save_experiment_and_generation_strategy,
)
from ax.storage.json_store.decoder import (
    generation_strategy_from_json,
    object_from_json,
)
from ax.storage.json_store.encoder import object_to_json
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none
from botorch.utils.sampling import manual_seed


logger = get_logger(__name__)


try:  # We don't require SQLAlchemy by default.
    from ax.storage.sqa_store.structs import DBSettings
except ModuleNotFoundError:  # pragma: no cover
    DBSettings = None


class AxClient:
    """
    Convenience handler for management of experimentation cycle through a
    service-like API. External system manages scheduling of the cycle and makes
    calls to this client to get next suggestion in the experiment and log back
    data from the evaluation of that suggestion.

    Note: `AxClient` expects to only propose 1 arm (suggestion) per trial; support
    for use cases that require use of batches is coming soon.

    Two custom types used in this class for convenience are `TParamValue` and
    `TParameterization`. Those are shortcuts for `Union[str, bool, float, int]`
    and `Dict[str, Union[str, bool, float, int]]`, respectively.

    Args:
        generation_strategy: Optional generation strategy. If not set, one is
            intelligently chosen based on properties of search space.

        db_settings: Settings for saving and reloading the underlying experiment
            to a database.

        enforce_sequential_optimization: Whether to enforce that when it is
            reasonable to switch models during the optimization (as prescribed
            by `num_arms` in generation strategy), Ax will wait for enough trials
            to be completed with data to proceed. Defaults to True. If set to
            False, Ax will keep generating new trials from the previous model
            until enough data is gathered. Use this only if necessary;
            otherwise, it is more resource-efficient to
            optimize sequentially, by waiting until enough data is available to
            use the next model.

        random_seed: Optional integer random seed, set to fix the optimization
            random seed for reproducibility. Works only for Sobol quasi-random
            generator and for BoTorch-powered models. For the latter models, the
            trials generated from the same optimization setup with the same seed,
            will be mostly similar, but the exact parameter values may still vary
            and trials latter in the optimizations will diverge more and more.
            This is because a degree of randomness is essential for high performance
            of the Bayesian optimization models and is not controlled by the seed.

            Note: In multi-threaded environments, the random seed is thread-safe,
            but does not actually guarantee reproducibility. Whether the outcomes
            will be exactly the same for two same operations that use the random
            seed, depends on whether the threads modify the random state in the
            same order across the two operations.
    """

    def __init__(
        self,
        generation_strategy: Optional[GenerationStrategy] = None,
        db_settings: Any = None,
        enforce_sequential_optimization: bool = True,
        random_seed: Optional[int] = None,
    ) -> None:
        self.generation_strategy = generation_strategy
        if db_settings and (not DBSettings or not isinstance(db_settings, DBSettings)):
            raise ValueError(
                "`db_settings` argument should be of type ax.storage.sqa_store."
                "structs.DBSettings. To use `DBSettings`, you will need SQLAlchemy "
                "installed in your environment (can be installed through pip)."
            )
        self.db_settings = db_settings
        self._experiment: Optional[Experiment] = None
        self._enforce_sequential_optimization = enforce_sequential_optimization
        self._random_seed = random_seed
        if random_seed is not None:
            logger.info(
                f"Random seed set to {random_seed}. Note that this setting "
                "only affects the Sobol quasi-random generator "
                "and BoTorch-powered Bayesian optimization models. For the latter "
                "models, setting random seed to the same number for two optimizations "
                "will make the generated trials similar, but not exactly the same, "
                "and over time the trials will diverge more."
            )
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
        status_quo: Optional[TParameterization] = None,
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
                This metric must be present in `raw_data` argument to `complete_trial`.
            name: Name of the experiment to be created.
            minimize: Whether this experiment represents a minimization problem.
            parameter_constraints: List of string representation of parameter
                constraints, such as "x3 >= x4" or "x3 + x4 + x5 >= 2". For sum
                constraints, any number of arguments is accepted, and acceptable
                operators are "<=" and ">=".
            outcome_constraints: List of string representation of outcome
                constraints of form "metric_name >= bound", like "m1 <= 3."
            status_quo: Parameterization of the current state of the system.
                If set, this will be added to each trial to be evaluated alongside
                test configurations.
        """
        if self.db_settings and not name:
            raise ValueError(  # pragma: no cover
                "Must give the experiment a name if `db_settings` is not None."
            )

        self._experiment = make_experiment(
            name=name,
            parameters=parameters,
            objective_name=objective_name,
            minimize=minimize,
            parameter_constraints=parameter_constraints,
            outcome_constraints=outcome_constraints,
            status_quo=status_quo,
        )
        if self.generation_strategy is None:
            self.generation_strategy = choose_generation_strategy(
                search_space=self._experiment.search_space,
                enforce_sequential_optimization=self._enforce_sequential_optimization,
                random_seed=self._random_seed,
            )
        self._save_experiment_and_generation_strategy_if_possible()

    def get_next_trial(self) -> Tuple[TParameterization, int]:
        """
        Generate trial with the next set of parameters to try in the iteration process.

        Note: Service API currently supports only 1-arm trials.

        Returns:
            Tuple of trial parameterization, trial index
        """
        with warnings.catch_warnings():
            # Filter out GPYTorch warnings to avoid confusing users.
            warnings.simplefilter("ignore")
            trial = self._suggest_new_trial()
        trial.mark_dispatched()
        self._updated_trials = []
        self._save_experiment_and_generation_strategy_if_possible()
        return not_none(trial.arm).parameters, trial.index

    def complete_trial(
        self,
        trial_index: int,
        raw_data: TEvaluationOutcome,
        metadata: Optional[Dict[str, str]] = None,
        sample_size: Optional[int] = None,
    ) -> None:
        """
        Completes the trial with given metric values and adds optional metadata
        to it.

        Args:
            trial_index: Index of trial within the experiment.
            raw_data: Evaluation data for the trial. Can be a mapping from
                metric name to a tuple of mean and SEM, just a tuple of mean and
                SEM if only one metric in optimization, or just the mean if there
                is no SEM.  Can also be a list of (fidelities, mapping from
                metric name to a tuple of mean and SEM).
            metadata: Additional metadata to track about this run.
        """
        assert isinstance(
            trial_index, int
        ), f"Trial index must be an int, got: {trial_index}."  # pragma: no cover
        trial = self.experiment.trials[trial_index]
        if not isinstance(trial, Trial):
            raise NotImplementedError(
                "Batch trial functionality is not yet available through Service API."
            )

        if metadata is not None:
            trial._run_metadata = metadata

        arm_name = not_none(trial.arm).name
        objective_name = self.experiment.optimization_config.objective.metric.name
        evaluations = {
            arm_name: raw_data_to_evaluation(
                raw_data=raw_data, objective_name=objective_name
            )
        }
        sample_sizes = {arm_name: sample_size} if sample_size else {}
        # evaluations[arm_name] is either a trial evaluation
        # {metric_name -> (mean, SEM)} or a fidelity trial evaluation
        # [(fidelities, {metric_name -> (mean, SEM)})]
        if isinstance(evaluations[arm_name], dict):
            data = Data.from_evaluations(
                evaluations=cast(Dict[str, TTrialEvaluation], evaluations),
                trial_index=trial.index,
                sample_sizes=sample_sizes,
            )
        else:
            data = Data.from_fidelity_evaluations(
                evaluations=cast(Dict[str, TFidelityTrialEvaluation], evaluations),
                trial_index=trial.index,
                sample_sizes=sample_sizes,
            )
        trial.mark_completed()
        self.experiment.attach_data(data)
        self._updated_trials.append(trial_index)
        self._save_experiment_and_generation_strategy_if_possible()

    def log_trial_failure(
        self, trial_index: int, metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """Mark that the given trial has failed while running.

        Args:
            trial_index: Index of trial within the experiment.
            metadata: Additional metadata to track about this run.
        """
        trial = self.experiment.trials[trial_index]
        trial.mark_failed()
        if metadata is not None:
            trial._run_metadata = metadata
        self._save_experiment_and_generation_strategy_if_possible()

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
        trial.mark_dispatched()
        self._save_experiment_and_generation_strategy_if_possible()
        return not_none(trial.arm).parameters, trial.index

    def get_trial_parameters(self, trial_index: int) -> TParameterization:
        """Retrieve the parameterization of the trial by the given index."""
        if trial_index not in self.experiment.trials:
            raise ValueError(f"Trial {trial_index} does not yet exist.")
        trial = checked_cast(Trial, self.experiment.trials.get(trial_index))
        return not_none(trial.arm).parameters

    @copy_doc(best_point_utils.get_best_parameters)
    def get_best_parameters(
        self
    ) -> Optional[Tuple[TParameterization, Optional[TModelPredictArm]]]:
        return best_point_utils.get_best_parameters(self.experiment)

    def get_recommended_max_parallelism(self) -> List[Tuple[int, int]]:
        """Recommends maximum number of trials that can be scheduled in parallel
        at different stages of optimization.

        Some optimization algorithms profit significantly from sequential
        optimization (e.g. suggest a few points, get updated with data for them,
        repeat). This setting indicates how many trials should be in flight
        (generated, but not yet completed with data).

        The output of this method is mapping of form
        {num_trials -> max_parallelism_setting}, where the max_parallelism_setting
        is used for num_trials trials. If max_parallelism_setting is -1, as
        many of the trials can be ran in parallel, as necessary. If num_trials
        in a tuple is -1, then the corresponding max_parallelism_setting
        should be used for all subsequent trials.

        For example, if the returned list is [(5, -1), (12, 6), (-1, 3)],
        the schedule could be: run 5 trials in parallel, run 6 trials in
        parallel twice, run 3 trials in parallel for as long as needed. Here,
        'running' a trial means obtaining a next trial from `AxClient` through
        get_next_trials and completing it with data when available.

        Returns:
            Mapping of form {num_trials -> max_parallelism_setting}.
        """
        if not (self._experiment and self.generation_strategy):
            # Auto-selected generation strategy is set on experiment creation.
            raise ValueError(
                "`get_recommended_max_parallelism` requires an experiment to be "
                "set on AxClient first."
            )
        parallelism_settings = []
        # pyre-fixme[16]: `Optional` has no attribute `_steps`.
        for step in self.generation_strategy._steps:
            parallelism_settings.append(
                (step.num_arms, step.recommended_max_parallelism or step.num_arms)
            )
        return parallelism_settings

    def get_optimization_trace(
        self, objective_optimum: Optional[float] = None
    ) -> AxPlotConfig:
        """Retrieves the plot configuration for optimization trace, which shows
        the evolution of the objective mean over iterations.

        Args:
            objective_optimum: Optimal objective, if known, for display in the
                visualization.
        """
        if not self.experiment.trials:
            raise ValueError("Cannot generate plot as there are no trials.")
        objective_name = self.experiment.optimization_config.objective.metric.name
        best_objectives = np.array(
            [
                [
                    checked_cast(Trial, trial).objective_mean
                    for trial in self.experiment.trials.values()
                ]
            ]
        )
        return optimization_trace_single_method(
            y=(
                np.minimum.accumulate(best_objectives, axis=1)
                if self.experiment.optimization_config.objective.minimize
                else np.maximum.accumulate(best_objectives, axis=1)
            ),
            optimum=objective_optimum,
            title="Model performance vs. # of iterations",
            ylabel=objective_name.capitalize(),
        )

    def get_contour_plot(
        self,
        param_x: Optional[str] = None,
        param_y: Optional[str] = None,
        metric_name: Optional[str] = None,
    ) -> AxPlotConfig:
        """Retrieves a plot configuration for a contour plot of the response
        surface. For response surfaces with more than two parameters,
        selected two parameters will appear on the axes, and remaining parameters
        will be affixed to the middle of their range. If contour params arguments
        are not provided, the first two parameters in the search space will be
        used. If contour metrics are not provided, objective will be used.

        Args:
            param_x: name of parameters to use on x-axis for
                the contour response surface plots.
            param_y: name of parameters to use on y-axis for
                the contour response surface plots.
            metric_name: Name of the metric, for which to plot the response
                surface.
        """
        if not self.experiment.trials:
            raise ValueError("Cannot generate plot as there are no trials.")
        assert (
            self.generation_strategy
        ), "Cannot plot response surface without generation strategy."
        if len(self.experiment.parameters) < 2:
            raise ValueError(
                "Cannot create a contour plot as experiment has less than 2 "
                "parameters, but a contour-related argument was provided."
            )
        if (param_x or param_y) and not (param_x and param_y):
            raise ValueError(
                "If `param_x` is provided, `param_y` is "
                "required as well, and vice-versa."
            )
        objective_name = self.experiment.optimization_config.objective.metric.name
        if not metric_name:
            metric_name = objective_name

        if not param_x or not param_y:
            parameter_names = list(self.experiment.parameters.keys())
            param_x = parameter_names[0]
            param_y = parameter_names[1]

        if param_x not in self.experiment.parameters:
            raise ValueError(
                f'Parameter "{param_x}" not found in the optimization search space.'
            )
        if param_y not in self.experiment.parameters:
            raise ValueError(
                f'Parameter "{param_y}" not found in the optimization search space.'
            )
        if metric_name not in self.experiment.metrics:
            raise ValueError(
                f'Metric "{metric_name}" is not associated with this optimization.'
            )
        # pyre-fixme[16]: `Optional` has no attribute `model`.
        if self.generation_strategy.model is not None:
            try:
                return plot_contour(
                    model=self.generation_strategy.model,
                    param_x=param_x,
                    param_y=param_y,
                    metric_name=metric_name,
                )
            except NotImplementedError:
                # Some models don't implement '_predict', which is needed
                # for the contour plots.
                pass
        raise ValueError(
            f'Could not obtain contour plot of "{metric_name}" for parameters '
            f'"{param_x}" and "{param_y}", as a model with predictive ability, '
            "such as a Gaussian Process, has not yet been trained in the course "
            "of this optimization."
        )

    def load_experiment(self, experiment_name: str) -> None:
        """[Work in progress] Load an existing experiment.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            Experiment object.
        """
        if not self.db_settings:
            raise ValueError(  # pragma: no cover
                "Cannot load an experiment in the absence of the DB settings."
                "Please initialize `AxClient` with DBSettings."
            )
        experiment, generation_strategy = load_experiment_and_generation_strategy(
            experiment_name=experiment_name, db_settings=self.db_settings
        )
        self._experiment = experiment
        self.generation_strategy = generation_strategy

    def get_report(self) -> str:
        """Returns HTML of a generated report containing vizualizations."""
        raise NotImplementedError(  # pragma: no cover
            "Report generation not supported for `AxClient` yet."
        )

    def should_stop_early(self, trial_index: int, data: TEvaluationOutcome) -> bool:
        """Whether to stop the given parameterization given early data."""
        raise NotImplementedError(  # pragma: no cover
            "Early stopping of trials not supported for `AxClient` yet."
        )

    def to_json_snapshot(self) -> Dict[str, Any]:
        """Serialize this `AxClient` to JSON to be able to interrupt and restart
        optimization and save it to file by the provided path.
        """
        return {
            "_type": self.__class__.__name__,
            "experiment": object_to_json(self.experiment),
            "generation_strategy": object_to_json(self.generation_strategy),
            "_enforce_sequential_optimization": self._enforce_sequential_optimization,
            "_updated_trials": object_to_json(self._updated_trials),
        }

    @staticmethod
    def from_json_snapshot(serialized: Dict[str, Any]) -> "AxClient":
        """Recreate an `AxClient` from a JSON snapshot."""
        experiment = object_from_json(serialized.pop("experiment"))
        ax_client = AxClient(
            generation_strategy=generation_strategy_from_json(
                generation_strategy_json=serialized.pop("generation_strategy")
            ),
            enforce_sequential_optimization=serialized.pop(
                "_enforce_sequential_optimization"
            ),
        )
        ax_client._experiment = experiment
        ax_client._updated_trials = object_from_json(serialized.pop("_updated_trials"))
        return ax_client

    def save(self, filepath: str = "ax_client_snapshot.json") -> None:
        """Save a JSON-serialized snapshot of this `AxClient`'s settings and state
        to a .json file by the given path.
        """
        with open(filepath, "w+") as file:  # pragma: no cover
            file.write(json.dumps(self.to_json_snapshot()))

    @staticmethod
    def load(filepath: str = "ax_client_snapshot.json") -> "AxClient":
        """Restore an `AxClient` and its state from a JSON-serialized snapshot,
        residing in a .json file by the given path.
        """
        with open(filepath, "r") as file:  # pragma: no cover
            serialized = json.loads(file.read())
            return AxClient.from_json_snapshot(serialized=serialized)

    # ---------------------- Private helper methods. ---------------------

    @property
    def experiment(self) -> Experiment:
        """Returns the experiment set on this Ax client"""
        if self._experiment is None:
            raise ValueError(
                "Experiment not set on Ax client. Must first "
                "call load_experiment or create_experiment to use handler functions."
            )
        # pyre-fixme[7]: Expected `Experiment` but got `Optional[Experiment]`.
        return self._experiment

    def _save_experiment_and_generation_strategy_if_possible(self) -> bool:
        """Saves attached experiment and generation strategy if DB settings are
        set on this AxClient instance.

        Returns:
            bool: Whether the experiment was saved.
        """
        if self.db_settings is not None and self._experiment is not None:
            assert (
                self.generation_strategy is not None
            ), "If experiment is set, generation strategy should be too."
            save_experiment_and_generation_strategy(
                experiment=not_none(self._experiment),
                generation_strategy=not_none(self.generation_strategy),
                db_settings=self.db_settings,
            )
        return False

    def _get_new_data(self) -> Data:
        """
        Returns new data since the last run of the generator.

        Returns:
            Latest data.
        """
        return Data.from_multiple_data(
            [self.experiment.lookup_data_for_trial(idx) for idx in self._updated_trials]
        )

    def _suggest_new_trial(self) -> Trial:
        """
        Suggest new candidate for this experiment.

        Args:
            n: Number of candidates to generate.

        Returns:
            Trial with candidate.
        """
        new_data = self._get_new_data()
        # If random seed is not set for this optimization, context manager does
        # nothing; otherwise, it sets the random seed for torch, but only for the
        # scope of this call. This is important because torch seed is set globally,
        # so if we just set the seed without the context manager, it can have
        # serious negative impact on the performance of the models that employ
        # stochasticity.
        with manual_seed(seed=self._random_seed):
            generator_run = not_none(self.generation_strategy).gen(
                experiment=self.experiment,
                new_data=new_data,
                pending_observations=get_pending_observation_features(
                    experiment=self.experiment
                ),
            )
        return self.experiment.new_trial(generator_run=generator_run)
