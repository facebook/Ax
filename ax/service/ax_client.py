#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import ax.service.utils.best_point as best_point_utils
import numpy as np
import pandas as pd
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.trial import Trial
from ax.core.types import (
    TEvaluationOutcome,
    TModelPredictArm,
    TParameterization,
    TParamValue,
)
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.modelbridge_utils import get_pending_observation_features
from ax.plot.base import AxPlotConfig
from ax.plot.contour import plot_contour
from ax.plot.exp_utils import exp_to_df
from ax.plot.feature_importances import plot_feature_importance_by_feature
from ax.plot.helper import _format_dict, _get_in_sample_arms
from ax.plot.trace import optimization_trace_single_method
from ax.service.utils.instantiation import (
    data_from_evaluations,
    make_experiment,
    raw_data_to_evaluation,
)
from ax.service.utils.with_db_settings_base import DBSettings, WithDBSettingsBase
from ax.storage.json_store.decoder import (
    generation_strategy_from_json,
    object_from_json,
)
from ax.storage.json_store.encoder import object_to_json
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import _round_floats_for_logging, get_logger
from ax.utils.common.typeutils import (
    checked_cast,
    checked_cast_dict,
    checked_cast_optional,
    not_none,
)
from botorch.utils.sampling import manual_seed


logger = get_logger(__name__)


class AxClient(WithDBSettingsBase):
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
            to a database. Expected to be of type
            ax.storage.sqa_store.structs.DBSettings and require SQLAlchemy.

        enforce_sequential_optimization: Whether to enforce that when it is
            reasonable to switch models during the optimization (as prescribed
            by `num_trials` in generation strategy), Ax will wait for enough trials
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

        verbose_logging: Whether Ax should log significant optimization events,
            defaults to `True`.

        suppress_storage_errors: Whether to suppress SQL storage-related errors if
            encounted. Only use if SQL storage is not important for the given use
            case, since this will only log, but not raise, an exception if its
            encountered while saving to DB or loading from it.
    """

    def __init__(
        self,
        generation_strategy: Optional[GenerationStrategy] = None,
        db_settings: Optional[DBSettings] = None,
        enforce_sequential_optimization: bool = True,
        random_seed: Optional[int] = None,
        verbose_logging: bool = True,
        suppress_storage_errors: bool = False,
    ) -> None:
        super().__init__(db_settings=db_settings)
        if not verbose_logging:
            logger.setLevel(logging.WARNING)  # pragma: no cover
        else:
            logger.info(
                "Starting optimization with verbose logging. To disable logging, "
                "set the `verbose_logging` argument to `False`. Note that float "
                "values in the logs are rounded to 2 decimal points."
            )
        self._generation_strategy = generation_strategy
        self._experiment: Optional[Experiment] = None
        self._enforce_sequential_optimization = enforce_sequential_optimization
        self._random_seed = random_seed
        self._suppress_storage_errors = suppress_storage_errors
        if random_seed is not None:
            logger.warning(
                f"Random seed set to {random_seed}. Note that this setting "
                "only affects the Sobol quasi-random generator "
                "and BoTorch-powered Bayesian optimization models. For the latter "
                "models, setting random seed to the same number for two optimizations "
                "will make the generated trials similar, but not exactly the same, "
                "and over time the trials will diverge more."
            )

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
        overwrite_existing_experiment: bool = False,
        experiment_type: Optional[str] = None,
        choose_generation_strategy_kwargs: Optional[Dict[str, Any]] = None,
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
                constraints, such as "x3 >= x4" or "-x3 + 2*x4 - 3.5*x5 >= 2". For
                the latter constraints, any number of arguments is accepted, and
                acceptable operators are "<=" and ">=".
            outcome_constraints: List of string representation of outcome
                constraints of form "metric_name >= bound", like "m1 <= 3."
            status_quo: Parameterization of the current state of the system.
                If set, this will be added to each trial to be evaluated alongside
                test configurations.
            overwrite_existing_experiment: If an experiment has already been set
                on this `AxClient` instance, whether to reset it to the new one.
                If overwriting the experiment, generation strategy will be
                re-selected for the new experiment and restarted.
                To protect experiments in production, one cannot overwrite existing
                experiments if the experiment is already stored in the database,
                regardless of the value of `overwrite_existing_experiment`.
            choose_generation_strategy_kwargs: Keyword arguments to pass to
                `choose_generation_strategy` function which determines what
                generation strategy should be used when none was specified on init.
        """
        if self.db_settings_set and not name:
            raise ValueError(  # pragma: no cover
                "Must give the experiment a name if `db_settings` is not None."
            )
        if self.db_settings_set:
            experiment, _ = self._load_experiment_and_generation_strategy(
                experiment_name=not_none(name)
            )
            if experiment:
                raise ValueError(
                    f"Experiment {name} already exists in the database. "
                    "To protect experiments that are running in production, "
                    "overwriting stored experiments is not allowed. To "
                    "start a new experiment and store it, change the "
                    "experiment's name."
                )
        if self._experiment is not None:
            if overwrite_existing_experiment:
                exp_name = self.experiment._name or "untitled"
                new_exp_name = name or "untitled"
                logger.info(
                    f"Overwriting existing experiment ({exp_name}) on this client "
                    f"with new experiment ({new_exp_name}) and restarting the "
                    "generation strategy."
                )
                self._generation_strategy = None
            else:
                raise ValueError(
                    "Experiment already created for this client instance. "
                    "Set the `overwrite_existing_experiment` to `True` to overwrite "
                    "with new experiment."
                )

        self._experiment = make_experiment(
            name=name,
            parameters=parameters,
            objective_name=objective_name,
            minimize=minimize,
            parameter_constraints=parameter_constraints,
            outcome_constraints=outcome_constraints,
            status_quo=status_quo,
            experiment_type=experiment_type,
        )

        try:
            self._save_experiment_to_db_if_possible(
                experiment=self.experiment,
                suppress_all_errors=self._suppress_storage_errors,
            )
        except Exception:
            # Unset the experiment on this `AxClient` instance if encountered and
            # raising an error from saving the experiment, to avoid a case where
            # overall `create_experiment` call fails with a storage error, but
            # `self._experiment` is still set and user has to specify the
            # `ooverwrite_existing_experiment` kwarg to re-attempt exp. creation.
            self._experiment = None
            raise

        self._set_generation_strategy(
            choose_generation_strategy_kwargs=choose_generation_strategy_kwargs
        )
        self._save_generation_strategy_to_db_if_possible(
            generation_strategy=self.generation_strategy,
            suppress_all_errors=self._suppress_storage_errors,
        )

    def get_next_trial(self) -> Tuple[TParameterization, int]:
        """
        Generate trial with the next set of parameters to try in the iteration process.

        Note: Service API currently supports only 1-arm trials.

        Returns:
            Tuple of trial parameterization, trial index
        """
        trial = self.experiment.new_trial(generator_run=self._gen_new_generator_run())
        logger.info(
            f"Generated new trial {trial.index} with parameters "
            f"{_round_floats_for_logging(item=not_none(trial.arm).parameters)}."
        )
        trial.mark_running(no_runner_required=True)
        self._save_new_trial_to_db_if_possible(
            experiment=self.experiment,
            trial=trial,
            suppress_all_errors=self._suppress_storage_errors,
        )
        self._save_generation_strategy_to_db_if_possible(
            generation_strategy=self.generation_strategy,
            suppress_all_errors=self._suppress_storage_errors,
        )
        return not_none(trial.arm).parameters, trial.index

    def abandon_trial(self, trial_index: int, reason: Optional[str] = None) -> None:
        """Abandons a trial and adds optional metadata to it.

        Args:
            trial_index: Index of trial within the experiment.
        """
        trial = self._get_trial(trial_index=trial_index)
        trial.mark_abandoned(reason=reason)

    def complete_trial(
        self,
        trial_index: int,
        raw_data: TEvaluationOutcome,
        metadata: Optional[Dict[str, Union[str, int]]] = None,
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
            sample_size: Number of samples collected for the underlying arm,
                optional.
        """
        # Validate that trial can be completed.
        if not isinstance(trial_index, int):  # pragma: no cover
            raise ValueError(f"Trial index must be an int, got: {trial_index}.")
        trial = self._get_trial(trial_index=trial_index)
        self._validate_can_complete_trial(trial=trial)

        # Format the data to save.
        sample_sizes = {not_none(trial.arm).name: sample_size} if sample_size else {}
        evaluations, data = self._make_evaluations_and_data(
            trial=trial, raw_data=raw_data, metadata=metadata, sample_sizes=sample_sizes
        )
        trial._run_metadata = metadata or {}
        self.experiment.attach_data(data=data)
        trial.mark_completed()
        data_for_logging = _round_floats_for_logging(
            item=evaluations[next(iter(evaluations.keys()))]
        )
        logger.info(
            f"Completed trial {trial_index} with data: "
            f"{_round_floats_for_logging(item=data_for_logging)}."
        )
        self._save_updated_trial_to_db_if_possible(
            experiment=self.experiment,
            trial=trial,
            suppress_all_errors=self._suppress_storage_errors,
        )

    def update_trial_data(
        self,
        trial_index: int,
        raw_data: TEvaluationOutcome,
        metadata: Optional[Dict[str, Union[str, int]]] = None,
        sample_size: Optional[int] = None,
    ) -> None:
        """
        Attaches additional data for completed trial (for example, if trial was
        completed with data for only one of the required metrics and more data
        needs to be attached).

        Args:
            trial_index: Index of trial within the experiment.
            raw_data: Evaluation data for the trial. Can be a mapping from
                metric name to a tuple of mean and SEM, just a tuple of mean and
                SEM if only one metric in optimization, or just the mean if there
                is no SEM.  Can also be a list of (fidelities, mapping from
                metric name to a tuple of mean and SEM).
            metadata: Additional metadata to track about this run.
            sample_size: Number of samples collected for the underlying arm,
                optional.
        """
        assert isinstance(
            trial_index, int
        ), f"Trial index must be an int, got: {trial_index}."  # pragma: no cover
        trial = self._get_trial(trial_index=trial_index)
        if not trial.status.is_completed:
            raise ValueError(
                f"Trial {trial.index} has not yet been completed with data."
                "To complete it, use `ax_client.complete_trial`."
            )
        sample_sizes = {not_none(trial.arm).name: sample_size} if sample_size else {}
        evaluations, data = self._make_evaluations_and_data(
            trial=trial, raw_data=raw_data, metadata=metadata, sample_sizes=sample_sizes
        )
        trial._run_metadata.update(metadata or {})
        # Registering trial data update is needed for generation strategies that
        # leverage the `update` functionality of model and bridge setup and therefore
        # need to be aware of new data added to experiment. Usually this happends
        # seamlessly, by looking at newly completed trials, but in this case trial
        # status does not change, so we manually register the new data.
        # Currently this call will only result in a `NotImplementedError` if generation
        # strategy uses `update` (`GenerationStep.use_update` is False by default).
        self.generation_strategy._register_trial_data_update(trial=trial, data=data)
        self.experiment.attach_data(data, combine_with_last_data=True)
        data_for_logging = _round_floats_for_logging(
            item=evaluations[next(iter(evaluations.keys()))]
        )
        logger.info(
            f"Added data: {_round_floats_for_logging(item=data_for_logging)} "
            f"to trial {trial.index}."
        )
        self._save_experiment_to_db_if_possible(
            experiment=self.experiment,
            suppress_all_errors=self._suppress_storage_errors,
        )

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
        logger.info(f"Registered failure of trial {trial_index}.")
        if metadata is not None:
            trial._run_metadata = metadata
        self._save_experiment_to_db_if_possible(
            experiment=self.experiment,
            suppress_all_errors=self._suppress_storage_errors,
        )

    def attach_trial(
        self, parameters: TParameterization
    ) -> Tuple[TParameterization, int]:
        """Attach a new trial with the given parameterization to the experiment.

        Args:
            parameters: Parameterization of the new trial.

        Returns:
            Tuple of parameterization and trial index from newly created trial.
        """
        self._validate_search_space_membership(parameters=parameters)
        trial = self.experiment.new_trial().add_arm(Arm(parameters=parameters))
        trial.mark_running(no_runner_required=True)
        logger.info(
            "Attached custom parameterization "
            f"{_round_floats_for_logging(item=parameters)} as trial {trial.index}."
        )
        self._save_new_trial_to_db_if_possible(
            experiment=self.experiment,
            trial=trial,
            suppress_all_errors=self._suppress_storage_errors,
        )
        return not_none(trial.arm).parameters, trial.index

    def get_trial_parameters(self, trial_index: int) -> TParameterization:
        """Retrieve the parameterization of the trial by the given index."""
        return not_none(self._get_trial(trial_index).arm).parameters

    @copy_doc(best_point_utils.get_best_parameters)
    def get_best_parameters(
        self,
    ) -> Optional[Tuple[TParameterization, Optional[TModelPredictArm]]]:
        return best_point_utils.get_best_parameters(self.experiment)

    def get_trials_data_frame(self) -> pd.DataFrame:
        return exp_to_df(exp=self.experiment)

    def get_max_parallelism(self) -> List[Tuple[int, int]]:
        """Retrieves maximum number of trials that can be scheduled in parallel
        at different stages of optimization.

        Some optimization algorithms profit significantly from sequential
        optimization (i.e. suggest a few points, get updated with data for them,
        repeat, see https://ax.dev/docs/bayesopt.html).
        Parallelism setting indicates how many trials should be running simulteneously
        (generated, but not yet completed with data).

        The output of this method is mapping of form
        {num_trials -> max_parallelism_setting}, where the max_parallelism_setting
        is used for num_trials trials. If max_parallelism_setting is -1, as
        many of the trials can be ran in parallel, as necessary. If num_trials
        in a tuple is -1, then the corresponding max_parallelism_setting
        should be used for all subsequent trials.

        For example, if the returned list is [(5, -1), (12, 6), (-1, 3)],
        the schedule could be: run 5 trials with any parallelism, run 6 trials in
        parallel twice, run 3 trials in parallel for as long as needed. Here,
        'running' a trial means obtaining a next trial from `AxClient` through
        get_next_trials and completing it with data when available.

        Returns:
            Mapping of form {num_trials -> max_parallelism_setting}.
        """
        parallelism_settings = []
        for step in self.generation_strategy._steps:
            parallelism_settings.append(
                (step.num_trials, step.max_parallelism or step.num_trials)
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
        hover_labels = [
            _format_dict(not_none(checked_cast(Trial, trial).arm).parameters)
            for trial in self.experiment.trials.values()
        ]
        return optimization_trace_single_method(
            y=(
                np.minimum.accumulate(best_objectives, axis=1)
                if self.experiment.optimization_config.objective.minimize
                else np.maximum.accumulate(best_objectives, axis=1)
            ),
            optimum=objective_optimum,
            title="Model performance vs. # of iterations",
            ylabel=objective_name.capitalize(),
            hover_labels=hover_labels,
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
        objective_name = self.objective_name
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
        if self.generation_strategy.model is not None:
            try:
                logger.info(
                    f"Retrieving contour plot with parameter '{param_x}' on X-axis "
                    f"and '{param_y}' on Y-axis, for metric '{metric_name}'. "
                    "Ramaining parameters are affixed to the middle of their range."
                )
                return plot_contour(
                    model=not_none(self.generation_strategy.model),
                    param_x=param_x,
                    param_y=param_y,
                    metric_name=metric_name,
                )

            except NotImplementedError:
                # Some models don't implement '_predict', which is needed
                # for the contour plots.
                logger.info(
                    f"Model {self.generation_strategy.model} does not implement "
                    "`predict`, so it cannot be used to generate a response "
                    "surface plot."
                )
        raise ValueError(
            f'Could not obtain contour plot of "{metric_name}" for parameters '
            f'"{param_x}" and "{param_y}", as a model with predictive ability, '
            "such as a Gaussian Process, has not yet been trained in the course "
            "of this optimization."
        )

    def get_feature_importances(self, relative: bool = True) -> AxPlotConfig:
        """
        Get a bar chart showing feature_importances for a metric.

        A drop-down controls the metric for which the importances are displayed.

        Args:
            relative: Whether the values are displayed as percentiles or
                as raw importance metrics.
        """
        if not self.experiment.trials:
            raise ValueError("Cannot generate plot as there are no trials.")
        cur_model = self.generation_strategy.model
        if cur_model is not None:
            try:
                return plot_feature_importance_by_feature(cur_model, relative=relative)
            except NotImplementedError:
                logger.info(
                    f"Model {self.generation_strategy.model} does not implement "
                    "`feature_importances`, so it cannot be used to generate "
                    "this plot. Only certain models, specifically GPEI, implement "
                    "feature importances."
                )

        raise ValueError(
            "Could not obtain feature_importances for any metrics "
            " as a model that can produce feature importances, such as a "
            "Gaussian Process, has not yet been trained in the course "
            "of this optimization."
        )

    def load_experiment_from_database(
        self,
        experiment_name: str,
        choose_generation_strategy_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Load an existing experiment from database using the `DBSettings`
        passed to this `AxClient` on instantiation.

        Args:
            experiment_name: Name of the experiment.

        Returns:
            Experiment object.
        """
        experiment, generation_strategy = self._load_experiment_and_generation_strategy(
            experiment_name=experiment_name
        )
        if experiment is None:
            raise ValueError(f"Experiment by name '{experiment_name}' not found.")
        self._experiment = experiment
        logger.info(f"Loaded {experiment}.")
        if generation_strategy is None:  # pragma: no cover
            self._set_generation_strategy(
                choose_generation_strategy_kwargs=choose_generation_strategy_kwargs
            )
        else:
            self._generation_strategy = generation_strategy
            logger.info(
                f"Using generation strategy associated with the loaded experiment:"
                f" {generation_strategy}."
            )

    def get_model_predictions(
        self, metric_names: Optional[List[str]] = None
    ) -> Dict[int, Dict[str, Tuple[float, float]]]:
        """Retrieve model-estimated means and covariances for all metrics.
        Note: this function retrieves the predictions for the 'in-sample' arms,
        which means that the return mapping on this function will only contain
        predictions for trials that have been completed with data.

        Args:
            metric_names: Names of the metrics, for which to retrieve predictions.
                All metrics on experiment will be retrieved if this argument was
                not specified.

        Returns:
            A mapping from trial index to a mapping of metric names to tuples
            of predicted metric mean and SEM, of form:
            { trial_index -> { metric_name: ( mean, SEM ) } }.
        """
        if self.generation_strategy.model is None:  # pragma: no cover
            raise ValueError("No model has been instantiated yet.")
        if metric_names is None and self.experiment.metrics is None:
            raise ValueError(  # pragma: no cover
                "No metrics to retrieve specified on the experiment or as "
                "argument to `get_model_predictions`."
            )
        arm_info, _, _ = _get_in_sample_arms(
            model=not_none(self.generation_strategy.model),
            metric_names=set(metric_names)
            if metric_names is not None
            else set(not_none(self.experiment.metrics).keys()),
        )
        trials = checked_cast_dict(int, Trial, self.experiment.trials)

        return {
            trial_index: {
                m: (
                    arm_info[not_none(trials[trial_index].arm).name].y_hat[m],
                    arm_info[not_none(trials[trial_index].arm).name].se_hat[m],
                )
                for m in arm_info[not_none(trials[trial_index].arm).name].y_hat
            }
            for trial_index in trials
            if not_none(trials[trial_index].arm).name in arm_info
        }

    def verify_trial_parameterization(
        self, trial_index: int, parameterization: TParameterization
    ) -> bool:
        """Whether the given parameterization matches that of the arm in the trial
        specified in the trial index.
        """
        return (
            not_none(self._get_trial(trial_index=trial_index).arm).parameters
            == parameterization
        )

    # ------------------ JSON serialization & storage methods. -----------------

    def save_to_json_file(self, filepath: str = "ax_client_snapshot.json") -> None:
        """Save a JSON-serialized snapshot of this `AxClient`'s settings and state
        to a .json file by the given path.
        """
        with open(filepath, "w+") as file:  # pragma: no cover
            file.write(json.dumps(self.to_json_snapshot()))
            logger.info(f"Saved JSON-serialized state of optimization to `{filepath}`.")

    @staticmethod
    def load_from_json_file(
        filepath: str = "ax_client_snapshot.json", **kwargs
    ) -> "AxClient":
        """Restore an `AxClient` and its state from a JSON-serialized snapshot,
        residing in a .json file by the given path.
        """
        with open(filepath, "r") as file:  # pragma: no cover
            serialized = json.loads(file.read())
            return AxClient.from_json_snapshot(serialized=serialized, **kwargs)

    def to_json_snapshot(self) -> Dict[str, Any]:
        """Serialize this `AxClient` to JSON to be able to interrupt and restart
        optimization and save it to file by the provided path.

        Returns:
            A JSON-safe dict representation of this `AxClient`.
        """
        return {
            "_type": self.__class__.__name__,
            "experiment": object_to_json(self._experiment),
            "generation_strategy": object_to_json(self._generation_strategy),
            "_enforce_sequential_optimization": self._enforce_sequential_optimization,
        }

    @staticmethod
    def from_json_snapshot(serialized: Dict[str, Any], **kwargs) -> "AxClient":
        """Recreate an `AxClient` from a JSON snapshot."""
        experiment = object_from_json(serialized.pop("experiment"))
        serialized_generation_strategy = serialized.pop("generation_strategy")
        ax_client = AxClient(
            generation_strategy=generation_strategy_from_json(
                generation_strategy_json=serialized_generation_strategy
            )
            if serialized_generation_strategy is not None
            else None,
            enforce_sequential_optimization=serialized.pop(
                "_enforce_sequential_optimization"
            ),
            **kwargs,
        )
        ax_client._experiment = experiment
        return ax_client

    # ---------------------- Private helper methods. ---------------------

    @property
    def experiment(self) -> Experiment:
        """Returns the experiment set on this Ax client."""
        if self._experiment is None:
            raise ValueError(
                "Experiment not set on Ax client. Must first "
                "call load_experiment or create_experiment to use handler functions."
            )
        return not_none(self._experiment)

    @property
    def generation_strategy(self) -> GenerationStrategy:
        """Returns the generation strategy, set on this experiment."""
        if self._generation_strategy is None:
            raise ValueError(
                "No generation strategy has been set on this optimization yet."
            )
        return not_none(self._generation_strategy)

    @property
    def objective_name(self) -> str:
        """Returns the name of the objective in this optimization."""
        opt_config = not_none(self.experiment.optimization_config)
        return opt_config.objective.metric.name

    def _set_generation_strategy(
        self, choose_generation_strategy_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Selects the generation strategy and applies specified dispatch kwargs,
        if any.
        """
        choose_generation_strategy_kwargs = choose_generation_strategy_kwargs or {}
        random_seed = choose_generation_strategy_kwargs.pop(
            "random_seed", self._random_seed
        )
        enforce_sequential_optimization = choose_generation_strategy_kwargs.pop(
            "enforce_sequential_optimization", self._enforce_sequential_optimization
        )
        if self._generation_strategy is None:
            self._generation_strategy = choose_generation_strategy(
                search_space=self.experiment.search_space,
                enforce_sequential_optimization=enforce_sequential_optimization,
                random_seed=random_seed,
                **choose_generation_strategy_kwargs,
            )

    def _gen_new_generator_run(self, n: int = 1) -> GeneratorRun:
        """Generate new generator run for this experiment.

        Args:
            n: Number of arms to generate.
        """
        # If random seed is not set for this optimization, context manager does
        # nothing; otherwise, it sets the random seed for torch, but only for the
        # scope of this call. This is important because torch seed is set globally,
        # so if we just set the seed without the context manager, it can have
        # serious negative impact on the performance of the models that employ
        # stochasticity.
        with manual_seed(seed=self._random_seed) and warnings.catch_warnings():
            # Filter out GPYTorch warnings to avoid confusing users.
            warnings.simplefilter("ignore")
            return not_none(self.generation_strategy).gen(
                experiment=self.experiment,
                n=n,
                pending_observations=get_pending_observation_features(
                    experiment=self.experiment
                ),
            )

    def _get_trial(self, trial_index: int) -> Trial:
        """Gets trial by given index or raises an error if it does not exist."""
        if trial_index in self.experiment.trials:
            trial = self.experiment.trials.get(trial_index)
            if not isinstance(trial, Trial):
                raise NotImplementedError(
                    "`AxClient` only supports `Trial`, not `BatchTrial`."
                )
            return trial
        raise ValueError(f"Trial {trial_index} does not yet exist.")

    def _find_last_trial_with_parameterization(
        self, parameterization: TParameterization
    ) -> int:
        """Given a parameterization, find the last trial in the experiment that
        contains an arm with that parameterization.
        """
        for trial_idx in sorted(self.experiment.trials.keys(), reverse=True):
            if not_none(self._get_trial(trial_idx).arm).parameters == parameterization:
                return trial_idx
        raise ValueError(
            f"No trial on experiment matches parameterization {parameterization}."
        )

    def _make_evaluations_and_data(
        self,
        trial: BaseTrial,
        raw_data: Union[TEvaluationOutcome, Dict[str, TEvaluationOutcome]],
        metadata: Optional[Dict[str, Union[str, int]]],
        sample_sizes: Optional[Dict[str, int]] = None,
    ) -> Tuple[Dict[str, TEvaluationOutcome], Data]:
        """Formats given raw data as Ax evaluations and `Data`.

        Args:
            trial: Trial within the experiment.
            raw_data: Metric outcomes for 1-arm trials, map from arm name to
                metric outcomes for batched trials.
            sample_size: Integer sample size for 1-arm trials, dict from arm
                name to sample size for batched trials. Optional.
            metadata: Additional metadata to track about this run.
            data_is_for_batched_trials: Whether making evaluations and data for
                a batched trial or a 1-arm trial.
        """
        if isinstance(trial, BatchTrial):
            assert isinstance(  # pragma: no cover
                raw_data, dict
            ), "Raw data must be a dict for batched trials."
        elif isinstance(trial, Trial):
            arm_name = not_none(trial.arm).name
            raw_data = {arm_name: raw_data}  # pyre-ignore[9]
        else:  # pragma: no cover
            raise ValueError(f"Unexpected trial type: {type(trial)}.")
        assert isinstance(raw_data, dict)
        evaluations = {
            arm_name: raw_data_to_evaluation(
                raw_data=raw_data[arm_name], objective_name=self.objective_name
            )
            for arm_name in raw_data
        }
        data = data_from_evaluations(
            evaluations=evaluations,
            trial_index=trial.index,
            sample_sizes=sample_sizes or {},
            start_time=(
                checked_cast_optional(int, metadata.get("start_time"))
                if metadata is not None
                else None
            ),
            end_time=(
                checked_cast_optional(int, metadata.get("end_time"))
                if metadata is not None
                else None
            ),
        )
        return evaluations, data

    # ------------------------------ Validators. -------------------------------

    @staticmethod
    def _validate_can_complete_trial(trial: BaseTrial) -> None:
        if trial.status.is_completed:
            raise ValueError(
                f"Trial {trial.index} has already been completed with data."
                "To add more data to it (for example, for a different metric), "
                "use `ax_client.update_trial_data`."
            )
        if trial.status.is_abandoned or trial.status.is_failed:
            raise ValueError(
                f"Trial {trial.index} has been marked {trial.status.name}, so it "
                "no longer expects data."
            )

    def _validate_search_space_membership(self, parameters: TParameterization) -> None:
        self.experiment.search_space.check_membership(
            parameterization=parameters, raise_error=True
        )
        # `check_membership` uses int and float interchangeably, which we don't
        # want here.
        for p_name, parameter in self.experiment.search_space.parameters.items():
            if not isinstance(parameters[p_name], parameter.python_type):
                typ = type(parameters[p_name])
                raise ValueError(
                    f"Value for parameter {p_name} is of type {typ}, expected "
                    f"{parameter.python_type}. If the intention was to have the "
                    f"parameter on experiment be of type {typ}, set `value_type` "
                    f"on experiment creation for {p_name}."
                )

    # -------- Backward-compatibility with old save / load method names. -------

    @staticmethod
    def get_recommended_max_parallelism() -> None:
        raise NotImplementedError(
            "Use `get_max_parallelism` instead; parallelism levels are now "
            "enforced in generation strategy, so max parallelism is no longer "
            "just recommended."
        )

    @staticmethod
    def load_experiment(experiment_name: str) -> None:
        raise NotImplementedError(
            "Use `load_experiment_from_database` to load from SQL database or "
            "`load_from_json_file` to load optimization state from .json file."
        )

    @staticmethod
    def load(filepath: Optional[str] = None) -> None:
        raise NotImplementedError(
            "Use `load_experiment_from_database` to load from SQL database or "
            "`load_from_json_file` to load optimization state from .json file."
        )

    @staticmethod
    def save(filepath: Optional[str] = None) -> None:
        raise NotImplementedError(
            "Use `save_to_json_file` to save optimization state to .json file."
        )
