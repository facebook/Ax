#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import logging
import warnings
from collections.abc import Callable, Iterable, Sequence
from functools import partial

from logging import Logger
from typing import Any, Optional, TypeVar

import ax.service.utils.early_stopping as early_stopping_utils
import numpy as np
import pandas as pd
import torch
from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial, TrialStatus
from ax.core.experiment import DataType, Experiment
from ax.core.formatting_utils import data_and_evaluations_from_raw_data
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.core.generator_run import GeneratorRun
from ax.core.map_data import MapData
from ax.core.map_metric import MapMetric
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.objective import MultiObjective, Objective
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.runner import Runner
from ax.core.trial import Trial
from ax.core.types import (
    TEvaluationOutcome,
    TModelPredictArm,
    TParameterization,
    TParamValue,
)

from ax.core.utils import get_pending_observation_features_based_on_trial_status
from ax.early_stopping.strategies import BaseEarlyStoppingStrategy
from ax.early_stopping.utils import estimate_early_stopping_savings
from ax.exceptions.constants import CHOLESKY_ERROR_ANNOTATION
from ax.exceptions.core import (
    DataRequiredError,
    OptimizationComplete,
    OptimizationShouldStop,
    UnsupportedError,
    UnsupportedPlotError,
    UserInputError,
)
from ax.exceptions.generation_strategy import MaxParallelismReachedException
from ax.global_stopping.strategies.base import BaseGlobalStoppingStrategy
from ax.global_stopping.strategies.improvement import constraint_satisfaction
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.prediction_utils import predict_by_features
from ax.plot.base import AxPlotConfig
from ax.plot.contour import plot_contour
from ax.plot.feature_importances import plot_feature_importance_by_feature
from ax.plot.helper import _format_dict
from ax.plot.trace import optimization_trace_single_method
from ax.service.utils.analysis_base import AnalysisBase
from ax.service.utils.best_point_mixin import BestPointMixin
from ax.service.utils.instantiation import (
    FixedFeatures,
    InstantiationBase,
    ObjectiveProperties,
)
from ax.service.utils.with_db_settings_base import DBSettings
from ax.storage.json_store.decoder import (
    generation_strategy_from_json,
    object_from_json,
)
from ax.storage.json_store.encoder import object_to_json
from ax.storage.json_store.registry import (
    CORE_CLASS_DECODER_REGISTRY,
    CORE_CLASS_ENCODER_REGISTRY,
    CORE_DECODER_REGISTRY,
    CORE_ENCODER_REGISTRY,
    TDecoderRegistry,
)
from ax.utils.common.docutils import copy_doc
from ax.utils.common.executils import retry_on_exception
from ax.utils.common.logger import _round_floats_for_logging, get_logger
from ax.utils.common.random import with_rng_seed
from pyre_extensions import assert_is_instance, none_throws


logger: Logger = get_logger(__name__)


AxClientSubclass = TypeVar("AxClientSubclass", bound="AxClient")

ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES: int = 6

# pyre-fixme[5]: Global expression must be annotated.
round_floats_for_logging = partial(
    _round_floats_for_logging,
    decimal_places=ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES,
)


class AxClient(AnalysisBase, BestPointMixin, InstantiationBase):
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

        torch_device: An optional `torch.device` object, used to choose the device
            used for generating new points for trials. Works only for torch-based
            models, such as MBM. Ignored if a `generation_strategy` is passed in
            manually. To specify the device for a custom `generation_strategy`,
            pass in `torch_device` as part of `model_kwargs`. See
            https://ax.dev/tutorials/generation_strategy.html for a tutorial on
            generation strategies.

        verbose_logging: Whether Ax should log significant optimization events,
            defaults to `True`.

        suppress_storage_errors: Whether to suppress SQL storage-related errors if
            encountered. Only use if SQL storage is not important for the given use
            case, since this will only log, but not raise, an exception if its
            encountered while saving to DB or loading from it.

        early_stopping_strategy: A ``BaseEarlyStoppingStrategy`` that determines
            whether a trial should be stopped given the current state of
            the experiment. Used in ``should_stop_trials_early``.

        global_stopping_strategy: A ``BaseGlobalStoppingStrategy`` that determines
            whether the full optimization should be stopped or not.
    """

    _experiment: Experiment | None = None

    def __init__(
        self,
        generation_strategy: GenerationStrategy | None = None,
        db_settings: Optional[DBSettings] = None,
        enforce_sequential_optimization: bool = True,
        random_seed: int | None = None,
        torch_device: torch.device | None = None,
        verbose_logging: bool = True,
        suppress_storage_errors: bool = False,
        early_stopping_strategy: BaseEarlyStoppingStrategy | None = None,
        global_stopping_strategy: BaseGlobalStoppingStrategy | None = None,
    ) -> None:
        super().__init__(
            db_settings=db_settings,
            suppress_all_errors=suppress_storage_errors,
        )

        if not verbose_logging:
            logger.setLevel(logging.WARNING)
        else:
            logger.info(
                "Starting optimization with verbose logging. To disable logging, "
                "set the `verbose_logging` argument to `False`. Note that float "
                "values in the logs are rounded to "
                f"{ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES} decimal points."
            )
        if generation_strategy is not None and torch_device is not None:
            warnings.warn(
                "Both a `generation_strategy` and a `torch_device` were specified. "
                "`torch_device` will be ignored. Instead, specify `torch_device` "
                "by passing it in `model_kwargs` while creating the "
                "`generation_strategy`.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._generation_strategy = generation_strategy
        self._enforce_sequential_optimization = enforce_sequential_optimization
        self._random_seed = random_seed
        self._torch_device = torch_device
        self._suppress_storage_errors = suppress_storage_errors
        self._early_stopping_strategy = early_stopping_strategy
        self._global_stopping_strategy = global_stopping_strategy
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
        parameters: list[
            dict[str, TParamValue | Sequence[TParamValue] | dict[str, list[str]]]
        ],
        name: str | None = None,
        description: str | None = None,
        owners: list[str] | None = None,
        objectives: dict[str, ObjectiveProperties] | None = None,
        parameter_constraints: list[str] | None = None,
        outcome_constraints: list[str] | None = None,
        status_quo: TParameterization | None = None,
        overwrite_existing_experiment: bool = False,
        experiment_type: str | None = None,
        tracking_metric_names: list[str] | None = None,
        choose_generation_strategy_kwargs: dict[str, Any] | None = None,
        support_intermediate_data: bool = False,
        immutable_search_space_and_opt_config: bool = True,
        is_test: bool = False,
        metric_definitions: dict[str, dict[str, Any]] | None = None,
        default_trial_type: str | None = None,
        default_runner: Runner | None = None,
    ) -> None:
        """Create a new experiment and save it if DBSettings available.

        Args:
            parameters: List of dictionaries representing parameters in the
                experiment search space.
                Required elements in the dictionaries are:
                1. "name" (name of parameter, string),
                2. "type" (type of parameter: "range", "fixed", or "choice", string),
                and one of the following:
                3a. "bounds" for range parameters (list of two values, lower bound
                first),
                3b. "values" for choice parameters (list of values), or
                3c. "value" for fixed parameters (single value).
                Optional elements are:
                1. "log_scale" (for float-valued range parameters, bool),
                2. "value_type" (to specify type that values of this parameter should
                take; expects "float", "int", "bool" or "str"),
                3. "is_fidelity" (bool) and "target_value" (float) for fidelity
                parameters,
                4. "is_ordered" (bool) for choice parameters, and
                5. "is_task" (bool) for task parameters.
                6. "digits" (int) for float-valued range parameters.
            name: Name of the experiment to be created.
            description: Description of the experiment to be created.
            objectives: Mapping from an objective name to object containing:
                minimize: Whether this experiment represents a minimization problem.
                threshold: The bound in the objective's threshold constraint.
            parameter_constraints: List of string representation of parameter
                constraints, such as "x3 >= x4" or "-x3 + 2*x4 - 3.5*x5 >= 2". For
                the latter constraints, any number of arguments is accepted, and
                acceptable operators are "<=" and ">=". Note that parameter
                constraints may only be placed on range parameters, not choice
                parameters or fixed parameters.
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
            tracking_metric_names: Names of additional tracking metrics not used for
                optimization.
            choose_generation_strategy_kwargs: Keyword arguments to pass to
                `choose_generation_strategy` function which determines what
                generation strategy should be used when none was specified on init.
            support_intermediate_data: Whether trials may report intermediate results
                for trials that are still running (i.e. have not been completed via
                `ax_client.complete_trial`).
            immutable_search_space_and_opt_config: Whether it's possible to update the
                search space and optimization config on this experiment after creation.
                Defaults to True. If set to True, we won't store or load copies of the
                search space and optimization config on each generator run, which will
                improve storage performance.
            is_test: Whether this experiment will be a test experiment (useful for
                marking test experiments in storage etc). Defaults to False.
            metric_definitions: A mapping of metric names to extra kwargs to pass
                to that metric. Note these are modified in-place. Each
                Metric must have its own dictionary (metrics cannot share a
                single dictionary object).
            default_trial_type: The default trial type if multiple
                trial types are intended to be used in the experiment.  If specified,
                a MultiTypeExperiment will be created. Otherwise, a single-type
                Experiment will be created.
            default_runner: The default runner in this experiment.
                This applies to MultiTypeExperiment (when default_trial_type
                is specified) and needs to be specified together with
                default_trial_type. This will be ignored for single-type Experiment
                (when default_trial_type is not specified).
        """
        self._validate_early_stopping_strategy(support_intermediate_data)

        objective_kwargs = {}
        if objectives is not None:
            objective_kwargs["objectives"] = {
                objective: ("minimize" if properties.minimize else "maximize")
                for objective, properties in objectives.items()
            }
            if len(objectives.keys()) > 1:
                objective_kwargs["objective_thresholds"] = (
                    self.build_objective_thresholds(objectives)
                )

        experiment = self.make_experiment(
            name=name,
            description=description,
            owners=owners,
            parameters=parameters,
            parameter_constraints=parameter_constraints,
            outcome_constraints=outcome_constraints,
            status_quo=status_quo,
            experiment_type=experiment_type,
            tracking_metric_names=tracking_metric_names,
            metric_definitions=metric_definitions,
            support_intermediate_data=support_intermediate_data,
            immutable_search_space_and_opt_config=immutable_search_space_and_opt_config,
            is_test=is_test,
            default_trial_type=default_trial_type,
            default_runner=default_runner,
            **objective_kwargs,
        )
        self._set_runner(experiment=experiment)
        self._set_experiment(
            experiment=experiment,
            overwrite_existing_experiment=overwrite_existing_experiment,
        )
        self._set_generation_strategy(
            choose_generation_strategy_kwargs=choose_generation_strategy_kwargs
        )
        self._save_generation_strategy_to_db_if_possible()

    @property
    def status_quo(self) -> TParameterization | None:
        """The parameterization of the status quo arm of the experiment."""
        if self.experiment.status_quo:
            return self.experiment.status_quo.parameters
        return None

    def set_status_quo(self, params: TParameterization | None) -> None:
        """Set, or unset status quo on the experiment.  There may be risk
        in using this after a trial with the status quo arm has run.

        Args:
            status_quo: Parameterization of the current state of the system.
                If set, this will be added to each trial to be evaluated alongside
                test configurations.
        """
        self.experiment.status_quo = None if params is None else Arm(parameters=params)

    def set_optimization_config(
        self,
        objectives: dict[str, ObjectiveProperties] | None = None,
        outcome_constraints: list[str] | None = None,
        metric_definitions: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Overwrite experiment's optimization config

        Args:
            objectives: Mapping from an objective name to object containing:
                minimize: Whether this experiment represents a minimization problem.
                threshold: The bound in the objective's threshold constraint.
            outcome_constraints: List of string representation of outcome
                constraints of form "metric_name >= bound", like "m1 <= 3."
            metric_definitions: A mapping of metric names to extra kwargs to pass
                to that metric
        """
        metric_definitions = (
            self.metric_definitions
            if metric_definitions is None
            else metric_definitions
        )
        optimization_config = self.make_optimization_config_from_properties(
            objectives=objectives,
            outcome_constraints=outcome_constraints,
            status_quo_defined=self.experiment.status_quo is not None,
            metric_definitions=metric_definitions,
        )
        if optimization_config:
            self.experiment.optimization_config = optimization_config
            self._save_experiment_to_db_if_possible(
                experiment=self.experiment,
            )
        else:
            raise ValueError(
                "optimization config not set because it was missing objectives"
            )

    def add_tracking_metrics(
        self,
        metric_names: list[str],
        metric_definitions: dict[str, dict[str, Any]] | None = None,
        metrics_to_trial_types: dict[str, str] | None = None,
        canonical_names: dict[str, str] | None = None,
    ) -> None:
        """Add a list of new metrics to the experiment.

        If any of the metrics are already defined on the experiment,
        we raise an error and don't add any of them to the experiment

        Args:
            metric_names: Names of metrics to be added.
            metric_definitions: A mapping of metric names to extra kwargs to pass
                to that metric. Note these are modified in-place. Each
                Metric must have its is own dictionary (metrics cannot share a
                single dictionary object).
            metrics_to_trial_types: Only applicable to MultiTypeExperiment.
                The mapping from metric names to corresponding
                trial types for each metric. If provided, the metrics will be
                added with their respective trial types. If not provided, then the
                default trial type will be used.
            canonical_names: A mapping from metric name (of a particular trial type)
                to the metric name of the default trial type. Only applicable to
                MultiTypeExperiment.
        """
        metric_definitions = (
            self.metric_definitions
            if metric_definitions is None
            else metric_definitions
        )
        metric_objects = [
            self._make_metric(name=metric_name, metric_definitions=metric_definitions)
            for metric_name in metric_names
        ]

        if isinstance(self.experiment, MultiTypeExperiment):
            experiment = assert_is_instance(self.experiment, MultiTypeExperiment)
            experiment.add_tracking_metrics(
                metrics=metric_objects,
                metrics_to_trial_types=metrics_to_trial_types,
                canonical_names=canonical_names,
            )
        else:
            self.experiment.add_tracking_metrics(metrics=metric_objects)

    @copy_doc(Experiment.remove_tracking_metric)
    def remove_tracking_metric(self, metric_name: str) -> None:
        self.experiment.remove_tracking_metric(metric_name=metric_name)

    def set_search_space(
        self,
        parameters: list[
            dict[str, TParamValue | Sequence[TParamValue] | dict[str, list[str]]]
        ],
        parameter_constraints: list[str] | None = None,
    ) -> None:
        """Sets the search space on the experiment and saves.
        This is expected to fail on base AxClient as experiment will have
        immutable search space and optimization config set to True by default

        Args:
            parameters: List of dictionaries representing parameters in the
                experiment search space.
                Required elements in the dictionaries are:
                1. "name" (name of parameter, string),
                2. "type" (type of parameter: "range", "fixed", or "choice", string),
                and one of the following:
                3a. "bounds" for range parameters (list of two values, lower bound
                first),
                3b. "values" for choice parameters (list of values), or
                3c. "value" for fixed parameters (single value).
                Optional elements are:
                1. "log_scale" (for float-valued range parameters, bool),
                2. "value_type" (to specify type that values of this parameter should
                take; expects "float", "int", "bool" or "str"),
                3. "is_fidelity" (bool) and "target_value" (float) for fidelity
                parameters,
                4. "is_ordered" (bool) for choice parameters, and
                5. "is_task" (bool) for task parameters.
                6. "digits" (int) for float-valued range parameters.
            parameter_constraints: List of string representation of parameter
                constraints, such as "x3 >= x4" or "-x3 + 2*x4 - 3.5*x5 >= 2". For
                the latter constraints, any number of arguments is accepted, and
                acceptable operators are "<=" and ">=". Note that parameter
                constraints may only be placed on range parameters, not choice
                parameters or fixed parameters.
        """
        self.experiment.search_space = self.make_search_space(
            parameters=parameters, parameter_constraints=parameter_constraints
        )
        self._save_experiment_to_db_if_possible(
            experiment=self.experiment,
        )

    @retry_on_exception(
        logger=logger,
        exception_types=(RuntimeError,),
        check_message_contains=["Cholesky", "cholesky"],
        suppress_all_errors=False,
        wrap_error_message_in=CHOLESKY_ERROR_ANNOTATION,
    )
    def get_next_trial(
        self,
        ttl_seconds: int | None = None,
        force: bool = False,
        fixed_features: FixedFeatures | None = None,
    ) -> tuple[TParameterization, int]:
        """
        Generate trial with the next set of parameters to try in the iteration process.

        Note: Service API currently supports only 1-arm trials.

        Args:
            ttl_seconds: If specified, will consider the trial failed after this
                many seconds. Used to detect dead trials that were not marked
                failed properly.
            force: If set to True, this function will bypass the global stopping
                strategy's decision and generate a new trial anyway.
            fixed_features: A FixedFeatures object containing any
                features that should be fixed at specified values during
                generation.

        Returns:
            Tuple of trial parameterization, trial index
        """

        # Check if the global stopping strategy suggests to stop the optimization.
        # This is needed only if there is actually a stopping strategy specified,
        # and if this function is not forced to generate a new trial.
        if self.global_stopping_strategy and (not force):
            # The strategy itself will check if enough trials have already been
            # completed.
            (
                stop_optimization,
                global_stopping_message,
            ) = self.global_stopping_strategy.should_stop_optimization(
                experiment=self.experiment
            )
            if stop_optimization:
                raise OptimizationShouldStop(message=global_stopping_message)

        try:
            trial = self.experiment.new_trial(
                generator_run=self._gen_new_generator_run(
                    fixed_features=fixed_features
                ),
                ttl_seconds=ttl_seconds,
            )
        except MaxParallelismReachedException as e:
            if self._early_stopping_strategy is not None:
                e.message += (  # noqa: B306
                    " When stopping trials early, make sure to call `stop_trial_early` "
                    "on the stopped trial."
                )
            raise e
        logger.info(
            f"Generated new trial {trial.index} with parameters "
            f"{round_floats_for_logging(item=none_throws(trial.arm).parameters)} "
            f"using model {none_throws(trial.generator_run)._model_key}."
        )
        trial.mark_running(no_runner_required=True)
        self._save_or_update_trial_in_db_if_possible(
            experiment=self.experiment, trial=trial
        )
        # TODO[T79183560]: Ensure correct handling of generator run when using
        # foreign keys.
        self._update_generation_strategy_in_db_if_possible(
            generation_strategy=self.generation_strategy,
            new_generator_runs=[self.generation_strategy._generator_runs[-1]],
        )
        return none_throws(trial.arm).parameters, trial.index

    def get_current_trial_generation_limit(self) -> tuple[int, bool]:
        """How many trials this ``AxClient`` instance can currently produce via
        calls to ``get_next_trial``, before more trials are completed, and whether
        the optimization is complete.

        NOTE: If return value of this function is ``(0, False)``, no more trials
        can currently be procuded by this ``AxClient`` instance, but optimization
        is not completed; once more trials are completed with data, more new
        trials can be generated.

        Returns: a two-item tuple of:
              - the number of trials that can currently be produced, with -1
                meaning unlimited trials,
              - whether no more trials can be produced by this ``AxClient``
                instance at any point (e.g. if the search space is exhausted or
                generation strategy is completed.
        """
        # Ensure that experiment is set on the generation strategy.
        if self.generation_strategy._experiment is None:
            self.generation_strategy.experiment = self.experiment

        return self.generation_strategy.current_generator_run_limit()

    def get_next_trials(
        self,
        max_trials: int,
        ttl_seconds: int | None = None,
        fixed_features: FixedFeatures | None = None,
    ) -> tuple[dict[int, TParameterization], bool]:
        """Generate as many trials as currently possible.

        NOTE: Useful for running multiple trials in parallel: produces multiple trials,
        with their number limited by:
          - parallelism limit on current generation step,
          - number of trials in current generation step,
          - number of trials required to complete before moving to next generation step,
            if applicable,
          - and ``max_trials`` argument to this method.

        Args:
            max_trials: Limit on how many trials the call to this method should produce.
            ttl_seconds: If specified, will consider the trial failed after this
                many seconds. Used to detect dead trials that were not marked
                failed properly.
            fixed_features: A FixedFeatures object containing any
                features that should be fixed at specified values during
                generation.

        Returns: two-item tuple of:
              - mapping from trial indices to parameterizations in those trials,
              - boolean indicator of whether optimization is completed and no more
                trials can be generated going forward.
        """
        gen_limit, optimization_complete = self.get_current_trial_generation_limit()
        if optimization_complete:
            return {}, True

        # Trial generation limit of -1 indicates that unlimited trials can be
        # generated, so we only want to limit `max_trials` if `trial_generation_
        # limit` is non-negative.
        if gen_limit >= 0:
            max_trials = min(gen_limit, max_trials)

        trials_dict = {}
        for _ in range(max_trials):
            try:
                params, trial_index = self.get_next_trial(
                    ttl_seconds=ttl_seconds, fixed_features=fixed_features
                )
                trials_dict[trial_index] = params
            except OptimizationComplete as err:
                logger.info(
                    f"Encountered exception indicating optimization completion: {err}"
                )
                return trials_dict, True

        # Check whether optimization is complete now that we generated a batch
        # of trials.
        _, optimization_complete = self.get_current_trial_generation_limit()
        return trials_dict, optimization_complete

    def abandon_trial(self, trial_index: int, reason: str | None = None) -> None:
        """Abandons a trial and adds optional metadata to it.

        Args:
            trial_index: Index of trial within the experiment.
        """
        trial = self.get_trial(trial_index)
        trial.mark_abandoned(reason=reason)

    def update_running_trial_with_intermediate_data(
        self,
        trial_index: int,
        raw_data: TEvaluationOutcome,
        metadata: dict[str, str | int] | None = None,
        sample_size: int | None = None,
    ) -> None:
        """
        Updates the trial with given metric values without completing it. Also
        adds optional metadata to it. Useful for intermediate results like
        the metrics of a partially optimized machine learning model. In these
        cases it should be called instead of `complete_trial` until it is
        time to complete the trial.

        NOTE: This method will raise an Exception if it is called multiple times
        with the same ``raw_data``. These cases typically arise when ``raw_data``
        does not change over time. To avoid this, pass a timestep metric in
        ``raw_data``, for example:

        .. code-block:: python

            for ts in range(100):
                raw_data = [({"ts": ts}, {"my_objective": (1.0, 0.0)})]
                ax_client.update_running_trial_with_intermediate_data(
                    trial_index=0, raw_data=raw_data
                )

        NOTE: When ``raw_data`` does not specify SEM for a given metric, Ax
        will default to the assumption that the data is noisy (specifically,
        corrupted by additive zero-mean Gaussian noise) and that the
        level of noise should be inferred by the optimization model. To
        indicate that the data is noiseless, set SEM to 0.0, for example:

        .. code-block:: python

            ax_client.update_running_trial_with_intermediate_data(
                trial_index=0,
                raw_data={"my_objective": (objective_mean_value, 0.0)}
            )

        Args:
            trial_index: Index of trial within the experiment.
            raw_data: Evaluation data for the trial. Can be a mapping from
                metric name to a tuple of mean and SEM, just a tuple of mean and
                SEM if only one metric in optimization, or just the mean if SEM is
                unknown (then Ax will infer observation noise level).
                Can also be a list of (fidelities, mapping from
                metric name to a tuple of mean and SEM).
            metadata: Additional metadata to track about this run.
            sample_size: Number of samples collected for the underlying arm,
                optional.
        """
        if not isinstance(trial_index, int):
            raise ValueError(f"Trial index must be an int, got: {trial_index}.")
        if not self.experiment.default_data_type == DataType.MAP_DATA:
            raise ValueError(
                "`update_running_trial_with_intermediate_data` requires that "
                "this client's `experiment` be constructed with "
                "`support_intermediate_data=True` and have `default_data_type` of "
                "`DataType.MAP_DATA`."
            )
        data_update_repr = self._update_trial_with_raw_data(
            trial_index=trial_index,
            raw_data=raw_data,
            metadata=metadata,
            sample_size=sample_size,
            combine_with_last_data=True,
        )
        logger.info(f"Updated trial {trial_index} with data: " f"{data_update_repr}.")

    def complete_trial(
        self,
        trial_index: int,
        raw_data: TEvaluationOutcome,
        metadata: dict[str, str | int] | None = None,
        sample_size: int | None = None,
    ) -> None:
        """
        Completes the trial with given metric values and adds optional metadata
        to it.

        NOTE: When ``raw_data`` does not specify SEM for a given metric, Ax
        will default to the assumption that the data is noisy (specifically,
        corrupted by additive zero-mean Gaussian noise) and that the
        level of noise should be inferred by the optimization model. To
        indicate that the data is noiseless, set SEM to 0.0, for example:

        .. code-block:: python

          ax_client.complete_trial(
              trial_index=0,
              raw_data={"my_objective": (objective_mean_value, 0.0)}
          )

        Args:
            trial_index: Index of trial within the experiment.
            raw_data: Evaluation data for the trial. Can be a mapping from
                metric name to a tuple of mean and SEM, just a tuple of mean and
                SEM if only one metric in optimization, or just the mean if SEM is
                unknown (then Ax will infer observation noise level).
                Can also be a list of (fidelities, mapping from
                metric name to a tuple of mean and SEM).
            metadata: Additional metadata to track about this run.
            sample_size: Number of samples collected for the underlying arm,
                optional.
        """
        # Validate that trial can be completed.
        trial = self.get_trial(trial_index)
        trial._validate_can_attach_data()
        if not isinstance(trial_index, int):
            raise ValueError(f"Trial index must be an int, got: {trial_index}.")
        data_update_repr = self._update_trial_with_raw_data(
            trial_index=trial_index,
            raw_data=raw_data,
            metadata=metadata,
            sample_size=sample_size,
            complete_trial=True,
            combine_with_last_data=True,
        )
        logger.info(f"Completed trial {trial_index} with data: " f"{data_update_repr}.")

    def update_trial_data(
        self,
        trial_index: int,
        raw_data: TEvaluationOutcome,
        metadata: dict[str, str | int] | None = None,
        sample_size: int | None = None,
    ) -> None:
        """
        Attaches additional data or updates the existing data for a trial in a
        terminal state. For example, if trial was completed with data for only
        one of the required metrics, this can be used to attach data for the
        remaining metrics.

        NOTE: This does not change the trial status.

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
        if not isinstance(trial_index, int):
            raise ValueError(f"Trial index must be an int, got: {trial_index}.")
        trial = self.get_trial(trial_index)
        if not trial.status.is_terminal:
            raise ValueError(
                f"Trial {trial.index} is not in a terminal state. Use "
                "`ax_client.complete_trial` to complete the trial with new data "
                "or use `ax_client.update_running_trial_with_intermediate_data` "
                "to attach intermediate data to a running trial."
            )
        data_update_repr = self._update_trial_with_raw_data(
            trial_index=trial_index,
            raw_data=raw_data,
            metadata=metadata,
            sample_size=sample_size,
            combine_with_last_data=True,
        )
        logger.info(f"Added data: {data_update_repr} to trial {trial.index}.")

    def log_trial_failure(
        self, trial_index: int, metadata: dict[str, str] | None = None
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
        )

    def attach_trial(
        self,
        parameters: TParameterization,
        ttl_seconds: int | None = None,
        run_metadata: dict[str, Any] | None = None,
        arm_name: str | None = None,
    ) -> tuple[TParameterization, int]:
        """Attach a new trial with the given parameterization to the experiment.

        Args:
            parameters: Parameterization of the new trial.
            ttl_seconds: If specified, will consider the trial failed after this
                many seconds. Used to detect dead trials that were not marked
                failed properly.

        Returns:
            Tuple of parameterization and trial index from newly created trial.
        """

        output_parameters, trial_index = self.experiment.attach_trial(
            parameterizations=[parameters],
            arm_names=[arm_name] if arm_name else None,
            ttl_seconds=ttl_seconds,
            run_metadata=run_metadata,
        )
        self._save_or_update_trial_in_db_if_possible(
            experiment=self.experiment,
            trial=self.experiment.trials[trial_index],
        )

        return list(output_parameters.values())[0], trial_index

    def get_trial_parameters(self, trial_index: int) -> TParameterization:
        """Retrieve the parameterization of the trial by the given index."""
        return none_throws(self.get_trial(trial_index).arm).parameters

    def get_trials_data_frame(self) -> pd.DataFrame:
        """Get a Pandas DataFrame representation of this experiment. The columns
        will include all the parameters in the search space and all the metrics
        on this experiment. The rows will each correspond to a trial (if using
        one-arm trials, which is the case in base ``AxClient``; will correspond
        to arms in trials in the batch-trial case).
        """
        return self.experiment.to_df()

    def get_max_parallelism(self) -> list[tuple[int, int]]:
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
        self, objective_optimum: float | None = None
    ) -> AxPlotConfig:
        """Retrieves the plot configuration for optimization trace, which shows
        the evolution of the objective mean over iterations.

        Args:
            objective_optimum: Optimal objective, if known, for display in the
                visualization.
        """
        if not self.experiment.trials:
            raise ValueError("Cannot generate plot as there are no trials.")

        objective = self.objective
        if isinstance(objective, MultiObjective):
            raise UnsupportedError(
                "`get_optimization_trace` is not supported "
                "for multi-objective experiments"
            )

        # Setting the objective values of infeasible points to be infinitely
        # bad prevents them from increasing or decreasing the
        # optimization trace.
        def _constrained_trial_objective_mean(trial: BaseTrial) -> float:
            if constraint_satisfaction(trial):
                return assert_is_instance(trial, Trial).objective_mean
            return float("inf") if self.objective.minimize else float("-inf")

        objective_name = self.objective_name
        best_objectives = np.array(
            [
                [
                    _constrained_trial_objective_mean(trial)
                    for trial in self.experiment.trials.values()
                    if trial.status.is_completed
                ]
            ]
        )
        hover_labels = [
            _format_dict(none_throws(assert_is_instance(trial, Trial).arm).parameters)
            for trial in self.experiment.trials.values()
            if trial.status.is_completed
        ]
        return optimization_trace_single_method(
            y=(
                np.minimum.accumulate(best_objectives, axis=1)
                if objective.minimize
                else np.maximum.accumulate(best_objectives, axis=1)
            ),
            optimum=objective_optimum,
            title="Best objective found vs. # of iterations",
            ylabel=objective_name.capitalize(),
            hover_labels=hover_labels,
        )

    def get_contour_plot(
        self,
        param_x: str | None = None,
        param_y: str | None = None,
        metric_name: str | None = None,
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

        if not metric_name:
            if isinstance(self.objective, MultiObjective):
                raise UnsupportedError(
                    "`get_contour_plot` requires a `metric_name` "
                    "for multi-objective experiments"
                )

            metric_name = self.objective_name

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
                    "Remaining parameters are affixed to the middle of their range."
                )
                return plot_contour(
                    model=none_throws(self.generation_strategy.model),
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
        raise UnsupportedPlotError(
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
                    "this plot. Only certain models, implement feature importances."
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
        choose_generation_strategy_kwargs: dict[str, Any] | None = None,
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
        self._experiment = none_throws(
            experiment, f"Experiment by name '{experiment_name}' not found."
        )
        logger.info(f"Loaded {experiment}.")
        if generation_strategy is None:
            if choose_generation_strategy_kwargs is None:
                raise UserInputError(
                    f"No generation strategy was found for {experiment}. Please "
                    "pass `choose_generation_strategy_kwargs` to load it with one."
                )
            self._set_generation_strategy(
                choose_generation_strategy_kwargs=choose_generation_strategy_kwargs
            )
            self._save_generation_strategy_to_db_if_possible()
        else:
            self._generation_strategy = generation_strategy
            logger.info(
                f"Using generation strategy associated with the loaded experiment:"
                f" {generation_strategy}."
            )

    def get_model_predictions_for_parameterizations(
        self,
        parameterizations: list[TParameterization],
        metric_names: list[str] | None = None,
    ) -> list[dict[str, tuple[float, float]]]:
        """Retrieve model-estimated means and covariances for all metrics
        for the provided parameterizations.

        Args:
            metric_names: Names of the metrics for which to predict.
                All metrics will be predicted if this argument is
                not specified.
            parameterizations: List of Parameterizations for which to predict.

        Returns:
            A list of predicted metric mean and SEM of form:
            List[Tuple[float, float]].
        """

        parameterization_dict = dict(enumerate(parameterizations))

        predictions_dict = self.get_model_predictions(
            metric_names=metric_names, parameterizations=parameterization_dict
        )

        predictions_array = [
            predictions_dict[i] for i, _ in enumerate(parameterizations)
        ]

        return predictions_array

    def get_model_predictions(
        self,
        metric_names: list[str] | None = None,
        include_out_of_sample: bool | None = True,
        parameterizations: dict[int, TParameterization] | None = None,
    ) -> dict[int, dict[str, tuple[float, float]]]:
        """Retrieve model-estimated means and covariances for all metrics.

        NOTE: This method currently only supports one-arm trials.

        Args:
            metric_names: Names of the metrics, for which to retrieve predictions.
                All metrics on experiment will be retrieved if this argument was
                not specified.
            include_out_of_sample: Defaults to True. Return predictions for
                out-of-sample (i.e. not yet completed trials) data in
                addition to in-sample (i.e. completed trials) data.
            parameterizations: Optional mapping from an int label to
                Parameterizations. When provided, predictions are performed *only*
                on these data points, no predictions from trial data is performed,
                and include_out_of_sample parameters is ignored.

        Returns:
            A mapping from trial index to a mapping of metric names to tuples
            of predicted metric mean and SEM, of form:
            { trial_index -> { metric_name: ( mean, SEM ) } }.
        """

        # Ensure there are metrics specified
        if metric_names is None and self.experiment.metrics is None:
            raise ValueError(
                "No metrics to retrieve specified on the experiment or as "
                "argument to `get_model_predictions`."
            )

        # Model update is normally tied to the GenerationStrategy.gen() call,
        # which is called from get_next_trial(). In order to ensure that predictions
        # can be performed without the need to call get_next_trial(), we update the
        # model with all attached data.
        self.fit_model()

        # Shared info for subsequent calls
        metric_names_to_predict = (
            set(metric_names)
            if metric_names is not None
            else set(none_throws(self.experiment.metrics).keys())
        )
        model = none_throws(
            self.generation_strategy.model, "No model has been instantiated yet."
        )

        # Construct a dictionary that maps from a label to an
        # ObservationFeature to predict.
        # - If returning trial predictions, the label is the trial index.
        # - If predictions are for user-provided parameterization, the label
        #   is provided in the input (also an int).
        label_to_feature_dict = {}

        # Predict on user-provided data
        if parameterizations is not None:
            logger.info(
                '"parameterizations" have been provided, only these data '
                "points will be predicted. No trial data prediction will be "
                "returned."
            )
            for label in parameterizations.keys():
                label_to_feature_dict[label] = ObservationFeatures(
                    parameters=parameterizations[label]
                )
        # Predict on associated trials
        else:
            trials_dict = self.experiment.trials
            for trial_index, trial in trials_dict.items():
                # filter trials based on input params and trial statuses
                if include_out_of_sample or trial.status.is_completed:
                    arms = trial.arms
                    if len(arms) > 1:
                        raise ValueError("Currently only 1-arm trials are supported.")
                    label_to_feature_dict[trial_index] = ObservationFeatures.from_arm(
                        arms[0]
                    )

        return predict_by_features(
            model=model,
            label_to_feature_dict=label_to_feature_dict,
            metric_names=metric_names_to_predict,
        )

    def fit_model(self) -> None:
        """Fit a model using data collected from the trials so far.

        This method will attempt to fit the same model that would be used for generating
        the next trial. The resulting model may be different from the model that was
        used to generate the last trial, if the generation node is ready to transition.

        This method rarely needs to be called by the user, because model-fitting is
        usually handled indirectly through ``AxClient.get_next_trial()``. This method
        instantiates a new model if none is yet available, which may be the case if
        no trials have been generated using a model-based method.

        NOTE: If the current generation node is not model-based, no model may be fit.
        """
        if not self.experiment.trial_indices_by_status[TrialStatus.COMPLETED]:
            raise DataRequiredError(
                "At least one trial must be completed with data to fit a model."
            )
        # Check if we should transition before generating the next candidate.
        self.generation_strategy._maybe_transition_to_next_node()
        self.generation_strategy._fit_current_model(data=None)

    def verify_trial_parameterization(
        self, trial_index: int, parameterization: TParameterization
    ) -> bool:
        """Whether the given parameterization matches that of the arm in the trial
        specified in the trial index.
        """
        return (
            none_throws(self.get_trial(trial_index).arm).parameters == parameterization
        )

    def should_stop_trials_early(
        self, trial_indices: set[int]
    ) -> dict[int, str | None]:
        """Evaluate whether to early-stop running trials.

        Args:
            trial_indices: Indices of trials to consider for early stopping.

        Returns:
            A dictionary mapping trial indices that should be early stopped to
            (optional) messages with the associated reason.
        """
        if self._early_stopping_strategy is None:
            logger.warning(
                "No early_stopping_strategy was passed to AxClient. "
                "Defaulting to never stopping any trials early."
            )
        return early_stopping_utils.should_stop_trials_early(
            early_stopping_strategy=self._early_stopping_strategy,
            trial_indices=trial_indices,
            experiment=self.experiment,
        )

    def stop_trial_early(self, trial_index: int) -> None:
        trial = self.get_trial(trial_index)
        trial.mark_early_stopped()
        logger.info(f"Early stopped trial {trial_index}.")
        self._save_or_update_trial_in_db_if_possible(
            experiment=self.experiment, trial=trial
        )

    def estimate_early_stopping_savings(self, map_key: str | None = None) -> float:
        """Estimate early stopping savings using progressions of the MapMetric present
        on the EarlyStoppingConfig as a proxy for resource usage.

        Args:
            map_key: The name of the map_key by which to estimate early stopping
                savings, usually steps. If none is specified use some arbitrary map_key
                in the experiment's MapData

        Returns:
            The estimated resource savings as a fraction of total resource usage (i.e.
            0.11 estimated savings indicates we would expect the experiment to have used
            11% more resources without early stopping present)
        """
        if self.experiment.default_data_constructor is not MapData:
            return 0

        strategy = self.early_stopping_strategy
        map_key = (
            map_key
            if map_key is not None
            else (
                assert_is_instance(
                    self.experiment.metrics[list(strategy.metric_names)[0]],
                    MapMetric,
                ).map_key_info.key
                if strategy is not None
                and strategy.metric_names is not None
                and len(list(strategy.metric_names)) > 0
                else None
            )
        )

        return estimate_early_stopping_savings(
            experiment=self.experiment,
            map_key=map_key,
        )

    # ------------------ JSON serialization & storage methods. -----------------

    def save_to_json_file(self, filepath: str = "ax_client_snapshot.json") -> None:
        """Save a JSON-serialized snapshot of this `AxClient`'s settings and state
        to a .json file by the given path.
        """
        with open(filepath, "w+") as file:
            file.write(json.dumps(self.to_json_snapshot()))
            logger.info(f"Saved JSON-serialized state of optimization to `{filepath}`.")

    @classmethod
    def load_from_json_file(
        cls: type[AxClientSubclass],
        filepath: str = "ax_client_snapshot.json",
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> AxClientSubclass:
        """Restore an `AxClient` and its state from a JSON-serialized snapshot,
        residing in a .json file by the given path.
        """
        with open(filepath) as file:
            serialized = json.loads(file.read())
            return cls.from_json_snapshot(serialized=serialized, **kwargs)

    def to_json_snapshot(
        self,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        encoder_registry: dict[type, Callable[[Any], dict[str, Any]]] | None = None,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type` to avoid runtime subscripting errors.
        class_encoder_registry: None
        | (dict[type, Callable[[Any], dict[str, Any]]]) = None,
    ) -> dict[str, Any]:
        """Serialize this `AxClient` to JSON to be able to interrupt and restart
        optimization and save it to file by the provided path.

        Returns:
            A JSON-safe dict representation of this `AxClient`.
        """
        if encoder_registry is None:
            encoder_registry = CORE_ENCODER_REGISTRY

        if class_encoder_registry is None:
            class_encoder_registry = CORE_CLASS_ENCODER_REGISTRY

        return {
            "_type": self.__class__.__name__,
            "experiment": object_to_json(
                self._experiment,
                encoder_registry=encoder_registry,
                class_encoder_registry=class_encoder_registry,
            ),
            "generation_strategy": object_to_json(
                self._generation_strategy,
                encoder_registry=encoder_registry,
                class_encoder_registry=class_encoder_registry,
            ),
            "_enforce_sequential_optimization": self._enforce_sequential_optimization,
        }

    @classmethod
    def from_json_snapshot(
        cls: type[AxClientSubclass],
        serialized: dict[str, Any],
        decoder_registry: TDecoderRegistry | None = None,
        # pyre-fixme[2]: Parameter annotation cannot contain `Any`.
        class_decoder_registry: None
        | (dict[str, Callable[[dict[str, Any]], Any]]) = None,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> AxClientSubclass:
        """Recreate an `AxClient` from a JSON snapshot."""
        if decoder_registry is None:
            decoder_registry = CORE_DECODER_REGISTRY

        if class_decoder_registry is None:
            class_decoder_registry = CORE_CLASS_DECODER_REGISTRY

        experiment = object_from_json(
            serialized.pop("experiment"),
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        )
        serialized_generation_strategy = serialized.pop("generation_strategy")
        ax_client = cls(
            generation_strategy=(
                generation_strategy_from_json(
                    generation_strategy_json=serialized_generation_strategy,
                    experiment=experiment,
                    decoder_registry=decoder_registry,
                    class_decoder_registry=class_decoder_registry,
                )
                if serialized_generation_strategy is not None
                else None
            ),
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
        return none_throws(
            self._experiment,
            (
                "Experiment not set on Ax client. Must first "
                "call load_experiment or create_experiment to use handler functions."
            ),
        )

    def get_trial(self, trial_index: int) -> Trial:
        """Return a trial on experiment cast as Trial"""
        return assert_is_instance(self.experiment.trials[trial_index], Trial)

    @property
    def generation_strategy(self) -> GenerationStrategy:
        """Returns the generation strategy, set on this experiment."""
        return none_throws(
            self._generation_strategy,
            "No generation strategy has been set on this optimization yet.",
        )

    @property
    def objective(self) -> Objective:
        return none_throws(self.experiment.optimization_config).objective

    @property
    def objective_name(self) -> str:
        """Returns the name of the objective in this optimization."""
        objective = self.objective
        if isinstance(objective, MultiObjective):
            raise UnsupportedError(
                "Multi-objective experiments contain multiple objectives"
            )
        return objective.metric.name

    @property
    def objective_names(self) -> list[str]:
        """Returns the name of the objective in this optimization."""
        objective = self.objective
        return [m.name for m in objective.metrics]

    @property
    def metric_definitions(self) -> dict[str, dict[str, Any]]:
        """Returns metric definitions for all experiment metrics that can
        be passed into functions requiring metric_definitions
        """
        return {
            m.serialize_init_args(m)["name"]: {
                "metric_class": m.__class__,
                **{k: v for k, v in m.serialize_init_args(m).items() if k != "name"},
            }
            for m in self.experiment.metrics.values()
        }

    @property
    def metric_names(self) -> set[str]:
        """Returns the names of all metrics on the attached experiment."""
        return set(self.experiment.metrics)

    @property
    def early_stopping_strategy(self) -> BaseEarlyStoppingStrategy | None:
        """The early stopping strategy used on the experiment."""
        return self._early_stopping_strategy

    @early_stopping_strategy.setter
    def early_stopping_strategy(self, ess: BaseEarlyStoppingStrategy) -> None:
        """Update the early stopping strategy."""
        self._early_stopping_strategy = ess

    @property
    def global_stopping_strategy(self) -> BaseGlobalStoppingStrategy | None:
        """The global stopping strategy used on the experiment."""
        return self._global_stopping_strategy

    @global_stopping_strategy.setter
    def global_stopping_strategy(self, gss: BaseGlobalStoppingStrategy) -> None:
        """Update the global stopping strategy."""
        self._global_stopping_strategy = gss

    @copy_doc(BestPointMixin.get_best_trial)
    def get_best_trial(
        self,
        optimization_config: OptimizationConfig | None = None,
        trial_indices: Iterable[int] | None = None,
        use_model_predictions: bool = True,
    ) -> tuple[int, TParameterization, TModelPredictArm | None] | None:
        return self._get_best_trial(
            experiment=self.experiment,
            generation_strategy=self.generation_strategy,
            trial_indices=trial_indices,
            use_model_predictions=use_model_predictions,
        )

    @copy_doc(BestPointMixin.get_pareto_optimal_parameters)
    def get_pareto_optimal_parameters(
        self,
        optimization_config: OptimizationConfig | None = None,
        trial_indices: Iterable[int] | None = None,
        use_model_predictions: bool = True,
    ) -> dict[int, tuple[TParameterization, TModelPredictArm]]:
        return self._get_pareto_optimal_parameters(
            experiment=self.experiment,
            generation_strategy=self.generation_strategy,
            trial_indices=trial_indices,
            use_model_predictions=use_model_predictions,
        )

    @copy_doc(BestPointMixin.get_hypervolume)
    def get_hypervolume(
        self,
        optimization_config: MultiObjectiveOptimizationConfig | None = None,
        trial_indices: Iterable[int] | None = None,
        use_model_predictions: bool = True,
    ) -> float:
        return BestPointMixin._get_hypervolume(
            experiment=self.experiment,
            generation_strategy=self.generation_strategy,
            optimization_config=optimization_config,
            trial_indices=trial_indices,
            use_model_predictions=use_model_predictions,
        )

    @copy_doc(BestPointMixin.get_trace)
    def get_trace(
        self,
        optimization_config: MultiObjectiveOptimizationConfig | None = None,
    ) -> list[float]:
        return BestPointMixin._get_trace(
            experiment=self.experiment,
            optimization_config=optimization_config,
        )

    @copy_doc(BestPointMixin.get_trace_by_progression)
    def get_trace_by_progression(
        self,
        optimization_config: OptimizationConfig | None = None,
        bins: list[float] | None = None,
        final_progression_only: bool = False,
    ) -> tuple[list[float], list[float]]:
        return BestPointMixin._get_trace_by_progression(
            experiment=self.experiment,
            optimization_config=optimization_config,
            bins=bins,
            final_progression_only=final_progression_only,
        )

    def _update_trial_with_raw_data(
        self,
        trial_index: int,
        raw_data: TEvaluationOutcome,
        metadata: dict[str, str | int] | None = None,
        sample_size: int | None = None,
        complete_trial: bool = False,
        combine_with_last_data: bool = False,
    ) -> str:
        """Helper method attaches data to a trial, returns a str of update."""
        # Format the data to save.
        trial = self.get_trial(trial_index)
        update_info = trial.update_trial_data(
            raw_data=raw_data,
            metadata=metadata,
            sample_size=sample_size,
            combine_with_last_data=combine_with_last_data,
        )

        if complete_trial:
            if not self._validate_all_required_metrics_present(
                raw_data=raw_data, trial_index=trial_index
            ):
                logger.warning(
                    "Marking the trial as failed because it is missing one"
                    "or more required metrics."
                )
                trial.mark_failed()
            else:
                trial.mark_completed()

        self._save_or_update_trial_in_db_if_possible(
            experiment=self.experiment, trial=trial
        )

        return update_info

    def _set_experiment(
        self,
        experiment: Experiment,
        overwrite_existing_experiment: bool = False,
    ) -> None:
        """Sets the ``_experiment`` attribute on this `AxClient`` instance and saves the
        experiment if this instance uses SQL storage.

        NOTE: This setter **should not be used outside of this file in production**.
        It can be leveraged in development, but all checked-in code that uses the
        Service API should leverage ``AxClient.create_experiment`` instead and extend it
        as needed. If using ``create_experiment`` is impossible and this setter is
        required, please raise your use case to the AE team or on our Github.
        """
        name = experiment._name

        if self.db_settings_set and not name:
            raise ValueError(
                "Must give the experiment a name if `db_settings` is not None."
            )
        if self.db_settings_set:
            experiment_id, _ = self._get_experiment_and_generation_strategy_db_id(
                experiment_name=none_throws(name)
            )
            if experiment_id:
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

        self._experiment = experiment

        try:
            self._save_experiment_to_db_if_possible(
                experiment=self.experiment,
            )
        except Exception:
            # Unset the experiment on this `AxClient` instance if encountered and
            # raising an error from saving the experiment, to avoid a case where
            # overall `create_experiment` call fails with a storage error, but
            # `self._experiment` is still set and user has to specify the
            # `overwrite_existing_experiment` kwarg to re-attempt exp. creation.
            self._experiment = None
            raise

    def _set_runner(self, experiment: Experiment) -> None:
        """Overridable method to sets a runner on the experiment."""
        experiment.runner = None

    def _set_generation_strategy(
        self, choose_generation_strategy_kwargs: dict[str, Any] | None = None
    ) -> None:
        """Selects the generation strategy and applies specified dispatch kwargs,
        if any.
        """
        choose_generation_strategy_kwargs = choose_generation_strategy_kwargs or {}
        if (
            "use_batch_trials" in choose_generation_strategy_kwargs
            and type(self) is AxClient
        ):
            logger.warning(
                "Selecting a GenerationStrategy when using BatchTrials is in beta. "
                "Double check the recommended strategy matches your expectations."
            )
        random_seed = choose_generation_strategy_kwargs.pop(
            "random_seed", self._random_seed
        )
        enforce_sequential_optimization = choose_generation_strategy_kwargs.pop(
            "enforce_sequential_optimization", self._enforce_sequential_optimization
        )
        if self._generation_strategy is None:
            self._generation_strategy = choose_generation_strategy(
                search_space=self.experiment.search_space,
                optimization_config=self.experiment.optimization_config,
                enforce_sequential_optimization=enforce_sequential_optimization,
                random_seed=random_seed,
                torch_device=self._torch_device,
                experiment=self.experiment,
                **choose_generation_strategy_kwargs,
            )
        elif self._experiment:
            self._generation_strategy.experiment = self.experiment

    def _save_generation_strategy_to_db_if_possible(
        self,
        generation_strategy: GenerationStrategyInterface | None = None,
    ) -> bool:
        return super()._save_generation_strategy_to_db_if_possible(
            generation_strategy=generation_strategy or self.generation_strategy,
        )

    def _gen_new_generator_run(
        self, n: int = 1, fixed_features: FixedFeatures | None = None
    ) -> GeneratorRun:
        """Generate new generator run for this experiment.

        Args:
            n: Number of arms to generate.
            fixed_features: A FixedFeatures object containing any
                features that should be fixed at specified values during
                generation.
        """
        # If random seed is not set for this optimization, context manager does
        # nothing; otherwise, it sets the random seed for torch, but only for the
        # scope of this call. This is important because torch seed is set globally,
        # so if we just set the seed without the context manager, it can have
        # serious negative impact on the performance of the models that employ
        # stochasticity.

        fixed_feats = (
            InstantiationBase.make_fixed_observation_features(
                fixed_features=fixed_features
            )
            if fixed_features
            else None
        )
        with with_rng_seed(seed=self._random_seed):
            return none_throws(self.generation_strategy).gen(
                experiment=self.experiment,
                n=n,
                pending_observations=self._get_pending_observation_features(
                    experiment=self.experiment
                ),
                fixed_features=fixed_feats,
            )

    def _find_last_trial_with_parameterization(
        self, parameterization: TParameterization
    ) -> int:
        """Given a parameterization, find the last trial in the experiment that
        contains an arm with that parameterization.
        """
        for trial_idx in sorted(self.experiment.trials.keys(), reverse=True):
            if (
                none_throws(self.get_trial(trial_idx).arm).parameters
                == parameterization
            ):
                return trial_idx
        raise ValueError(
            f"No trial on experiment matches parameterization {parameterization}."
        )

    def _validate_all_required_metrics_present(
        self, raw_data: TEvaluationOutcome, trial_index: int
    ) -> bool:
        """Check if all required metrics are present in the given raw data."""
        opt_config = self.experiment.optimization_config
        if opt_config is None:
            return True

        _, data = data_and_evaluations_from_raw_data(
            raw_data={"data": raw_data},
            sample_sizes={},
            trial_index=trial_index,
            data_type=self.experiment.default_data_type,
            metric_names=opt_config.objective.metric_names,
        )
        required_metrics = set(opt_config.metrics.keys())
        provided_metrics = data.metric_names
        missing_metrics = required_metrics - provided_metrics
        return not missing_metrics

    @classmethod
    def _get_pending_observation_features(
        cls,
        experiment: Experiment,
    ) -> dict[str, list[ObservationFeatures]] | None:
        """Extract pending points for the given experiment.

        NOTE: With one-arm `Trial`-s, we use a more performant
        ``get_pending_observation_features_based_on_trial_status`` utility instead
        of ``get_pending_observation_features``, since we can determine whether a point
        is pending based on the status of the corresponding trial.
        """
        return get_pending_observation_features_based_on_trial_status(
            experiment=experiment
        )

    # ------------------------------ Validators. -------------------------------

    def _validate_early_stopping_strategy(
        self, support_intermediate_data: bool
    ) -> None:
        if self._early_stopping_strategy is not None and not support_intermediate_data:
            raise ValueError(
                "Early stopping is only supported for experiments which allow "
                " reporting intermediate trial data by setting passing "
                "`support_intermediate_data=True`."
            )

    def __repr__(self) -> str:
        """String representation of this client."""
        return f"{self.__class__.__name__}(experiment={self._experiment})"

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
    def load(filepath: str | None = None) -> None:
        raise NotImplementedError(
            "Use `load_experiment_from_database` to load from SQL database or "
            "`load_from_json_file` to load optimization state from .json file."
        )

    @staticmethod
    def save(filepath: str | None = None) -> None:
        raise NotImplementedError(
            "Use `save_to_json_file` to save optimization state to .json file."
        )
