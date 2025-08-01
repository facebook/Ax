# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
from collections.abc import Sequence
from logging import Logger
from typing import Any, Literal

import numpy as np
import pandas as pd
from ax.analysis.analysis import Analysis, display_cards
from ax.analysis.analysis_card import AnalysisCardBase
from ax.analysis.overview import OverviewAnalysis
from ax.analysis.summary import Summary
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig, StorageConfig
from ax.api.protocols.metric import IMetric
from ax.api.protocols.runner import IRunner
from ax.api.types import TOutcome, TParameterization
from ax.api.utils.generation_strategy_dispatch import choose_generation_strategy
from ax.api.utils.instantiation.from_string import optimization_config_from_string
from ax.api.utils.instantiation.from_struct import experiment_from_struct
from ax.api.utils.storage import db_settings_from_storage_config
from ax.api.utils.structs import ExperimentStruct, GenerationStrategyDispatchStruct
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.runner import Runner
from ax.core.trial import Trial
from ax.core.trial_status import TrialStatus  # Used as a return type
from ax.core.utils import get_pending_observation_features_based_on_trial_status
from ax.early_stopping.strategies import (
    BaseEarlyStoppingStrategy,
    PercentileEarlyStoppingStrategy,
)
from ax.exceptions.core import ObjectNotFoundError, UnsupportedError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.orchestrator import Orchestrator, OrchestratorOptions
from ax.service.utils.best_point_mixin import BestPointMixin
from ax.service.utils.with_db_settings_base import WithDBSettingsBase
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
)
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from ax.utils.common.random import with_rng_seed

from pyre_extensions import assert_is_instance, none_throws
from typing_extensions import Self

logger: Logger = get_logger(__name__)


class Client(WithDBSettingsBase):
    _maybe_experiment: Experiment | None = None
    _maybe_generation_strategy: GenerationStrategy | None = None
    _maybe_early_stopping_strategy: BaseEarlyStoppingStrategy | None = None

    def __init__(
        self,
        storage_config: StorageConfig | None = None,
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize a ``Client``, which manages state across the lifecycle of an
        experiment.

        Args:
            storage_config: Configuration for saving to and loading from a database. If
                elided the experiment will not automatically be saved to a database.
            random_seed: An optional integer to set the random seed for reproducibility
                of the experiment's results. If not provided, the random seed will not
                be set, leading to potentially different results on different runs.
        """

        super().__init__(  # Initialize WithDBSettingsBase
            db_settings=db_settings_from_storage_config(storage_config=storage_config)
            if storage_config is not None
            else None,
        )

        self._storage_config = storage_config
        self._random_seed = random_seed

    # -------------------- Section 1: Configure -------------------------------------
    def configure_experiment(
        self,
        parameters: Sequence[RangeParameterConfig | ChoiceParameterConfig],
        parameter_constraints: Sequence[str] | None = None,
        name: str | None = None,
        description: str | None = None,
        experiment_type: str | None = None,
        owner: str | None = None,
    ) -> None:
        """
        Given an ``ExperimentConfig``, construct the Ax ``Experiment`` object. Note that
        validation occurs at time of config instantiation, not at
        ``configure_experiment``.

        This method only constitutes defining the search space and misc. metadata
        like name, description, and owners.

        Saves to database on completion if ``storage_config`` is present.
        """
        if self._maybe_experiment is not None:
            raise UnsupportedError(
                "Experiment already configured. Please create a new Client if you "
                "would like a new experiment."
            )

        experiment_struct = ExperimentStruct(
            parameters=[*parameters],
            parameter_constraints=[*parameter_constraints]
            if parameter_constraints
            else [],
            name=name,
            description=description,
            experiment_type=experiment_type,
            owner=owner,
        )

        self._maybe_experiment = experiment_from_struct(struct=experiment_struct)

        self._save_experiment_to_db_if_possible(experiment=self._experiment)

    def configure_optimization(
        self,
        objective: str,
        outcome_constraints: Sequence[str] | None = None,
    ) -> None:
        """
        Configures the goals of the optimization by setting the ``OptimizationConfig``.
        ``Metrics`` referenced here by their name will be moved from the Experiment's
        ``tracking_metrics`` if they were were already present (i.e. they were attached
        via ``configure_metrics``) or added as base ``Metrics``.

        Args:
            objective: Objective is a string and allows us to express single,
                scalarized, and multi-objective goals. Ex: "loss", "ne1 + 2 * ne2",
                "-ne, qps"
            outcome_constraints: Outcome constraints are also strings and allow us to
                express a desire to have a metric clear a threshold but not be
                further optimized. These constraints are expressed as inequalities.
                Ex: "qps >= 100", "0.5 * ne1 + 0.5 * ne2 >= 0.95".
                To indicate a relative constraint multiply your bound by "baseline"
                Ex: "qps >= 0.95 * baseline" will constrain such that the QPS is at
                least 95% of the baseline arm's QPS.
                Note that scalarized outcome constraints cannot be relative.


        Saves to database on completion if ``storage_config`` is present.
        """
        old_metrics = self._experiment.metrics
        self._experiment.optimization_config = optimization_config_from_string(
            objective_str=objective,
            outcome_constraint_strs=outcome_constraints,
        )
        self._set_metrics(metrics=list(old_metrics.values()))
        self._save_experiment_to_db_if_possible(experiment=self._experiment)

    def configure_generation_strategy(
        self,
        method: Literal["quality", "fast", "random_search"] = "fast",
        # Initialization options
        initialization_budget: int | None = None,
        initialization_random_seed: int | None = None,
        initialize_with_center: bool = True,
        use_existing_trials_for_initialization: bool = True,
        min_observed_initialization_trials: int | None = None,
        allow_exceeding_initialization_budget: bool = False,
        # Misc options
        torch_device: str | None = None,
    ) -> None:
        """
        Optional method to configure the way candidate parameterizations are generated
        during the optimization; if not called a default ``GenerationStrategy`` will be
        used.

        Saves to database on completion if ``storage_config`` is present.
        """
        generation_strategy = self._choose_generation_strategy(
            method=method,
            initialization_budget=initialization_budget,
            initialization_random_seed=initialization_random_seed,
            initialize_with_center=initialize_with_center,
            use_existing_trials_for_initialization=use_existing_trials_for_initialization,  # noqa[E501]
            min_observed_initialization_trials=min_observed_initialization_trials,
            allow_exceeding_initialization_budget=allow_exceeding_initialization_budget,
            torch_device=torch_device,
        )
        self.set_generation_strategy(generation_strategy=generation_strategy)

    # -------------------- Section 1.1: Configure Automation ------------------------
    def configure_runner(self, runner: IRunner) -> None:
        """
        Attaches a ``Runner`` to the ``Experiment``, to be used for automating trial
        deployment when using ``run_n_trials``.

        Saves to database on completion if ``storage_config`` is present.
        """
        self._set_runner(runner=runner)

    def configure_metrics(self, metrics: Sequence[IMetric]) -> None:
        """
        Attach a ``Metric`` with logic for autmating fetching of a given metric by
        replacing its instance with the provided ``Metric`` from metrics sequence input,
        or adds the ``Metric`` provided to the ``Experiment`` as a tracking metric if
        that metric was not already present.
        """
        self._set_metrics(metrics=metrics)

    @copy_doc(Experiment.remove_tracking_metric)
    def remove_tracking_metric(self, metric_name: str) -> None:
        self._experiment.remove_tracking_metric(metric_name=metric_name)

    # -------------------- Section 1.2: Set (not API) -------------------------------
    def set_experiment(self, experiment: Experiment) -> None:
        """
        This method is not part of the API and is provided (without guarantees of
        method signature stability) for the convenience of some developers, power
        users, and partners.

        Overwrite the existing ``Experiment`` with the provided ``Experiment``.

        Saves to database on completion if ``storage_config`` is present.
        """
        self._maybe_experiment = experiment

        self._save_experiment_to_db_if_possible(experiment=self._experiment)

    def set_optimization_config(self, optimization_config: OptimizationConfig) -> None:
        """
        This method is not part of the API and is provided (without guarantees of
        method signature stability) for the convenience of some developers, power
        users, and partners.

        Overwrite the existing ``OptimizationConfig`` with the provided
        ``OptimizationConfig``.

        Saves to database on completion if ``storage_config`` is present.
        """
        self._experiment.optimization_config = optimization_config

        self._save_experiment_to_db_if_possible(experiment=self._experiment)

    def set_generation_strategy(self, generation_strategy: GenerationStrategy) -> None:
        """
        This method is not part of the API and is provided (without guarantees of
        method signature stability) for the convenience of some developers, power
        users, and partners.

        Overwrite the existing ``GenerationStrategy`` with the provided
        ``GenerationStrategy``.

        Saves to database on completion if ``storage_config`` is present.
        """
        self._maybe_generation_strategy = generation_strategy

        self._generation_strategy._experiment = self._experiment

        self._save_generation_strategy_to_db_if_possible(
            generation_strategy=self._generation_strategy
        )

    def set_early_stopping_strategy(
        self, early_stopping_strategy: BaseEarlyStoppingStrategy
    ) -> None:
        """
        This method is not part of the API and is provided (without guarantees of
        method signature stability) for the convenience of some developers, power
        users, and partners.

        Overwrite the existing ``EarlyStoppingStrategy`` with the provided
        ``EarlyStoppingStrategy``.

        Saves to database on completion if ``storage_config`` is present.
        """
        self._maybe_early_stopping_strategy = early_stopping_strategy

    def _set_runner(self, runner: Runner) -> None:
        """
        This method is not part of the API and is provided (without guarantees of
        method signature stability) for the convenience of some developers and power
        users.

        Attaches a ``Runner`` to the ``Experiment``.

        Saves to database on completion if ``storage_config`` is present.
        """
        self._experiment.runner = runner

        self._update_runner_on_experiment_in_db_if_possible(
            experiment=self._experiment, runner=runner
        )

    def _set_metrics(self, metrics: Sequence[Metric]) -> None:
        """
        This method is not part of the API and is provided (without guarantees of
        method signature stability) for the convenience of some developers, power
        users, and partners.

        Attach a ``Metric`` with logic for autmating fetching of a given metric by
        replacing its instance with the provided ``Metric`` from metrics sequence input,
        or adds the ``Metric`` provided to the Experiment as a tracking metric if that
        metric was not already present.

        Saves to database on completion if ``storage_config`` is present.
        """
        # If an equivalently named Metric already exists on the Experiment, replace it
        # with the Metric provided. Otherwise, add the Metric to the Experiment as a
        # tracking metric.
        for metric in metrics:
            # Check the optimization config first
            self._overwrite_metric(metric=metric)

        self._save_experiment_to_db_if_possible(experiment=self._experiment)

    # -------------------- Section 2. Conduct Experiment ----------------------------
    def get_next_trials(
        self,
        max_trials: int,
        fixed_parameters: TParameterization | None = None,
    ) -> dict[int, TParameterization]:
        """
        Create up to ``max_trials`` trials using the ``GenerationStrategy`` (or as many
        as possible before reaching the maximum parellelism defined by the
        ``GenerationNode``), attach them to the ``Experiment`` with status RUNNING, and
        return a mapping from trial index to its parameterization. If a partial
        parameterization is provided via ``fixed_parameters`` each parameterization will
        have those parameters set to the provided values.

        Saves to database on completion if ``storage_config`` is present.

        Returns:
            A mapping of trial index to parameterization.
        """

        if self._experiment.optimization_config is None:
            raise UnsupportedError(
                "OptimizationConfig not set. Please call configure_optimization before "
                "generating trials."
            )

        trials: list[Trial] = []
        with with_rng_seed(seed=self._random_seed):
            gs = self._generation_strategy_or_choose()

            # This will be changed to use gen directly post gen-unfication cc @mgarrard
            generator_runs = gs.gen(
                experiment=self._experiment,
                pending_observations=(
                    get_pending_observation_features_based_on_trial_status(
                        experiment=self._experiment
                    )
                ),
                n=1,
                fixed_features=(
                    # pyre-fixme[6]: Type narrowing broken because core Ax
                    # TParameterization is dict not Mapping
                    ObservationFeatures(parameters=fixed_parameters)
                    if fixed_parameters is not None
                    else None
                ),
                num_trials=max_trials,
            )

        for generator_run in generator_runs:
            trial = assert_is_instance(
                self._experiment.new_trial(
                    generator_run=generator_run[0],
                ),
                Trial,
            )
            trial.mark_running(no_runner_required=True)

            trials.append(trial)

        # Save GS to db
        self._save_generation_strategy_to_db_if_possible(
            generation_strategy=self._generation_strategy
        )

        # Bulk save all trials to the database if possible
        self._save_or_update_trials_in_db_if_possible(
            experiment=self._experiment, trials=trials
        )

        res = {trial.index: none_throws(trial.arm).parameters for trial in trials}

        if len(res) != max_trials:
            logger.warning(
                f"{max_trials} trials requested but only {len(res)} could be "
                "generated."
            )

        # pyre-fixme[7]: Core Ax allows users to specify TParameterization values as
        # None, but we do not allow this in the API.
        return res

    def complete_trial(
        self,
        trial_index: int,
        raw_data: TOutcome | None = None,
        progression: int | None = None,
    ) -> TrialStatus:
        """
        Indicate the trial is complete and optionally attach data. In non-timeseries
        settings users should prefer to use ``complete_trial`` with ``raw_data`` over
        ``attach_data``. Ax will determine the trial's status automatically:
            - If all metrics on the ``OptimizationConfig`` are present the trial will be
                marked as COMPLETED
            - If any metrics on the ``OptimizationConfig`` are missing the trial will be
                marked as FAILED

        Saves to database on completion if ``storage_config`` is present.
        """
        if raw_data is not None:
            self.attach_data(
                trial_index=trial_index, raw_data=raw_data, progression=progression
            )

        # If no OptimizationConfig is set, mark the trial as COMPLETED
        if (optimization_config := self._experiment.optimization_config) is None:
            self._experiment.trials[trial_index].mark_completed()
        else:
            trial_data = self._experiment.lookup_data(trial_indices=[trial_index])
            missing_metrics = {*optimization_config.metrics.keys()} - {
                *trial_data.metric_names
            }

            # If all necessary metrics are present mark the trial as COMPLETED
            if len(missing_metrics) == 0:
                self._experiment.trials[trial_index].mark_completed()

            # If any metrics are missing mark the trial as FAILED
            else:
                logger.warning(
                    f"Trial {trial_index} marked completed but metrics "
                    f"{missing_metrics} are missing, marking trial FAILED."
                )
                self.mark_trial_failed(
                    trial_index=trial_index,
                    failed_reason=f"{missing_metrics} are missing, marking trial\
                    FAILED.",
                )

        self._save_or_update_trial_in_db_if_possible(
            experiment=self._experiment, trial=self._experiment.trials[trial_index]
        )

        return self._experiment.trials[trial_index].status

    def attach_data(
        self,
        trial_index: int,
        raw_data: TOutcome,
        progression: int | None = None,
    ) -> None:
        """
        Attach data without indicating the trial is complete. Missing metrics are,
        allowed, and unexpected metric values will be added to the Experiment as
        tracking metrics.

        Saves to database on completion if ``storage_config`` is present.
        """

        # If no progression is provided assume the data is not timeseries-like and
        # set step=NaN
        data_with_progression = [
            ({"step": progression if progression is not None else np.nan}, raw_data)
        ]

        trial = assert_is_instance(self._experiment.trials[trial_index], Trial)
        trial.update_trial_data(
            raw_data=data_with_progression, combine_with_last_data=True
        )

        self._save_or_update_trial_in_db_if_possible(
            experiment=self._experiment, trial=trial
        )

    # -------------------- Section 2.1 Custom trials --------------------------------
    def attach_trial(
        self, parameters: TParameterization, arm_name: str | None = None
    ) -> int:
        """
        Attach a single-arm trial to the ``Experiment`` with the provided parameters.
        The trial will be marked as RUNNING and must be completed manually by the
        user.

        Saves to database on completion if ``storage_config`` is present.

        Returns:
            The index of the attached trial.
        """
        _, trial_index = self._experiment.attach_trial(
            # pyre-fixme[6]: Type narrowing broken because core Ax TParameterization
            # is dict not Mapping
            parameterizations=[parameters],
            arm_names=[arm_name] if arm_name else None,
        )

        self._save_or_update_trial_in_db_if_possible(
            experiment=self._experiment, trial=self._experiment.trials[trial_index]
        )

        return trial_index

    def attach_baseline(
        self, parameters: TParameterization, arm_name: str | None = None
    ) -> int:
        """
        Attaches custom single-arm trial to an ``Experiment`` specifically for use as
        the baseline or status quo in evaluating relative outcome constraints and
        improvement over baseline objective value. The trial will be marked as RUNNING
        and must be completed manually by the user.

        Returns:
            The index of the attached trial.

        Saves to database on completion if ``storage_config`` is present.
        """
        trial_index = self.attach_trial(
            parameters=parameters,
            arm_name=arm_name or "baseline",
        )

        self._experiment.status_quo = assert_is_instance(
            self._experiment.trials[trial_index], Trial
        ).arm

        self._save_experiment_to_db_if_possible(experiment=self._experiment)

        return trial_index

    # -------------------- Section 2.2 Early Stopping -------------------------------
    def should_stop_trial_early(self, trial_index: int) -> bool:
        """
        Check if the trial should be stopped early. If True and the user wishes to heed
        Ax's recommendation the user should manually stop the trial and call
        ``mark_trial_early_stopped(trial_index)``. The ``EarlyStoppingStrategy`` may be
        selected automatically or set manually via ``set_early_stopping_strategy``.

        Returns:
            Whether the trial should be stopped early.
        """

        es_response = none_throws(
            self._early_stopping_strategy_or_choose()
        ).should_stop_trials_early(
            trial_indices={trial_index},
            experiment=self._experiment,
            current_node=self._generation_strategy_or_choose()._curr,
        )

        # TODO[mpolson64]: log the returned reason for stopping the trial
        return trial_index in es_response

    # -------------------- Section 2.3 Marking trial status manually ----------------
    def mark_trial_failed(
        self, trial_index: int, failed_reason: str | None = None
    ) -> None:
        """
        Manually mark a trial as FAILED. FAILED trials typically may be re-suggested by
        ``get_next_trials``, though this is controlled by the ``GenerationStrategy``.

        Saves to database on completion if ``storage_config`` is present.
        """
        self._experiment.trials[trial_index].mark_failed(reason=failed_reason)

        self._save_or_update_trial_in_db_if_possible(
            experiment=self._experiment, trial=self._experiment.trials[trial_index]
        )

    def mark_trial_abandoned(self, trial_index: int) -> None:
        """
        Manually mark a trial as ABANDONED. ABANDONED trials are typically not able to
        be re-suggested by ``get_next_trials``, though this is controlled by the
        ``GenerationStrategy``.

        Saves to database on completion if ``storage_config`` is present.
        """
        self._experiment.trials[trial_index].mark_abandoned()

        self._save_or_update_trial_in_db_if_possible(
            experiment=self._experiment, trial=self._experiment.trials[trial_index]
        )

    def mark_trial_early_stopped(self, trial_index: int) -> None:
        """
        Manually mark a trial as EARLY_STOPPED. This is used when the user has decided
        (with or without Ax's recommendation) to stop the trial after some data has
        been attached but before the trial is completed. Note that if data has not been
        attached for the trial yet users should instead call ``mark_trial_abandoned``.
        EARLY_STOPPED trials will not be re-suggested by ``get_next_trials``.

        Saves to database on completion if ``storage_config`` is present.
        """
        self._experiment.trials[trial_index].mark_early_stopped()

        self._save_or_update_trial_in_db_if_possible(
            experiment=self._experiment, trial=self._experiment.trials[trial_index]
        )

    def run_trials(
        self,
        max_trials: int,
        parallelism: int = 1,
        tolerated_trial_failure_rate: float = 0.5,
        initial_seconds_between_polls: int = 1,
    ) -> None:
        """
        Run maximum_trials trials in a loop by creating an ephemeral Orchestrator under
        the hood using the Experiment, GenerationStrategy, Metrics, and Runner attached
        to this AxClient along with the provided OrchestrationConfig.

        Saves to database on completion if ``storage_config`` is present.
        """

        orchestrator = Orchestrator(
            experiment=self._experiment,
            generation_strategy=self._generation_strategy_or_choose(),
            options=OrchestratorOptions(
                max_pending_trials=parallelism,
                tolerated_trial_failure_rate=tolerated_trial_failure_rate,
                init_seconds_between_polls=initial_seconds_between_polls,
            ),
            db_settings=db_settings_from_storage_config(self._storage_config)
            if self._storage_config is not None
            else None,
        )

        # Note: This Orchestrator call will handle storage internally
        orchestrator.run_n_trials(max_trials=max_trials)

    # -------------------- Section 3. Analyze ---------------------------------------
    def compute_analyses(
        self,
        analyses: Sequence[Analysis] | None = None,
        display: bool = True,
    ) -> list[AnalysisCardBase]:
        """
        Compute ``AnalysisCards`` (data about the optimization for end-user consumption)
        using the ``Experiment`` and ``GenerationStrategy``. If no analyses are
        provided use some heuristic to determine which analyses to run. If some
        analyses fail, log failure and continue to compute the rest.

        Note that the Analysis class is NOT part of the API and its methods are subject
        to change incompatibly between minor versions. Users are encouraged to use the
        provided analyses or leave this argument as ``None`` to use the default
        analyses.

        Saves cards to database on completion if ``storage_config`` is present.

        Args:
            analyses: A list of Analysis classes to run. If None Ax will choose which
                analyses to run based on the state of the experiment.
            display: Whether to display the AnalysisCards if executed in an interactive
                environment (e.g. Jupyter). Defaults to True. If not in an interactive
                environment this setting has no effect.
        Returns:
            A list of AnalysisCards.
        """

        analyses = analyses if analyses is not None else [OverviewAnalysis()]

        # Compute Analyses. If any fails to compute, catch and instead return an
        # ErrorAnalysisCard which contains the Exception and its associated traceback.
        cards = [
            analysis.compute_or_error_card(
                experiment=self._experiment,
                generation_strategy=self._generation_strategy,
            )
            for analysis in analyses
        ]

        # Display the AnalysisCards if requested and if the user is in a notebook
        if display:
            display_cards(cards=cards)

        return cards

    def summarize(self) -> pd.DataFrame:
        """
        Special convenience method for producing the ``DataFrame`` produced by the
        ``Summary`` ``Analysis``. This method is a convenient way to inspect the state
        of the ``Experiment``, but because the shape of the resultant DataFrame can
        change based on the ``Experiment`` state both users and Ax developers should
        prefer to use other methods for extracting information from the experiment to
        consume downstream.

        The ``DataFrame`` computed will contain one row per arm and the following
        columns (though empty columns are omitted):
            - trial_index: The trial index of the arm
            - arm_name: The name of the arm
            - trial_status: The status of the trial (e.g. RUNNING, SUCCEDED, FAILED)
            - failure_reason: The reason for the failure, if applicable
            - generation_node: The name of the ``GenerationNode`` that generated the arm
            - **METADATA: Any metadata associated with the trial, as specified by the
                Experiment's ``runner.run_metadata_report_keys`` field
            - **METRIC_NAME: The observed mean of the metric specified, for each metric
            - **PARAMETER_NAME: The parameter value for the arm, for each parameter
        """

        card = Summary(omit_empty_columns=True).compute(
            experiment=self._experiment,
            generation_strategy=self._maybe_generation_strategy,
        )

        return card.df

    def get_best_parameterization(
        self, use_model_predictions: bool = True
    ) -> tuple[TParameterization, TOutcome, int, str]:
        """
        Identifies the best parameterization tried in the experiment so far, also
        called the best in-sample arm.

        If ``use_model_predictions`` is ``True``, first attempts to do so with the
        model used in optimization and its corresponding predictions if available. If
        ``use_model_predictions`` is ``False`` or attempts to use the model fails,
        falls back to the best raw objective based on the data fetched from the
        ``Experiment``.

        Parameterizations which were observed to violate outcome constraints are not
        eligible to be the best parameterization.

        Returns:
            - The parameters predicted to have the best optimization value without
                violating any outcome constraints.
            - The metric values for the best parameterization. Uses model prediction if
                ``use_model_predictions=True``, otherwise returns observed data.
            - The trial which most recently ran the best parameterization
            - The name of the best arm (each trial has a unique name associated with
                each parameterization)
        """

        if self._experiment.optimization_config is None:
            raise UnsupportedError(
                "No optimization config has been set. Please configure the "
                "optimization before calling get_best_parameterization."
            )

        if self._experiment.optimization_config.is_moo_problem:
            raise UnsupportedError(
                "The client is currently configured to jointly optimize "
                f"{self._experiment.optimization_config}. "
                "Multi-objective optimization does not return a single best "
                "parameterization -- it returns a Pareto frontier. Please call "
                "get_pareto_frontier instead."
            )

        if len(self._experiment.trials) < 1:
            raise UnsupportedError(
                "No trials have been run yet. Please run at least one trial before "
                "calling get_best_parameterization."
            )

        # Note: Using BestPointMixin directly instead of inheriting to avoid exposing
        # unwanted public methods
        trial_index, parameterization, model_prediction = none_throws(
            BestPointMixin._get_best_trial(
                experiment=self._experiment,
                generation_strategy=self._generation_strategy_or_choose(),
                use_model_predictions=use_model_predictions,
            )
        )

        # pyre-fixme[7]: Core Ax allows users to specify TParameterization values as
        # None but we do not allow this in the API.
        return BestPointMixin._to_best_point_tuple(
            experiment=self._experiment,
            trial_index=trial_index,
            parameterization=parameterization,
            model_prediction=model_prediction,
        )

    def get_pareto_frontier(
        self, use_model_predictions: bool = True
    ) -> list[tuple[TParameterization, TOutcome, int, str]]:
        """
        Identifies the parameterizations which are predicted to efficiently trade-off
        between all objectives in a multi-objective optimization, also called the
        in-sample Pareto frontier.

        Returns:
            A list of tuples containing:
                - The parameters predicted to have the best optimization value without
                violating any outcome constraints.
                - The metric values for the best parameterization. Uses model
                    prediction if ``use_model_predictions=True``, otherwise returns
                    observed data.
                - The trial which most recently ran the best parameterization
                - The name of the best arm (each trial has a unique name associated
                    with each parameterization).
        """
        if self._experiment.optimization_config is None:
            raise UnsupportedError(
                "No optimization config has been set. Please configure the "
                "optimization before calling get_pareto_frontier."
            )

        if not self._experiment.optimization_config.is_moo_problem:
            raise UnsupportedError(
                "The client is currently configured to optimize "
                f"{self._experiment.optimization_config.objective}. "
                "Single-objective optimization does not return a Pareto frontier -- "
                "it returns a single best point. Please call "
                "get_best_parameterization instead."
            )

        if len(self._experiment.trials) < 1:
            raise UnsupportedError(
                "No trials have been run yet. Please run at least one trial before "
                "calling get_pareto_frontier."
            )

        frontier = BestPointMixin._get_pareto_optimal_parameters(
            experiment=self._experiment,
            # Requiring true GenerationStrategy here, ideally we will loosen this
            # in the future
            generation_strategy=self._generation_strategy,
            use_model_predictions=use_model_predictions,
        )

        # pyre-fixme[7]: Core Ax allows users to specify TParameterization values as
        # None but we do not allow this in the API.
        return [
            BestPointMixin._to_best_point_tuple(
                experiment=self._experiment,
                trial_index=trial_index,
                parameterization=parameterization,
                model_prediction=model_prediction,
            )
            for trial_index, (parameterization, model_prediction) in frontier.items()
        ]

    def predict(
        self,
        points: Sequence[TParameterization],
    ) -> list[dict[str, tuple[float, float]]]:
        """
        Use the current surrogate model to predict the outcome of the provided
        list of parameterizations.

        Returns:
            A list of mappings from metric name to predicted mean and SEM
        """
        for parameters in points:
            self._experiment.search_space.check_membership(
                parameterization=parameters,
                raise_error=True,
                check_all_parameters_present=True,
            )

        try:
            mean, covariance = none_throws(self._generation_strategy.adapter).predict(
                observation_features=[
                    # pyre-fixme[6]: Core Ax allows users to specify TParameterization
                    # values as None but we do not allow this in the API.
                    ObservationFeatures(parameters=parameters)
                    for parameters in points
                ]
            )
        except (NotImplementedError, AssertionError) as e:
            raise UnsupportedError(
                "Predicting with the GenerationStrategy's adapter failed. This "
                "could be because the current GenerationNode is not predictive -- try "
                "running more trials to progress to a predictive GenerationNode."
            ) from e

        return [
            {
                metric_name: (
                    mean[metric_name][i],
                    covariance[metric_name][metric_name][i] ** 0.5,
                )
                for metric_name in mean.keys()
            }
            for i in range(len(points))
        ]

    # -------------------- Section 4: Save/Load -------------------------------------
    # Note: SQL storage handled automatically during regular usage
    def save_to_json_file(self, filepath: str = "ax_client_snapshot.json") -> None:
        """
        Save a JSON-serialized snapshot of this ``Client``'s settings and state
        to a .json file by the given path.
        """
        with open(filepath, "w+") as file:
            file.write(json.dumps(self._to_json_snapshot()))
            logger.debug(
                f"Saved JSON-serialized state of optimization to `{filepath}`."
            )

    @classmethod
    def load_from_json_file(
        cls,
        filepath: str = "ax_client_snapshot.json",
        storage_config: StorageConfig | None = None,
    ) -> Self:
        """
        Restore a ``Client`` and its state from a JSON-serialized snapshot,
        residing in a .json file by the given path.

        Returns:
            The restored ``Client``.
        """
        with open(filepath) as file:
            return cls._from_json_snapshot(
                snapshot=json.loads(file.read()), storage_config=storage_config
            )

    @classmethod
    def load_from_database(
        cls,
        experiment_name: str,
        storage_config: StorageConfig | None = None,
    ) -> Self:
        """
        Restore an ``Client`` and its state from database by the given name.

        Returns:
            The restored ``Client``.
        """
        db_settings_base = WithDBSettingsBase(
            db_settings=db_settings_from_storage_config(storage_config=storage_config)
            if storage_config is not None
            else None
        )

        maybe_experiment, maybe_generation_strategy = (
            db_settings_base._load_experiment_and_generation_strategy(
                experiment_name=experiment_name
            )
        )
        if (experiment := maybe_experiment) is None:
            raise ObjectNotFoundError(
                f"Experiment {experiment_name} not found in database. Please check "
                "its name is correct, check your StorageConfig is correct, or create "
                "a new experiment."
            )

        client = cls(storage_config=storage_config)
        client.set_experiment(experiment=experiment)
        if maybe_generation_strategy is not None:
            client.set_generation_strategy(
                generation_strategy=maybe_generation_strategy
            )

        return client

    # -------------------- Section 5: Private Methods -------------------------------
    # -------------------- Section 5.1: Getters and defaults ------------------------
    @property
    def _experiment(self) -> Experiment:
        return none_throws(
            self._maybe_experiment,
            (
                "Experiment not set. Please call configure_experiment or load an "
                "experiment before utilizing any other methods on the Client."
            ),
        )

    @property
    def _generation_strategy(self) -> GenerationStrategy:
        return none_throws(
            self._maybe_generation_strategy,
            (
                "GenerationStrategy not set. Please call "
                "configure_generation_strategy, load a GenerationStrategy, or call "
                "get_next_trials or run_trials to automatically choose a "
                "GenerationStrategy before utilizing any other methods on the Client "
                "which require one."
            ),
        )

    @property
    def _early_stopping_strategy(self) -> BaseEarlyStoppingStrategy:
        return none_throws(
            self._maybe_early_stopping_strategy,
            "Early stopping strategy not set. Please set an early stopping strategy "
            "before calling should_stop_trial_early.",
        )

    def _generation_strategy_or_choose(
        self,
    ) -> GenerationStrategy:
        """
        If a GenerationStrategy is not set, choose a default one (save to database) and
        return it.
        """

        try:
            return self._generation_strategy
        except AssertionError:
            self.configure_generation_strategy()

            return self._generation_strategy

    def _early_stopping_strategy_or_choose(
        self,
    ) -> BaseEarlyStoppingStrategy:
        """
        If an EarlyStoppingStrategy is not set choose a default one and return it.
        """

        try:
            return self._early_stopping_strategy
        except AssertionError:
            # PercetinleEarlyStoppingStrategy may or may not have sensible defaults at
            # current moment -- we will need to be critical of these settings during
            # benchmarking
            self.set_early_stopping_strategy(
                early_stopping_strategy=PercentileEarlyStoppingStrategy()
            )

            return self._early_stopping_strategy

    def _choose_generation_strategy(
        self,
        method: Literal["quality", "fast", "random_search"] = "fast",
        # Initialization options
        initialization_budget: int | None = None,
        initialization_random_seed: int | None = None,
        initialize_with_center: bool = True,
        use_existing_trials_for_initialization: bool = True,
        min_observed_initialization_trials: int | None = None,
        allow_exceeding_initialization_budget: bool = False,
        # Misc options
        torch_device: str | None = None,
        _is_quickBO: bool = False,
    ) -> GenerationStrategy:
        """
        Choose a generation strategy based on the provided method and options.

        Args:
            method: The method to use for generating candidates. Options are:
                - "fast": Uses Bayesian optimization, configured specifically for
                  the current experiment.
                - "random_search": Uses random search.
            initialization_budget: Number of initialization trials. If None, will be
                automatically determined based on the search space.
            initialization_random_seed: Random seed for initialization. If None, no
                seed will be set.
            initialize_with_center: Whether to include the center of the search space
                in the initialization trials.
            use_existing_trials_for_initialization: Whether to use existing trials
                for initialization.
            min_observed_initialization_trials: Minimum number of observed
                init trials required before moving to the next generation step.
            allow_exceeding_initialization_budget: Whether to allow exceeding the
                initialization budget if more trials are needed.
            torch_device: The torch device to use for model fitting. If None, will
                use the default device.
            _is_quickbo: Internal parameter for QuickBO mode.

        Returns:
            A GenerationStrategy instance configured according to the specified options.
        """
        return choose_generation_strategy(
            struct=GenerationStrategyDispatchStruct(
                method=method,
                initialization_budget=initialization_budget,
                initialization_random_seed=initialization_random_seed,
                initialize_with_center=initialize_with_center,
                use_existing_trials_for_initialization=(
                    use_existing_trials_for_initialization
                ),
                min_observed_initialization_trials=min_observed_initialization_trials,
                allow_exceeding_initialization_budget=(
                    allow_exceeding_initialization_budget
                ),
                torch_device=torch_device,
            )
        )

    # -------------------- Section 5.2: Metric configuration --------------------------
    def _overwrite_metric(self, metric: Metric) -> None:
        """
        Overwrite an existing Metric on the Experiment with the provided Metric if they
        share the same name. If not Metric with the same name exists, add the Metric as
        a tracking metric.

        Note that this method does not save the Experiment to the database (this is
        handled in self._set_metrics).
        """

        # Check the OptimizationConfig first
        if (optimization_config := self._experiment.optimization_config) is not None:
            # Check the objective
            if isinstance(
                multi_objective := optimization_config.objective, MultiObjective
            ):
                for i in range(len(multi_objective.objectives)):
                    if metric.name == multi_objective.objectives[i].metric.name:
                        multi_objective._objectives[i]._metric = metric
                        return

            if isinstance(
                scalarized_objective := optimization_config.objective,
                ScalarizedObjective,
            ):
                for i in range(len(scalarized_objective.metrics)):
                    if metric.name == scalarized_objective.metrics[i].name:
                        scalarized_objective._metrics[i] = metric
                        return

            if (
                isinstance(optimization_config.objective, Objective)
                and metric.name == optimization_config.objective.metric.name
            ):
                optimization_config.objective._metric = metric
                return

            # Check the outcome constraints
            for i in range(len(optimization_config.outcome_constraints)):
                if (
                    metric.name
                    == optimization_config.outcome_constraints[i].metric.name
                ):
                    optimization_config._outcome_constraints[i]._metric = metric
                    return

        # Check the tracking metrics
        if metric.name in self._experiment._tracking_metrics.keys():
            self._experiment._tracking_metrics[metric.name] = metric
            return

        # If an equivalently named Metric does not exist, add it as a tracking
        # metric.
        self._experiment.add_tracking_metric(metric=metric)
        logger.warning(
            f"Metric {metric} not found in optimization config, added as tracking "
            "metric."
        )

    # -------------------- Section 5.3: Storage utilies -------------------------------
    def _to_json_snapshot(self) -> dict[str, Any]:
        """Serialize this `AxClient` to JSON to be able to interrupt and restart
        optimization and save it to file by the provided path.

        Returns:
            A JSON-safe dict representation of this `AxClient`.
        """

        # If the user has supplied custom encoder registries, use them. Otherwise use
        # the core encoder registries.
        if (
            self._storage_config is not None
            and self._storage_config.registry_bundle is not None
        ):
            encoder_registry = (
                self._storage_config.registry_bundle.sqa_config.json_encoder_registry
            )
            class_encoder_registry = self._storage_config.registry_bundle.sqa_config.json_class_encoder_registry  # noqa: E501
        else:
            encoder_registry = CORE_ENCODER_REGISTRY
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
            )
            if self._maybe_generation_strategy is not None
            else None,
        }

    @classmethod
    def _from_json_snapshot(
        cls,
        snapshot: dict[str, Any],
        storage_config: StorageConfig | None = None,
    ) -> Self:
        # If the user has supplied custom encoder registries, use them. Otherwise use
        # the core encoder registries.
        if storage_config is not None and storage_config.registry_bundle is not None:
            decoder_registry = (
                storage_config.registry_bundle.sqa_config.json_decoder_registry
            )
            class_decoder_registry = (
                storage_config.registry_bundle.sqa_config.json_class_decoder_registry
            )
        else:
            decoder_registry = CORE_DECODER_REGISTRY
            class_decoder_registry = CORE_CLASS_DECODER_REGISTRY

        # Decode the experiment, and generation strategy if present
        experiment = object_from_json(
            object_json=snapshot["experiment"],
            decoder_registry=decoder_registry,
            class_decoder_registry=class_decoder_registry,
        )

        generation_strategy = (
            generation_strategy_from_json(
                generation_strategy_json=snapshot["generation_strategy"],
                experiment=experiment,
                decoder_registry=decoder_registry,
                class_decoder_registry=class_decoder_registry,
            )
            if "generation_strategy" in snapshot
            else None
        )

        client = cls(storage_config=storage_config)
        client.set_experiment(experiment=experiment)
        if generation_strategy is not None:
            client.set_generation_strategy(generation_strategy=generation_strategy)

        return client
