# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

from ax.analysis.analysis import Analysis, AnalysisCard  # Used as a return type

from ax.core.base_trial import TrialStatus  # Used as a return type

from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.core.metric import Metric
from ax.core.optimization_config import OptimizationConfig
from ax.core.runner import Runner
from ax.core.search_space import SearchSpace
from ax.early_stopping.strategies import BaseEarlyStoppingStrategy

from ax.preview.api.configs import (
    DatabaseConfig,
    ExperimentConfig,
    GenerationStrategyConfig,
    OrchestrationConfig,
)
from ax.preview.api.protocols.metric import IMetric
from ax.preview.api.protocols.runner import IRunner
from ax.preview.api.types import TOutcome, TParameterization
from typing_extensions import Self


class Client:
    def __init__(
        self,
        db_config: DatabaseConfig | None = None,
        random_seed: int | None = None,
    ) -> None:
        """
        Many parameter are intentionally omitted from __init__ that were present
        in AxClient.__init__, including:

        generation_strategy: Now set via configure_generation_strategy or
            set_generation_strategy
        enforce_sequential_optimization: Now set via GenerationStrategyConfig
        torch_device: Now set via GenerationStrategyConfig
        verbose_logging: Omitted, user can set the logger level on their root config
        suppress_storage_errors: Omitted
        early_stopping_strategy: Now set via set_early_stopping_strategy
        global_stopping_strategy: Global stopping is not yet supported in API
        """
        ...

    # -------------------- Section 1: Configure --------------------------------------
    def configure_experiment(self, experiment_config: ExperimentConfig) -> None:
        """
        Given an ExperimentConfig, construct the Ax Experiment object. Note that
        validation occurs at time of config instantiation, not at
        configure_experiment.

        This method only constitutes defining the search space and misc. metadata
        like name, description, and owners.

        Saves to database on completion if db_config is present.
        """
        ...

    def configure_optimization(
        self,
        # Objective string is an expression and allows us to express single,
        # scalarized, and multi-objective via SymPy parsing.
        # Ex: "loss", "ne1 + ne1", "-ne, qps"
        objective: str,
        # Outcome constraints will also be parsed via SymPy
        # Ex: "num_layers1 <= num_layers2", "compound_a + compound_b <= 1"
        outcome_constraints: Sequence[str] | None = None,
    ) -> None:
        """
        Configures the goals of the optimization by setting the OptimizationConfig.
        Metrics referenced here by their name will be moved from the Experiment's
        tracking_metrics if they were were already present (i.e. they were attached via
        configure_metrics) or added as base Metrics.

        Saves to database on completion if db_config is present.
        """
        ...

    def configure_generation_strategy(
        self, generation_strategy_config: GenerationStrategyConfig
    ) -> None:
        """
        Overwrite the existing GenerationStrategy by calling choose_gs using the
        arguments of the GenerationStrategyConfig as parameters.

        Saves to database on completion if db_config is present.
        """
        ...

    # -------------------- Section 1.1: Configure Automation ------------------------
    def configure_runner(self, runner: IRunner) -> None:
        """
        Attaches a Runner to the Experiment.

        Saves to database on completion if db_config is present.
        """
        ...

    def configure_metrics(self, metrics: Sequence[IMetric]) -> None:
        """
        Finds equivallently named Metric that already exists on the Experiment and
        replaces it with the Metric provided, or adds the Metric provided to the
        Experiment as tracking metrics.
        """
        ...

    # -------------------- Section 1.2: Set (not API) -------------------------------
    def set_experiment(self, experiment: Experiment) -> None:
        """
        Overwrite the existing Experiment with the provided Experiment.

        Saves to database on completion if db_config is present.
        """
        ...

    def set_search_space(self, search_space: SearchSpace) -> None:
        """
        Overwrite the existing SearchSpace with the provided SearchSpace.

        Saves to database on completion if db_config is present.
        """
        ...

    def set_optimization_config(self, optimization_config: OptimizationConfig) -> None:
        """
        Overwrite the existing OptimizationConfig with the provided OptimizationConfig.

        Saves to database on completion if db_config is present.
        """
        ...

    def set_generation_strategy(
        self, generation_strategy: GenerationStrategyInterface
    ) -> None:
        """
        Overwrite the existing GenerationStrategy with the provided GenerationStrategy.

        Saves to database on completion if db_config is present.
        """
        ...

    def set_early_stopping_strategy(
        self, early_stopping_strategy: BaseEarlyStoppingStrategy
    ) -> None:
        """
        Overwrite the existing EarlyStoppingStrategy with the provided
        EarlyStoppingStrategy.

        Saves to database on completion if db_config is present.
        """
        ...

    def set_runner(self, runner: Runner) -> None:
        """
        Attaches a Runner to the Experiment.

        Saves to database on completion if db_config is present.
        """
        ...

    def set_metrics(self, metrics: Sequence[Metric]) -> None:
        """
        Finds equivallently named Metric that already exists on the Experiment and
        replaces it with the Metric provided, or adds the Metric provided to the
        Experiment as tracking metrics.
        """
        ...

    # -------------------- Section 2. Conduct Experiment ----------------------------
    def get_next_trials(
        self, maximum_trials: int = 1, fixed_parameters: TParameterization | None = None
    ) -> dict[int, TParameterization]:
        """
        Create up to `maximum_trials` trials using the `GenerationStrategy`, attach
        them to the Experiment, with status RUNNING, and return a mapping of trial
        index to its parameterization. If a partial parameterization is provided via
        fixed_parameters those parameters will be locked for all trials.

        This will need to be rethought somewhat when we add support for BatchTrials,
        but will be okay for current supported functionality.

        Saves to database on completion if db_config is present.

        Returns:
            A mapping of trial index to parameterization.
        """
        ...

    def complete_trial(
        self,
        trial_index: int,
        raw_data: TOutcome | None = None,
        progression: int | None = None,
    ) -> TrialStatus:
        """
        Indicate the trial is complete while optionally attach data. In non-timeseries
        settings users should prefer to use complete_trial with raw_data over
        attach_data. Ax will determine the trial's status automatically:
            - If all metrics on the OptimizationConfig are present the trial will be
                marked as COMPLETED
            - If any metrics on the OptimizationConfig are missing the trial will be
                marked as FAILED

        Saves to database on completion if db_config is present.
        """
        ...

    def attach_data(
        self,
        trial_index: int,
        raw_data: TOutcome,
        progression: int | None = None,
    ) -> None:
        """
        Attach data without indicating the trial is complete. Missing metrics are,
        allowed, and unexpected metric values will be added to the Experiment as
        tracking metrics. If progression is provided the Experiment will be updated to
        use MapData and the data will be attached to the appropriate step.

        Saves to database on completion if db_config is present.
        """
        ...

    # -------------------- Section 2.1 Custom trials --------------------------------
    def attach_trial(
        self, parameters: TParameterization, arm_name: str | None = None
    ) -> int:
        """
        Attach a single-arm trial to the experiment with the provided parameters.
        The trial will be marked as RUNNING and must be completed manually by the
        user.

        Saves to database on completion if db_config is present.

        Returns:
            The index of the attached trial.
        """
        ...

    def attach_baseline(self, baseline: TParameterization) -> int:
        """
        Attaches custom single-arm trial to an experiment specifically for use as the
        baseline or status quo in evaluating relative outcome constraints and
        improvement over baseline objective value. The trial will be marked as RUNNING
        and must be completed manually by the user.

        Returns:
            The index of the attached trial.

        Saves to database on completion if db_config is present.
        """
        ...

    # -------------------- Section 2.2 Early Stopping -------------------------------
    def should_stop_trial_early(self, trial_index: int) -> bool:
        """
        Check if the trial should be stopped early. If True and the user wishes to heed
        Ax's recommendation the user should manually stop the trial and call
        mark_trial_early_stopped(trial_index). The EarlyStoppingStrategy may be selected
        automatically or set manually via set_early_stopping_strategy.

        Returns:
            Whether the trial should be stopped early.
        """
        ...

    # -------------------- Section 2.3 Marking trial status manually ----------------
    def mark_trial_failed(self, trial_index: int) -> None:
        """
        Manually mark a trial as FAILED. FAILED trials may be re-suggested by
        get_next_trials.

        Saves to database on completion if db_config is present.
        """
        ...

    def mark_trial_abandoned(self, trial_index: int) -> None:
        """
        Manually mark a trial as ABANDONED. ABANDONED trials may be re-suggested by
        get_next_trials.

        Saves to database on completion if db_config is present.
        """
        ...

    def mark_trial_early_stopped(self, trial_index: int) -> None:
        """
        Manually mark a trial as EARLY_STOPPED. This is used when the user has decided
        (with or without Ax's recommendation) to stop the trial early. EARLY_STOPPED
        trials will not be re-suggested by get_next_trials.

        Saves to database on completion if db_config is present.
        """
        ...

    def run_trials(self, maximum_trials: int, options: OrchestrationConfig) -> None:
        """
        Run maximum_trials trials in a loop by creating an ephemeral Scheduler under
        the hood using the Experiment, GenerationStrategy, Metrics, and Runner attached
        to this AxClient along with the provided OrchestrationConfig.

        Saves to database on completion if db_config is present.
        """
        ...

    # -------------------- Section 3. Analyze ---------------------------------------
    def compute_analyses(
        self,
        analyses: Sequence[Analysis] | None = None,
    ) -> list[AnalysisCard]:
        """
        Compute AnalysisCards (data about the optimization for end-user consumption)
        using the Experiment and GenerationStrategy. If no analyses are provided use
        some heuristic to determine which analyses to run. If some analyses fail, log
        failure and continue to compute the rest.

        Note that the Analysis class is NOT part of the API and its methods are subject
        to change incompatibly between minor versions. Users are encouraged to use the
        provided analyses or leave this argument as None to use the default analyses.

        Saves to database on completion if db_config is present.

        Returns:
            A list of AnalysisCards.
        """
        ...

    def get_best_trial(
        self, use_model_predictions: bool = True
    ) -> tuple[int, TParameterization, TOutcome]:
        """
        Calculates the best in-sample trial.

        Returns:
            - The index of the best trial
            - The parameters of the best trial
            - The metric values associated withthe best trial
        """
        ...

    def get_pareto_frontier(
        self, use_model_predictions: bool = True
    ) -> dict[int, tuple[TParameterization, TOutcome]]:
        """
        Calculates the in-sample Pareto frontier.

        Returns:
            A mapping of trial index to its parameterization and metric values.
        """
        ...

    def predict(
        self,
        parameters: TParameterization,
        # If None predict for all Metrics
        metrics: Sequence[str] | None = None,
    ) -> TOutcome:
        """
        Use the GenerationStrategy to predict the outcome of the provided
        parameterization. If metrics is provided only predict for those metrics.

        Returns:
            A mapping of metric name to predicted mean and SEM.
        """
        ...

    # -------------------- Section 4: Save/Load -------------------------------------
    # Note: SQL storage handled automatically during regular usage
    def save_to_json_file(self, filepath: str = "ax_client_snapshot.json") -> None:
        """
        Save a JSON-serialized snapshot of this `AxClient`'s settings and state
        to a .json file by the given path.
        """
        ...

    @classmethod
    def load_from_json_file(
        cls,
        filepath: str = "ax_client_snapshot.json",
    ) -> Self:
        """
        Restore an `AxClient` and its state from a JSON-serialized snapshot,
        residing in a .json file by the given path.

        Returns:
            The restored `AxClient`.
        """
        ...

    def load_from_database(
        self,
        experiment_name: str,
    ) -> Self:
        """
        Restore an `AxClient` and its state from database by the given name.

        Returns:
            The restored `AxClient`.
        """
        ...
