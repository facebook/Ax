#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import inspect

import logging
import warnings
from collections import defaultdict, OrderedDict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from datetime import datetime
from functools import partial, reduce
from typing import Any, cast, Union

import ax.core.observation as observation
import pandas as pd
from ax.core.arm import Arm
from ax.core.auxiliary import (
    AuxiliaryExperiment,
    AuxiliaryExperimentPurpose,
    AuxiliaryExperimentValidation,
    TransferLearningMetadata,
)
from ax.core.base_trial import BaseTrial, sort_by_trial_index_and_arm_name
from ax.core.batch_trial import BatchTrial
from ax.core.data import combine_dfs_favoring_recent, Data
from ax.core.evaluations_to_data import DATA_TYPE_LOOKUP, DataType
from ax.core.generator_run import GeneratorRun
from ax.core.map_data import MapData
from ax.core.map_metric import MapMetric
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.core.objective import MultiObjective
from ax.core.optimization_config import ObjectiveThreshold, OptimizationConfig
from ax.core.parameter import DerivedParameter, Parameter
from ax.core.runner import Runner
from ax.core.search_space import SearchSpace
from ax.core.trial import Trial
from ax.core.trial_status import (
    DEFAULT_STATUSES_TO_WARM_START,
    STATUSES_EXPECTING_DATA,
    TrialStatus,
)
from ax.core.types import ComparisonOp, TParameterization
from ax.exceptions.core import (
    AxError,
    OptimizationNotConfiguredError,
    RunnerNotFoundError,
    UnsupportedError,
    UserInputError,
)
from ax.utils.common.base import Base
from ax.utils.common.constants import Keys
from ax.utils.common.docutils import copy_doc
from ax.utils.common.executils import retry_on_exception
from ax.utils.common.logger import _round_floats_for_logging, get_logger
from ax.utils.common.result import Err, Ok
from ax.utils.common.timeutils import current_timestamp_in_millis
from pyre_extensions import assert_is_instance, none_throws

logger: logging.Logger = get_logger(__name__)

NO_RETRY_EXCEPTIONS: tuple[type[Exception], ...] = (
    cast(type[Exception], RunnerNotFoundError),
    cast(type[Exception], NotImplementedError),
    cast(type[Exception], UnsupportedError),
)

ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES: int = 6

# pyre-fixme[5]: Global expression must be annotated.
round_floats_for_logging = partial(
    _round_floats_for_logging,
    decimal_places=ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES,
)
METRIC_DF_COLNAMES: Mapping[Hashable, str] = {
    "name": "Name",
    "type": "Type",
    "goal": "Goal",
    "bound": "Bound",
    "lower_is_better": "Lower is Better",
}


class Experiment(Base):
    """Base class for defining an experiment."""

    def __init__(
        self,
        search_space: SearchSpace,
        name: str | None = None,
        optimization_config: OptimizationConfig | None = None,
        tracking_metrics: list[Metric] | None = None,
        runner: Runner | None = None,
        status_quo: Arm | None = None,
        description: str | None = None,
        is_test: bool = False,
        experiment_type: str | None = None,
        properties: dict[str, Any] | None = None,
        default_data_type: DataType | None = None,
        auxiliary_experiments_by_purpose: None
        | (dict[AuxiliaryExperimentPurpose, list[AuxiliaryExperiment]]) = None,
        default_trial_type: str | None = None,
    ) -> None:
        """Inits Experiment.

        Args:
            search_space: Search space of the experiment.
            name: Name of the experiment.
            optimization_config: Optimization config of the experiment.
            tracking_metrics: Additional tracking metrics not used for optimization.
            runner: Default runner used for trials on this experiment.
            status_quo: Arm representing existing "control" arm.
            description: Description of the experiment.
            is_test: Mark experiment as test in metadata. This flag is meant purely
                for development and integration testing purposes. Leave as False for
                live experiments.
            experiment_type: The class of experiments this one belongs to.
            properties: Dictionary of this experiment's properties.  It is meant to
                only store primitives that pertain to Ax experiment state. Any trial
                deployment-related information and modeling-layer configuration
                should be stored elsewhere, e.g. in ``run_metadata`` of the trials.
            default_data_type: Enum representing the data type this experiment uses.
            auxiliary_experiments_by_purpose: Dictionary of auxiliary experiments
                for different purposes (e.g., transfer learning).
        """
        # appease pyre
        # pyre-fixme[13]: Attribute `_search_space` is never initialized.
        self._search_space: SearchSpace
        self._status_quo: Arm | None = None

        self._name = name
        self.description = description
        self._runner = runner
        self.is_test: bool = is_test

        self._data_by_trial: dict[int, OrderedDict[int, Data]] = {}
        self._experiment_type: str | None = experiment_type
        self._optimization_config: OptimizationConfig | None = None
        self._tracking_metrics: dict[str, Metric] = {}
        self._time_created: datetime = datetime.now()
        self._trials: dict[int, BaseTrial] = {}
        self._properties: dict[str, Any] = properties or {}
        self._default_data_type: DataType = default_data_type or DataType.DATA

        # Initialize trial type to runner mapping
        self._default_trial_type = default_trial_type
        self._trial_type_to_runner: dict[str | None, Runner | None] = {
            default_trial_type: runner
        }
        # Used to keep track of whether any trials on the experiment
        # specify a TTL. Since trials need to be checked for their TTL's
        # expiration often, having this attribute helps avoid unnecessary
        # TTL checks for experiments that do not use TTL.
        self._trials_have_ttl = False
        # Make sure all statuses appear in this dict, to avoid key errors.
        self._trial_indices_by_status: dict[TrialStatus, set[int]] = {
            status: set() for status in TrialStatus
        }
        self._arms_by_signature: dict[str, Arm] = {}
        self._arms_by_name: dict[str, Arm] = {}

        # Used to keep track of auxiliary experiments that were removed.
        self._initial_auxiliary_experiments_by_purpose: dict[
            AuxiliaryExperimentPurpose, list[AuxiliaryExperiment]
        ] = auxiliary_experiments_by_purpose or {}

        # Only tracks active auxiliary experiments.
        self.auxiliary_experiments_by_purpose: dict[
            AuxiliaryExperimentPurpose, list[AuxiliaryExperiment]
        ] = {
            purpose: [
                auxiliary_experiment
                for auxiliary_experiment in auxiliary_experiments
                if auxiliary_experiment.is_active
            ]
            for (
                purpose,
                auxiliary_experiments,
            ) in self._initial_auxiliary_experiments_by_purpose.items()
        }

        self.add_tracking_metrics(tracking_metrics or [])

        # call setters defined below
        self.search_space: SearchSpace = search_space
        self.status_quo = status_quo
        if optimization_config is not None:
            self.optimization_config = optimization_config

        # Keyed on tuple[trial_index, metric_name].
        self._metric_fetching_errors: dict[
            tuple[int, str], dict[str, Union[int, str]]
        ] = {}

    @property
    def has_name(self) -> bool:
        """Return true if experiment's name is not None."""
        return self._name is not None

    @property
    def name(self) -> str:
        """Get experiment name. Throws if name is None."""
        if self._name is None:
            raise ValueError("Experiment's name is None.")
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        """Set experiment name."""
        self._name = name

    @property
    def time_created(self) -> datetime:
        """Creation time of the experiment."""
        return self._time_created

    @property
    def experiment_type(self) -> str | None:
        """The type of the experiment."""
        return self._experiment_type

    @experiment_type.setter
    def experiment_type(self, experiment_type: str | None) -> None:
        """Set the type of the experiment."""
        self._experiment_type = experiment_type

    @property
    def search_space(self) -> SearchSpace:
        """The search space for this experiment.

        When setting a new search space, all parameter names and types
        must be preserved. However, if no trials have been created, all
        modifications are allowed.
        """
        # TODO: maybe return a copy here to guard against implicit changes
        return self._search_space

    @search_space.setter
    def search_space(self, search_space: SearchSpace) -> None:
        # Allow all modifications when no trials present.
        if not hasattr(self, "_search_space") or len(self.trials) < 1:
            self._search_space = search_space
            return

        # At least 1 trial is present.
        if self.immutable_search_space_and_opt_config:
            raise UnsupportedError(
                "Modifications of search space are disabled by the "
                f"`{Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF.value}` "
                "property that is set to `True` on this experiment."
            )
        if len(search_space.parameters) < len(self._search_space.parameters):
            raise ValueError(
                "New search_space must contain all parameters in the existing."
            )
        for param_name, parameter in search_space.parameters.items():
            if param_name not in self._search_space.parameters:
                raise ValueError(
                    f"Cannot add new parameter `{param_name}` because "
                    "it is not defined in the existing search space."
                )
            elif (
                parameter.parameter_type
                != self._search_space.parameters[param_name].parameter_type
            ):
                raise ValueError(
                    f"Expected parameter `{param_name}` to be of type "
                    f"{self._search_space.parameters[param_name].parameter_type}, "
                    f"got {parameter.parameter_type}."
                )
        self._search_space = search_space

    def add_parameters_to_search_space(
        self,
        parameters: Sequence[Parameter],
        status_quo_values: TParameterization | None = None,
    ) -> None:
        """
        Add new parameters to the experiment's search space. This allows extending
        the search space after the experiment has run some trials.

        Backfill values must be provided for all new parameters if the experiment has
        already run some trials. The backfill values represent the parameter values
        that were used in the existing trials.

        Args:
            parameters: A sequence of parameter configurations to add to the search
                space.
            status_quo_values: Optional parameter values for the new parameters to
                use in the status quo (baseline) arm, if one is defined.
        """
        status_quo_values = status_quo_values or {}

        # Additional checks iff a trial exists
        if len(self.trials) != 0:
            if any(parameter.backfill_value is None for parameter in parameters):
                raise UserInputError(
                    "Must provide backfill values for all new parameters when "
                    "adding parameters to an experiment with existing trials."
                )
            if any(isinstance(parameter, DerivedParameter) for parameter in parameters):
                raise UserInputError(
                    "Cannot add derived parameters to an experiment with existing "
                    "trials."
                )

        # Validate status quo values
        status_quo = self._status_quo
        if status_quo_values is not None and status_quo is None:
            logger.warning(
                "Status quo values specified, but experiment does not have a "
                "status quo. Ignoring provided status quo values."
            )
        if status_quo is not None:
            parameter_names = {parameter.name for parameter in parameters}
            status_quo_parameters = status_quo_values.keys()
            disabled_parameters = {
                parameter.name
                for parameter in self._search_space.parameters.values()
                if parameter.is_disabled
            }
            extra_status_quo_values = status_quo_parameters - parameter_names
            if extra_status_quo_values:
                logger.warning(
                    "Status quo value provided for parameters "
                    f"`{extra_status_quo_values}` which is are being added to "
                    "the search space. Ignoring provided status quo values."
                )
            mising_status_quo_values = (
                parameter_names - disabled_parameters - status_quo_parameters
            )
            if mising_status_quo_values:
                raise UserInputError(
                    "No status quo value provided for parameters "
                    f"`{mising_status_quo_values}` which are being added to "
                    "the search space."
                )
            for parameter_name, value in status_quo_values.items():
                status_quo._parameters[parameter_name] = value

        # Add parameters to search space
        self._search_space.add_parameters(parameters)

    def disable_parameters_in_search_space(
        self, default_parameter_values: TParameterization
    ) -> None:
        """
        Disable parameters in the experiment. This allows narrowing the search space
        after the experiment has run some trials.

        When parameters are disabled, they are effectively removed from the search
        space for future trial generation. Existing trials remain valid, and the
        disabled parameters are replaced with fixed default values for all subsequent
        trials.

        Args:
            default_parameter_values: Fixed values to use for the disabled parameters
                in all future trials. These values will be used for the parameter in
                all subsequent trials.
        """
        self._search_space.disable_parameters(default_parameter_values)

    @property
    def status_quo(self) -> Arm | None:
        """The existing arm that new arms will be compared against."""
        return self._status_quo

    @status_quo.setter
    def status_quo(self, status_quo: Arm | None) -> None:
        # Make status_quo immutable once any trial has been created.
        if self._status_quo is not None and len(self.trials) > 0:
            raise UnsupportedError(
                "Modifications of status_quo are disabled after trials have been "
                "created."
            )
        if status_quo == self.status_quo:
            return  # No need to update the SQ arm.

        if status_quo is not None:
            self.search_space.check_types(
                parameterization=status_quo.parameters,
                allow_extra_params=False,
                raise_error=True,
            )
            self.search_space.check_all_parameters_present(
                parameterization=status_quo.parameters, raise_error=True
            )

            # Compute a unique name if "status_quo" is taken
            name = "status_quo"
            sq_idx = 0
            arms_by_name = self.arms_by_name
            while name in arms_by_name:
                name = f"status_quo_e{sq_idx}"
                sq_idx += 1
            self._name_and_store_arm_if_not_exists(arm=status_quo, proposed_name=name)

        self._status_quo = status_quo

    @property
    def runner(self) -> Runner | None:
        """Default runner used for trials on this experiment."""
        return self._runner

    @runner.setter
    def runner(self, runner: Runner | None) -> None:
        """Set the default runner and update trial type mapping."""
        self._runner = runner
        if runner is not None:
            self._trial_type_to_runner[self._default_trial_type] = runner
        else:
            self._trial_type_to_runner = {None: None}

    @runner.deleter
    def runner(self) -> None:
        """Delete the runner."""
        self._runner = None
        self._trial_type_to_runner = {None: None}

    @property
    def parameters(self) -> dict[str, Parameter]:
        """The parameters in the experiment's search space."""
        return self.search_space.parameters

    @property
    def arms_by_name(self) -> dict[str, Arm]:
        """The arms belonging to this experiment, by their name."""
        return self._arms_by_name

    @property
    def arms_by_signature(self) -> dict[str, Arm]:
        """The arms belonging to this experiment, by their signature."""
        return self._arms_by_signature

    @property
    def arms_by_signature_for_deduplication(self) -> dict[str, Arm]:
        """The arms belonging to this experiment that should be used for deduplication
        in ``GenerationStrategy``, by their signature.

        In its current form, this includes all arms except for those that are
        associated with a ``FAILED`` trial.
        - The ``CANDIDATE``, ``STAGED``, ``RUNNING``, and ``ABANDONED`` arms are
        included as pending points during generation, so they should be less likely
         to get suggested by the model again.
        - The ``EARLY_STOPPED`` and ``COMPLETED`` trials were already evaluated, so
        the model will have data for these and is unlikely to suggest them again.
        """
        arms_dict = self.arms_by_signature.copy()
        for trial in self.trials_by_status[TrialStatus.FAILED]:
            for arm in trial.arms:
                arms_dict.pop(arm.signature, None)
        return arms_dict

    @property
    def sum_trial_sizes(self) -> int:
        """Sum of numbers of arms attached to each trial in this experiment."""
        return reduce(lambda a, b: a + len(b.arms_by_name), self._trials.values(), 0)

    @property
    def num_abandoned_arms(self) -> int:
        """How many arms attached to this experiment are abandoned."""
        abandoned = set()
        for trial in self.trials.values():
            for x in trial.abandoned_arms:
                abandoned.add(x)
        return len(abandoned)

    @property
    def optimization_config(self) -> OptimizationConfig | None:
        """The experiment's optimization config."""
        return self._optimization_config

    @optimization_config.setter
    def optimization_config(self, optimization_config: OptimizationConfig) -> None:
        if (
            len(self.trials) > 0
            and getattr(self, "_optimization_config", None) is not None
            and self.immutable_search_space_and_opt_config
        ):
            raise UnsupportedError(
                "Modifications of optimization config are disabled by the "
                f"`{Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF.value}` "
                "property that is set to `True` on this experiment."
            )
        for metric_name in optimization_config.metrics.keys():
            if metric_name in self._tracking_metrics:
                self.remove_tracking_metric(metric_name)
        # add metrics from the previous optimization config that are not in the new
        # optimization config as tracking metrics
        prev_optimization_config = self._optimization_config
        self._optimization_config = optimization_config
        if prev_optimization_config is not None:
            metrics_to_track = (
                set(prev_optimization_config.metrics.keys())
                - set(optimization_config.metrics.keys())
                - {Keys.DEFAULT_OBJECTIVE_NAME.value}  # remove default objective
            )
            for metric_name in metrics_to_track:
                self.add_tracking_metric(prev_optimization_config.metrics[metric_name])

        if any(metric.has_map_data for metric in optimization_config.metrics.values()):
            self._default_data_type = DataType.MAP_DATA

    @property
    def is_moo_problem(self) -> bool:
        """Whether the experiment's optimization config contains multiple objectives."""
        if self.optimization_config is None:
            return False
        return none_throws(self.optimization_config).is_moo_problem

    @property
    def is_bope_problem(self) -> bool:
        """Whether this experiment is a BO with Preference Exploration (BOPE)
        experiment.

        An experiment is considered a preference learning experiment if:
        1. It has a PreferenceOptimizationConfig as its optimization config, OR
        2. It has a PE_EXPERIMENT (preference exploration) auxiliary experiment attached

        Returns:
            True if this is a preference learning experiment, False otherwise.
        """
        # Check if optimization config indicates preference learning
        if self.optimization_config is not None:
            if self.optimization_config.is_bope_problem:
                return True

        # Check if experiment has a PE_EXPERIMENT auxiliary experiment
        return bool(
            self.auxiliary_experiments_by_purpose.get(
                AuxiliaryExperimentPurpose.PE_EXPERIMENT, []
            )
        )

    @property
    def immutable_search_space_and_opt_config(self) -> bool:
        """Boolean representing whether search space and metrics on this experiment
        are mutable (by default they are).

        NOTE: For experiments with immutable search spaces and metrics, generator
        runs will not store copies of search space and metrics, which improves
        storage layer performance. Not keeping copies of those on generator runs
        also disables keeping track of changes to search space and metrics,
        thereby necessitating that those attributes be immutable on experiment.
        """
        return self._properties.get(Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF, False)

    @property
    def tracking_metrics(self) -> list[Metric]:
        return list(self._tracking_metrics.values())

    def add_tracking_metric(self, metric: Metric) -> Experiment:
        """Add a new metric to the experiment.

        Args:
            metric: Metric to be added.
        """
        if metric.name in self._tracking_metrics:
            raise ValueError(
                f"Metric `{metric.name}` already defined on experiment. "
                "Use `update_tracking_metric` to update an existing metric definition."
            )

        optimization_config = self.optimization_config
        if optimization_config and metric.name in optimization_config.metrics:
            raise ValueError(
                f"Metric `{metric.name}` already present in experiment's "
                "OptimizationConfig. Set a new OptimizationConfig without this metric "
                "before adding it to tracking metrics."
            )

        if metric.has_map_data:
            self._default_data_type = DataType.MAP_DATA

        self._tracking_metrics[metric.name] = metric
        return self

    def add_tracking_metrics(self, metrics: list[Metric]) -> Experiment:
        """Add a list of new metrics to the experiment.

        If any of the metrics are already defined on the experiment,
        we raise an error and don't add any of them to the experiment

        Args:
            metrics: Metrics to be added.
        """
        # Before setting any metrics, we validate none are already on
        # the experiment
        for metric in metrics:
            self.add_tracking_metric(metric)
        return self

    def update_tracking_metric(self, metric: Metric) -> Experiment:
        """Redefine a metric that already exists on the experiment.

        Args:
            metric: New metric definition.
        """
        if metric.name not in self._tracking_metrics:
            raise ValueError(f"Metric `{metric.name}` doesn't exist on experiment.")

        self._tracking_metrics[metric.name] = metric
        return self

    def remove_tracking_metric(self, metric_name: str) -> Experiment:
        """Remove a metric that already exists on the experiment.

        Args:
            metric_name: Unique name of metric to remove.
        """
        if metric_name not in self._tracking_metrics:
            raise ValueError(f"Metric `{metric_name}` doesn't exist on experiment.")

        del self._tracking_metrics[metric_name]
        return self

    @property
    def metrics(self) -> dict[str, Metric]:
        """The metrics attached to the experiment."""
        optimization_config_metrics: dict[str, Metric] = {}
        if self.optimization_config is not None:
            optimization_config_metrics = self.optimization_config.metrics
        return {**self._tracking_metrics, **optimization_config_metrics}

    @property
    def signature_to_metric(self) -> dict[str, Metric]:
        """Returns a dictionary of metrics attached to the experiment, keyed by
        their signature. Useful for cases that require accessing metric objects
        by their signature (e.g. plotting from observation data/adapter).
        """
        return {metric.signature: metric for metric in self.metrics.values()}

    def _metrics_by_class(
        self, metrics: list[Metric] | None = None
    ) -> dict[type[Metric], list[Metric]]:
        metrics_by_class: dict[type[Metric], list[Metric]] = defaultdict(list)
        for metric in metrics or list(self.metrics.values()):
            # By default, all metrics are grouped by their class for fetch;
            # however, for some metrics, `fetch_trial_data_multi` of a
            # superclass is used for fetch the subclassing metrics' data. In
            # those cases, "fetch_multi_group_by_metric" property on metric
            # will be set to a class other than its own (likely a superclass).
            metrics_by_class[metric.fetch_multi_group_by_metric].append(metric)
        return metrics_by_class

    def get_metrics(self, metric_names: list[str] | None) -> list[Metric]:
        """Get a list of metrics from the experiment by name.

        Args:
            metric_names: List of metric names to retrieve. If None, returns all metrics
                defined on the experiment.

        Returns:
            List of Metric objects corresponding to the requested metric names,
            deduplicated so the same `Metric` does not occur twice.

        Raises:
            UserInputError: If any of the requested metric names are not found in the
                experiment.
        """
        if metric_names is None:
            return list(self.metrics.values())
        try:
            return [self.metrics[name] for name in metric_names]
        except KeyError as e:
            raise AxError(
                "One of the requested metrics was not present on the "
                f"experiment; original error: {e}."
            )

    def bulk_configure_metrics_of_class(
        self,
        metric_class: type[Metric],
        attributes_to_update: dict[str, Any],
    ) -> None:
        """Apply the same set of updates to all metrics of a specified type at once.

        Args:
            metric_class: the class of metric to update
            attributes_to_update: A dictionary that maps the parameter to be updated
                with it's new value.

        Raises:
            * If there are no metrics of the specified class on the experiment
            * If any of the specified parameters to update are not args in the
            metric class's initialization method

        Note: All metrics of the specified class will receive the same update. For
        custom updates to specific metrics, use `update_tracking_metric`
        """
        metrics_of_class = self._metrics_by_class(metrics=list(self.metrics.values()))[
            metric_class
        ]
        if len(metrics_of_class) == 0:
            raise UserInputError(
                f"No metrics of class {metric_class} found on experiment."
            )
        metric_attributes = set(
            inspect.signature(metric_class.__init__).parameters.keys()
        )
        if not set(attributes_to_update.keys()).issubset(metric_attributes):
            raise (
                UserInputError(
                    f"Metric class {metric_class} does not contain the requested "
                    "attributes to update. Requested updates to attributes: "
                    f"{set(attributes_to_update.keys())} but metric class defines"
                    f"{metric_attributes}."
                )
            )
        # Update each metric on the experiment with the specified new values
        # for each parameter to update
        for metric in metrics_of_class:
            new_metric = metric
            for param_name, param_value in attributes_to_update.items():
                setattr(new_metric, param_name, param_value)
            self.metrics[metric.name] = new_metric
        return

    def fetch_data_results(
        self,
        metrics: list[Metric] | None = None,
        **kwargs: Any,
    ) -> dict[int, dict[str, MetricFetchResult]]:
        """Fetches data for all trials on this experiment and for either the
        specified metrics or all metrics currently on the experiment, if `metrics`
        argument is not specified.

        If a metric fetch fails, the Exception will be captured in the
        MetricFetchResult along with a message.

        NOTE: For metrics that are not available while trial is running, the data
        may be retrieved from cache on the experiment. Data is cached on the experiment
        via calls to `experiment.attach_data` and whether a given metric class is
        available while trial is running is determined by the boolean returned from its
        `is_available_while_running` class method.

        Args:
            metrics: If provided, fetch data for these metrics instead of the ones
                defined on the experiment.
            kwargs: keyword args to pass to underlying metrics' fetch data functions.

        Returns:
            A nested Dictionary from trial_index => metric_name => result
        """

        return self._lookup_or_fetch_trials_results(
            trials=list(self.trials.values()),
            metrics=metrics,
            **kwargs,
        )

    def fetch_trials_data_results(
        self,
        trial_indices: Iterable[int],
        metrics: list[Metric] | None = None,
        **kwargs: Any,
    ) -> dict[int, dict[str, MetricFetchResult]]:
        """Fetches data for specific trials on the experiment.

        If a metric fetch fails, the Exception will be captured in the
        MetricFetchResult along with a message.

        NOTE: For metrics that are not available while trial is running, the data
        may be retrieved from cache on the experiment. Data is cached on the experiment
        via calls to `experiment.attach_data` and whether a given metric class is
        available while trial is running is determined by the boolean returned from its
        `is_available_while_running` class method.

        Args:
            trial_indices: Indices of trials, for which to fetch data.
            metrics: If provided, fetch data for these metrics instead of the ones
                defined on the experiment.
            kwargs: keyword args to pass to underlying metrics' fetch data functions.

        Returns:
            A nested Dictionary from trial_index => metric_name => result
        """
        return self._lookup_or_fetch_trials_results(
            trials=self.get_trials_by_indices(trial_indices=trial_indices),
            metrics=metrics,
            **kwargs,
        )

    def fetch_data(
        self,
        trial_indices: Iterable[int] | None = None,
        metrics: list[Metric] | None = None,
        **kwargs: Any,
    ) -> Data:
        """Fetches data for all trials on this experiment and for either the
        specified metrics or all metrics currently on the experiment, if `metrics`
        argument is not specified.

        NOTE: For metrics that are not available while trial is running, the data
        may be retrieved from cache on the experiment. Data is cached on the experiment
        via calls to `experiment.attach_data` and whether a given metric class is
        available while trial is running is determined by the boolean returned from its
        `is_available_while_running` class method.

        NOTE: This can be lossy (ex. a MapData could get implicitly cast to a Data and
        lose rows) if Experiment.default_data_type is misconfigured!

        Args:
            metrics: If provided, fetch data for these metrics; otherwise, fetch
                data for all metrics defined on the experiment.
            kwargs: keyword args to pass to underlying metrics' fetch data functions.

        Returns:
            Data for the experiment.
        """
        results = self._lookup_or_fetch_trials_results(
            trials=list(self.trials.values())
            if trial_indices is None
            else self.get_trials_by_indices(trial_indices=trial_indices),
            metrics=metrics,
            **kwargs,
        )

        base_metric_cls = (
            MapMetric if self.default_data_constructor == MapData else Metric
        )

        return base_metric_cls._unwrap_experiment_data_multi(
            results=results,
        )

    def _lookup_or_fetch_trials_results(
        self,
        trials: list[BaseTrial],
        metrics: Iterable[Metric] | None = None,
        **kwargs: Any,
    ) -> dict[int, dict[str, MetricFetchResult]]:
        if not self.metrics and not metrics:
            raise ValueError(
                "No metrics to fetch data for, as no metrics are defined for "
                "this experiment, and none were passed in to `fetch_data`."
            )
        if not any(t.status.expecting_data for t in trials):
            logger.debug("No trials are in a state expecting data.")
            return {}
        metrics_to_fetch = list(metrics or self.metrics.values())
        metrics_by_class = self._metrics_by_class(metrics=metrics_to_fetch)

        results: dict[int, dict[str, MetricFetchResult]] = {}
        contains_new_data = False

        for metric_cls, metrics in metrics_by_class.items():
            first_metric_of_group = metrics[0]
            (
                new_fetch_results,
                new_results_contains_new_data,
            ) = first_metric_of_group.fetch_data_prefer_lookup(
                experiment=self,
                metrics=metrics_by_class[metric_cls],
                trials=trials,
                **kwargs,
            )

            contains_new_data = contains_new_data or new_results_contains_new_data

            # Merge in results
            results = {
                trial.index: {
                    **(
                        new_fetch_results[trial.index]
                        if trial.index in new_fetch_results
                        else {}
                    ),
                    **(results[trial.index] if trial.index in results else {}),
                }
                for trial in trials
            }

        if contains_new_data:
            try:
                self.attach_fetch_results(results=results)
            except ValueError as e:
                logger.error(
                    f"Encountered ValueError {e} while attaching results. Proceeding "
                    "and returning Results fetched without attaching."
                )

        return results

    @copy_doc(BaseTrial.fetch_data)
    def _fetch_trial_data(
        self, trial_index: int, metrics: list[Metric] | None = None, **kwargs: Any
    ) -> dict[str, MetricFetchResult]:
        trial = self.trials[trial_index]

        trial_data = self._lookup_or_fetch_trials_results(
            trials=[trial], metrics=metrics, **kwargs
        )

        if trial_index in trial_data:
            return trial_data[trial_index]

        return {}

    def attach_data(self, data: Data, **kwargs: Any) -> int:
        """
        Attach data to the experiment's `_data_by_trial` attribute.

        Store data in `experiment._data_by_trial`, to be looked up via
        ``experiment.lookup_data_for_trial`` or ``experiment.lookup_data()``.
        When a new observation is attached to a trial that already has an
        observation for that arm name, metric, and (if present) step, the new
        observation replaces the old one.

        Args:
            data: Data to attach.
            kwargs: Deprecated arguments.

        Returns:
            Timestamp of storage in millis.
        """
        deprecated_arguments = ["combine_with_last_data", "overwrite_existing_data"]
        for arg in deprecated_arguments:
            if arg in kwargs:
                warnings.warn(
                    f"Passing {arg} to `attach_data` is deprecated. "
                    "`attach_data` will behave in a way similar to the old "
                    "`combine_with_last_data=True`, "
                    "`overwrite_existing_data=False`. behavior.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        unexpected_args = set(kwargs.keys()) - set(deprecated_arguments)
        if unexpected_args:
            raise ValueError(
                f"Unexpected arguments {unexpected_args} passed to `attach_data`."
            )
        data_type = type(data)
        if data.full_df.empty:
            raise ValueError("Data to attach is empty.")
        metrics_not_on_exp = set(data.full_df["metric_name"].values) - set(
            self.metrics.keys()
        )
        if metrics_not_on_exp:
            logger.debug(
                f"Attached data has some metrics ({metrics_not_on_exp}) that are "
                "not among the metrics on this experiment. Note that attaching data "
                "will not automatically add those metrics to the experiment. "
                "For these metrics to be automatically fetched by `experiment."
                "fetch_data`, add them via `experiment.add_tracking_metric` or update "
                "the experiment's optimization config."
            )
        cur_time_millis = current_timestamp_in_millis()
        for trial_index, trial_df in data.full_df.groupby("trial_index"):
            if not isinstance(data, MapData):
                trial_df = sort_by_trial_index_and_arm_name(df=trial_df)
            current_trial_data = (
                self._data_by_trial[trial_index]
                if trial_index in self._data_by_trial
                else OrderedDict()
            )
            if len(current_trial_data) == 1:
                _, last_data = current_trial_data.popitem()
                combined_df = combine_dfs_favoring_recent(
                    last_df=last_data.full_df, new_df=trial_df
                )
                data_type = (
                    MapData
                    if isinstance(last_data, MapData) or isinstance(data, MapData)
                    else Data
                )
            elif len(current_trial_data) == 0:
                combined_df = trial_df
            else:
                raise ValueError(
                    "Each dict within `_data_by_trial` should have at most one "
                    "element."
                )
            current_trial_data[cur_time_millis] = data_type(df=combined_df)
            self._data_by_trial[trial_index] = current_trial_data

        return cur_time_millis

    def attach_fetch_results(
        self,
        results: Mapping[int, Mapping[str, MetricFetchResult]],
    ) -> int | None:
        """
        UNSAFE: Prefer to use attach_data directly instead.

        Attach fetched data results to the Experiment so they will not have to be
        fetched again. Additionally caches any metric fetching errors that occurred
        to the experiment. Returns the timestamp from attachment, which is used
        as a dict key for _data_by_trial.

        NOTE: Any Errs in the results passed in will silently be dropped! This will
        cause the Experiment to fail to find them in the _data_by_trial cache and
        attempt to refetch at fetch time. If this is not your intended behavior you
        MUST resolve your results first and use attach_data directly instead.
        """

        completed_trial_indices = self.trial_indices_by_status[TrialStatus.COMPLETED]
        oks: list[Ok[Data, MetricFetchE]] = []
        for trial_index, metrics in results.items():
            for metric_name, result in metrics.items():
                if isinstance(result, Ok):
                    oks.append(result)
                    self._metric_fetching_errors.pop((trial_index, metric_name), None)
                elif isinstance(result, Err):
                    msg = (
                        "Discovered Metric fetching Err while attaching data "
                        f"{result.err}. "
                        "Ignoring for now -- will retry query on next call to fetch."
                    )
                    self._cache_metric_fetch_error(
                        trial_index=trial_index,
                        metric_name=metric_name,
                        metric_fetch_e=result.err,
                    )
                    if trial_index in completed_trial_indices:
                        logger.error(msg)
                    else:
                        msg += (
                            f" Suppressing error for trial {trial_index} not in "
                            "COMPLETED state."
                        )
                        logger.debug(msg)

        if len(oks) < 1:
            return None

        data = self.default_data_constructor.from_multiple_data(
            data=[ok.ok for ok in oks]
        )

        return self.attach_data(data=data)

    def lookup_data_for_trial(self, trial_index: int) -> Data:
        """Look up stored data for a specific trial.

        Returns latest data object for this trial. Returns empty data if no data
        is present. This method will not fetch data from metrics - to do that,
        use `fetch_data()` instead.

        Args:
            trial_index: The index of the trial to lookup data for.

        Returns:
            The requested data object.
        """
        try:
            trial_data_dict = self._data_by_trial[trial_index]
        except KeyError:
            return self.default_data_constructor()

        if len(trial_data_dict) == 0:
            return self.default_data_constructor()

        storage_time = max(trial_data_dict.keys())
        return trial_data_dict[storage_time]

    def lookup_data(
        self,
        trial_indices: Iterable[int] | None = None,
    ) -> Data:
        """
        Combine stored ``Data``s for trials ``trial_indices`` into one ``Data``.

        For each trial, returns latest data object present for this trial.
        Returns empty data if no data is present. In particular, this method
        will not fetch data from metrics - to do that, use `fetch_data()` instead.

        Args:
            trial_indices: Indices of trials for which to fetch data. If omitted,
                lookup data for all trials on the experiment.

        Returns:
            Data for trials ``trial_indices`` on the experiment.
        """
        trial_indices = (
            list(self.trials.keys()) if trial_indices is None else list(trial_indices)
        )
        if len(trial_indices) == 0:
            return self.default_data_constructor()

        data_by_trial = []
        has_map_data = False
        for trial_index in trial_indices:
            trial_data = self.lookup_data_for_trial(trial_index=trial_index)
            data_by_trial.append(trial_data)
            has_map_data = has_map_data or isinstance(trial_data, MapData)

        data_type = MapData if has_map_data else Data
        return data_type.from_multiple_data(data_by_trial)

    @property
    def num_trials(self) -> int:
        """How many trials are associated with this experiment."""
        return len(self._trials)

    @property
    def trials(self) -> dict[int, BaseTrial]:
        """The trials associated with the experiment.

        NOTE: If some trials on this experiment specify their TTL, `CANDIDATE` trials
        will be checked for whether their TTL elapsed during this call. Found past-
        TTL trials will be marked as `STALE`.
        """
        self._check_TTL_on_candidate_trials()
        return self._trials

    @property
    def trials_by_status(self) -> dict[TrialStatus, list[BaseTrial]]:
        """Trials associated with the experiment, grouped by trial status."""
        # Make sure all statuses appear in this dict, to avoid key errors.
        return {
            status: self.get_trials_by_indices(trial_indices=idcs)
            for status, idcs in self.trial_indices_by_status.items()
        }

    @property
    def trials_expecting_data(self) -> list[BaseTrial]:
        """list[BaseTrial]: the list of all trials for which data has arrived
        or is expected to arrive.
        """
        return [trial for trial in self.trials.values() if trial.status.expecting_data]

    @property
    def completed_trials(self) -> list[BaseTrial]:
        """list[BaseTrial]: the list of all trials for which data has arrived
        or is expected to arrive.
        """
        return self.trials_by_status[TrialStatus.COMPLETED]

    @property
    def trial_indices_by_status(self) -> dict[TrialStatus, set[int]]:
        """Indices of trials associated with the experiment, grouped by trial
        status.
        """
        self._check_TTL_on_candidate_trials()  # Marks past-TTL trials as stale.
        return self._trial_indices_by_status

    @property
    def running_trial_indices(self) -> set[int]:
        """Indices of running trials, associated with the experiment."""
        return self._trial_indices_by_status[TrialStatus.RUNNING]

    @property
    def trial_indices_expecting_data(self) -> set[int]:
        """Set of indices of trials, statuses of which indicate that we expect
        these trials to have data, either already or in the future.
        """
        return set.union(
            *(
                self.trial_indices_by_status[status]
                for status in STATUSES_EXPECTING_DATA
            )
        )

    def trial_indices_with_data(
        self, critical_metrics_only: bool | None = True
    ) -> set[int]:
        """Set of indices of trials for which we have data for either all metrics on
        the experiment, or all metrics in the optimization config. Helpful for
        determining which trials currently have data for modeling.

        Args:
            critical_metrics_only: If True, only return trials for which we have
            metrics for the optimization config. If False, return trials for which
            we have data for all metrics on the experiment, including tracking metrics
        """
        if critical_metrics_only:
            if self.optimization_config is None:
                raise OptimizationNotConfiguredError(
                    "Cannot find trials with data for optimization config metrics "
                    "because no optimization config has been defined."
                )
            metric_names = set(self.optimization_config.metrics.keys())
        else:
            metric_names = set(self.metrics.keys())
            if len(metric_names) == 0:
                return set()

        exp_data = self.lookup_data().filter(metric_names=metric_names)
        trials_with_data = set()
        for trial_idx in self.trials.keys():
            metrics_in_trial_data = set(
                exp_data.df[exp_data.df["trial_index"] == trial_idx][
                    "metric_name"
                ].unique()
            )
            if metrics_in_trial_data == metric_names:
                trials_with_data.add(trial_idx)
            else:
                logger.debug(
                    f"Trial {trial_idx} does not have data for required metrics "
                    f"({metric_names}) on the experiment. "
                    f"Metrics present in trial data: {metrics_in_trial_data}"
                )

        return trials_with_data

    @property
    def default_data_type(self) -> DataType:
        return self._default_data_type

    @property
    def default_data_constructor(self) -> type[Data]:
        return DATA_TYPE_LOOKUP[self.default_data_type]

    def new_trial(
        self,
        generator_run: GeneratorRun | None = None,
        trial_type: str | None = None,
        ttl_seconds: int | None = None,
    ) -> Trial:
        """Create a new trial associated with this experiment.

        Args:
            generator_run: GeneratorRun, associated with this trial.
                Trial has only one arm attached to it and this generator_run
                must therefore contain one arm. This arm can also be set later
                through `add_arm` or `add_generator_run`, but a trial's
                associated generator run is immutable once set.
            trial_type: Type of this trial, if used in MultiTypeExperiment.
            ttl_seconds: If specified, trials will be considered stale after
                this many seconds since the time the trial was ran, unless the
                trial is completed before then. Meant to be used to detect
                'dead' trials, for which the evaluation process might have
                crashed etc., and which should be considered stale after
                their 'time to live' has passed.
        """
        if ttl_seconds is not None:
            self._trials_have_ttl = True
        return Trial(
            experiment=self,
            trial_type=trial_type,
            generator_run=generator_run,
            ttl_seconds=ttl_seconds,
        )

    def new_batch_trial(
        self,
        generator_run: GeneratorRun | None = None,
        generator_runs: list[GeneratorRun] | None = None,
        should_add_status_quo_arm: bool | None = False,
        trial_type: str | None = None,
        ttl_seconds: int | None = None,
    ) -> BatchTrial:
        """Create a new batch trial associated with this experiment.

        Args:
            generator_run: GeneratorRun, associated with this trial. This can a
                also be set later through `add_arm` or `add_generator_run`, but a
                trial's associated generator run is immutable once set.
            generator_runs: GeneratorRuns, associated with this trial. This can a
                also be set later through `add_arm` or `add_generator_run`, but a
                trial's associated generator run is immutable once set.  This cannot
                be combined with the `generator_run` argument.
            should_add_status_quo_arm: If True, adds the status quo arm to the trial
                with a weight of 1.0. If False, the _status_quo is still set on the
                trial for tracking purposes, but without a weight it will not be an
                Arm present on the trial
            trial_type: Type of this trial, if used in MultiTypeExperiment.
            ttl_seconds: If specified, trials will be considered stale after
                this many seconds since the time the trial was ran, unless the
                trial is completed before then. Meant to be used to detect
                'dead' trials, for which the evaluation process might have
                crashed etc., and which should be considered stale after
                their 'time to live' has passed.
        """
        if ttl_seconds is not None:
            self._trials_have_ttl = True
        return BatchTrial(
            experiment=self,
            trial_type=trial_type,
            generator_run=generator_run,
            generator_runs=generator_runs,
            should_add_status_quo_arm=should_add_status_quo_arm,
            ttl_seconds=ttl_seconds,
        )

    def get_batch_trial(self, trial_index: int) -> BatchTrial:
        """
        Return a trial on experiment cast as BatchTrial
        Args:
            trial_index: The index of the trial to lookup data for.
        Returns:
            The requested trial cast as BatchTrial
        """
        return assert_is_instance(
            self.get_trials_by_indices(trial_indices=[trial_index])[0], BatchTrial
        )

    def get_trials_by_indices(self, trial_indices: Iterable[int]) -> list[BaseTrial]:
        """Grabs trials on this experiment by their indices."""
        trial_indices = list(trial_indices)
        try:
            trials = self.trials
            return [trials[idx] for idx in trial_indices]
        except KeyError:
            missing = set(trial_indices) - set(self.trials)
            raise ValueError(
                f"Trial indices {missing} are not associated with the experiment.\n"
                f"Trials indices available on this experiment: {list(self.trials)}."
            )

    def extract_relevant_trials(
        self,
        trial_indices: Sequence[int] | None = None,
        trial_statuses: Sequence[TrialStatus] | None = None,
    ) -> list[BaseTrial]:
        """
        Find the trials on this experiment which meet both the trial_indices
        filtering condition and the trial_statuses filtering condition. If None is
        provided for either condition, the condition is not applied.

        Args:
            trial_indices: Indices of trials to include. If None, include all trials.
                If empty list, include no trials.
            trial_statuses: Statuses to filter by. If None, no filtering by status.
        """
        trials = (
            [self.trials[i] for i in trial_indices]
            if trial_indices is not None
            else [*self.trials.values()]
        )

        filtered_by_status = [
            trial
            for trial in trials
            if (trial_statuses is None or trial.status in trial_statuses)
        ]

        return filtered_by_status

    @retry_on_exception(retries=3, no_retry_on_exception_types=NO_RETRY_EXCEPTIONS)
    def stop_trial_runs(
        self, trials: list[BaseTrial], reasons: list[str | None] | None = None
    ) -> None:
        """Stops the jobs that execute given trials.

        Used if, for example, TTL for a trial was specified and expired, or poor
        early results suggest the trial is not worth running to completion.

        Override default implementation on the ``Runner`` if its desirable to stop
        trials in bulk.

        Args:
            trials: Trials to be stopped.
            reasons: A list of strings describing the reasons for why the
                trials are to be stopped (in the same order).
        """
        if len(trials) == 0:
            return

        if reasons is None:
            reasons = [None] * len(trials)

        for trial, reason in zip(trials, reasons):
            runner = self.runner_for_trial_type(trial_type=trial.trial_type)
            if runner is None:
                raise RunnerNotFoundError(
                    "Unable to stop trial runs: Runner not configured "
                    "for experiment or trial."
                )
            runner.stop(trial=trial, reason=reason)
            trial.mark_early_stopped()

    def _attach_trial(self, trial: BaseTrial, index: int | None = None) -> int:
        """Attach a trial to this experiment.

        Should only be called within the trial constructor.

        Args:
            trial: The trial to be attached.
            index: If specified, the trial's index will be set accordingly.
                This should generally not be specified, as the index
                will be automatically determined based on the number
                of existing trials. This is only used for the purpose
                of loading from storage.

        Returns:
            The index of the trial within the experiment's trial list.
        """

        if trial.experiment is not self:
            raise ValueError("BatchTrial does not belong to this experiment.")

        for existing_trial in self._trials.values():
            if existing_trial is trial:
                raise ValueError("BatchTrial already attached to experiment.")

        if index is not None and index in self._trials:
            logger.debug(
                f"Trial index {index} already exists on the experiment. Overwriting."
            )
        index = (
            index
            if index is not None
            else (0 if len(self._trials) == 0 else max(self._trials.keys()) + 1)
        )
        self._trials[index] = trial
        return index

    def validate_trials(self, trials: Iterable[BaseTrial]) -> None:
        """Raise ValueError if any of the trials in the input are not from
        this experiment.
        """
        for t in trials:
            if t.experiment is not self:
                raise ValueError(
                    "All trials must be from the input experiment (name: "
                    f"{self.name}), but trial #{t.index} is from "
                    f"another experiment (name: {t.experiment.name})."
                )

    def warm_start_from_old_experiment(
        self,
        old_experiment: Experiment,
        copy_run_metadata_keys: list[str] | None = None,
        trial_statuses_to_copy: list[TrialStatus] | None = None,
        search_space_check_membership_raise_error: bool = True,
    ) -> list[Trial]:
        """Copy all completed trials with data from an old Ax experiment to this one.
        This function checks that the parameters of each trial are members of the
        current experiment's search_space.

        NOTE: Currently only handles experiments with 1-arm ``Trial``-s, not
        ``BatchTrial``-s as there has not yet been need for support of the latter.

        Args:
            old_experiment: The experiment from which to transfer trials and data
            copy_run_metadata_keys: A list of keys denoting which items to copy over
                from each trial's run_metadata. Defaults to copying all run_metadata.
            trial_statuses_to_copy: All trials with a status in this list will be
                copied. By default, copies all ``RUNNING``, ``COMPLETED``,
                ``ABANDONED``, and ``EARLY_STOPPED`` trials.
            search_space_check_membership_raise_error: Whether to raise an exception
                if the warm started trials being imported fall outside of the
                defined search space.

        Returns:
            List of trials successfully copied from old_experiment to this one
        """
        if len(self.trials) > 0:
            raise ValueError(
                f"Can only warm-start experiments that don't yet have trials. "
                f"Experiment {self._name} has {len(self.trials)} trials."
            )

        old_parameter_names = set(old_experiment.search_space.parameters.keys())
        parameter_names = set(self.search_space.parameters.keys())
        if old_parameter_names.symmetric_difference(parameter_names):
            raise ValueError(
                f"Cannot warm-start experiment '{self._name}' from experiment "
                f"'{old_experiment._name}' due to mismatch in search space parameters."
                f"Parameters in '{self._name}' but not in '{old_experiment._name}': "
                f"{old_parameter_names - parameter_names}. Vice-versa: "
                f"{parameter_names - old_parameter_names}."
            )

        trial_statuses_to_copy = (
            trial_statuses_to_copy
            if trial_statuses_to_copy is not None
            else DEFAULT_STATUSES_TO_WARM_START
        )

        warm_start_trials = [
            trial
            for trial in old_experiment.trials.values()
            if trial.status in trial_statuses_to_copy
        ]
        copied_trials = []
        for trial in warm_start_trials:
            if not isinstance(trial, Trial):
                raise NotImplementedError(
                    "Only experiments with 1-arm trials currently supported."
                )
            self.search_space.check_membership(
                none_throws(trial.arm).parameters,
                raise_error=search_space_check_membership_raise_error,
            )
            dat = old_experiment.lookup_data_for_trial(trial_index=trial.index)
            # Set trial index and arm name to their values in new trial.
            new_trial = self.new_trial()
            add_arm_and_prevent_naming_collision(
                new_trial=new_trial,
                old_trial=trial,
                old_experiment_name=old_experiment._name,
            )
            new_trial.mark_running(no_runner_required=True)
            new_trial._properties["source"] = (
                f"Warm start from Experiment: `{old_experiment._name}`, "
                f"trial: `{trial.index}`"
            )
            # Associates a generation_model_key to the new trial.
            generation_model_key = trial._properties.get("generation_model_key")
            if generation_model_key is None and trial.generator_run is not None:
                generation_model_key = trial.generator_run._generator_key or "Manual"
            new_trial._properties["generation_model_key"] = generation_model_key

            # Copy all run_metadata by default.
            if copy_run_metadata_keys is None:
                copy_run_metadata_keys = list(trial.run_metadata.keys())

            for run_metadata_field in copy_run_metadata_keys:
                new_trial.update_run_metadata(
                    {run_metadata_field: trial.run_metadata.get(run_metadata_field)}
                )
            # Trial has data, so we replicate it on the new experiment.
            has_data = not dat.df.empty
            if has_data:
                new_df = dat.full_df.copy()
                new_df["trial_index"].replace(
                    {trial.index: new_trial.index}, inplace=True
                )
                new_df["arm_name"].replace(
                    {none_throws(trial.arm).name: none_throws(new_trial.arm).name},
                    inplace=True,
                )
                # Attach updated data to new trial on experiment.
                old_data = old_experiment.default_data_constructor(df=new_df)
                self.attach_data(data=old_data)
            if trial.status == TrialStatus.ABANDONED:
                new_trial.mark_abandoned(reason=trial.abandoned_reason)
            elif trial.status is not TrialStatus.RUNNING:
                new_trial.mark_as(trial.status)
            copied_trials.append(new_trial)

        if self._name is not None:
            logger.debug(
                f"Copied {len(copied_trials)} completed trials and their data "
                f"from {old_experiment._name} to {self._name}."
            )
        else:
            logger.debug(
                f"Copied {len(copied_trials)} completed trials and their data "
                f"from {old_experiment._name}."
            )

        return copied_trials

    def _name_and_store_arm_if_not_exists(
        self, arm: Arm, proposed_name: str, replace: bool = False
    ) -> None:
        """Tries to lookup arm with same signature, otherwise names and stores it.

        - Looks up if arm already exists on experiment
            - If so, name the input arm the same as the existing arm
            - else name the arm with given name and store in _arms_by_signature

        Args:
            arm: The arm object to name.
            proposed_name: The name to assign if it doesn't have one already.
            replace: If true, override arm w/ same name and different signature.
                If false, raise an error if this conflict occurs.
        """

        # If arm is identical to an existing arm, return that
        # so that the names match.
        if arm.signature in self.arms_by_signature:
            existing_arm = self.arms_by_signature[arm.signature]
            if arm.has_name:
                if arm.name != existing_arm.name:
                    raise ValueError(
                        f"Arm already exists with name {existing_arm.name}, "
                        f"which doesn't match given arm name of {arm.name}."
                    )
            else:
                arm.name = existing_arm.name
        else:
            if not arm.has_name:
                arm.name = proposed_name

            # Check for signature conflict by arm name/proposed name
            if (
                arm.name in self.arms_by_name
                and arm.signature != self.arms_by_name[arm.name].signature
            ):
                error_msg = (
                    f"Arm with name {arm.name} already exists on experiment "
                    + "with different signature."
                )
                if replace:
                    logger.warning(f"{error_msg} Replacing the existing arm. ")
                else:
                    raise AxError(error_msg)

            # Add the new arm
            self._register_arm(arm)

    def _register_arm(self, arm: Arm) -> None:
        """Add a new arm to the experiment, updating the relevant
        lookup dictionaries.

        Args:
            arm: Arm to add
        """
        self._arms_by_signature[arm.signature] = arm
        self._arms_by_name[arm.name] = arm

    def _check_TTL_on_candidate_trials(self) -> None:
        """Checks whether any past-TTL trials are still marked as `CANDIDATE`
        and marks them as stale if so.

        NOTE: this function just calls `trial.status` for each trial, as the
        computation of that property checks the TTL for trials.
        """
        if not self._trials_have_ttl:
            return

        # The trial status changes during TTL check modifies the original set
        # _trial_indices_by_status, hence create a copy of indices for iteration
        candidate_indices = self._trial_indices_by_status[TrialStatus.CANDIDATE].copy()

        for idx in candidate_indices:
            self._trials[idx].status  # `status` property checks TTL if applicable.

    def _cache_metric_fetch_error(
        self, trial_index: int, metric_name: str, metric_fetch_e: MetricFetchE | None
    ) -> None:
        """Caches a given metric fetch error to the experiment.
        Args:
            trial_index: Index of the trial that encountered the metric fetch error.
            metric_name: Name of the metric that failed to be fetched.
            metric_fetch_e: The metric fetch error to cache.
        Returns:
            None
        """
        error_data = {
            "trial_index": trial_index,
            "metric_name": metric_name,
            "reason": "",
            "timestamp": datetime.now().isoformat(),
            "traceback": "",
        }

        if metric_fetch_e is not None:
            reason_for_failure = metric_fetch_e.message
            if metric_fetch_e.exception is not None:
                exception_str = (
                    f"{type(metric_fetch_e.exception).__name__}: "
                    f"{metric_fetch_e.exception}"
                )
                reason_for_failure = (
                    f"Ran into the following exception: {exception_str}"
                )

            error_data["reason"] = reason_for_failure
            error_data["traceback"] = metric_fetch_e.tb_str() or ""

        self._metric_fetching_errors[(trial_index, metric_name)] = error_data

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({self._name})"

    # --- MultiTypeExperiment convenience functions ---
    #
    # Certain functionalities have special behavior for multi-type experiments.
    # This defines the base behavior for regular experiments that will be
    # overridden in the MultiTypeExperiment class.

    @property
    def default_trial_type(self) -> str | None:
        """Default trial type assigned to trials in this experiment.

        In the base experiment class this is always None. For experiments
        with multiple trial types, use the MultiTypeExperiment class.
        """
        return self._default_trial_type

    def runner_for_trial_type(self, trial_type: str | None) -> Runner | None:
        """The default runner to use for a given trial type.

        Looks up the appropriate runner for this trial type in the trial_type_to_runner.
        """
        if not self.supports_trial_type(trial_type):
            raise ValueError(f"Trial type `{trial_type}` is not supported.")
        if (runner := self._trial_type_to_runner.get(trial_type)) is None:
            return self.runner  # return the default runner
        return runner

    def supports_trial_type(self, trial_type: str | None) -> bool:
        """Whether this experiment allows trials of the given type.

        The base experiment class only supports None. For experiments
        with multiple trial types, use the MultiTypeExperiment class.
        """
        return (
            trial_type is None
            # We temporarily allow "short run" and "long run" trial
            # types in single-type experiments during development of
            # a new ``GenerationStrategy`` that needs them.
            or trial_type == Keys.SHORT_RUN
            or trial_type == Keys.LONG_RUN
        )

    def attach_trial(
        self,
        parameterizations: list[TParameterization],
        arm_names: list[str] | None = None,
        should_add_status_quo_arm: bool = False,
        ttl_seconds: int | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, TParameterization], int]:
        """Attach a new trial with the given parameterization to the experiment.

        Args:
            parameterizations: List of parameterization for the new trial. If
                only one is provided a single-arm Trial is created. If multiple
                arms are provided a BatchTrial is created.
            arm_names: Names of arm(s) in the new trial.
            should_add_status_quo_arm: If True, adds the status quo arm to the trial
                with a weight of 1.0. If False, the _status_quo is still set on the
                trial for tracking purposes, but without a weight it will not be an
                Arm present on the trial
            ttl_seconds: If specified, will consider the trial stale after this
                many seconds. Used to detect dead trials that did not complete.
            run_metadata: Metadata to attach to the trial.

        Returns:
            Tuple of arm name to parameterization dict, and trial index from
            newly created trial.
        """

        # If more than one parameterization is provided,
        # proceed with a batch trial
        is_batch = len(parameterizations) > 1

        # Validate search space membership for all parameterizations
        for parameterization in parameterizations:
            try:
                self.search_space.validate_membership(parameters=parameterization)
            except ValueError as e:
                # To not raise on out-of-design parameterizations
                if "is not a valid value for parameter" in str(e):
                    warnings.warn(
                        f"Parameterization {parameterization} is in out-of-design. "
                        "Ax will still attach the trial for use in candidate "
                        "generation.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    raise e

        # Validate number of arm names if any arm names are provided.
        named_arms = False
        arm_names = arm_names or []
        if len(arm_names) > 0:
            named_arms = True
            if len(arm_names) != len(parameterizations):
                raise UserInputError(
                    f"Number of arm names ({len(arm_names)} "
                    "does not match number of parameterizations "
                    f"({len(parameterizations)})."
                )

        # Prepare arm(s) to be added to the trial created later
        arms = [
            Arm(
                parameters=parameterization,
                name=arm_names[i] if named_arms else None,
            )
            for i, parameterization in enumerate(parameterizations)
        ]

        # Create the trial and add arm(s)
        trial = None
        if is_batch:
            trial = self.new_batch_trial(
                ttl_seconds=ttl_seconds,
                should_add_status_quo_arm=should_add_status_quo_arm,
            ).add_arms_and_weights(arms=arms)

        else:
            # If search space is hierarchical, we need to store dummy values of
            # parameters that are not in the arm (but are in flattened
            # search space), as metadata, so later we are able to make the
            # data for this arm "complete" in the flattened search space.
            candidate_metadata = None
            if self.search_space.is_hierarchical:
                candidate_metadata = self.search_space.cast_observation_features(
                    observation_features=self.search_space.flatten_observation_features(
                        observation_features=observation.ObservationFeatures(
                            parameters=parameterizations[0]
                        ),
                        inject_dummy_values_to_complete_flat_parameterization=True,
                    )
                ).metadata

            trial = self.new_trial(ttl_seconds=ttl_seconds).add_arm(
                arms[0],
                candidate_metadata=candidate_metadata,
            )

        trial.mark_running(no_runner_required=True)

        logger.debug(
            "Attached custom parameterizations "
            f"{round_floats_for_logging(item=parameterizations)} "
            f"as trial {trial.index}."
        )

        if run_metadata is not None:
            trial.update_run_metadata(metadata=run_metadata)

        return {arm.name: arm.parameters for arm in trial.arms}, trial.index

    def clone_with(
        self,
        search_space: SearchSpace | None = None,
        name: str | None = None,
        optimization_config: OptimizationConfig | None = None,
        tracking_metrics: list[Metric] | None = None,
        runner: Runner | None = None,
        status_quo: Arm | None = None,
        description: str | None = None,
        is_test: bool | None = None,
        properties_to_keep: list[str] | None = None,
        trial_indices: list[int] | None = None,
        data: Data | None = None,
        clear_trial_type: bool = False,
    ) -> Experiment:
        r"""
        Return a copy of this experiment with some attributes replaced.

        NOTE: This method only retains the latest data attached to the experiment.
        This is the same data that would be accessed using common APIs such as
        ``Experiment.lookup_data()``.

        Args:
            search_space: New search space. If None, it uses the cloned search space
                of the original experiment.
            name: New experiment name. If None, it adds cloned_experiment_  prefix
                to the original experiment name.
            optimization_config: New optimization config. If None, it clones the same
                optimization_config from the orignal experiment.
            tracking_metrics: New list of metrics to track. If None, it clones the
                tracking metrics already attached to the main experiment.
            runner: New runner. If None, it clones the existing runner.
            status_quo: New status quo arm. If None, it clones the existing status quo.
            description: New description. If None, it uses the same description.
            is_test: Whether the cloned experiment should be considered a test. If None,
                it uses the same value.
            properties_to_keep: List of property keys to retain in the cloned
                experiment. Defaults to ["owners"].
            trial_indices: If specified, only clones the specified trials. If None,
                clones all trials.
            data: If specified, attach this data to the cloned experiment. If None,
                clones the latest data attached to the original experiment if
                the experiment has any data.
            clear_trial_type: If True, all cloned trials on the cloned experiment have
                `trial_type` set to `None`.
        """
        if properties_to_keep is None:
            properties_to_keep = ["owners"]
        search_space = (
            self.search_space.clone() if (search_space is None) else search_space
        )
        name = (
            "cloned_experiment_" + self.name
            if (name is None and self.name is not None)
            else name
        )
        optimization_config = (
            self.optimization_config.clone()
            if (optimization_config is None and self.optimization_config is not None)
            else optimization_config
        )
        tracking_metrics = (
            [m.clone() for m in self.tracking_metrics]
            if (tracking_metrics is None and self.tracking_metrics is not None)
            else tracking_metrics
        )

        runner = (
            self.runner.clone()
            if (runner is None and self.runner is not None)
            else runner
        )
        status_quo = (
            self.status_quo.clone()
            if (status_quo is None and self.status_quo is not None)
            else status_quo
        )
        description = self.description if description is None else description
        is_test = self.is_test if is_test is None else is_test

        properties = {
            k: v for k, v in self._properties.items() if k in properties_to_keep
        }
        dropped_keys = set(self._properties.keys()) - set(properties.keys())
        if dropped_keys:
            logger.warning(
                f"When cloning the experiment, the following fields were dropped from "
                f"properties: {', '.join(dropped_keys)}.",
            )

        cloned_experiment = Experiment(
            search_space=search_space,
            name=name,
            optimization_config=optimization_config,
            tracking_metrics=tracking_metrics,
            runner=runner,
            status_quo=status_quo,
            description=description,
            is_test=is_test,
            experiment_type=self.experiment_type,
            properties=properties,
            default_data_type=self._default_data_type,
        )

        # Clone only the specified trials.
        original_trial_indices = self.trials.keys()
        trial_indices_to_keep = (
            set(original_trial_indices) if trial_indices is None else set(trial_indices)
        )
        if trial_indices_diff := trial_indices_to_keep.difference(
            original_trial_indices
        ):
            warnings.warn(
                f"Trials indexed with {trial_indices_diff} are not a part "
                "of the original experiment. ",
                stacklevel=2,
            )

        data_by_trial = {}
        for trial_index in trial_indices_to_keep.intersection(original_trial_indices):
            trial = self.trials[trial_index]
            if not isinstance(trial, (Trial, BatchTrial)):
                raise NotImplementedError(f"Cloning of {type(trial)} is not supported.")
            new_trial = trial.clone_to(
                cloned_experiment, clear_trial_type=clear_trial_type
            )
            new_index = new_trial.index
            if (
                trial_index in self._data_by_trial
                and len(trial_data_dict := self._data_by_trial[trial_index]) > 0
            ):
                timestamp = max(trial_data_dict.keys())
                # Clone the data to avoid overwriting the original in the DB.
                trial_data = trial_data_dict[timestamp].clone()
                trial_data.df["trial_index"] = new_index
                data_by_trial[new_index] = OrderedDict([(timestamp, trial_data)])
        if data is not None:
            # If user passed in data, use it.
            cloned_experiment.attach_data(data.clone())
        else:
            # Otherwise, attach the data extracted from the original experiment.
            cloned_experiment._data_by_trial = data_by_trial

        return cloned_experiment

    @property
    def metric_config_summary_df(self) -> pd.DataFrame:
        """Creates a dataframe with information about each metric in the
        experiment. The resulting dataframe has one row per metric, and the
        following columns:
            - Name: the name of the metric.
            - Type: the metric subclass (e.g., Metric, BraninMetric).
            - Goal: the goal for this for this metric, based on the optimization
              config (minimize, maximize, constraint or track).
            - Bound: the bound of this metric (e.g., "<=10.0") if it is being used
              as part of an ObjectiveThreshold or OutcomeConstraint.
            - Lower is Better: whether the user prefers this metric to be lower,
              if provided.

        """
        records = {}
        for metric_name in self.metrics.keys():
            m = self.metrics[metric_name]
            records[m.name] = m.summary_dict
        if self.optimization_config is not None:
            opt_config = self.optimization_config
            if self.is_moo_problem:
                multi_objective = assert_is_instance(
                    opt_config.objective, MultiObjective
                )
                objectives = multi_objective.objectives
            else:
                objectives = [opt_config.objective]
            for objective in objectives:
                records[objective.metric.name][METRIC_DF_COLNAMES["goal"]] = (
                    "minimize" if objective.minimize else "maximize"
                )

            for constraint in opt_config.all_constraints:
                if not isinstance(constraint, ObjectiveThreshold):
                    records[constraint.metric.name][METRIC_DF_COLNAMES["goal"]] = (
                        "constrain"
                    )
                op = ">= " if constraint.op == ComparisonOp.GEQ else "<= "
                relative = "%" if constraint.relative else ""
                records[constraint.metric.name][METRIC_DF_COLNAMES["bound"]] = (
                    f"{op}{constraint.bound}{relative}"
                )

        for metric in self.tracking_metrics or []:
            records[metric.name][METRIC_DF_COLNAMES["goal"]] = "track"

        # Sort metrics rows by purpose and name, then sort columns.
        df = pd.DataFrame(records.values()).fillna(value="None")
        df.rename(columns=METRIC_DF_COLNAMES, inplace=True)
        df[METRIC_DF_COLNAMES["goal"]] = pd.Categorical(
            df[METRIC_DF_COLNAMES["goal"]],
            categories=["minimize", "maximize", "constrain", "track", "None"],
            ordered=True,
        )
        df = df.sort_values(
            by=[METRIC_DF_COLNAMES["goal"], METRIC_DF_COLNAMES["name"]]
        ).reset_index()
        # Reorder columns.
        df = df[
            [
                colname
                for colname in METRIC_DF_COLNAMES.values()
                if colname in df.columns
            ]
        ]
        return df

    def to_df(
        self,
        trial_indices: Iterable[int] | None = None,
        trial_statuses: Sequence[TrialStatus] | None = None,
        omit_empty_columns: bool = True,
        relativize: bool = False,
    ) -> pd.DataFrame:
        """
        High-level summary of the Experiment with one row per arm. Any values missing at
        compute time will be represented as None. Columns where every value is None will
        be omitted by default.

        The DataFrame computed will contain one row per arm and the following columns:
            - trial_index: The trial index of the arm
            - arm_name: The name of the arm
            - trial_status: The status of the trial (e.g. RUNNING, SUCCEDED, FAILED)
            - failure_reason: The reason for the failure, if applicable
            - generation_node: The name of the ``GenerationNode`` that generated the arm
            - **METADATA: Any metadata associated with the trial, as specified by the
                Experiment's runner.run_metadata_report_keys field
            - **METRIC_NAME: The observed mean of the metric specified, for each metric
            - **PARAMETER_NAME: The parameter value for the arm, for each parameter

        Args:
            trial_indices: If specified, only include these trial indices.
            omit_empty_columns: If True, omit columns where every value is None.
            trial_status: If specified, only include trials with this status.
            relativize: If True and:
                * experiment has a status quo on all of its ``BatchTrial``-s
                * OR a status quo trial among its ``Trial``-s,
                , relativize metrics against the status quo.
        """

        records = []
        data = self.lookup_data(trial_indices=trial_indices)

        # Relativize metrics if requested
        if relativize:
            if self.status_quo is None:
                raise UserInputError(
                    "Attempting to relativize the experiment data, however, "
                    "the experiment status quo is None. Please set the experiment "
                    "status quo, or set `relativize` = False"
                )

            data_df = data.relativize(
                status_quo_name=self.status_quo.name,
                as_percent=True,
                include_sq=True,
            ).df
        else:
            data_df = data.df

        # Filter trials by trial_indices and trial_statuses
        trials = self.extract_relevant_trials(
            trial_indices=list(trial_indices) if trial_indices is not None else None,
            trial_statuses=trial_statuses,
        )
        # Iterate through trials, and for each trial, iterate through its arms
        # and add a record for each arm.
        for trial in trials:
            for arm in trial.arms:
                # Find the observed means for each metric, placing None if not found
                observed_means = {}
                for metric in self.metrics.keys():
                    try:
                        observed_means[metric] = data_df[
                            (data_df["trial_index"] == trial.index)
                            & (data_df["arm_name"] == arm.name)
                            & (data_df["metric_name"] == metric)
                        ]["mean"].item()
                    except (ValueError, KeyError):
                        # ValueError if there is no row for the (trial, arm, metric).
                        # KeyError if the df is empty and missing one of the columns.
                        observed_means[metric] = None

                # Find the arm's associated generation method from the trial via the
                # GeneratorRuns if possible
                grs = [gr for gr in trial.generator_runs if arm in gr.arms]
                generation_node = grs[0]._generation_node_name if len(grs) > 0 else None

                # Find other metadata from the trial to include from the trial based
                # on the experiment's runner
                metadata = (
                    {
                        key: value
                        for key, value in trial.run_metadata.items()
                        if key in none_throws(self.runner).run_metadata_report_keys
                    }
                    if self.runner is not None
                    else {}
                )

                # Construct the record
                record = {
                    "trial_index": trial.index,
                    "arm_name": arm.name,
                    "trial_status": trial.status.name,
                    "fail_reason": trial.run_metadata.get("fail_reason", None),
                    "generation_node": generation_node,
                    **metadata,
                    **observed_means,
                    **arm.parameters,
                }

                records.append(record)

        df = pd.DataFrame(records)
        if omit_empty_columns:
            df = df.loc[:, df.notnull().any()]

        # Format metric columns as percentages with 4 significant figures when
        # relativized
        if relativize:
            for metric_name in self.metrics.keys():
                if metric_name in df.columns:
                    df[metric_name] = df[metric_name].apply(
                        lambda x: (
                            f"{x:.4g}%"
                            if pd.notna(x) and x != 0.0
                            else ("0%" if pd.notna(x) else None)
                        )
                    )

        return df

    def add_auxiliary_experiment(
        self,
        purpose: AuxiliaryExperimentPurpose,
        auxiliary_experiment: AuxiliaryExperiment,
    ) -> None:
        """Add a (non-duplicated) auxiliary experiment to this experiment.

        This method adds the auxiliary experiment as the first element in the list
        of auxiliary experiments with the specified purpose. If the auxiliary is
        already present, it is moved to the first position in the list.

        Args:
            purpose: The purpose of the auxiliary experiment.
            auxiliary_experiment: The auxiliary experiment to add.
        """
        if purpose not in self.auxiliary_experiments_by_purpose:
            # if no aux experiment, make aux the first one
            self.auxiliary_experiments_by_purpose[purpose] = [auxiliary_experiment]
            return

        # Add or move auxiliary_experiment to be the first element
        # Adding to the first and use the order as a default tie-breaker when multiple
        # auxiliary experiments are present but only one is going to be used.
        self.auxiliary_experiments_by_purpose[purpose] = [auxiliary_experiment] + [
            item
            for item in self.auxiliary_experiments_by_purpose[purpose]
            if item != auxiliary_experiment
        ]

    def find_auxiliary_experiment_by_name(
        self,
        purpose: AuxiliaryExperimentPurpose,
        auxiliary_experiment_name: str,
        raise_if_not_found: bool = False,
    ) -> AuxiliaryExperiment | None:
        """Find the aux experiment with the given name and purpose in the experiment.

        Args:
            purpose: The purpose of the aux experiment.
            auxiliary_experiment_name: The name of the aux experiment.

        Returns:
            The aux experiment with the given name and purpose, or None if not found.
            if raise_if_not_found is True, raises a ValueError if not found.
        """
        found_aux_exp = None
        if purpose in self.auxiliary_experiments_by_purpose:
            for auxiliary_experiment in self.auxiliary_experiments_by_purpose[purpose]:
                if auxiliary_experiment.experiment.name == auxiliary_experiment_name:
                    found_aux_exp = auxiliary_experiment
                    break

        if raise_if_not_found:
            if found_aux_exp is None:
                raise UserInputError(
                    f"Auxiliary experiment {auxiliary_experiment_name} is not "
                    f"found for purpose {purpose}."
                )
        return found_aux_exp

    def validate_auxiliary_experiment(
        self,
        source_experiment: Experiment,
        purpose: AuxiliaryExperimentPurpose,
    ) -> AuxiliaryExperimentValidation:
        """Validate a source auxiliary experiment against the current experiment as a
           target experiment based on a given purpose.

        Args:
            source_experiment: The source experiment to validate.
            purpose: The purpose of the auxiliary experiment.

        Returns:
            An AuxiliaryExperimentValidation object containing the validation result.
        """

        match purpose:
            case AuxiliaryExperimentPurpose.TRANSFERABLE_EXPERIMENT:
                overlapping_parameters = (
                    source_experiment.search_space.get_overlapping_parameters(
                        self.search_space
                    )
                )
                is_valid = len(overlapping_parameters) > 0
                invalid_reason = None if is_valid else "No overlapping parameters."
                return AuxiliaryExperimentValidation(
                    is_valid=is_valid,
                    invalid_reason=invalid_reason,
                    metadata=TransferLearningMetadata(
                        overlap_parameters=overlapping_parameters,
                    ),
                )
            case _:
                return AuxiliaryExperimentValidation(
                    is_valid=False,
                    invalid_reason="Validation not supported for auxiliary "
                    f"experiment purpose: {purpose}",
                )

    @property
    def auxiliary_experiments_by_purpose_for_storage(
        self,
    ) -> dict[AuxiliaryExperimentPurpose, list[AuxiliaryExperiment]]:
        """Tracks removed auxiliary experiments to be stored as inactive auxiliary
        experiments."""
        # Start with the current active auxiliary experiments.
        result = {
            purpose: list(auxiliary_experiments)
            for (
                purpose,
                auxiliary_experiments,
            ) in self.auxiliary_experiments_by_purpose.items()
        }
        # Iterate through the auxiliary experiments that were loaded and mark any
        # deleted ones as inactive.
        for (
            purpose,
            prev_auxiliary_experiments,
        ) in self._initial_auxiliary_experiments_by_purpose.items():
            # If the purpose is not in the new auxiliary experiments, mark all
            # previous auxiliary experiments as inactive.
            if purpose not in result:
                for pre_experiment in prev_auxiliary_experiments:
                    pre_experiment.is_active = False
                result[purpose] = prev_auxiliary_experiments
                continue
            # Mark any removed auxiliary experiments as inactive.
            for prev_auxiliary_experiment in prev_auxiliary_experiments:
                if prev_auxiliary_experiment not in result[purpose]:
                    prev_auxiliary_experiment.is_active = False
                    result[purpose].append(prev_auxiliary_experiment)
        return result


def add_arm_and_prevent_naming_collision(
    new_trial: Trial, old_trial: Trial, old_experiment_name: str | None = None
) -> None:
    # Add all of an old trial's arms to a new trial. Rename any arm with auto-generated
    # naming format to prevent naming collisions during warm-start. If an old
    # experiment name is provided, append that to the original arm name. Else, clear
    # the arm name. Preserves all names not matching the automatic naming format.
    # experiment is not named, clear the arm's name.
    # `arm_index` is 0 since all trials are single-armed.
    old_arm_name = none_throws(old_trial.arm).name
    has_default_name = bool(old_arm_name == old_trial._get_default_name(arm_index=0))
    if has_default_name:
        new_arm = none_throws(old_trial.arm).clone(clear_name=True)
        if old_experiment_name is not None:
            new_arm.name = f"{old_arm_name}_{old_experiment_name}"
        new_trial.add_arm(new_arm)
    else:
        try:
            new_trial.add_arm(none_throws(old_trial.arm).clone(clear_name=False))
        except ValueError as e:
            warnings.warn(
                f"Attaching arm {old_trial.arm} to trial {new_trial} while preserving "
                f"its name failed with error: {e}. Retrying with `clear_name=True`.",
                stacklevel=2,
            )
            new_trial.add_arm(none_throws(old_trial.arm).clone(clear_name=True))
