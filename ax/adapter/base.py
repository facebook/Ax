#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import time
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from logging import Logger
from typing import Any

import numpy as np
import pandas as pd
from ax.adapter.data_utils import (
    DataLoaderConfig,
    ExperimentData,
    extract_experiment_data,
)
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.cast import Cast
from ax.adapter.transforms.fill_missing_parameters import FillMissingParameters
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import extract_arm_predictions, GeneratorRun
from ax.core.map_data import MAP_KEY
from ax.core.observation import (
    Observation,
    ObservationData,
    ObservationFeatures,
    recombine_observations,
)
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import (
    TCandidateMetadata,
    TModelCov,
    TModelMean,
    TModelPredict,
    TParameterization,
)
from ax.core.utils import get_target_trial_index, has_map_metrics
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.exceptions.model import AdapterMethodNotImplementedError, ModelError
from ax.generators.base import Generator
from ax.generators.types import TConfig
from ax.utils.common.logger import get_logger
from botorch.settings import validate_input_scaling
from pandas import DataFrame
from pyre_extensions import assert_is_instance, none_throws

logger: Logger = get_logger(__name__)


@dataclass(frozen=True)
class BaseGenArgs:
    search_space: SearchSpace
    optimization_config: OptimizationConfig | None
    pending_observations: dict[str, list[ObservationFeatures]]
    fixed_features: ObservationFeatures


@dataclass(frozen=True)
class GenResults:
    observation_features: list[ObservationFeatures]
    weights: list[float]
    best_observation_features: ObservationFeatures | None = None
    gen_metadata: dict[str, Any] = field(default_factory=dict)


class Adapter:
    """The main object for using generators in Ax.

    Adapter specifies 3 methods for using generators:

    - predict: Make predictions with the generator. This method is not optimized for
      speed and so should be used primarily for plotting or similar tasks
      and not inside an optimization loop.
    - gen: Use the generator to generate new candidates.
    - cross_validate: Do cross validation to assess generator predictions.

    Adapter converts Ax types like Data and Arm to types that are
    meant to be consumed by the generators. The data sent to the generator will depend
    on the implementation of the subclass, which will specify the actual API
    for external generator.

    This class also applies a sequence of transforms to the input data and
    problem specification which can be used to ensure that the external generator
    receives appropriate inputs.

    Subclasses will implement what is here referred to as the "terminal
    transform", which is a transform that changes types of the data and problem
    specification.
    """

    def __init__(
        self,
        *,
        experiment: Experiment,
        generator: Generator,
        search_space: SearchSpace | None = None,
        data: Data | None = None,
        transforms: Sequence[type[Transform]] | None = None,
        transform_configs: Mapping[str, TConfig] | None = None,
        optimization_config: OptimizationConfig | None = None,
        expand_model_space: bool = True,
        fit_tracking_metrics: bool = True,
        fit_on_init: bool = True,
        data_loader_config: DataLoaderConfig | None = None,
        # fit_out_of_design, fit_abandoned, and fit_only_completed_map_metrics
        # were deprecated in Ax 1.0.0, so they can now be reaped.
        fit_out_of_design: bool | None = None,
        fit_abandoned: bool | None = None,
        fit_only_completed_map_metrics: bool | None = None,
    ) -> None:
        """
        Applies transforms and fits the generator.

        Args:
            experiment: An ``Experiment`` object representing the setup and the
                current state of the experiment, including the search space,
                trials and observation data. It is used to extract various
                attributes, and is not mutated.
            generator: A ``Generator`` that is used for generating candidates.
                Its interface will be specified in subclasses. If generator requires
                initialization, that should be done prior to its use here.
            search_space: An optional ``SearchSpace`` for fitting the generator.
                If not provided, `experiment.search_space` is used.
                The search space may be modified during ``Adapter.gen``, e.g.,
                to try out a different set of parameter bounds or constraints.
                The bounds of the ``RangeParameter``s are considered soft and
                will be expanded to match the range of the data sent in for fitting,
                if `expand_model_space` is True.
            data: An optional ``Data`` object, containing mean and SEM observations.
                If `None`, extracted using `experiment.lookup_data()`.
            transforms: List of uninitialized transform classes. Forward
                transforms will be applied in this order, and untransforms in
                the reverse order.
            transform_configs: A dictionary from transform name to the
                transform config dictionary.
            optimization_config: An optional ``OptimizationConfig`` defining how to
                optimize the generator. Defaults to `experiment.optimization_config`.
            expand_model_space: If True, expand range parameter bounds in model
                space to cover given training data. This will make the modeling
                space larger than the search space if training data fall outside
                the search space. Will also include training points that violate
                parameter constraints in the modeling.
            fit_tracking_metrics: Whether to fit a surrogate model for tracking metrics.
                Setting this to False will improve runtime at the expense of
                models not being available for predicting tracking metrics.
                NOTE: This can only be set to False when the optimization config
                is provided.
            fit_on_init: Whether to fit the generator on initialization. This can
                be used to skip generator fitting when a fitted generator is not needed.
                To fit the generator afterwards, use `_process_and_transform_data`
                to get the transformed inputs and call `_fit_if_implemented` with
                the transformed inputs.
            data_loader_config: A DataLoaderConfig of options for loading data. See the
                docstring of DataLoaderConfig for more details.
            fit_out_of_design: Deprecation warning: `fit_out_of_design` is deprecated.
                Overwrites `data_loader_config.fit_out_of_design` if not None.
            fit_abandoned: Deprecation warning: `fit_out_of_design` is deprecated.
                Overwrites `data_loader_config.fit_abandoned` if not None.
            fit_only_completed_map_metrics: Deprecation warning: `fit_out_of_design`
                is deprecated. If not None, overwrites
                `data_loader_config.fit_only_completed_map_metrics`.
        """
        if data_loader_config is None:
            data_loader_config = DataLoaderConfig()
        self._data_loader_config: DataLoaderConfig = (
            _legacy_overwrite_data_loader_config(
                data_loader_config=data_loader_config,
                fit_out_of_design=fit_out_of_design,
                fit_abandoned=fit_abandoned,
                fit_only_completed_map_metrics=fit_only_completed_map_metrics,
            )
        )

        t_fit_start = time.monotonic()
        transforms = transforms or []
        transforms = [Cast] + list(transforms)
        transform_configs = {} if transform_configs is None else transform_configs
        if "FillMissingParameters" in transform_configs:
            transforms = [FillMissingParameters] + transforms
        self._raw_transforms = transforms
        self._transform_configs: Mapping[str, TConfig] = transform_configs

        self.fit_time: float = 0.0
        self.fit_time_since_gen: float = 0.0
        self._metric_signatures: set[str] = set()
        # pyre-ignore [13] Assigned in _set_and_filter_training_data.
        self._training_data: ExperimentData
        self._optimization_config: OptimizationConfig | None = optimization_config
        self._training_in_design_idx: list[bool] = []
        self._status_quo: Observation | None = None
        self._status_quo_name: str | None = None
        self.transforms: MutableMapping[str, Transform] = OrderedDict()
        self._model_key: str | None = None
        self._model_kwargs: dict[str, Any] | None = None
        self._bridge_kwargs: dict[str, Any] | None = None
        # The space used for optimization.
        search_space = search_space or experiment.search_space
        self._search_space: SearchSpace = search_space.clone()
        # The space used for modeling. Might be larger than the optimization
        # space to cover training data.
        self._model_space: SearchSpace = search_space.clone()
        self._fit_tracking_metrics = fit_tracking_metrics
        self.outcomes: list[str] = []
        self._experiment_has_immutable_search_space_and_opt_config: bool = (
            experiment is not None and experiment.immutable_search_space_and_opt_config
        )
        self._experiment_properties: dict[str, Any] = experiment._properties
        self._experiment: Experiment = experiment

        if self._optimization_config is None:
            self._optimization_config = experiment.optimization_config
        self._arms_by_signature: dict[str, Arm] = experiment.arms_by_signature

        if self._fit_tracking_metrics is False:
            if self._optimization_config is None:
                raise UserInputError(
                    "Optimization config is required when "
                    "`fit_tracking_metrics` is False."
                )
            self.outcomes = sorted(self._optimization_config.metrics.keys())

        # Set training data (in the raw / untransformed space). This also omits
        # out-of-design and abandoned observations depending on the corresponding flags.
        experiment_data = extract_experiment_data(
            experiment=experiment,
            data_loader_config=self._data_loader_config,
            data=data,
        )
        if expand_model_space:
            self._set_model_space(arm_data=experiment_data.arm_data)
        experiment_data = self._set_and_filter_training_data(
            experiment_data=experiment_data, search_space=self._model_space
        )

        # Set model status quo.
        # NOTE: training data must be set before setting the status quo.
        self._set_status_quo(experiment=experiment)

        # Save generator, apply terminal transform, and fit.
        self.generator = generator
        if fit_on_init:
            experiment_data, search_space = self._transform_data(
                experiment_data=experiment_data,
                search_space=self._model_space,
                transforms=self._raw_transforms,
                transform_configs=self._transform_configs,
            )
            self._fit_if_implemented(
                search_space=search_space,
                experiment_data=experiment_data,
                time_so_far=time.monotonic() - t_fit_start,
            )

    @property
    def can_predict(self) -> bool:
        """Whether this adapter can predict outcomes for new parameterizations."""
        return self.generator.can_predict

    @property
    def can_model_in_sample(self) -> bool:
        """Whether this adapter can model (e.g. apply shrinkage) on observed
        parameterizations (in this case, it needs to support calling `predict`()
        on points in the training data / provided during `fit()`)."""
        return self.generator.can_model_in_sample

    def _fit_if_implemented(
        self,
        search_space: SearchSpace,
        experiment_data: ExperimentData,
        time_so_far: float,
    ) -> None:
        r"""Fits the generator if `_fit` is implemented and stores fit time.

        Args:
            search_space: A transformed search space for fitting the generator.
            experiment_data: The ``ExperimentData`` to fit the generator on, with
                the transforms already applied.
            time_so_far: Time spent in initializing the generator up to
                `_fit_if_implemented` call.
        """
        try:
            t_fit_start = time.monotonic()
            self._fit(search_space=search_space, experiment_data=experiment_data)
            increment = time.monotonic() - t_fit_start + time_so_far
            self.fit_time += increment
            self.fit_time_since_gen += increment
        except AdapterMethodNotImplementedError:
            pass

    def _process_and_transform_data(
        self, experiment: Experiment, data: Data | None = None
    ) -> tuple[ExperimentData, SearchSpace]:
        r"""Processes the data into ``ExperimentData`` and returns the transformed
        ``ExperimentData`` and the search space. This packages the following methods:
        * self._set_and_filter_training_data
        * self._set_status_quo
        * self._transform_data
        """
        experiment_data = extract_experiment_data(
            experiment=experiment,
            data_loader_config=self._data_loader_config,
            data=data,
        )
        experiment_data = self._set_and_filter_training_data(
            experiment_data=experiment_data, search_space=self._model_space
        )
        # This ensures that SQ is up to date when we re-fit the existing Adapter
        # in GeneratorSpec.fit.
        self._set_status_quo(experiment=experiment)
        return self._transform_data(
            experiment_data=experiment_data,
            search_space=self._model_space,
            transforms=self._raw_transforms,
            transform_configs=self._transform_configs,
        )

    def _transform_data(
        self,
        experiment_data: ExperimentData,
        search_space: SearchSpace,
        transforms: Sequence[type[Transform]] | None,
        transform_configs: Mapping[str, TConfig],
        assign_transforms: bool = True,
    ) -> tuple[ExperimentData, SearchSpace]:
        """Initialize transforms and apply them to provided data."""
        search_space = search_space.clone()
        if transforms is not None:
            for t in transforms:
                t_instance = t(
                    search_space=search_space.clone(),
                    experiment_data=experiment_data,
                    adapter=self,
                    config=transform_configs.get(t.__name__, None),
                )
                search_space = t_instance.transform_search_space(
                    search_space=search_space
                )
                experiment_data = t_instance.transform_experiment_data(
                    experiment_data=experiment_data
                )
                if assign_transforms:
                    self.transforms[t.__name__] = t_instance
        return experiment_data, search_space

    def _set_and_filter_training_data(
        self, experiment_data: ExperimentData, search_space: SearchSpace
    ) -> ExperimentData:
        """Store non-transformed training data, and return it after filtering to
        include only in-design points if
        ``self._data_loader_config._fit_out_of_design=True``.
        """
        # NOTE: This is copied in get_training_data, so it won't be modified in-place.
        self._training_data = experiment_data
        self._metric_signatures: set[str] = set(experiment_data.metric_signatures)
        # Filter out-of-design points if `fit_out_of_design` is False.
        if self._data_loader_config.fit_out_of_design:
            self._training_in_design_idx = [True] * len(experiment_data.arm_data)
        else:
            self._training_in_design_idx = self._compute_in_design(
                search_space=search_space, experiment_data=experiment_data
            )
        return self.get_training_data(
            filter_in_design=self._data_loader_config.fit_out_of_design
        )

    def _compute_in_design(
        self,
        search_space: SearchSpace,
        experiment_data: ExperimentData,
    ) -> list[bool]:
        """Compute in-design status for each row of ``experiment_data``, after
        filling missing values if ``FillMissingParameters`` transform is used.
        """
        if "FillMissingParameters" in self._transform_configs:
            t = FillMissingParameters(
                config=self._transform_configs["FillMissingParameters"]
            )
            experiment_data = t.transform_experiment_data(
                experiment_data=experiment_data,
            )
        # TODO [T230585235]: Implement more efficient membership checks.
        return [
            search_space.check_membership(
                parameterization={k: v for k, v in params.items() if not pd.isnull(v)}
            )
            for params in experiment_data.arm_data.drop(
                # Ignoring errors which can be raised by missing metadata column
                # when the data is empty.
                columns=["metadata"],
                inplace=False,
                errors="ignore",
            ).to_dict(orient="records")
        ]

    def _set_model_space(self, arm_data: DataFrame) -> None:
        """Set model space, possibly expanding range parameters to cover data."""
        # If fill for missing values, include those in expansion.
        fill_values: TParameterization | None = None
        if "FillMissingParameters" in self._transform_configs:
            fill_values = self._transform_configs[  # pyre-ignore[9]
                "FillMissingParameters"
            ].get("fill_values", None)
        # Update model space. Expand bounds as needed to cover the values found
        # in the data. Only applies to range parameters.
        for p_name, p in self._model_space.parameters.items():
            if not isinstance(p, RangeParameter):
                continue
            if p_name in arm_data:
                param_vals = arm_data[p_name].dropna().tolist()
            else:
                param_vals = []
            if fill_values is not None and p_name in fill_values:
                param_vals.append(fill_values[p_name])
            if len(param_vals) == 0:
                continue
            p.lower = min(p.lower, min(param_vals))
            p.upper = max(p.upper, max(param_vals))
        # Remove parameter constraints from the model space.
        self._model_space.set_parameter_constraints([])

    def _set_status_quo(self, experiment: Experiment) -> None:
        """Set the status quo by extracting it from the experiment.
        The ``experiment.status_quo`` is an Arm that contains the parameterization
        and the name of the status quo arm. This method extracts the target
        trial index from the experiment, then matches the parameterization and
        trial index to the training data to make a status quo ``Observation``,
        complete with the parameterization, trial index, and data.

        NOTE: The status quo will not be set if the target trial index is None.
        If there are multiple observations for the status quo arm in the training
        data for the target trial index, we check for the map keys of the map
        metrics in the optimization config. If there is a single map key, the
        observation with the maximal map key value in the metadata will be used.
        If there are multiple map keys, the status quo will not be set.

        Args:
            experiment: The experiment to extract the status quo from.
        """
        self._status_quo: Observation | None = None  # reset the SQ.
        status_quo_arm = experiment.status_quo
        if status_quo_arm is None:
            self._status_quo_name = None
            return
        self._status_quo_name = status_quo_arm.name
        target_trial_index = get_target_trial_index(experiment=experiment)
        if target_trial_index is None:
            logger.warning(
                f"Status quo {self._status_quo_name} is not present in the "
                "training data."
            )
            return
        # Filter the training data to find the observations for status quo.
        sq_arm_observations = self._training_data.filter_by_arm_names(
            arm_names=[none_throws(self.status_quo_name)]
        ).convert_to_list_of_observations()
        status_quo_observations = [
            obs
            for obs in sq_arm_observations
            if obs.features.trial_index == target_trial_index
        ]
        if len(status_quo_observations) == 0:
            logger.warning(
                f"Status quo {self._status_quo_name} is not present in the "
                "training data."
            )
            return
        elif len(status_quo_observations) > 1:
            if self._optimization_config is None:
                logger.warning(
                    f"Status quo {self._status_quo_name} was found in the data with "
                    "multiple observations, and the Adapter does not have an "
                    "optimization config. `Adapter.status_quo` will not be set."
                )
                return

            if has_map_metrics(optimization_config=self._optimization_config):
                self._status_quo = _combine_multiple_status_quo_observations(
                    status_quo_observations=status_quo_observations,
                    metrics=set(none_throws(self._optimization_config).metrics),
                )
            else:
                logger.warning(
                    f"Status quo {self._status_quo_name} was found in the data with "
                    "multiple observations, and the optimization config does not "
                    "include any MapMetrics. `Adapter.status_quo` will not be "
                    "set."
                )
        else:
            self._status_quo = status_quo_observations[-1]

    @property
    def status_quo_data_by_trial(self) -> dict[int, ObservationData] | None:
        """A map of trial index to the status quo observation data of each trial.

        If status quo does not exist, return None.
        """
        # TODO: We could possibly extract from the experiment directly. See D72685267.
        # Status quo name will be set if status quo exists. We can just filter by name.
        if self.status_quo_name is None:
            return None
        # Identify status quo data by arm name.
        obs_data = self._training_data.observation_data
        sq_data = obs_data.loc[
            obs_data.index.get_level_values("arm_name") == self.status_quo_name
        ]
        metric_signatures = list(sq_data["mean"].columns)
        return {
            index[0]: ObservationData(
                metric_signatures=metric_signatures,
                means=row["mean"].to_numpy(),
                covariance=np.diag(row["sem"].to_numpy() ** 2),
            )
            for index, row in sq_data.iterrows()
        }

    @property
    def status_quo(self) -> Observation | None:
        """Observation corresponding to status quo, if any."""
        return self._status_quo

    @property
    def status_quo_name(self) -> str | None:
        """Name of status quo, if any."""
        return self._status_quo_name

    @property
    def metric_signatures(self) -> set[str]:
        """Metric signatures present in training data."""
        return self._metric_signatures

    @property
    def model_space(self) -> SearchSpace:
        """SearchSpace used to fit model."""
        return self._model_space

    def get_training_data(self, filter_in_design: bool = False) -> ExperimentData:
        """A copy of the (untransformed) data with which the generator was fit.

        Args:
            filter_in_design: If True, the data is filtered by
                ``self.training_in_design``. Note that this will include all
                points if ``self._data_loader_config.fit_out_of_design is True``,
                since all points will be marked as in-design.
        """
        experiment_data = deepcopy(self._training_data)
        if not filter_in_design or self._data_loader_config.fit_out_of_design:
            return experiment_data
        arm_data = experiment_data.arm_data.loc[self.training_in_design]
        obs_data = experiment_data.observation_data
        # In-design-ness is determined by the arm parameterization. So, we can
        # just filter out the observation data by the arms that are in-design.
        # We can't just use `self.training_in_design`, since there may be multiple
        # rows of observations for the same arm.
        obs_data = obs_data[
            obs_data.index.get_level_values("arm_name").isin(
                arm_data.index.get_level_values("arm_name")
            )
        ]
        return ExperimentData(arm_data=arm_data, observation_data=obs_data)

    @property
    def training_in_design(self) -> list[bool]:
        """For each observation in the training data, a bool indicating if it
        is in-design for the generator.
        """
        return self._training_in_design_idx

    def _fit(
        self,
        search_space: SearchSpace,
        experiment_data: ExperimentData,
    ) -> None:
        """Apply terminal transform and fit the generator."""
        raise AdapterMethodNotImplementedError(
            f"{self.__class__.__name__} does not implement `_fit`."
        )

    def _predict_observation_data(
        self,
        observation_features: list[ObservationFeatures],
        use_posterior_predictive: bool = False,
        untransform: bool = True,
    ) -> list[ObservationData]:
        """
        Like 'predict' method, but returns results as a list of ObservationData

        Args:
            observation_features: A list of observation features to predict.
            use_posterior_predictive: A boolean indicating if the predictions
                should be from the posterior predictive (i.e. including
                observation noise).
                This option is only supported by the ``BoTorchGenerator``.
            untransform: Whether to untransform the predictions to the original
                scale before returning.

        Returns:
            List of `ObservationData`, each representing (independent) predictions
            for the corresponding `ObservationFeatures` input.
        """
        input_len = len(observation_features)
        observation_features = deepcopy(observation_features)
        # Transform
        for t in self.transforms.values():
            observation_features = t.transform_observation_features(
                observation_features=observation_features
            )
        # Apply terminal transform and predict
        observation_data = self._predict(
            observation_features=observation_features,
            use_posterior_predictive=use_posterior_predictive,
        )
        if untransform:
            # Apply reverse transforms, in reverse order
            pred_observations = recombine_observations(
                observation_features=observation_features,
                observation_data=observation_data,
            )

            for t in reversed(list(self.transforms.values())):
                pred_observations = t.untransform_observations(pred_observations)
            observation_data = [obs.data for obs in pred_observations]
        if (output_len := len(observation_data)) != input_len:
            raise ModelError(
                f"Predictions resulted in fewer outcomes ({output_len}) than "
                f"expected ({input_len}). This can happen if a transform modifies "
                "the number of observation features, such as `Cast` dropping any "
                "observation features with `None` parameter values. "
            )
        return observation_data

    def predict(
        self,
        observation_features: list[ObservationFeatures],
        use_posterior_predictive: bool = False,
    ) -> TModelPredict:
        """Make model predictions (mean and covariance) for the given
        observation features.

        Predictions are made for all outcomes.
        If an out-of-design observation can successfully be transformed,
        the predicted value will be returned.
        Otherwise, we will attempt to find that observation in the training data
        and return the raw value.

        Args:
            observation_features: A list of observation features to predict.
            use_posterior_predictive: A boolean indicating if the predictions
                should be from the posterior predictive (i.e. including
                observation noise).
                This option is only supported by the ``BoTorchGenerator``.

        Returns:
            2-element tuple containing

            - Dictionary from metric name to list of mean estimates, in same
              order as observation_features.
            - Nested dictionary with cov['metric1']['metric2'] a list of
              cov(metric1@x, metric2@x) for x in observation_features.
        """
        # Make sure that input is a list of ObservationFeatures. If you pass in
        # arms, the code runs but it doesn't apply the transforms.
        if not all(
            isinstance(obsf, ObservationFeatures) for obsf in observation_features
        ):
            raise UserInputError(
                "Input to predict must be a list of `ObservationFeatures`."
            )
        observation_data = self._predict_observation_data(
            observation_features=observation_features,
            use_posterior_predictive=use_posterior_predictive,
        )
        f, cov = unwrap_observation_data(observation_data)
        return f, cov

    def _predict(
        self,
        observation_features: list[ObservationFeatures],
        use_posterior_predictive: bool = False,
    ) -> list[ObservationData]:
        """Apply terminal transform, predict, and reverse terminal transform on
        output.
        """
        raise AdapterMethodNotImplementedError(
            f"{self.__class__.__name__} does not implement `_predict`."
        )

    def _get_transformed_gen_args(
        self,
        search_space: SearchSpace,
        optimization_config: OptimizationConfig | None = None,
        pending_observations: dict[str, list[ObservationFeatures]] | None = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> BaseGenArgs:
        if pending_observations is None:
            pending_observations = {}
        if optimization_config is None:
            optimization_config = (
                self._optimization_config.clone()
                if self._optimization_config is not None
                else None
            )
        else:
            if not self._fit_tracking_metrics:
                # Check that the optimization config has the same metrics as
                # the original one. Otherwise, we may attempt to optimize over
                # metrics that do not have a fitted model.
                outcomes = set(optimization_config.metrics.keys())
                if not outcomes.issubset(self.outcomes):
                    raise UnsupportedError(
                        "When fit_tracking_metrics is False, the optimization config "
                        "can only include metrics that were included in the "
                        "optimization config used while initializing the Adapter. "
                        f"Metrics {outcomes} is not a subset of {self.outcomes}."
                    )
            optimization_config = optimization_config.clone()

        pending_observations = deepcopy(pending_observations)
        fixed_features = (
            ObservationFeatures(parameters={})
            if fixed_features is None
            else fixed_features.clone()
        )
        search_space = search_space.clone()

        # Transform
        for t in self.transforms.values():
            search_space = t.transform_search_space(search_space)
            if optimization_config is not None:
                optimization_config = t.transform_optimization_config(
                    optimization_config=optimization_config,
                    adapter=self,
                    fixed_features=fixed_features,
                )
            for metric, po in pending_observations.items():
                pending_observations[metric] = t.transform_observation_features(po)
            if not isinstance(t, FillMissingParameters):
                (fixed_features,) = t.transform_observation_features([fixed_features])
        return BaseGenArgs(
            search_space=search_space,
            optimization_config=optimization_config,
            pending_observations=pending_observations,
            fixed_features=fixed_features,
        )

    def _validate_gen_inputs(
        self,
        n: int,
        search_space: SearchSpace | None = None,
        optimization_config: OptimizationConfig | None = None,
        pending_observations: dict[str, list[ObservationFeatures]] | None = None,
        fixed_features: ObservationFeatures | None = None,
        model_gen_options: TConfig | None = None,
    ) -> None:
        """Validate inputs to `Adapter.gen`.

        Currently, this is only used to ensure that `n` is a positive integer.
        """
        if n < 1:
            raise UserInputError(
                f"Attempted to generate n={n} points. Number of points to generate "
                "must be a positive integer."
            )

    def gen(
        self,
        n: int,
        search_space: SearchSpace | None = None,
        optimization_config: OptimizationConfig | None = None,
        pending_observations: dict[str, list[ObservationFeatures]] | None = None,
        fixed_features: ObservationFeatures | None = None,
        model_gen_options: TConfig | None = None,
    ) -> GeneratorRun:
        """
        Generate new points from the underlying generator according to
        search_space, optimization_config and other parameters.

        Args:
            n: Number of points to generate
            search_space: Search space
            optimization_config: Optimization config
            pending_observations: A map from metric name to pending
                observations for that metric.
            fixed_features: An ObservationFeatures object containing any
                features that should be fixed at specified values during
                generation.
            model_gen_options: A config dictionary that is passed along to the
                generator. See `TorchOptConfig` for details.

        Returns:
            A GeneratorRun object that contains the generated points and other metadata.
        """
        self._validate_gen_inputs(
            n=n,
            search_space=search_space,
            optimization_config=optimization_config,
            pending_observations=pending_observations,
            fixed_features=fixed_features,
            model_gen_options=model_gen_options,
        )
        t_gen_start = time.monotonic()
        # Get modifiable versions
        if search_space is None:
            search_space = self._search_space
        orig_search_space = search_space.clone()
        base_gen_args = self._get_transformed_gen_args(
            search_space=search_space,
            optimization_config=optimization_config,
            pending_observations=pending_observations,
            fixed_features=fixed_features,
        )
        # Apply terminal transform and gen
        gen_results = self._gen(
            n=n,
            search_space=base_gen_args.search_space,
            optimization_config=base_gen_args.optimization_config,
            pending_observations=base_gen_args.pending_observations,
            fixed_features=base_gen_args.fixed_features,
            model_gen_options=model_gen_options,
        )

        observation_features = gen_results.observation_features
        best_obsf = gen_results.best_observation_features

        # Apply reverse transforms
        for t in reversed(list(self.transforms.values())):
            observation_features = t.untransform_observation_features(
                observation_features
            )
            if best_obsf is not None:
                best_obsf = t.untransform_observation_features([best_obsf])[0]

        # Clamp the untransformed data to the original search space if
        # we don't fit/gen OOD points
        if not self._data_loader_config.fit_out_of_design:
            observation_features = clamp_observation_features(
                observation_features, orig_search_space
            )
            if best_obsf is not None:
                best_obsf = clamp_observation_features([best_obsf], orig_search_space)[
                    0
                ]
        best_point_predictions = None
        try:
            model_predictions = self.predict(observation_features)
            if best_obsf is not None:
                best_point_predictions = extract_arm_predictions(
                    model_predictions=self.predict([best_obsf]), arm_idx=0
                )
        except Exception as e:
            logger.debug(f"Generator predictions failed with error {e}.")
            model_predictions = None

        if best_obsf is None:
            best_arm = None
        else:
            best_arms, _ = gen_arms(
                observation_features=[best_obsf],
                arms_by_signature=self._arms_by_signature,
            )
            best_arm = best_arms[0]

        arms, candidate_metadata = gen_arms(
            observation_features=observation_features,
            arms_by_signature=self._arms_by_signature,
        )

        # If experiment has immutable search space and metrics, no need to
        # save them on generator runs.
        immutable = getattr(
            self, "_experiment_has_immutable_search_space_and_opt_config", False
        )
        optimization_config = None if immutable else base_gen_args.optimization_config
        # Remove information about the objective thresholds - we do not want to save
        # these as `ObjectiveThreshold` objects, as this causes storage headaches.
        gen_metadata = gen_results.gen_metadata
        gen_metadata.pop("objective_thresholds", None)

        generator_run = GeneratorRun(
            arms=arms,
            weights=gen_results.weights,
            optimization_config=optimization_config,
            search_space=None if immutable else orig_search_space,
            model_predictions=model_predictions,
            best_arm_predictions=(
                None if best_arm is None else (best_arm, best_point_predictions)
            ),
            fit_time=self.fit_time_since_gen,
            gen_time=time.monotonic() - t_gen_start,
            model_key=self._model_key,
            model_kwargs=self._model_kwargs,
            bridge_kwargs=self._bridge_kwargs,
            gen_metadata=gen_metadata,
            model_state_after_gen=self._get_serialized_model_state(),
            candidate_metadata_by_arm_signature=candidate_metadata,
        )
        if len(generator_run.arms) < n:
            logger.warning(
                f"{self} was not able to generate {n} unique candidates. "
                "Generated arms have the following weights, as there are repeats:\n"
                f"{generator_run.weights}"
            )
        self.fit_time_since_gen = 0.0
        return generator_run

    def _gen(
        self,
        n: int,
        search_space: SearchSpace,
        optimization_config: OptimizationConfig | None,
        pending_observations: dict[str, list[ObservationFeatures]],
        fixed_features: ObservationFeatures | None,
        model_gen_options: TConfig | None,
    ) -> GenResults:
        """Apply terminal transform, gen, and reverse terminal transform on
        output.
        """
        raise AdapterMethodNotImplementedError(
            f"{self.__class__.__name__} does not implement `_gen`."
        )

    def cross_validate(
        self,
        cv_training_data: ExperimentData,
        cv_test_points: list[ObservationFeatures],
        use_posterior_predictive: bool = False,
    ) -> list[ObservationData]:
        """Make a set of cross-validation predictions.

        Args:
            cv_training_data: The training data to use for cross validation.
            cv_test_points: The test points at which predictions will be made.
            use_posterior_predictive: A boolean indicating if the predictions
                should be from the posterior predictive (i.e. including
                observation noise).

        Returns:
            A list of predictions at the test points.
        """
        # Apply transforms to cv_training_data and cv_test_points
        cv_training_data, cv_test_points, search_space = self._transform_inputs_for_cv(
            cv_training_data=cv_training_data, cv_test_points=cv_test_points
        )

        # Apply terminal transform, and get predictions.
        # Since each CV fold removes points from the training data, the
        # remaining observations will not pass the input scaling checks.
        # To avoid confusing users with warnings, we disable these checks.
        with validate_input_scaling(False):
            cv_predictions = self._cross_validate(
                search_space=search_space,
                cv_training_data=cv_training_data,
                cv_test_points=cv_test_points,
                use_posterior_predictive=use_posterior_predictive,
            )
        # Apply reverse transforms, in reverse order
        cv_test_observations = [
            Observation(features=obsf, data=cv_predictions[i])
            for i, obsf in enumerate(cv_test_points)
        ]

        for t in reversed(list(self.transforms.values())):
            cv_test_observations = t.untransform_observations(cv_test_observations)
        return [obs.data for obs in cv_test_observations]

    def _cross_validate(
        self,
        search_space: SearchSpace,
        cv_training_data: ExperimentData,
        cv_test_points: list[ObservationFeatures],
        use_posterior_predictive: bool = False,
    ) -> list[ObservationData]:
        """Apply the terminal transform, make predictions on the test points,
        and reverse terminal transform on the results.
        """
        raise AdapterMethodNotImplementedError(
            f"{self.__class__.__name__} does not implement `_cross_validate`."
        )

    def _transform_inputs_for_cv(
        self,
        cv_training_data: ExperimentData,
        cv_test_points: list[ObservationFeatures],
    ) -> tuple[ExperimentData, list[ObservationFeatures], SearchSpace]:
        """Apply transforms to cv_training_data and cv_test_points,
        and return cv_training_data, cv_test_points, and search space in
        transformed space. This is to prepare data to be used in _cross_validate.

        Args:
            cv_training_data: The training data to use for cross validation.
            cv_test_points: The test points at which predictions will be made.

        Returns:
            cv_training_data, cv_test_points, and search_space in transformed space.
        """
        cv_test_points = deepcopy(cv_test_points)
        cv_training_data = deepcopy(cv_training_data)
        search_space = self._model_space.clone()
        for t in self.transforms.values():
            cv_training_data = t.transform_experiment_data(
                experiment_data=cv_training_data
            )
            cv_test_points = t.transform_observation_features(
                observation_features=cv_test_points
            )
            search_space = t.transform_search_space(search_space=search_space)
        return cv_training_data, cv_test_points, search_space

    def _set_kwargs_to_save(
        self,
        model_key: str,
        model_kwargs: dict[str, Any],
        bridge_kwargs: dict[str, Any],
    ) -> None:
        """Set properties used to save the model that created a given generator
        run, on the `GeneratorRun` object. Each generator run produced by the
        `gen` method of this adapter will have the model key and kwargs
        fields set as provided in arguments to this function.
        """
        self._model_key = model_key
        self._model_kwargs = model_kwargs
        self._bridge_kwargs = bridge_kwargs

    def _get_serialized_model_state(self) -> dict[str, Any]:
        """Obtains the state of the underlying generator (if using a stateful one)
        in a readily JSON-serializable form.
        """
        return self.generator.serialize_state(raw_state=self.generator._get_state())

    def _deserialize_model_state(
        self, serialized_state: dict[str, Any]
    ) -> dict[str, Any]:
        return self.generator.deserialize_state(serialized_state=serialized_state)

    def feature_importances(self, metric_signature: str) -> dict[str, float]:
        """Computes feature importances for a single metric.

        Depending on the type of the generator, this method will approach sensitivity
        analysis (calculating the sensitivity of the metric to changes in the search
        space's parameters, a.k.a. features) differently.

        For Bayesian optimization generators (BoTorch generators), this method uses
        parameter inverse lengthscales to compute normalized feature importances.

        NOTE: Currently, this is only implemented for GP-based generators.

        Args:
            metric_signatures: Signature of metric to compute feature importances for.

        Returns:
            A dictionary mapping parameter names to their corresponding feature
            importances.

        """
        raise AdapterMethodNotImplementedError(
            f"{self.__class__.__name__} does not implement `feature_importances`."
        )

    def transform_observations(self, observations: list[Observation]) -> Any:
        """Applies transforms to given observation features and returns them in the
        model space.

        Args:
            observation_features: ObservationFeatures to be transformed.

        Returns:
            Transformed values. This could be e.g. a torch Tensor, depending
            on the Adapter subclass.
        """
        observations = deepcopy(observations)
        for t in self.transforms.values():
            observations = t.transform_observations(observations)
        # Apply terminal transform and return
        return self._transform_observations(observations)

    def _transform_observations(self, observations: list[Observation]) -> Any:
        """Apply terminal transform to given observations and return result."""
        raise AdapterMethodNotImplementedError(
            f"{self.__class__.__name__} does not implement `_transform_observations`."
        )

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> Any:
        """Applies transforms to given observation features and returns them in the
        model space.

        Args:
            observation_features: ObservationFeatures to be transformed.

        Returns:
            Transformed values. This could be e.g. a torch Tensor, depending
            on the Adapter subclass.
        """
        obsf = deepcopy(observation_features)
        for t in self.transforms.values():
            obsf = t.transform_observation_features(obsf)
        # Apply terminal transform and return
        return self._transform_observation_features(obsf)

    def _transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> Any:
        """Apply terminal transform to given observation features and return result."""
        raise AdapterMethodNotImplementedError(
            f"{self.__class__.__name__} does not implement "
            "`_transform_observation_features`."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(generator={self.generator})"


def unwrap_observation_data(observation_data: list[ObservationData]) -> TModelPredict:
    """Converts observation data to the format for model prediction outputs.
    That format assumes each observation data has the same set of metrics.
    """
    metrics = set(observation_data[0].metric_signatures)
    f: TModelMean = {metric: [] for metric in metrics}
    cov: TModelCov = {m1: {m2: [] for m2 in metrics} for m1 in metrics}
    for od in observation_data:
        if set(od.metric_signatures) != metrics:
            raise ValueError(
                "Each ObservationData should use same set of metrics. "
                "Expected {exp}, got {got}.".format(
                    exp=metrics, got=set(od.metric_signatures)
                )
            )
        for i, m1 in enumerate(od.metric_signatures):
            f[m1].append(od.means[i])
            for j, m2 in enumerate(od.metric_signatures):
                cov[m1][m2].append(od.covariance[i, j])
    return f, cov


def gen_arms(
    observation_features: list[ObservationFeatures],
    arms_by_signature: dict[str, Arm] | None = None,
) -> tuple[list[Arm], dict[str, TCandidateMetadata] | None]:
    """Converts observation features to a tuple of arms list and candidate metadata
    dict, where arm signatures are mapped to their respective candidate metadata.
    """
    # TODO(T34225939): handle static context (which is stored on observation_features)
    arms = []
    candidate_metadata = {}
    for of in observation_features:
        arm = Arm(parameters=of.parameters)
        if arms_by_signature is not None and arm.signature in arms_by_signature:
            existing_arm = arms_by_signature[arm.signature]
            arm = Arm(name=existing_arm.name, parameters=existing_arm.parameters)
        arms.append(arm)
        if of.metadata:
            candidate_metadata[arm.signature] = of.metadata
    return arms, candidate_metadata or None  # None if empty cand. metadata.


def clamp_observation_features(
    observation_features: list[ObservationFeatures], search_space: SearchSpace
) -> list[ObservationFeatures]:
    range_parameters = [
        p for p in search_space.parameters.values() if isinstance(p, RangeParameter)
    ]
    for obsf in observation_features:
        for p in range_parameters:
            if p.name not in obsf.parameters:
                continue
            if p.parameter_type == ParameterType.FLOAT:
                val = assert_is_instance(obsf.parameters[p.name], float)
            else:
                val = assert_is_instance(obsf.parameters[p.name], int)
            if val < p.lower:
                logger.debug(
                    f"Untransformed parameter {val} "
                    f"less than lower bound {p.lower}, clamping"
                )
                obsf.parameters[p.name] = p.lower
            elif val > p.upper:
                logger.debug(
                    f"Untransformed parameter {val} "
                    f"greater than upper bound {p.upper}, clamping"
                )
                obsf.parameters[p.name] = p.upper
    return observation_features


def _legacy_overwrite_data_loader_config(
    data_loader_config: DataLoaderConfig,
    fit_out_of_design: bool | None = None,
    fit_abandoned: bool | None = None,
    fit_only_completed_map_metrics: bool | None = None,
    warn_if_legacy: bool = True,
) -> DataLoaderConfig:
    """Overwrites data loader config with legacy keyword arguments.

    Args:
        data_loader_config: Data loader config.
        fit_out_of_design: Whether to fit out-of-design points.
        fit_abandoned: Whether to fit abandoned arms.
        fit_only_completed_map_metrics: Whether to fit only completed map metrics.
        warn_if_legacy: Whether to warn if legacy keyword arguments are used.

    Returns:
        Updated data loader config.
    """
    data_loader_config_dict = {}
    for var_name, deprecated_var in (
        ("fit_out_of_design", fit_out_of_design),
        ("fit_abandoned", fit_abandoned),
        ("fit_only_completed_map_metrics", fit_only_completed_map_metrics),
    ):
        if deprecated_var is not None:
            if warn_if_legacy:
                logger.warning(
                    f"`{var_name}` is deprecated. Please pass as "
                    f"`data_loader_config.{var_name}` instead."
                )
            data_loader_config_dict[var_name] = deprecated_var
        else:
            data_loader_config_dict[var_name] = getattr(data_loader_config, var_name)

    data_loader_config = DataLoaderConfig(
        latest_rows_per_group=data_loader_config.latest_rows_per_group,
        limit_rows_per_metric=data_loader_config.limit_rows_per_metric,
        limit_rows_per_group=data_loader_config.limit_rows_per_group,
        **data_loader_config_dict,
    )
    return data_loader_config


def _combine_multiple_status_quo_observations(
    status_quo_observations: list[Observation],
    metrics: set[str],
) -> Observation | None:
    """Finds the maximal (in terms of map key value) observation for each metric
    in `status_quo_observations`, and combines them into a single ``Observation``
    object, representing the status quo observations for all metrics.

    NOTE: The resulting ``ObservationFeatures`` will not have any ``metadata``.
    If there are multiple ``Observation``s for the status quo, this is due to
    them having different ``metadata``, so we discard it here to avoid having
    misleading or incomplete information for some metrics.

    Args:
        status_quo_observations: List of observations for the status quo
            arm at target trial index. Extracted in ``Adapter._set_status_quo``.
        metrics: The metrics to include in the combined observation.
            This should include all metrics in the optimization config.

    Returns:
        A single ``Observation`` object that includes the maximal observation
        for each metric.
    """
    # Pick the observation with maximal map key value.
    partial_obs = [
        max(
            status_quo_observations,
            key=lambda obs: obs.features.metadata[MAP_KEY],
        )
    ]
    # Check if the it includes all metrics in the opt config.
    # If not, search for observations of the remaining metrics as well.
    while remaining_metrics := metrics.difference(
        sum((obs.data.metric_signatures for obs in partial_obs), [])
    ):
        # Find observations of the remaining metrics.
        # Search using one metric at a time.
        lookup_metric = remaining_metrics.pop()
        obs_w_lookup_metric = [
            obs
            for obs in status_quo_observations
            if lookup_metric in obs.data.metric_signatures
        ]
        if len(obs_w_lookup_metric) == 0:
            logger.warning(
                f"Could not find observations of metric {lookup_metric} for the "
                f"status quo {status_quo_observations[0].arm_name} in the training "
                "data. `Adapter.status_quo` will not be set."
            )
            return
        partial_obs.append(
            max(
                obs_w_lookup_metric,
                key=lambda obs: obs.features.metadata[MAP_KEY],
            )
        )
    # Combine into a single Observation object.
    return Observation(
        features=ObservationFeatures(
            parameters=partial_obs[0].features.parameters,
            trial_index=partial_obs[0].features.trial_index,
            # NOTE: omitting the metadata since it can be different in each obs.
        ),
        data=ObservationData(
            metric_signatures=sum(
                (obs.data.metric_signatures for obs in partial_obs), []
            ),
            means=np.concatenate([obs.data.means for obs in partial_obs], axis=0),
            covariance=np.diag(
                np.concatenate(
                    [np.diag(obs.data.covariance) for obs in partial_obs], axis=0
                )
            ),
        ),
        arm_name=partial_obs[0].arm_name,
    )
