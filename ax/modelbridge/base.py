#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import time
import warnings
from abc import ABC
from collections import OrderedDict
from collections.abc import MutableMapping
from copy import deepcopy
from dataclasses import dataclass, field

from logging import Logger
from typing import Any

from ax.core.arm import Arm
from ax.core.base_trial import NON_ABANDONED_STATUSES, TrialStatus
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import extract_arm_predictions, GeneratorRun
from ax.core.observation import (
    Observation,
    ObservationData,
    ObservationFeatures,
    observations_from_data,
    recombine_observations,
    separate_observations,
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
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.exceptions.model import ModelBridgeMethodNotImplementedError
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.cast import Cast
from ax.modelbridge.transforms.fill_missing_parameters import FillMissingParameters
from ax.models.types import TConfig
from ax.utils.common.logger import get_logger
from botorch.exceptions.warnings import InputDataWarning
from pyre_extensions import assert_is_instance, none_throws

logger: Logger = get_logger(__name__)


@dataclass(frozen=True)
class BaseGenArgs:
    search_space: SearchSpace
    optimization_config: OptimizationConfig | None
    pending_observations: dict[str, list[ObservationFeatures]]
    fixed_features: ObservationFeatures | None


@dataclass(frozen=True)
class GenResults:
    observation_features: list[ObservationFeatures]
    weights: list[float]
    best_observation_features: ObservationFeatures | None = None
    gen_metadata: dict[str, Any] = field(default_factory=dict)


class ModelBridge(ABC):  # noqa: B024 -- ModelBridge doesn't have any abstract methods.
    """The main object for using models in Ax.

    ModelBridge specifies 3 methods for using models:

    - predict: Make model predictions. This method is not optimized for
      speed and so should be used primarily for plotting or similar tasks
      and not inside an optimization loop.
    - gen: Use the model to generate new candidates.
    - cross_validate: Do cross validation to assess model predictions.

    ModelBridge converts Ax types like Data and Arm to types that are
    meant to be consumed by the models. The data sent to the model will depend
    on the implementation of the subclass, which will specify the actual API
    for external model.

    This class also applies a sequence of transforms to the input data and
    problem specification which can be used to ensure that the external model
    receives appropriate inputs.

    Subclasses will implement what is here referred to as the "terminal
    transform," which is a transform that changes types of the data and problem
    specification.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        model: Any,
        transforms: list[type[Transform]] | None = None,
        experiment: Experiment | None = None,
        data: Data | None = None,
        transform_configs: dict[str, TConfig] | None = None,
        status_quo_name: str | None = None,
        status_quo_features: ObservationFeatures | None = None,
        optimization_config: OptimizationConfig | None = None,
        expand_model_space: bool = True,
        fit_out_of_design: bool = False,
        fit_abandoned: bool = False,
        fit_tracking_metrics: bool = True,
        fit_on_init: bool = True,
    ) -> None:
        """
        Applies transforms and fits model.

        Args:
            experiment: Is used to get arm parameters. Is not mutated.
            search_space: Search space for fitting the model. Constraints need
                not be the same ones used in gen. RangeParameter bounds are
                considered soft and will be expanded to match the range of the
                data sent in for fitting, if expand_model_space is True.
            data: Ax Data.
            model: Interface will be specified in subclass. If model requires
                initialization, that should be done prior to its use here.
            transforms: List of uninitialized transform classes. Forward
                transforms will be applied in this order, and untransforms in
                the reverse order.
            transform_configs: A dictionary from transform name to the
                transform config dictionary.
            status_quo_name: Name of the status quo arm. Can only be used if
                Data has a single set of ObservationFeatures corresponding to
                that arm.
            status_quo_features: ObservationFeatures to use as status quo.
                Either this or status_quo_name should be specified, not both.
            optimization_config: Optimization config defining how to optimize
                the model.
            expand_model_space: If True, expand range parameter bounds in model
                space to cover given training data. This will make the modeling
                space larger than the search space if training data fall outside
                the search space.
            fit_out_of_design: If specified, all training data are used.
                Otherwise, only in design points are used.
            fit_abandoned: Whether data for abandoned arms or trials should be
                included in model training data. If ``False``, only
                non-abandoned points are returned.
            fit_tracking_metrics: Whether to fit a model for tracking metrics.
                Setting this to False will improve runtime at the expense of
                models not being available for predicting tracking metrics.
                NOTE: This can only be set to False when the optimization config
                is provided.
            fit_on_init: Whether to fit the model on initialization. This can
                be used to skip model fitting when a fitted model is not needed.
                To fit the model afterwards, use `_process_and_transform_data`
                to get the transformed inputs and call `_fit_if_implemented` with
                the transformed inputs.
        """
        t_fit_start = time.monotonic()
        transforms = transforms or []
        transforms = [Cast] + transforms

        self.fit_time: float = 0.0
        self.fit_time_since_gen: float = 0.0
        self._metric_names: set[str] = set()
        self._training_data: list[Observation] = []
        self._optimization_config: OptimizationConfig | None = optimization_config
        self._training_in_design: list[bool] = []
        self._status_quo: Observation | None = None
        self._status_quo_name: str | None = None
        self._arms_by_signature: dict[str, Arm] | None = None
        self.transforms: MutableMapping[str, Transform] = OrderedDict()
        self._model_key: str | None = None
        self._model_kwargs: dict[str, Any] | None = None
        self._bridge_kwargs: dict[str, Any] | None = None
        # The space used for optimization.
        self._search_space: SearchSpace = search_space.clone()
        # The space used for modeling. Might be larger than the optimization
        # space to cover training data.
        self._model_space: SearchSpace = search_space.clone()
        self._raw_transforms = transforms
        self._transform_configs: dict[str, TConfig] | None = transform_configs
        self._fit_out_of_design = fit_out_of_design
        self._fit_abandoned = fit_abandoned
        self._fit_tracking_metrics = fit_tracking_metrics
        self.outcomes: list[str] = []
        self._experiment_has_immutable_search_space_and_opt_config: bool = (
            experiment is not None and experiment.immutable_search_space_and_opt_config
        )
        self._experiment_properties: dict[str, Any] = {}
        self._experiment: Experiment | None = experiment

        if experiment is not None:
            if self._optimization_config is None:
                self._optimization_config = experiment.optimization_config
            self._arms_by_signature = experiment.arms_by_signature
            self._experiment_properties = experiment._properties

        if self._fit_tracking_metrics is False:
            if self._optimization_config is None:
                raise UserInputError(
                    "Optimization config is required when "
                    "`fit_tracking_metrics` is False."
                )
            self.outcomes = sorted(self._optimization_config.metrics.keys())

        # Set training data (in the raw / untransformed space). This also omits
        # out-of-design and abandoned observations depending on the corresponding flags.
        observations_raw = self._prepare_observations(experiment=experiment, data=data)
        if expand_model_space:
            self._set_model_space(observations=observations_raw)
        observations_raw = self._set_training_data(
            observations=observations_raw, search_space=self._model_space
        )

        # Set model status quo.
        # NOTE: training data must be set before setting the status quo.
        self._set_status_quo(
            experiment=experiment,
            status_quo_name=status_quo_name,
            status_quo_features=status_quo_features,
        )

        # Save model, apply terminal transform, and fit.
        self.model = model
        if fit_on_init:
            observations, search_space = self._transform_data(
                observations=observations_raw,
                search_space=self._model_space,
                transforms=self._raw_transforms,
                transform_configs=self._transform_configs,
            )
            self._fit_if_implemented(
                search_space=search_space,
                observations=observations,
                time_so_far=time.monotonic() - t_fit_start,
            )

    def _fit_if_implemented(
        self,
        search_space: SearchSpace,
        observations: list[Observation],
        time_so_far: float,
    ) -> None:
        r"""Fits the model if `_fit` is implemented and stores fit time.

        Args:
            search_space: A transformed search space for fitting the model.
            observations: The observations to fit the model with. These should
                also be transformed.
            time_so_far: Time spent in initializing the model up to
                `_fit_if_implemented` call.
        """
        try:
            t_fit_start = time.monotonic()
            self._fit(
                model=self.model,
                search_space=search_space,
                observations=observations,
            )
            increment = time.monotonic() - t_fit_start + time_so_far
            self.fit_time += increment
            self.fit_time_since_gen += increment
        except ModelBridgeMethodNotImplementedError:
            pass

    def _process_and_transform_data(
        self,
        experiment: Experiment | None = None,
        data: Data | None = None,
    ) -> tuple[list[Observation], SearchSpace]:
        r"""Processes the data into observations and returns transformed
        observations and the search space. This packages the following methods:
        * self._prepare_observations
        * self._set_training_data
        * self._transform_data
        """
        observations = self._prepare_observations(experiment=experiment, data=data)
        observations_raw = self._set_training_data(
            observations=observations, search_space=self._model_space
        )
        return self._transform_data(
            observations=observations_raw,
            search_space=self._model_space,
            transforms=self._raw_transforms,
            transform_configs=self._transform_configs,
        )

    def _prepare_observations(
        self, experiment: Experiment | None, data: Data | None
    ) -> list[Observation]:
        if experiment is None or data is None:
            return []
        return observations_from_data(
            experiment=experiment,
            data=data,
            statuses_to_include=self.statuses_to_fit,
            statuses_to_include_map_metric=self.statuses_to_fit_map_metric,
        )

    def _transform_data(
        self,
        observations: list[Observation],
        search_space: SearchSpace,
        transforms: list[type[Transform]] | None,
        transform_configs: dict[str, TConfig] | None,
        assign_transforms: bool = True,
    ) -> tuple[list[Observation], SearchSpace]:
        """Initialize transforms and apply them to provided data."""
        # Initialize transforms
        search_space = search_space.clone()
        if transforms is not None:
            if transform_configs is None:
                transform_configs = {}

            for t in transforms:
                t_instance = t(
                    search_space=search_space,
                    observations=observations,
                    modelbridge=self,
                    config=transform_configs.get(t.__name__, None),
                )
                search_space = t_instance.transform_search_space(search_space)
                observations = t_instance.transform_observations(observations)
                if assign_transforms:
                    self.transforms[t.__name__] = t_instance

        return observations, search_space

    def _prepare_training_data(
        self, observations: list[Observation]
    ) -> list[Observation]:
        observation_features, observation_data = separate_observations(observations)
        if len(observation_features) != len(set(observation_features)):
            raise ValueError(
                "Observation features are not unique. "
                "Something went wrong constructing training data..."
            )
        return observations

    def _set_training_data(
        self, observations: list[Observation], search_space: SearchSpace
    ) -> list[Observation]:
        """Store training data, not-transformed.

        If the modelbridge specifies _fit_out_of_design, all training data is
        returned. Otherwise, only in design points are returned.
        """
        observations = self._prepare_training_data(observations=observations)
        self._training_data = deepcopy(observations)
        self._metric_names: set[str] = set()
        for obs in observations:
            self._metric_names.update(obs.data.metric_names)
        return self._process_in_design(
            search_space=search_space,
            observations=observations,
        )

    def _process_in_design(
        self,
        search_space: SearchSpace,
        observations: list[Observation],
    ) -> list[Observation]:
        """Set training_in_design, and decide whether to filter out of design points."""
        # Don't filter points.
        if self._fit_out_of_design:
            # Use all data for training
            # Set training_in_design to True for all observations so that
            # all observations are used in CV and plotting
            self.training_in_design = [True] * len(observations)
            return observations
        in_design = self._compute_in_design(
            search_space=search_space, observations=observations
        )
        self.training_in_design = in_design
        in_design_obs = [
            observations[i] for i, is_in_design in enumerate(in_design) if is_in_design
        ]
        return in_design_obs

    def _compute_in_design(
        self,
        search_space: SearchSpace,
        observations: list[Observation],
    ) -> list[bool]:
        """Compute in-design status for each observation, after filling missing
        values if FillMissingParameters transform is used."""
        observation_features = [obs.features for obs in observations]
        if (
            self._transform_configs is not None
            and "FillMissingParameters" in self._transform_configs
        ):
            t = FillMissingParameters(
                config=self._transform_configs["FillMissingParameters"]
            )
            observation_features = t.transform_observation_features(
                deepcopy(observation_features)
            )
        return [
            search_space.check_membership(obsf.parameters)
            for obsf in observation_features
        ]

    def _set_model_space(self, observations: list[Observation]) -> None:
        """Set model space, possibly expanding range parameters to cover data."""
        # If fill for missing values, include those in expansion.
        fill_values: TParameterization | None = None
        if (
            self._transform_configs is not None
            and "FillMissingParameters" in self._transform_configs
        ):
            fill_values = self._transform_configs[  # pyre-ignore[9]
                "FillMissingParameters"
            ].get("fill_values", None)
        # Extract parameter values across arms
        parameter_dicts = [obs.features.parameters for obs in observations]
        if fill_values is not None:
            parameter_dicts.append(fill_values)
        param_vals = {p_name: [] for p_name in self._model_space.parameters.keys()}
        for parameter_dict in parameter_dicts:
            for p_name in self._model_space.parameters.keys():
                p_val = parameter_dict.get(p_name, None)
                if p_val is not None:
                    param_vals[p_name].append(p_val)

        # Update model space. Expand bounds as needed to cover the values found
        # in the data.
        for p in self._model_space.parameters.values():
            if len(param_vals[p.name]) == 0:
                continue
            if isinstance(p, RangeParameter):
                p.lower = min(p.lower, min(param_vals[p.name]))
                p.upper = max(p.upper, max(param_vals[p.name]))

    def _set_status_quo(
        self,
        experiment: Experiment | None,
        status_quo_name: str | None,
        status_quo_features: ObservationFeatures | None,
    ) -> None:
        """Set model status quo by matching status_quo_name or status_quo_features.

        First checks for status quo in inputs status_quo_name and
        status_quo_features. If neither of these is provided, checks the
        experiment for a status quo. If that is set, it is handled by name in
        the same way as input status_quo_name.

        Args:
            experiment: Experiment that will be checked for status quo.
            status_quo_name: Name of status quo arm.
            status_quo_features: Features for status quo.
        """
        self._status_quo: Observation | None = None
        sq_obs = None

        if (
            status_quo_name is None
            and status_quo_features is None
            and experiment is not None
            and experiment.status_quo is not None
        ):
            status_quo_name = experiment.status_quo.name

        if status_quo_name is not None:
            if status_quo_features is not None:
                raise ValueError(
                    "Specify either status_quo_name or status_quo_features, not both."
                )
            sq_obs = [
                obs for obs in self._training_data if obs.arm_name == status_quo_name
            ]
        elif status_quo_features is not None:
            sq_obs = [
                obs
                for obs in self._training_data
                if (obs.features.parameters == status_quo_features.parameters)
                and (obs.features.trial_index == status_quo_features.trial_index)
            ]

        # if status_quo_name or status_quo_features is used for matching status quo
        if sq_obs is not None:
            if len(sq_obs) == 0:
                logger.warning(f"Status quo {status_quo_name} not present in data")
            elif len(sq_obs) >= 1:
                # status quo name (not features as trial index is part of the
                # observation features) should be consistent even if we have multiple
                # observations of the status quo.
                # This is useful for getting status_quo_data_by_trial
                self._status_quo_name = sq_obs[0].arm_name
                if len(sq_obs) > 1:
                    logger.warning(
                        f"Status quo {status_quo_name} found in data with multiple "
                        "features. Use status_quo_features to specify which to use."
                    )
                else:
                    # if there is a unique status_quo, set it
                    # unique features verified in _set_training_data.
                    self._status_quo = sq_obs[0]

    @property
    def status_quo_data_by_trial(self) -> dict[int, ObservationData] | None:
        """A map of trial index to the status quo observation data of each trial"""
        return _get_status_quo_by_trial(
            observations=self._training_data,
            status_quo_name=(
                self._status_quo_name
                if self.status_quo is None
                else self.status_quo.arm_name
            ),
            status_quo_features=(
                None if self.status_quo is None else self.status_quo.features
            ),
        )

    @property
    def status_quo(self) -> Observation | None:
        """Observation corresponding to status quo, if any."""
        return self._status_quo

    @property
    def status_quo_name(self) -> str | None:
        """Name of status quo, if any."""
        if self._status_quo is not None:
            if self._status_quo.arm_name is not None:
                return self._status_quo.arm_name
        return self._status_quo_name

    @property
    def metric_names(self) -> set[str]:
        """Metric names present in training data."""
        return self._metric_names

    @property
    def model_space(self) -> SearchSpace:
        """SearchSpace used to fit model."""
        return self._model_space

    def get_training_data(self) -> list[Observation]:
        """A copy of the (untransformed) data with which the model was fit."""
        return deepcopy(self._training_data)

    @property
    def training_in_design(self) -> list[bool]:
        """For each observation in the training data, a bool indicating if it
        is in-design for the model.
        """
        return self._training_in_design

    @property
    def statuses_to_fit(self) -> set[TrialStatus]:
        """Statuses to fit the model on."""
        if self._fit_abandoned:
            return set(TrialStatus)
        return NON_ABANDONED_STATUSES

    @property
    def statuses_to_fit_map_metric(self) -> set[TrialStatus]:
        """Statuses to fit the model on."""
        return {TrialStatus.COMPLETED}

    @training_in_design.setter
    def training_in_design(self, training_in_design: list[bool]) -> None:
        if len(training_in_design) != len(self._training_data):
            raise ValueError(
                f"In-design indicators not same length ({len(training_in_design)})"
                f" as training data ({len(self._training_data)})."
            )
        # Identify out-of-design arms
        if sum(training_in_design) < len(training_in_design):
            ood_names = []
            for i, obs in enumerate(self._training_data):
                if not training_in_design[i] and obs.arm_name is not None:
                    ood_names.append(obs.arm_name)
            ood_str = ", ".join(set(ood_names))
            logger.info(f"Leaving out out-of-design observations for arms: {ood_str}")
        self._training_in_design = training_in_design

    def _fit(
        self,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        model: Any,
        search_space: SearchSpace,
        observations: list[Observation],
    ) -> None:
        """Apply terminal transform and fit model."""
        raise ModelBridgeMethodNotImplementedError(
            f"{self.__class__.__name__} does not implement `_fit`."
        )

    def _batch_predict(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationData]:
        """Predict a list of ObservationFeatures together."""
        # Get modifiable version
        observation_features = deepcopy(observation_features)

        # Transform
        for t in self.transforms.values():
            observation_features = t.transform_observation_features(
                observation_features
            )
        # Apply terminal transform and predict
        observation_data = self._predict(observation_features)

        # Apply reverse transforms, in reverse order
        pred_observations = recombine_observations(
            observation_features=observation_features, observation_data=observation_data
        )

        for t in reversed(list(self.transforms.values())):
            pred_observations = t.untransform_observations(pred_observations)
        return [obs.data for obs in pred_observations]

    def _single_predict(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationData]:
        """Predict one ObservationFeature at a time."""
        observation_data = []
        for obsf in observation_features:
            try:
                obsd = self._batch_predict([obsf])
                observation_data += obsd
            except (TypeError, ValueError) as e:
                # If the prediction is not out of design, this is a real error.
                # Let's re-raise.
                if self.model_space.check_membership(obsf.parameters):
                    logger.debug(obsf.parameters)
                    logger.debug(self.model_space)
                    raise e from None
                # Prediction is out of design.
                # Training data is untranformed already.
                observation = next(
                    (
                        data
                        for data in self.get_training_data()
                        if obsf.parameters == data.features.parameters
                        and obsf.trial_index == data.features.trial_index
                    ),
                    None,
                )
                if not observation:
                    raise ValueError(
                        "Out-of-design point could not be transformed, and was "
                        "not found in the training data."
                    )
                observation_data.append(observation.data)
        return observation_data

    def _predict_observation_data(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationData]:
        """
        Like 'predict' method, but returns results as a list of ObservationData

        Predictions are made for all outcomes.
        If an out-of-design observation can successfully be transformed,
        the predicted value will be returned.
        Othwerise, we will attempt to find that observation in the training data
        and return the raw value.

        Args:
            observation_features: observation features

        Returns:
            List of `ObservationData`
        """
        # Predict in single batch.
        try:
            observation_data = self._batch_predict(observation_features)
        # Predict one by one.
        except (TypeError, ValueError):
            observation_data = self._single_predict(observation_features)
        return observation_data

    def predict(self, observation_features: list[ObservationFeatures]) -> TModelPredict:
        """Make model predictions (mean and covariance) for the given
        observation features.

        Predictions are made for all outcomes.
        If an out-of-design observation can successfully be transformed,
        the predicted value will be returned.
        Othwerise, we will attempt to find that observation in the training data
        and return the raw value.

        Args:
            observation_features: observation features

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
            observation_features=observation_features
        )
        f, cov = unwrap_observation_data(observation_data)
        return f, cov

    def _predict(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationData]:
        """Apply terminal transform, predict, and reverse terminal transform on
        output.
        """
        raise ModelBridgeMethodNotImplementedError(
            f"{self.__class__.__name__} does not implement `_predict`."
        )

    def update(self, new_data: Data, experiment: Experiment) -> None:
        """Update the model bridge and the underlying model with new data. This
        method should be used instead of `fit`, in cases where the underlying
        model does not need to be re-fit from scratch, but rather updated.

        Note: `update` expects only new data (obtained since the model initialization
        or last update) to be passed in, not all data in the experiment.

        Args:
            new_data: Data from the experiment obtained since the last call to
                `update`.
            experiment: Experiment, in which this data was obtained.
        """
        raise DeprecationWarning("ModelBridge.update is deprecated. Use `fit` instead.")

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
                        "optimization config used while initializing the ModelBridge. "
                        f"Metrics {outcomes} is not a subset of {self.outcomes}."
                    )
            optimization_config = optimization_config.clone()

        # TODO(T34225037): replace deepcopy with native clone() in Ax
        pending_observations = deepcopy(pending_observations)
        fixed_features = deepcopy(fixed_features)
        search_space = search_space.clone()

        # Transform
        for t in self.transforms.values():
            search_space = t.transform_search_space(search_space)
            if optimization_config is not None:
                optimization_config = t.transform_optimization_config(
                    optimization_config=optimization_config,
                    modelbridge=self,
                    fixed_features=fixed_features,
                )
            for metric, po in pending_observations.items():
                pending_observations[metric] = t.transform_observation_features(po)
            fixed_features = (
                t.transform_observation_features([fixed_features])[0]
                if fixed_features is not None
                else None
            )
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
        """Validate inputs to `ModelBridge.gen`.

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
        Generate new points from the underlying model according to
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
                model. See `TorchOptConfig` for details.

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
        if not self._fit_out_of_design:
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
        except NotImplementedError:
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
        gr = GeneratorRun(
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
            gen_metadata=gen_results.gen_metadata,
            model_state_after_gen=self._get_serialized_model_state(),
            candidate_metadata_by_arm_signature=candidate_metadata,
        )
        if len(gr.arms) < n:
            logger.warning(
                f"{self} was not able to generate {n} unique candidates. "
                "Generated arms have the following weights, as there are repeats:\n"
                f"{gr.weights}"
            )
        self.fit_time_since_gen = 0.0
        return gr

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
        raise ModelBridgeMethodNotImplementedError(
            f"{self.__class__.__name__} does not implement `_gen`."
        )

    def cross_validate(
        self,
        cv_training_data: list[Observation],
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
        with warnings.catch_warnings():
            # Since each CV fold removes points from the training data, the remaining
            # observations will not pass the standardization test. To avoid confusing
            # users with this warning, we filter it out.
            warnings.filterwarnings(
                "ignore",
                message=r"Data \(outcome observations\) is not standardized",
                category=InputDataWarning,
            )
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
        cv_training_data: list[Observation],
        cv_test_points: list[ObservationFeatures],
        use_posterior_predictive: bool = False,
    ) -> list[ObservationData]:
        """Apply the terminal transform, make predictions on the test points,
        and reverse terminal transform on the results.
        """
        raise ModelBridgeMethodNotImplementedError(
            f"{self.__class__.__name__} does not implement `_cross_validate`."
        )

    def _transform_inputs_for_cv(
        self,
        cv_training_data: list[Observation],
        cv_test_points: list[ObservationFeatures],
    ) -> tuple[list[Observation], list[ObservationFeatures], SearchSpace]:
        """Apply transforms to cv_training_data and cv_test_points,
        and return cv_training_data, cv_test_points, and search space in
        transformed space. This is to prepare data to be used in _cross_validate.

        Args:
            cv_training_data: The training data to use for cross validation.
            cv_test_points: The test points at which predictions will be made.

        Returns:
            cv_training_data, cv_test_points, and search space
            in transformed space."""
        cv_test_points = deepcopy(cv_test_points)
        cv_training_data = deepcopy(cv_training_data)
        search_space = self._model_space.clone()
        for t in self.transforms.values():
            cv_training_data = t.transform_observations(cv_training_data)
            cv_test_points = t.transform_observation_features(cv_test_points)
            search_space = t.transform_search_space(search_space)
        return cv_training_data, cv_test_points, search_space

    def _set_kwargs_to_save(
        self,
        model_key: str,
        model_kwargs: dict[str, Any],
        bridge_kwargs: dict[str, Any],
    ) -> None:
        """Set properties used to save the model that created a given generator
        run, on the `GeneratorRun` object. Each generator run produced by the
        `gen` method of this model bridge will have the model key and kwargs
        fields set as provided in arguments to this function.
        """
        self._model_key = model_key
        self._model_kwargs = model_kwargs
        self._bridge_kwargs = bridge_kwargs

    def _get_serialized_model_state(self) -> dict[str, Any]:
        """Obtains the state of the underlying model (if using a stateful one)
        in a readily JSON-serializable form.
        """
        model = none_throws(self.model)
        return model.serialize_state(raw_state=model._get_state())

    def _deserialize_model_state(
        self, serialized_state: dict[str, Any]
    ) -> dict[str, Any]:
        model = none_throws(self.model)
        return model.deserialize_state(serialized_state=serialized_state)

    def feature_importances(self, metric_name: str) -> dict[str, float]:
        """Computes feature importances for a single metric.

        Depending on the type of the model, this method will approach sensitivity
        analysis (calculating the sensitivity of the metric to changes in the search
        space's parameters, a.k.a. features) differently.

        For Bayesian optimization models (BoTorch models), this method uses parameter
        inverse lengthscales to compute normalized feature importances.

        NOTE: Currently, this is only implemented for GP models.

        Args:
            metric_name: Name of metric to compute feature importances for.

        Returns:
            A dictionary mapping parameter names to their corresponding feature
            importances.

        """
        raise ModelBridgeMethodNotImplementedError(
            f"{self.__class__.__name__} does not implement `feature_importances`."
        )

    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def transform_observations(self, observations: list[Observation]) -> Any:
        """Applies transforms to given observation features and returns them in the
        model space.

        Args:
            observation_features: ObservationFeatures to be transformed.

        Returns:
            Transformed values. This could be e.g. a torch Tensor, depending
            on the ModelBridge subclass.
        """
        observations = deepcopy(observations)
        for t in self.transforms.values():
            observations = t.transform_observations(observations)
        # Apply terminal transform and return
        return self._transform_observations(observations)

    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def _transform_observations(self, observations: list[Observation]) -> Any:
        """Apply terminal transform to given observations and return result."""
        raise ModelBridgeMethodNotImplementedError(
            f"{self.__class__.__name__} does not implement `_transform_observations`."
        )

    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> Any:
        """Applies transforms to given observation features and returns them in the
        model space.

        Args:
            observation_features: ObservationFeatures to be transformed.

        Returns:
            Transformed values. This could be e.g. a torch Tensor, depending
            on the ModelBridge subclass.
        """
        obsf = deepcopy(observation_features)
        for t in self.transforms.values():
            obsf = t.transform_observation_features(obsf)
        # Apply terminal transform and return
        return self._transform_observation_features(obsf)

    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def _transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> Any:
        """Apply terminal transform to given observation features and return result."""
        raise ModelBridgeMethodNotImplementedError(
            f"{self.__class__.__name__} does not implement "
            "`_transform_observation_features`."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"


def unwrap_observation_data(observation_data: list[ObservationData]) -> TModelPredict:
    """Converts observation data to the format for model prediction outputs.
    That format assumes each observation data has the same set of metrics.
    """
    metrics = set(observation_data[0].metric_names)
    f: TModelMean = {metric: [] for metric in metrics}
    cov: TModelCov = {m1: {m2: [] for m2 in metrics} for m1 in metrics}
    for od in observation_data:
        if set(od.metric_names) != metrics:
            raise ValueError(
                "Each ObservationData should use same set of metrics. "
                "Expected {exp}, got {got}.".format(
                    exp=metrics, got=set(od.metric_names)
                )
            )
        for i, m1 in enumerate(od.metric_names):
            f[m1].append(od.means[i])
            for j, m2 in enumerate(od.metric_names):
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
                logger.info(
                    f"Untransformed parameter {val} "
                    f"less than lower bound {p.lower}, clamping"
                )
                obsf.parameters[p.name] = p.lower
            elif val > p.upper:
                logger.info(
                    f"Untransformed parameter {val} "
                    f"greater than upper bound {p.upper}, clamping"
                )
                obsf.parameters[p.name] = p.upper
    return observation_features


def _get_status_quo_by_trial(
    observations: list[Observation],
    status_quo_name: str | None = None,
    status_quo_features: ObservationFeatures | None = None,
) -> dict[int, ObservationData] | None:
    r"""
    Given a status quo observation, return a dictionary of trial index to
    the status quo observation data of each trial.

    When either `status_quo_name` or `status_quo_features` exists, return the dict;
    when both exist, use `status_quo_name`;
    when neither exists, return None.

    Args:
        observations: List of observations.
        status_quo_name: Name of the status quo.
        status_quo_features: ObservationFeatures for the status quo.

    Returns:
        A map from trial index to status quo observation data, or None
    """
    trial_idx_to_sq_data = None
    if status_quo_name is not None:
        # identify status quo by arm name
        trial_idx_to_sq_data = {
            int(none_throws(obs.features.trial_index)): obs.data
            for obs in observations
            if obs.arm_name == status_quo_name
        }
    elif status_quo_features is not None:
        # identify status quo by (untransformed) feature
        status_quo_signature = json.dumps(
            status_quo_features.parameters, sort_keys=True
        )
        trial_idx_to_sq_data = {
            int(none_throws(obs.features.trial_index)): obs.data
            for obs in observations
            if json.dumps(obs.features.parameters, sort_keys=True)
            == status_quo_signature
        }

    return trial_idx_to_sq_data
