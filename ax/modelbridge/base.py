#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from abc import ABC
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field

from logging import Logger
from typing import Any, Dict, List, MutableMapping, Optional, Set, Tuple, Type

from ax.core.arm import Arm
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
from ax.core.types import TCandidateMetadata, TModelCov, TModelMean, TModelPredict
from ax.exceptions.core import UserInputError
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.cast import Cast
from ax.models.types import TConfig
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none

logger: Logger = get_logger(__name__)


@dataclass(frozen=True)
class BaseGenArgs:
    search_space: SearchSpace
    optimization_config: Optional[OptimizationConfig]
    pending_observations: Dict[str, List[ObservationFeatures]]
    fixed_features: ObservationFeatures


@dataclass(frozen=True)
class GenResults:
    observation_features: List[ObservationFeatures]
    weights: List[float]
    best_observation_features: Optional[ObservationFeatures] = None
    gen_metadata: Dict[str, Any] = field(default_factory=dict)


class ModelBridge(ABC):
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
        transforms: Optional[List[Type[Transform]]] = None,
        experiment: Optional[Experiment] = None,
        data: Optional[Data] = None,
        transform_configs: Optional[Dict[str, TConfig]] = None,
        status_quo_name: Optional[str] = None,
        status_quo_features: Optional[ObservationFeatures] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        fit_out_of_design: bool = False,
        fit_abandoned: bool = False,
    ) -> None:
        """
        Applies transforms and fits model.

        Args:
            experiment: Is used to get arm parameters. Is not mutated.
            search_space: Search space for fitting the model. Constraints need
                not be the same ones used in gen.
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
            fit_out_of_design: If specified, all training data is returned.
                Otherwise, only in design points are returned.
            fit_abandoned: Whether data for abandoned arms or trials should be
                included in model training data. If ``False``, only
                non-abandoned points are returned.
        """
        t_fit_start = time.time()
        transforms = transforms or []
        # pyre-ignore: Cast is a Tranform
        transforms: List[Type[Transform]] = [Cast] + transforms

        self._metric_names: Set[str] = set()
        self._training_data: List[Observation] = []
        self._optimization_config: Optional[OptimizationConfig] = optimization_config
        self._training_in_design: List[bool] = []
        self._status_quo: Optional[Observation] = None
        self._arms_by_signature: Optional[Dict[str, Arm]] = None
        self.transforms: MutableMapping[str, Transform] = OrderedDict()
        self._model_key: Optional[str] = None
        self._model_kwargs: Optional[Dict[str, Any]] = None
        self._bridge_kwargs: Optional[Dict[str, Any]] = None

        # pyre-fixme[4]: Attribute must be annotated.
        self._model_space = search_space.clone()
        self._raw_transforms = transforms
        self._transform_configs: Optional[Dict[str, TConfig]] = transform_configs
        self._fit_out_of_design = fit_out_of_design
        self._fit_abandoned = fit_abandoned
        imm = experiment and experiment.immutable_search_space_and_opt_config
        # pyre-fixme[4]: Attribute must be annotated.
        self._experiment_has_immutable_search_space_and_opt_config = imm
        if experiment is not None:
            if self._optimization_config is None:
                self._optimization_config = experiment.optimization_config
            self._arms_by_signature = experiment.arms_by_signature

        # Convert Data to Observations
        observations = self._prepare_observations(experiment=experiment, data=data)

        observations_raw = self._set_training_data(
            observations=observations, search_space=search_space
        )
        # Set model status quo
        # NOTE: training data must be set before setting the status quo.
        self._set_status_quo(
            experiment=experiment,
            status_quo_name=status_quo_name,
            status_quo_features=status_quo_features,
        )
        observations, search_space = self._transform_data(
            observations=observations_raw,
            search_space=search_space,
            transforms=transforms,
            transform_configs=transform_configs,
        )

        # Save model, apply terminal transform, and fit
        self.model = model
        try:
            self._fit(
                model=model,
                search_space=search_space,
                observations=observations,
            )
            # pyre-fixme[4]: Attribute must be annotated.
            self.fit_time = time.time() - t_fit_start
            # pyre-fixme[4]: Attribute must be annotated.
            self.fit_time_since_gen = float(self.fit_time)
        except NotImplementedError:
            self.fit_time = 0.0
            self.fit_time_since_gen = 0.0

    def _prepare_observations(
        self, experiment: Optional[Experiment], data: Optional[Data]
    ) -> List[Observation]:
        if experiment is None or data is None:
            return []
        return observations_from_data(
            experiment=experiment, data=data, include_abandoned=self._fit_abandoned
        )

    def _transform_data(
        self,
        observations: List[Observation],
        search_space: SearchSpace,
        transforms: Optional[List[Type[Transform]]],
        transform_configs: Optional[Dict[str, TConfig]],
    ) -> Tuple[List[Observation], SearchSpace]:
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
                self.transforms[t.__name__] = t_instance

        return observations, search_space

    def _prepare_training_data(
        self, observations: List[Observation]
    ) -> List[Observation]:
        observation_features, observation_data = separate_observations(observations)
        if len(observation_features) != len(set(observation_features)):
            raise ValueError(
                "Observation features not unique."
                "Something went wrong constructing training data..."
            )
        return observations

    def _set_training_data(
        self, observations: List[Observation], search_space: SearchSpace
    ) -> List[Observation]:
        """Store training data, not-transformed.

        If the modelbridge specifies _fit_out_of_design, all training data is
        returned. Otherwise, only in design points are returned.
        """
        observations = self._prepare_training_data(observations=observations)
        self._training_data = deepcopy(observations)
        self._metric_names: Set[str] = set()
        for obs in observations:
            self._metric_names.update(obs.data.metric_names)
        return self._process_in_design(
            search_space=search_space,
            observations=observations,
        )

    def _extend_training_data(
        self, observations: List[Observation]
    ) -> List[Observation]:
        """Extend and return training data, not-transformed.

        If the modelbridge specifies _fit_out_of_design, all training data is
        returned. Otherwise, only in design points are returned.

        Args:
            observations: New observations.

        Returns: New + old observations.
        """
        observations = self._prepare_training_data(observations=observations)
        for obs in observations:
            for metric_name in obs.data.metric_names:
                if metric_name not in self._metric_names:
                    raise ValueError(
                        f"Unrecognised metric {metric_name}; cannot update "
                        "training data with metrics that were not in the original "
                        "training data."
                    )
        # Initialize with all points in design.
        self._training_data.extend(deepcopy(observations))
        all_observations = self.get_training_data()
        return self._process_in_design(
            search_space=self._model_space,
            observations=all_observations,
        )

    def _process_in_design(
        self,
        search_space: SearchSpace,
        observations: List[Observation],
    ) -> List[Observation]:
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
        observations: List[Observation],
    ) -> List[bool]:
        return [
            search_space.check_membership(obs.features.parameters)
            for obs in observations
        ]

    def _set_status_quo(
        self,
        experiment: Optional[Experiment],
        status_quo_name: Optional[str],
        status_quo_features: Optional[ObservationFeatures],
    ) -> None:
        """Set model status quo.

        First checks for status quo in inputs status_quo_name and
        status_quo_features. If neither of these is provided, checks the
        experiment for a status quo. If that is set, it is handled by name in
        the same way as input status_quo_name.

        Args:
            experiment: Experiment that will be checked for status quo.
            status_quo_name: Name of status quo arm.
            status_quo_features: Features for status quo.
        """
        self._status_quo: Optional[Observation] = None

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

            if len(sq_obs) == 0:
                logger.warning(f"Status quo {status_quo_name} not present in data")
            elif len(sq_obs) > 1:
                logger.warning(  # pragma: no cover
                    f"Status quo {status_quo_name} found in data with multiple "
                    "features. Use status_quo_features to specify which to use."
                )
            else:
                self._status_quo = sq_obs[0]

        elif status_quo_features is not None:
            sq_obs = [
                obs
                for obs in self._training_data
                if (obs.features.parameters == status_quo_features.parameters)
                and (obs.features.trial_index == status_quo_features.trial_index)
            ]

            if len(sq_obs) == 0:
                logger.warning(
                    f"Status quo features {status_quo_features} not found in data."
                )
            else:
                # len(sq_obs) will not be > 1,
                # unique features verified in _set_training_data.
                self._status_quo = sq_obs[0]

    @property
    def status_quo(self) -> Optional[Observation]:
        """Observation corresponding to status quo, if any."""
        return self._status_quo

    @property
    def metric_names(self) -> Set[str]:
        """Metric names present in training data."""
        return self._metric_names

    @property
    def model_space(self) -> SearchSpace:
        """SearchSpace used to fit model."""
        return self._model_space

    def get_training_data(self) -> List[Observation]:
        """A copy of the (untransformed) data with which the model was fit."""
        return deepcopy(self._training_data)

    @property
    def training_in_design(self) -> List[bool]:
        """For each observation in the training data, a bool indicating if it
        is in-design for the model.
        """
        return self._training_in_design

    @training_in_design.setter
    def training_in_design(self, training_in_design: List[bool]) -> None:
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
        observations: List[Observation],
    ) -> None:
        """Apply terminal transform and fit model."""
        raise NotImplementedError  # pragma: no cover

    def _batch_predict(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationData]:
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
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationData]:
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
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationData]:
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

    def predict(self, observation_features: List[ObservationFeatures]) -> TModelPredict:
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
        observation_data = self._predict_observation_data(
            observation_features=observation_features
        )
        f, cov = unwrap_observation_data(observation_data)
        return f, cov

    def _predict(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationData]:
        """Apply terminal transform, predict, and reverse terminal transform on
        output.
        """
        raise NotImplementedError  # pragma: no cover

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
        t_update_start = time.time()
        observations = self._prepare_observations(experiment=experiment, data=new_data)
        obs_raw = self._extend_training_data(observations=observations)
        observations, search_space = self._transform_data(
            observations=obs_raw,
            search_space=self._model_space,
            transforms=self._raw_transforms,
            transform_configs=self._transform_configs,
        )
        self._update(
            search_space=search_space,
            observations=observations,
        )
        self.fit_time += time.time() - t_update_start
        self.fit_time_since_gen += time.time() - t_update_start

    def _update(
        self,
        search_space: SearchSpace,
        observations: List[Observation],
    ) -> None:
        """Apply terminal transform and update model.

        Note: This function requires ALL observation_features and
        observation_data observed thus far, not just the new data to update with.

        Args:
            observation_features: All observation features observed so far.
            observation_data: All observation data observed so far.
        """
        raise NotImplementedError  # pragma: no cover

    def _get_transformed_gen_args(
        self,
        search_space: SearchSpace,
        optimization_config: Optional[OptimizationConfig] = None,
        pending_observations: Optional[Dict[str, List[ObservationFeatures]]] = None,
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> BaseGenArgs:
        if pending_observations is None:
            pending_observations = {}
        if fixed_features is None:
            fixed_features = ObservationFeatures({})
        if optimization_config is None:
            optimization_config = (
                self._optimization_config.clone()
                if self._optimization_config is not None
                else None
            )
        else:
            optimization_config = optimization_config.clone()

        # TODO(T34225037): replace deepcopy with native clone() in Ax
        pending_observations = deepcopy(pending_observations)
        fixed_features = deepcopy(fixed_features)

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
            fixed_features = t.transform_observation_features([fixed_features])[0]
        return BaseGenArgs(
            search_space=search_space,
            optimization_config=optimization_config,
            pending_observations=pending_observations,
            fixed_features=fixed_features,
        )

    def _validate_gen_inputs(
        self,
        n: int,
        search_space: Optional[SearchSpace] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        pending_observations: Optional[Dict[str, List[ObservationFeatures]]] = None,
        fixed_features: Optional[ObservationFeatures] = None,
        model_gen_options: Optional[TConfig] = None,
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
        search_space: Optional[SearchSpace] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        pending_observations: Optional[Dict[str, List[ObservationFeatures]]] = None,
        fixed_features: Optional[ObservationFeatures] = None,
        model_gen_options: Optional[TConfig] = None,
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
                model.

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
            search_space = self._model_space
        orig_search_space = search_space
        search_space = search_space.clone()
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
        except NotImplementedError:  # pragma: no cover
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
            search_space=None if immutable else base_gen_args.search_space,
            model_predictions=model_predictions,
            best_arm_predictions=None
            if best_arm is None
            else (best_arm, best_point_predictions),
            fit_time=self.fit_time_since_gen,
            gen_time=time.monotonic() - t_gen_start,
            model_key=self._model_key,
            model_kwargs=self._model_kwargs,
            bridge_kwargs=self._bridge_kwargs,
            gen_metadata=gen_results.gen_metadata,
            model_state_after_gen=self._get_serialized_model_state(),
            candidate_metadata_by_arm_signature=candidate_metadata,
        )
        self.fit_time_since_gen = 0.0
        return gr

    def _gen(
        self,
        n: int,
        search_space: SearchSpace,
        optimization_config: Optional[OptimizationConfig],
        pending_observations: Dict[str, List[ObservationFeatures]],
        fixed_features: ObservationFeatures,
        model_gen_options: Optional[TConfig],
    ) -> GenResults:
        """Apply terminal transform, gen, and reverse terminal transform on
        output.
        """
        raise NotImplementedError  # pragma: no cover

    def cross_validate(
        self,
        cv_training_data: List[Observation],
        cv_test_points: List[ObservationFeatures],
    ) -> List[ObservationData]:
        """Make a set of cross-validation predictions.

        Args:
            cv_training_data: The training data to use for cross validation.
            cv_test_points: The test points at which predictions will be made.

        Returns:
            A list of predictions at the test points.
        """
        # Apply transforms to cv_training_data and cv_test_points
        cv_test_points = deepcopy(cv_test_points)
        cv_training_data = deepcopy(cv_training_data)
        search_space = self._model_space.clone()
        for t in self.transforms.values():
            cv_training_data = t.transform_observations(cv_training_data)
            cv_test_points = t.transform_observation_features(cv_test_points)
            search_space = t.transform_search_space(search_space)

        obs_feats, obs_data = separate_observations(observations=cv_training_data)
        # Apply terminal transform, and get predictions.
        cv_predictions = self._cross_validate(
            search_space=search_space,
            cv_training_data=cv_training_data,
            cv_test_points=cv_test_points,
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
        cv_training_data: List[Observation],
        cv_test_points: List[ObservationFeatures],
    ) -> List[ObservationData]:
        """Apply the terminal transform, make predictions on the test points,
        and reverse terminal transform on the results.
        """
        raise NotImplementedError  # pragma: no cover

    def _set_kwargs_to_save(
        self,
        model_key: str,
        model_kwargs: Dict[str, Any],
        bridge_kwargs: Dict[str, Any],
    ) -> None:
        """Set properties used to save the model that created a given generator
        run, on the `GeneratorRun` object. Each generator run produced by the
        `gen` method of this model bridge will have the model key and kwargs
        fields set as provided in arguments to this function.
        """
        self._model_key = model_key
        self._model_kwargs = model_kwargs
        self._bridge_kwargs = bridge_kwargs

    def _get_serialized_model_state(self) -> Dict[str, Any]:
        """Obtains the state of the underlying model (if using a stateful one)
        in a readily JSON-serializable form.
        """
        model = not_none(self.model)
        return model.serialize_state(raw_state=model._get_state())

    def _deserialize_model_state(
        self, serialized_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        model = not_none(self.model)  # pragma: no cover
        return model.deserialize_state(  # pragma: no cover
            serialized_state=serialized_state
        )

    def feature_importances(self, metric_name: str) -> Dict[str, float]:
        raise NotImplementedError(
            "Feature importance not available for this model type"
        )

    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def transform_observations(self, observations: List[Observation]) -> Any:
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
    def _transform_observations(self, observations: List[Observation]) -> Any:
        """Apply terminal transform to given observations and return result."""
        raise NotImplementedError  # pragma: no cover

    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
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
        self, observation_features: List[ObservationFeatures]
    ) -> Any:
        """Apply terminal transform to given observation features and return result."""
        raise NotImplementedError  # pragma: no cover

    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        fixed_features: ObservationFeatures,
    ) -> Any:
        """Applies transforms to given optimization config.

        Args:
            optimization_config: OptimizationConfig to transform.
            fixed_features: features which should not be transformed.

        Returns:
            Transformed values. This could be e.g. a torch Tensor, depending
            on the ModelBridge subclass.
        """
        optimization_config = optimization_config.clone()
        for t in self.transforms.values():
            optimization_config = t.transform_optimization_config(
                optimization_config=optimization_config,
                modelbridge=self,
                fixed_features=fixed_features,
            )
        return optimization_config


def unwrap_observation_data(observation_data: List[ObservationData]) -> TModelPredict:
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
    observation_features: List[ObservationFeatures],
    arms_by_signature: Optional[Dict[str, Arm]] = None,
) -> Tuple[List[Arm], Optional[Dict[str, TCandidateMetadata]]]:
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
    observation_features: List[ObservationFeatures], search_space: SearchSpace
) -> List[ObservationFeatures]:
    range_parameters = [
        p for p in search_space.parameters.values() if isinstance(p, RangeParameter)
    ]
    for obsf in observation_features:
        for p in range_parameters:
            if p.name not in obsf.parameters:
                continue
            if p.parameter_type == ParameterType.FLOAT:
                val = checked_cast(float, obsf.parameters[p.name])
            else:
                val = checked_cast(int, obsf.parameters[p.name])
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
