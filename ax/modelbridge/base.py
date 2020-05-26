#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from abc import ABC
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, MutableMapping, Optional, Set, Tuple, Type

from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun, extract_arm_predictions
from ax.core.observation import (
    Observation,
    ObservationData,
    ObservationFeatures,
    observations_from_data,
    separate_observations,
)
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.core.types import (
    TCandidateMetadata,
    TConfig,
    TGenMetadata,
    TModelCov,
    TModelMean,
    TModelPredict,
)
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.cast import Cast
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none


logger = get_logger("ModelBridge")


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
        model: Any,
        transforms: Optional[List[Type[Transform]]] = None,
        experiment: Optional[Experiment] = None,
        data: Optional[Data] = None,
        transform_configs: Optional[Dict[str, TConfig]] = None,
        status_quo_name: Optional[str] = None,
        status_quo_features: Optional[ObservationFeatures] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        fit_out_of_design: bool = False,
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

        self._model_space = search_space.clone()
        self._raw_transforms = transforms
        self._transform_configs: Optional[Dict[str, TConfig]] = transform_configs
        self._fit_out_of_design = fit_out_of_design

        if experiment is not None:
            if self._optimization_config is None:
                self._optimization_config = experiment.optimization_config
            self._arms_by_signature = experiment.arms_by_signature

        observations = (
            observations_from_data(experiment, data)
            if experiment is not None and data is not None
            else []
        )
        obs_feats_raw, obs_data_raw = self._set_training_data(
            observations=observations, search_space=search_space
        )
        # Set model status quo
        # NOTE: training data must be set before setting the status quo.
        self._set_status_quo(
            experiment=experiment,
            status_quo_name=status_quo_name,
            status_quo_features=status_quo_features,
        )
        obs_feats, obs_data, search_space = self._transform_data(
            obs_feats=obs_feats_raw,
            obs_data=obs_data_raw,
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
                observation_features=obs_feats,
                observation_data=obs_data,
            )
            self.fit_time = time.time() - t_fit_start
            self.fit_time_since_gen = float(self.fit_time)
        except NotImplementedError:
            self.fit_time = 0.0
            self.fit_time_since_gen = 0.0

    def _transform_data(
        self,
        obs_feats: List[ObservationFeatures],
        obs_data: List[ObservationData],
        search_space: SearchSpace,
        transforms: Optional[List[Type[Transform]]],
        transform_configs: Optional[Dict[str, TConfig]],
    ) -> Tuple[List[ObservationFeatures], List[ObservationData], SearchSpace]:
        """Initialize transforms and apply them to provided data."""
        # Initialize transforms
        search_space = search_space.clone()
        if transforms is not None:
            if transform_configs is None:
                transform_configs = {}

            for t in transforms:
                t_instance = t(
                    search_space=search_space,
                    observation_features=obs_feats,
                    observation_data=obs_data,
                    config=transform_configs.get(t.__name__, None),
                )
                search_space = t_instance.transform_search_space(search_space)
                obs_feats = t_instance.transform_observation_features(obs_feats)
                obs_data = t_instance.transform_observation_data(obs_data, obs_feats)
                self.transforms[t.__name__] = t_instance

        return obs_feats, obs_data, search_space

    def _prepare_training_data(
        self, observations: List[Observation]
    ) -> Tuple[List[ObservationFeatures], List[ObservationData]]:
        observation_features, observation_data = separate_observations(observations)
        if len(observation_features) != len(set(observation_features)):
            raise ValueError(
                "Observation features not unique."
                "Something went wrong constructing training data..."
            )
        return observation_features, observation_data

    def _set_training_data(
        self, observations: List[Observation], search_space: SearchSpace
    ) -> Tuple[List[ObservationFeatures], List[ObservationData]]:
        """Store training data, not-transformed.

        If the modelbridge specifies _fit_out_of_design, all training data is
        returned. Otherwise, only in design points are returned.
        """
        observation_features, observation_data = self._prepare_training_data(
            observations=observations
        )
        self._training_data = deepcopy(observations)
        self._metric_names: Set[str] = set()
        for obsd in observation_data:
            self._metric_names.update(obsd.metric_names)
        return self._process_in_design(
            search_space=search_space,
            observation_features=observation_features,
            observation_data=observation_data,
        )

    def _extend_training_data(
        self, observations: List[Observation]
    ) -> Tuple[List[ObservationFeatures], List[ObservationData]]:
        """Extend and return training data, not-transformed.

        If the modelbridge specifies _fit_out_of_design, all training data is
        returned. Otherwise, only in design points are returned.

        Args:
            observations: New observations.

        Returns:
            observation_features: New + old observation features.
            observation_data: New + old observation data.
        """
        observation_features, observation_data = self._prepare_training_data(
            observations=observations
        )
        for obsd in observation_data:
            for metric_name in obsd.metric_names:
                if metric_name not in self._metric_names:
                    raise ValueError(
                        f"Unrecognised metric {metric_name}; cannot update "
                        "training data with metrics that were not in the original "
                        "training data."
                    )
        # Initialize with all points in design.
        self._training_data.extend(deepcopy(observations))
        all_observation_features, all_observation_data = separate_observations(
            self.get_training_data()
        )
        return self._process_in_design(
            search_space=self._model_space,
            observation_features=all_observation_features,
            observation_data=all_observation_data,
        )

    def _process_in_design(
        self,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
    ) -> Tuple[List[ObservationFeatures], List[ObservationData]]:
        """Set training_in_design, and decide whether to filter out of design points."""
        # Don't filter points.
        if self._fit_out_of_design:
            # Use all data for training
            # Set training_in_design to True for all observations so that
            # all observations are used in CV and plotting
            self.training_in_design = [True] * len(observation_features)
            return observation_features, observation_data
        in_design = [
            search_space.check_membership(obsf.parameters)
            for obsf in observation_features
        ]
        self.training_in_design = in_design
        in_design_indices = [i for i, in_design in enumerate(in_design) if in_design]
        in_design_features = [observation_features[i] for i in in_design_indices]
        in_design_data = [observation_data[i] for i in in_design_indices]
        return in_design_features, in_design_data

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
            # pyre-fixme[16]: `Optional` has no attribute `name`.
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
        """SearchSpace used to fit model.
        """
        return self._model_space

    def get_training_data(self) -> List[Observation]:
        """A copy of the (untransformed) data with which the model was fit.
        """
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
        model: Any,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
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
        for t in reversed(self.transforms.values()):  # noqa T484
            observation_features = t.untransform_observation_features(
                observation_features
            )
            observation_data = t.untransform_observation_data(
                observation_data, observation_features
            )
        return observation_data

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
        # Predict in single batch.
        try:
            observation_data = self._batch_predict(observation_features)
        # Predict one by one.
        except (TypeError, ValueError):
            observation_data = self._single_predict(observation_features)
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
        observations = (
            observations_from_data(experiment=experiment, data=new_data)
            if experiment is not None and new_data is not None
            else []
        )
        obs_feats_raw, obs_data_raw = self._extend_training_data(
            observations=observations
        )
        obs_feats, obs_data, search_space = self._transform_data(
            obs_feats=obs_feats_raw,
            obs_data=obs_data_raw,
            search_space=self._model_space,
            transforms=self._raw_transforms,
            transform_configs=self._transform_configs,
        )
        self._update(observation_features=obs_feats, observation_data=obs_data)
        self.fit_time += time.time() - t_update_start
        self.fit_time_since_gen += time.time() - t_update_start

    def _update(
        self,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
    ) -> None:
        """Apply terminal transform and update model.

        Note: This function requires ALL observation_features and
        observation_data observed thus far, not just the new data to update with.

        Args:
            observation_features: All observation features observed so far.
            observation_data: All observation data observed so far.
        """
        raise NotImplementedError  # pragma: no cover

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
        """
        t_gen_start = time.time()
        if pending_observations is None:
            pending_observations = {}
        if fixed_features is None:
            fixed_features = ObservationFeatures({})

        # Get modifiable versions
        if search_space is None:
            search_space = self._model_space
        search_space = search_space.clone()

        if optimization_config is None:
            optimization_config = (
                # pyre-fixme[16]: `Optional` has no attribute `clone`.
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

        # Apply terminal transform and gen
        observation_features, weights, best_obsf, gen_metadata = self._gen(
            n=n,
            search_space=search_space,
            optimization_config=optimization_config,
            pending_observations=pending_observations,
            fixed_features=fixed_features,
            model_gen_options=model_gen_options,
        )

        # Apply reverse transforms
        for t in reversed(self.transforms.values()):  # noqa T484
            observation_features = t.untransform_observation_features(
                observation_features
            )
            if best_obsf is not None:
                best_obsf = t.untransform_observation_features([best_obsf])[0]

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
        gr = GeneratorRun(
            arms=arms,
            weights=weights,
            optimization_config=optimization_config,
            search_space=search_space,
            model_predictions=model_predictions,
            best_arm_predictions=None
            if best_arm is None
            else (best_arm, best_point_predictions),
            fit_time=self.fit_time_since_gen,
            gen_time=time.time() - t_gen_start,
            model_key=self._model_key,
            model_kwargs=self._model_kwargs,
            bridge_kwargs=self._bridge_kwargs,
            gen_metadata=gen_metadata,
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
    ) -> Tuple[
        List[ObservationFeatures],
        List[float],
        Optional[ObservationFeatures],
        TGenMetadata,
    ]:
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
        obs_feats, obs_data = separate_observations(
            observations=cv_training_data, copy=True
        )
        for t in self.transforms.values():
            obs_feats = t.transform_observation_features(obs_feats)
            obs_data = t.transform_observation_data(obs_data, obs_feats)
            cv_test_points = t.transform_observation_features(cv_test_points)

        # Apply terminal transform, and get predictions.
        cv_predictions = self._cross_validate(
            obs_feats=obs_feats, obs_data=obs_data, cv_test_points=cv_test_points
        )
        # Apply reverse transforms, in reverse order
        for t in reversed(self.transforms.values()):
            cv_test_points = t.untransform_observation_features(cv_test_points)
            cv_predictions = t.untransform_observation_data(
                cv_predictions, cv_test_points
            )
        return cv_predictions

    def _cross_validate(
        self,
        obs_feats: List[ObservationFeatures],
        obs_data: List[ObservationData],
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
        model = not_none(self.model)
        return model.deserialize_state(serialized_state=serialized_state)

    def feature_importances(self, metric_name: str) -> Dict[str, float]:
        raise NotImplementedError(
            "Feature importance not available for this model type"
        )

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

    def _transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> Any:
        """Apply terminal transform to given observation features and return result.
        """
        raise NotImplementedError  # pragma: no cover


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
