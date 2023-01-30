# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple, Type

import numpy as np

import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.observation import (
    Observation,
    ObservationData,
    ObservationFeatures,
    observations_from_map_data,
    separate_observations,
)
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.core.types import TCandidateMetadata
from ax.modelbridge.base import GenResults
from ax.modelbridge.modelbridge_utils import (
    array_to_observation_data,
    observation_features_to_array,
    parse_observation_features,
)
from ax.modelbridge.torch import FIT_MODEL_ERROR, TorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.models.torch_base import TorchModel
from ax.models.types import TConfig
from ax.utils.common.constants import Keys
from ax.utils.common.typeutils import not_none


# A mapping from map_key to its target (or final) value; by default,
# we assume normalization to [0, 1], making 1.0 the target value.
# Used in both generation and prediction.
DEFAULT_TARGET_MAP_VALUES = {"steps": 1.0}


class MapTorchModelBridge(TorchModelBridge):
    """A model bridge for using torch-based models that fit on MapData. Most
    of the `TorchModelBridge` functionality is retained, except that this
    class should be used in the case where `model` makes use of map_key values.
    For example, the use case of fitting a joint surrogate model on
    `(parameters, map_key)`, while candidate generation is only for `parameters`.
    """

    def __init__(
        self,
        experiment: Experiment,
        search_space: SearchSpace,
        data: Data,
        model: TorchModel,
        transforms: List[Type[Transform]],
        transform_configs: Optional[Dict[str, TConfig]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        torch_device: Optional[torch.device] = None,
        status_quo_name: Optional[str] = None,
        status_quo_features: Optional[ObservationFeatures] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        fit_out_of_design: bool = False,
        default_model_gen_options: Optional[TConfig] = None,
        map_data_limit_total_rows: Optional[int] = None,
        map_data_limit_rows_per_group: Optional[int] = None,
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
            torch_dtype: Torch data type.
            torch_device: Torch device.
            status_quo_name: Name of the status quo arm. Can only be used if
                Data has a single set of ObservationFeatures corresponding to
                that arm.
            status_quo_features: ObservationFeatures to use as status quo.
                Either this or status_quo_name should be specified, not both.
            optimization_config: Optimization config defining how to optimize
                the model.
            fit_out_of_design: If specified, all training data is returned.
                Otherwise, only in design points are returned.
            default_model_gen_options: Options passed down to `model.gen(...)`.
            map_data_limit_total_rows: Subsample the map data so that the total
                number of rows is limited by this value.
            map_data_limit_rows_per_group: Subsample the map data so that the
                number of rows in the `map_key` column for each (arm, metric)
                is limited by this value.
        """

        if not isinstance(data, MapData):
            raise ValueError(  # pragma: no cover
                "`MapTorchModelBridge expects `MapData` instead of `Data`."
            )
        # pyre-fixme[4]: Attribute must be annotated.
        self._map_key_features = data.map_keys
        self._map_data_limit_total_rows = map_data_limit_total_rows
        self._map_data_limit_rows_per_group = map_data_limit_rows_per_group

        super().__init__(
            experiment=experiment,
            search_space=search_space,
            data=data,
            model=model,
            transforms=transforms,
            transform_configs=transform_configs,
            torch_dtype=torch_dtype,
            torch_device=torch_device,
            status_quo_name=status_quo_name,
            status_quo_features=status_quo_features,
            optimization_config=optimization_config,
            fit_out_of_design=fit_out_of_design,
            default_model_gen_options=default_model_gen_options,
        )

    @property
    def parameters_with_map_keys(self) -> List[str]:
        return self.parameters + self._map_key_features

    def _predict(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationData]:
        """This method is updated from `TorchModelBridge._predict(...) in that it
        will accept observation features with or without map_keys. If observation
        features do not contain map_keys, it will insert them based on
        `target_map_values`.
        """
        if not self.model:  # pragma: no cover
            raise ValueError(FIT_MODEL_ERROR.format(action="_model_predict"))
        # The fitted model expects map_keys. If they do not exist, we use the
        # target values.
        target_map_values = self._default_model_gen_options.get(
            "target_map_values", DEFAULT_TARGET_MAP_VALUES
        )
        for p in self._map_key_features:
            for obsf in observation_features:
                if p not in obsf.parameters:
                    obsf.parameters[p] = target_map_values[p]  # pyre-ignore[16]

        # Convert observation features to array
        X = observation_features_to_array(
            self.parameters_with_map_keys, observation_features
        )
        f, cov = not_none(self.model).predict(X=self._array_to_tensor(X))
        f = f.detach().cpu().clone().numpy()
        cov = cov.detach().cpu().clone().numpy()
        # Convert resulting arrays to observations
        return array_to_observation_data(f=f, cov=cov, outcomes=self.outcomes)

    def _fit(
        self,
        model: TorchModel,
        search_space: SearchSpace,
        observations: List[Observation],
        parameters: Optional[List[str]] = None,
    ) -> None:
        """The difference from `TorchModelBridge._fit(...)` is that we use
        `self.parameters_with_map_keys` instead of `self.parameters`.
        """
        self.parameters = list(search_space.parameters.keys())
        if parameters is None:
            parameters = self.parameters_with_map_keys
        super()._fit(
            model=model,
            search_space=search_space,
            observations=observations,
            parameters=parameters,
        )

    def _gen(
        self,
        n: int,
        search_space: SearchSpace,
        pending_observations: Dict[str, List[ObservationFeatures]],
        fixed_features: Optional[ObservationFeatures],
        model_gen_options: Optional[TConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ) -> GenResults:
        """An updated version of `TorchModelBridge._gen(...) that first injects
        `map_dim_to_target` (e.g., `{-1: 1.0}`) into `model_gen_options` so that
        the target values of the map_keys are known during candidate generation.
        """
        model_gen_options = self._add_map_dim_to_target(options=model_gen_options or {})
        return super()._gen(
            n=n,
            search_space=search_space,
            pending_observations=pending_observations,
            fixed_features=fixed_features,
            model_gen_options=model_gen_options,
            optimization_config=optimization_config,
        )

    def _array_to_observation_features(
        self, X: np.ndarray, candidate_metadata: Optional[List[TCandidateMetadata]]
    ) -> List[ObservationFeatures]:
        """The difference b/t this method and TorchModelBridge._update(...) is
        that this one makes use of `self.parameters_with_map_keys`.
        """
        return parse_observation_features(
            X=X,
            param_names=self.parameters_with_map_keys,
            candidate_metadata=candidate_metadata,
        )

    def _update(
        self,
        search_space: SearchSpace,
        observations: List[Observation],
        parameters: Optional[List[str]] = None,
    ) -> None:
        """The difference b/t this method and TorchModelBridge._update(...) is
        that this one makes use of `self.parameters_with_map_keys`.
        """
        return super()._update(  # pragma: no cover
            search_space=search_space,
            observations=observations,
            parameters=self.parameters_with_map_keys,
        )

    def _prepare_observations(
        self, experiment: Optional[Experiment], data: Optional[Data]
    ) -> List[Observation]:
        """The difference b/t this method and ModelBridge._prepare_observations(...)
        is that this one uses `observations_from_map_data`.
        """
        if experiment is None or data is None:
            return []  # pragma: no cover
        return observations_from_map_data(
            experiment=experiment,
            map_data=data,  # pyre-ignore[6]: Checked in __init__.
            map_keys_as_parameters=True,
            include_abandoned=self._fit_abandoned,
            limit_total_rows=self._map_data_limit_total_rows,
            limit_rows_per_group=self._map_data_limit_rows_per_group,
        )

    def _compute_in_design(
        self, search_space: SearchSpace, observations: List[Observation]
    ) -> List[bool]:
        """The difference b/t this method and ModelBridge._compute_in_design(...)
        is that this one correctly excludes map_keys when checking membership in
        search space (as map_keys are not explicitly in the search space).
        """
        return [  # pragma: no cover
            search_space.check_membership(
                # Exclude map key features when checking
                {
                    p: v
                    for p, v in obs.features.parameters.items()
                    if p not in self._map_key_features
                }
            )
            for obs in observations
        ]

    def _cross_validate(
        self,
        search_space: SearchSpace,
        cv_training_data: List[Observation],
        cv_test_points: List[ObservationFeatures],
        parameters: Optional[List[str]] = None,
    ) -> List[ObservationData]:
        """Make predictions at cv_test_points using only the data in obs_feats
        and obs_data. The difference from `TorchModelBridge._cross_validate`
        is that here we do cross validation on the parameters + map_keys. There
        is some extra logic to filter out out-of-design points in the map_key
        dimension.
        """
        if parameters is None:
            parameters = self.parameters_with_map_keys
        cv_test_data = super()._cross_validate(
            search_space=search_space,
            cv_training_data=cv_training_data,
            cv_test_points=cv_test_points,
            parameters=parameters,  # we pass the map_keys too by default
        )
        observation_features, observation_data = separate_observations(cv_training_data)
        # Since map_keys are used as features, there can be the possibility that
        # models for different outcomes were fit on different ranges of map_key
        # values; for example, this is the case if we (1) mix learning curve data with
        # standard data (taking default map value), or (2) are in a situation where
        # some learning curves start later than others. These prediction results are
        # "out-of-design" in the map_key dimension, so we should filter them out.
        map_key_ranges = self._get_map_key_ranges(
            observation_features=observation_features,
            observation_data=observation_data,
        )
        cv_test_data = self._filter_outcomes_out_of_map_range(
            observation_features=cv_test_points,
            observation_data=cv_test_data,
            map_key_ranges=map_key_ranges,
        )
        return cv_test_data

    def _filter_outcomes_out_of_map_range(
        self,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
        # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
        map_key_ranges: Dict[str, Dict[str, Optional[Tuple]]],
    ) -> List[ObservationData]:
        """Uses `map_key_ranges` to detect which `observation_features` have
        out-of-range map_keys and filters out the corresponding outcomes in
        `observation_data`.
        """
        filtered_observation_data = []
        for obsf, obsd in zip(observation_features, observation_data):
            metric_names = obsd.metric_names
            means = obsd.means
            covariance = obsd.covariance
            for o in self.outcomes:
                if o in metric_names:
                    for p in self._map_key_features:
                        map_key_value = obsf.parameters[p]
                        map_key_range = map_key_ranges[o][p]
                        if map_key_range is not None:
                            range_min, range_max = map_key_range
                            if map_key_value < range_min or map_key_value > range_max:
                                p_idx = metric_names.index(o)
                                metric_names.pop(p_idx)
                                means = np.delete(means, p_idx, axis=0)
                                covariance = np.delete(covariance, p_idx, axis=0)
                                covariance = np.delete(covariance, p_idx, axis=1)
                                break
            new_obsd = ObservationData(
                metric_names=metric_names, means=means, covariance=covariance
            )
            filtered_observation_data.append(new_obsd)
        return filtered_observation_data

    def _get_map_key_ranges(
        self,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
        # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    ) -> Dict[str, Dict[str, Optional[Tuple]]]:
        """Get ranges of map_key values in observation features. Returns a dict of the
        form: {"outcome": {"map_key": (min_val, max_val)}}.
        """
        map_values = {o: {p: [] for p in self._map_key_features} for o in self.outcomes}
        for obsd, obsf in zip(observation_data, observation_features):
            for p in self._map_key_features:
                param_value = obsf.parameters[p]
                for o in obsd.metric_names:
                    map_values[o][p].append(param_value)

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List` to avoid runtime subscripting errors.
        def get_range(values: List):
            return (min(values), max(values)) if len(values) > 0 else None

        return {
            o: {p: get_range(map_values[o][p]) for p in self._map_key_features}
            for o in self.outcomes
        }

    def _add_map_dim_to_target(self, options: TConfig) -> TConfig:
        """Convert `target_map_values` to `map_dim_to_target`, a form useable
        by the acquisition function and insert into options dict.
        """
        target_map_values = self._default_model_gen_options.get("target_map_values")
        if target_map_values is None:
            target_map_values = DEFAULT_TARGET_MAP_VALUES  # pragma: no cover
        param_and_map = self.parameters_with_map_keys
        map_dim_to_target = {
            param_and_map.index(p): target_map_values[p]  # pyre-ignore[16]
            for p in self._map_key_features
        }
        options[Keys.ACQF_KWARGS] = {  # pyre-ignore[32]
            **options.get(Keys.ACQF_KWARGS, {}),
            "map_dim_to_target": map_dim_to_target,
        }
        return options
