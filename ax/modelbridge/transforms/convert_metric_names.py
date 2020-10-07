#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.modelbridge.transforms.base import Transform
from ax.utils.common.docutils import copy_doc
from ax.utils.common.typeutils import not_none


class ConvertMetricNames(Transform):
    """Convert all metric names to canonical name as specified on a multi_type_experiment.

    For example, a multi-type experiment may have an offline simulator which attempts to
    approximate observations from some online system. We want to map the offline
    metric names to the corresponding online ones so the model can associate them.

    This is done by replacing metric names in the data with the corresponding
    online metric names.

    In the inverse transform, data will be mapped back onto the original metric names.
    By default, this transform is turned off. It can be enabled by passing the
    "perform_untransform" flag to the config.
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace],
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
        config: Optional[TConfig] = None,
    ) -> None:
        if config is None:
            raise ValueError("Config cannot be none.")

        self.metric_name_map: Dict[str, str] = config.get(  # pyre-ignore[8]
            "metric_name_map"
        )
        self.metric_name_to_trial_type: Dict[str, str] = config.get(  # pyre-ignore[8]
            "metric_name_to_trial_type"
        )
        self.trial_index_to_type: Dict[int, str] = config.get(  # pyre-ignore[8]
            "trial_index_to_type"
        )

        if self.metric_name_map is None:
            raise ValueError("Config must contain metric_name_map")

        if self.metric_name_to_trial_type is None:
            raise ValueError("Config must contain metric_name_to_trial_type")

        if self.trial_index_to_type is None:
            raise ValueError("Config must contain trial_index_to_type")

        for obsf in observation_features:
            if obsf.trial_index not in self.trial_index_to_type:
                raise ValueError("trial_index_to_type does not include all trials")

        # For each trial type, give a map from transformed name back to original
        # Usage: reverse_metric_name_map[trial_type][transformed_name] -> original_name
        self.reverse_metric_name_map: Dict[str, Dict[str, str]] = {}

        # For most practical cases we want to skip the untransform
        self.perform_untransform = config.get("perform_untransform", False)

        for orig_name, trans_name in self.metric_name_map.items():
            trial_type = self.metric_name_to_trial_type[orig_name]
            if trial_type in self.reverse_metric_name_map:
                self.reverse_metric_name_map[trial_type][trans_name] = orig_name
            else:
                self.reverse_metric_name_map[trial_type] = {trans_name: orig_name}

    @copy_doc(Transform.transform_observation_data)
    def transform_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
    ) -> List[ObservationData]:
        for obsd in observation_data:
            for i in range(len(obsd.metric_names)):
                if obsd.metric_names[i] in self.metric_name_map:
                    obsd.metric_names[i] = self.metric_name_map[obsd.metric_names[i]]
        return observation_data

    @copy_doc(Transform.untransform_observation_data)
    def untransform_observation_data(
        self,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
    ) -> List[ObservationData]:
        if not self.perform_untransform:
            return observation_data
        for i, obsd in enumerate(observation_data):
            trial_index = int(not_none(observation_features[i].trial_index))
            trial_type = self.trial_index_to_type[trial_index]
            reverse_map = self.reverse_metric_name_map.get(trial_type)

            if not reverse_map:
                continue

            for j in range(len(obsd.metric_names)):
                if obsd.metric_names[j] in reverse_map:
                    obsd.metric_names[j] = reverse_map[obsd.metric_names[j]]
        return observation_data


def tconfig_from_mt_experiment(experiment: MultiTypeExperiment) -> TConfig:
    """Generate the TConfig for this transform given a multi_type_experiment.

    Args:
        experiment: The experiment from which to generate the config.

    Returns:
        The transform config to pass into the ConvertMetricNames constructor.
    """

    trial_index_to_type = {t.index: t.trial_type for t in experiment.trials.values()}
    return {  # pyre-ignore[7]
        "metric_name_map": experiment._metric_to_canonical_name,
        "trial_index_to_type": trial_index_to_type,
        "metric_name_to_trial_type": experiment.metric_to_trial_type,
    }


def convert_mt_observations(
    observations: List[Observation], experiment: MultiTypeExperiment
) -> List[Observation]:
    """Apply ConvertMetricNames transform to observations for a MT experiment."""
    observation_data = [o.data for o in observations]
    observation_features = [o.features for o in observations]
    transform = ConvertMetricNames(
        search_space=None,
        observation_data=observation_data,
        observation_features=observation_features,
        config=tconfig_from_mt_experiment(experiment),
    )
    transformed_observations = transform.transform_observation_data(
        observation_data=observation_data, observation_features=observation_features
    )
    return [
        Observation(
            features=obs.features,
            data=transformed_observations[i],
            arm_name=obs.arm_name,
        )
        for i, obs in enumerate(observations)
    ]
