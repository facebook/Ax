#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.modelbridge.transforms.base import Transform
from ax.models.types import TConfig

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401  # pragma: no cover


class MetricsAsTask(Transform):
    """Convert metrics to a task parameter.

    For each metric to be used as a task, the config must specify a list of the
    target metrics for that particular task metric. So,

    config = {
        'metric_task_map': {
            'metric1': ['metric2', 'metric3'],
            'metric2': ['metric3'],
        }
    }

    means that metric2 will be given additional task observations of metric1,
    and metric3 will be given additional task observations of both metric1 and
    metric2. Note here that metric2 and metric3 are the target tasks, and this
    map is from base tasks to target tasks.
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        observations: Optional[List[Observation]] = None,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        # Use config to specify metric task map
        if config is None or "metric_task_map" not in config:
            raise ValueError("config must specify metric_task_map")
        self.metric_task_map: Dict[str, List[str]] = config[  # pyre-ignore
            "metric_task_map"
        ]
        self.task_values: List[str] = list(self.metric_task_map.keys())
        assert "TARGET" not in self.task_values
        self.task_values.append("TARGET")

    def transform_observations(
        self,
        observations: List[Observation],
    ) -> List[Observation]:
        new_observations = []
        for obs in observations:
            # For the original observation, all the metrics with the new task param
            params = obs.features.parameters.copy()
            params["METRIC_TASK"] = "TARGET"
            new_observations.append(
                Observation(
                    features=obs.features.clone(replace_parameters=params),
                    data=obs.data,
                    arm_name=obs.arm_name,
                )
            )
            # Split out observations for the task metrics
            for task_metric, target_metrics in self.metric_task_map.items():
                if task_metric in obs.data.metric_names:
                    # Make an observation for this task metric.
                    params = obs.features.parameters.copy()
                    params["METRIC_TASK"] = task_metric
                    new_obs_feats = obs.features.clone(replace_parameters=params)
                    new_obs_data = ObservationData(
                        metric_names=target_metrics,
                        means=obs.data.means_dict[task_metric]  # pyre-ignore
                        * np.ones(len(target_metrics)),
                        covariance=np.diag(
                            obs.data.covariance_matrix[task_metric][task_metric]
                            * np.ones(len(target_metrics))
                        ),
                    )
                    new_observations.append(
                        Observation(
                            features=new_obs_feats,
                            data=new_obs_data,
                            arm_name=obs.arm_name,
                        )
                    )
        return new_observations

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        """
        If transforming features without data, map them to the target.
        """
        for obsf in observation_features:
            obsf.parameters["METRIC_TASK"] = "TARGET"
        return observation_features

    def untransform_observations(
        self, observations: List[Observation]
    ) -> List[Observation]:
        # Drop any observations that are not TARGET, and remove the param.
        new_observations = []
        for obs in observations:
            task = obs.features.parameters.pop("METRIC_TASK")
            if task == "TARGET":
                new_observations.append(
                    Observation(
                        features=obs.features, data=obs.data, arm_name=obs.arm_name
                    )
                )
        return new_observations

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        # Add task parameter
        task_param = ChoiceParameter(
            name="METRIC_TASK",
            parameter_type=ParameterType.STRING,
            values=self.task_values,  # pyre-ignore
            is_ordered=False,
            is_task=True,
            sort_values=True,
        )
        search_space.add_parameter(task_param)
        return search_space

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        # This is called during gen. We shouldn't gen for any task other than
        # the target task.
        for obsf in observation_features:
            task = obsf.parameters.pop("METRIC_TASK")
            if task != "TARGET":
                raise ValueError(f"Got point for task {task}. Something went wrong.")
        return observation_features
