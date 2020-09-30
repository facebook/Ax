#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import numpy as np
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.modelbridge.transforms.base import Transform


TRIAL_PARAM = "TRIAL_PARAM"


class TrialAsTask(Transform):
    """Convert trial to one or more task parameters.

    How trial is mapped to parameter is specified with a map like
    {parameter_name: {trial_index: level name}}.
    For example,
    {"trial_param1": {0: "level1", 1: "level1", 2: "level2"},}
    will create choice parameters "trial_param1" with is_task=True.
    Observations with trial 0 or 1 will have "trial_param1" set to "level1",
    and those with trial 2 will have "trial_param1" set to "level2". Multiple
    parameter names and mappings can be specified in this dict.

    The trial level mapping can be specified in config["trial_level_map"]. If
    not specified, defaults to a parameter with a level for every trial index.

    For the reverse transform, if there are multiple mappings in the transform
    the trial will not be set.

    Will raise if trial not specified for every point in the training data.

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
        config: Optional[TConfig] = None,
    ) -> None:
        # Identify values of trial.
        trials = {obsf.trial_index for obsf in observation_features}
        if None in trials:
            raise ValueError(
                "Unable to use trial as task since not all observations have "
                "trial specified."
            )
        # Get trial level map
        if config is not None and "trial_level_map" in config:
            # pyre: Attribute `trial_level_map` declared in class `ax.
            # pyre: modelbridge.transforms.trial_as_task.TrialAsTask` has type
            # pyre: `Dict[str, Dict[int, str]]` but is used as type `typing.
            # pyre-fixme[8]: Union[float, int, str]`.
            self.trial_level_map: Dict[str, Dict[int, str]] = config["trial_level_map"]
            # Validate
            for _p_name, level_dict in self.trial_level_map.items():
                # Check that trials match those in data
                level_map = set(level_dict.keys())
                if not trials.issubset(level_map):
                    raise ValueError(
                        f"Not all trials in data ({trials}) contained "
                        f"in trial level map for {_p_name} ({level_map})"
                    )
        else:
            # Set TRIAL_PARAM for each trial to the corresponding trial_index.
            # pyre-fixme[6]: Expected `Union[bytes, str, typing.SupportsInt]` for
            #  1st param but got `Optional[np.int64]`.
            self.trial_level_map = {TRIAL_PARAM: {int(b): str(b) for b in trials}}
        if len(self.trial_level_map) == 1:
            level_dict = next(iter(self.trial_level_map.values()))
            self.inverse_map: Optional[Dict[str, int]] = {
                v: k for k, v in level_dict.items()
            }
        else:
            self.inverse_map = None

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            if obsf.trial_index is not None:
                for p_name, level_dict in self.trial_level_map.items():
                    # pyre-fixme[6]: Expected `Union[bytes, str,
                    #  typing.SupportsInt]` for 1st param but got `Optional[np.int64]`.
                    obsf.parameters[p_name] = level_dict[int(obsf.trial_index)]
                obsf.trial_index = None
        return observation_features

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        for p_name, level_dict in self.trial_level_map.items():
            level_values = list(set(level_dict.values()))
            if len(level_values) == 1:
                raise ValueError(
                    "TrialAsTask transform expects 2+ task params, "
                    + f"only 1 found: {level_values}"
                )
            trial_param = ChoiceParameter(
                name=p_name,
                parameter_type=ParameterType.STRING,
                # Expected `List[Optional[typing.Union[bool, float, str]]]` for 4th
                # parameter `values` to call
                # `ax.core.parameter.ChoiceParameter.__init__` but got
                # `List[str]`.
                # pyre-fixme[6]:
                values=level_values,
                is_ordered=False,
                is_task=True,
            )
            search_space.add_parameter(trial_param)
        return search_space

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in self.trial_level_map:
                pval = obsf.parameters.pop(p_name)
            if self.inverse_map is not None:
                # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
                obsf.trial_index = np.int64(self.inverse_map[pval])
        return observation_features
