#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger
from typing import Dict, List, Optional, TYPE_CHECKING, Union

from ax.core.observation import Observation, ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.transforms.base import Transform
from ax.models.types import TConfig
from ax.utils.common.logger import get_logger

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401


TRIAL_PARAM = "TRIAL_PARAM"
logger: Logger = get_logger(__name__)


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

    The created parameter will be given a target value that will default to the
    lowest trial index in the mapping, or can be provided in config["target_trial"].

    Will raise if trial not specified for every point in the training data.

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        observations: Optional[List[Observation]] = None,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        assert observations is not None, "TrialAsTask requires observations"
        # Identify values of trial.
        trials = {obs.features.trial_index for obs in observations}
        if isinstance(search_space, RobustSearchSpace):
            raise UnsupportedError(
                "TrialAsTask transform is not supported for RobustSearchSpace."
            )
        if None in trials:
            raise ValueError(
                "Unable to use trial as task since not all observations have "
                "trial specified."
            )
        # Get trial level map
        if config is not None and "trial_level_map" in config:
            # pyre-ignore [9]
            trial_level_map: Dict[str, Dict[Union[int, str], Union[int, str]]] = config[
                "trial_level_map"
            ]
            # Validate
            self.trial_level_map: Dict[str, Dict[int, Union[int, str]]] = {}
            for _p_name, level_dict in trial_level_map.items():
                # cast trial index as an integer
                int_keyed_level_dict = {
                    int(trial_index): v for trial_index, v in level_dict.items()
                }
                self.trial_level_map[_p_name] = int_keyed_level_dict
                # Check that trials match those in data
                level_map = set(int_keyed_level_dict.keys())
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
            self.inverse_map: Optional[Dict[Union[int, str], int]] = {
                v: k for k, v in level_dict.items()
            }
        else:
            self.inverse_map = None
        # Compute target values
        self.target_values: Dict[str, Union[int, str]] = {}
        for p_name, trial_map in self.trial_level_map.items():
            if config is not None and "target_trial" in config:
                target_trial = int(config["target_trial"])  # pyre-ignore [6]
            else:
                target_trial = min(trial_map.keys())
                logger.debug(f"Setting target value for {p_name} to {target_trial}")
            self.target_values[p_name] = trial_map[target_trial]

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

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        for p_name, level_dict in self.trial_level_map.items():
            level_values = sorted(set(level_dict.values()))
            if len(level_values) < 2:
                details = (
                    f"only 1 found: {level_values}" if level_values else "none found"
                )
                raise ValueError(
                    f"TrialAsTask transform expects 2+ task params, {details}"
                )
            is_int = all(isinstance(val, int) for val in level_values)
            trial_param = ChoiceParameter(
                name=p_name,
                parameter_type=ParameterType.INT if is_int else ParameterType.STRING,
                values=level_values,  # pyre-fixme [6]
                # if all values are integers, retain the original order
                # they are encoded in TaskEncode
                is_ordered=is_int,
                is_task=True,
                sort_values=True,
                target_value=self.target_values[p_name],
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
                # pyre-fixme[61]: `pval` may not be initialized here.
                obsf.trial_index = self.inverse_map[pval]
        return observation_features
