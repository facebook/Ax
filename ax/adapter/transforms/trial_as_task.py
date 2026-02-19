#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from logging import Logger
from typing import TYPE_CHECKING

from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.core.utils import get_target_trial_index
from ax.generators.types import TConfig
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from pyre_extensions import none_throws

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


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

    If there are fewer than 2 trials or levels, the transform is a no-op.

    Transform is done in-place.
    """

    requires_data_for_initialization: bool = True

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: adapter_module.base.Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        super().__init__(
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        assert adapter is not None, "TrialAsTask requires adapter"
        # Identify values of trial.
        trials = set(
            none_throws(experiment_data).arm_data.index.get_level_values("trial_index")
        )
        if None in trials:
            raise ValueError(
                "Unable to use trial as task since not all observations have "
                "trial specified."
            )
        # Get trial level map
        if "trial_level_map" in self.config:
            # pyre-ignore [9]
            trial_level_map: dict[str, dict[int | str, int | str]] = self.config[
                "trial_level_map"
            ]
            # Validate
            self.trial_level_map: dict[str, dict[int, int | str]] = {}
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
        elif self.config.get("trial_type_as_task", False):
            # map long run trials to 0 and short run trials to 1
            self.trial_level_map = {TRIAL_PARAM: {}}
            for trial_index in trials:
                trial_type = adapter._experiment.trials[trial_index].trial_type
                if trial_type is not None and trial_type == Keys.LONG_RUN:
                    self.trial_level_map[TRIAL_PARAM][trial_index] = "0"
                else:
                    self.trial_level_map[TRIAL_PARAM][trial_index] = "1"
        else:
            # Set TRIAL_PARAM for each trial to the corresponding trial_index.
            self.trial_level_map = {TRIAL_PARAM: {int(b): str(b) for b in trials}}

        if len(self.trial_level_map) == 1:
            level_dict = next(iter(self.trial_level_map.values()))
            self.inverse_map: dict[int | str, int] | None = {
                v: k for k, v in level_dict.items()
            }
        else:
            self.inverse_map = None
        # Compute target values
        self.target_values: dict[str, int | str] = {}

        for p_name, trial_map in list(self.trial_level_map.items()):
            if len(set(trial_map.values())) < 2:
                # If there are less than two distinct levels, then we don't need to
                # create a task parameter and the transform is a no-op.
                del self.trial_level_map[p_name]
                continue
            if "target_trial" in self.config:
                target_trial = int(self.config["target_trial"])
            else:
                target_trial = none_throws(
                    get_target_trial_index(
                        experiment=none_throws(none_throws(adapter)._experiment)
                    )
                )
                logger.debug(f"Setting target value for {p_name} to {target_trial}")
            self.target_values[p_name] = trial_map[target_trial]

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        """If the trial_index is in ``level_dict``, which is from trials with data,
        use the value for that trial index from ``level_dict``, which is probably
        the stringified trial index, as the task parameter. If the trial index is
        None or not in ``level_dict``, use the value from the most recent trial.

        Trial indices not observed are probably pending points from candidate or staged
        trials. Trial indices set to None are probably pending points passed in by the
        user.
        """
        if len(self.trial_level_map) == 0:
            # no-op
            return observation_features
        for obsf in observation_features:
            for p_name, level_dict in self.trial_level_map.items():
                if obsf.trial_index is not None and int(obsf.trial_index) in level_dict:
                    obsf.parameters[p_name] = level_dict[int(obsf.trial_index)]
                elif len(obsf.parameters) > 0:
                    # If the trial index is none, but the parameters are not empty
                    # perform the transform by assuming the observation is from the
                    # most recent trial. This is needed for generating trials composed
                    # of points from multiple models.
                    obsf.parameters[p_name] = level_dict[max(level_dict)]
        return observation_features

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        if len(self.trial_level_map) == 0:
            # no-op
            return search_space
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
                # they are encoded in TaskChoiceToIntTaskChoice
                is_ordered=is_int,
                is_task=True,
                sort_values=True,
                target_value=self.target_values[p_name],
            )
            search_space.add_parameter(trial_param)
        return search_space

    def untransform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        """If task parameters have been added to observation features by
        this parameter, then remove those task parameters and restore
        the trial index/

        Args:
            observation_features: List of observation features to untransform.

        Returns:
            List of observation features with task parameters removed and trial
                index restored.
        """
        if len(self.trial_level_map) == 0:
            # no-op
            return observation_features
        for obsf in observation_features:
            for p_name in self.trial_level_map:
                obsf.parameters.pop(p_name)
        return observation_features

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        arm_data = experiment_data.arm_data
        for p_name, level_dict in self.trial_level_map.items():
            arm_data[p_name] = arm_data.index.get_level_values("trial_index").map(
                level_dict
            )
        return ExperimentData(
            arm_data=arm_data, observation_data=experiment_data.observation_data
        )
