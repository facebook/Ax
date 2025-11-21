#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional, TYPE_CHECKING

from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.core.arm import Arm
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ChoiceParameter, FixedParameter, ParameterType
from ax.core.search_space import SearchSpace
from ax.core.types import TParameterization, TParamValue
from ax.generators.types import TConfig
from pyre_extensions import assert_is_instance, none_throws

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


class SearchSpaceToChoice(Transform):
    """Replaces the search space with a single choice parameter, whose values
    are the signatures of the arms observed in the data.

    This transform is meant to be used with ThompsonSampler.

    Choice parameter will be unordered unless config["use_ordered"] specifies
    otherwise.

    Transform is done in-place.
    """

    requires_data_for_initialization: bool = True

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: Optional["adapter_module.base.Adapter"] = None,
        config: TConfig | None = None,
    ) -> None:
        assert search_space is not None, "SearchSpaceToChoice requires search space"
        super().__init__(
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        if any(p.is_fidelity for p in search_space.parameters.values()):
            raise ValueError(
                "Cannot perform SearchSpaceToChoice conversion if fidelity "
                "parameters are present"
            )
        self.parameter_name = "arms"
        self.parameter_names: list[str] = list(search_space.parameters)
        arm_data = none_throws(experiment_data).arm_data[self.parameter_names]
        arm_data = arm_data[self.parameter_names]
        self.signature_to_parameterization: dict[str, TParameterization] = {
            Arm(parameters=row).signature: row.copy()
            for row in arm_data.to_dict(orient="records")
        }

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        values: list[TParamValue] = list(self.signature_to_parameterization.keys())
        if len(values) > 1:
            parameter = ChoiceParameter(
                name=self.parameter_name,
                parameter_type=ParameterType.STRING,
                values=values,
                is_ordered=assert_is_instance(
                    self.config.get("use_ordered", False),
                    bool,
                ),
                sort_values=False,
            )
        else:
            parameter = FixedParameter(
                name=self.parameter_name,
                parameter_type=ParameterType.STRING,
                value=values[0],
            )
        return SearchSpace(parameters=[parameter])

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            # if obsf.parameters is not an empty dict
            if len(obsf.parameters) != 0:
                obsf.parameters = {
                    self.parameter_name: Arm(parameters=obsf.parameters).signature
                }
        return observation_features

    def untransform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            # Do not untransform empty dict as it wasn't transformed in the first place
            if len(obsf.parameters) != 0:
                signature = assert_is_instance(
                    obsf.parameters[self.parameter_name], str
                )
                obsf.parameters = self.signature_to_parameterization[signature]
        return observation_features

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        arm_data = experiment_data.arm_data
        if arm_data.empty:
            return experiment_data
        arm_data[self.parameter_name] = arm_data[self.parameter_names].apply(
            lambda row: Arm(parameters=row).signature, axis=1
        )
        arm_data = arm_data[[self.parameter_name, "metadata"]]
        return ExperimentData(
            arm_data=arm_data,
            observation_data=experiment_data.observation_data,
        )
