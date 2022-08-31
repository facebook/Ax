#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, TYPE_CHECKING

from ax.core.arm import Arm
from ax.core.observation import Observation, ObservationFeatures
from ax.core.parameter import ChoiceParameter, FixedParameter, ParameterType
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.transforms.base import Transform
from ax.models.types import TConfig
from ax.utils.common.typeutils import checked_cast

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401  # pragma: no cover


class SearchSpaceToChoice(Transform):
    """Replaces the search space with a single choice parameter, whose values
    are the signatures of the arms observed in the data.

    This transform is meant to be used with ThompsonSampler.

    Choice parameter will be unordered unless config["use_ordered"] specifies
    otherwise.

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        observations: Optional[List[Observation]] = None,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        assert search_space is not None, "SearchSpaceToChoice requires search space"
        assert observations is not None, "SeachSpaceToChoice requires observations"
        super().__init__(
            search_space=search_space,
            observations=observations,
            config=config,
        )
        if any(p.is_fidelity for p in search_space.parameters.values()):
            raise ValueError(
                "Cannot perform SearchSpaceToChoice conversion if fidelity "
                "parameters are present"
            )
        if isinstance(search_space, RobustSearchSpace):
            raise UnsupportedError(
                "SearchSpaceToChoice transform is not supported for RobustSearchSpace."
            )
        self.parameter_name = "arms"
        # pyre-fixme[4]: Attribute must be annotated.
        self.signature_to_parameterization = {
            Arm(parameters=obs.features.parameters).signature: obs.features.parameters
            for obs in observations
        }

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        values = list(self.signature_to_parameterization.keys())
        if len(values) > 1:
            parameter = ChoiceParameter(
                name=self.parameter_name,
                parameter_type=ParameterType.STRING,
                values=values,
                is_ordered=checked_cast(bool, self.config.get("use_ordered", False)),
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
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            obsf.parameters = {
                self.parameter_name: Arm(parameters=obsf.parameters).signature
            }
        return observation_features

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            signature = obsf.parameters[self.parameter_name]
            obsf.parameters = self.signature_to_parameterization[signature]
        return observation_features
