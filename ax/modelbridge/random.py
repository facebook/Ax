#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.modelbridge.base import GenResults, ModelBridge
from ax.modelbridge.modelbridge_utils import (
    extract_parameter_constraints,
    extract_search_space_digest,
    get_fixed_features,
    parse_observation_features,
    transform_callback,
)
from ax.models.random.base import RandomModel
from ax.models.types import TConfig
from ax.utils.common.docutils import copy_doc


FIT_MODEL_ERROR = "Model must be fit before {action}."


# pyre-fixme[13]: Attribute `model` is never initialized.
# pyre-fixme[13]: Attribute `parameters` is never initialized.
class RandomModelBridge(ModelBridge):
    """A model bridge for using purely random 'models'.
    Data and optimization configs are not required.

    This model bridge interfaces with RandomModel.

    Attributes:
        model: A RandomModel used to generate candidates
            (note: this an awkward use of the word 'model').
        parameters: Params found in search space on modelbridge init.
    """

    model: RandomModel
    parameters: List[str]

    def _fit(
        self,
        model: RandomModel,
        search_space: SearchSpace,
        observations: Optional[List[Observation]] = None,
    ) -> None:
        self.model = model
        # Extract and fix parameters from initial search space.
        self.parameters = list(search_space.parameters.keys())

    @copy_doc(ModelBridge.update)
    def update(self, new_data: Data, experiment: Experiment) -> None:
        pass  # pragma: no cover

    def _gen(
        self,
        n: int,
        search_space: SearchSpace,
        pending_observations: Dict[str, List[ObservationFeatures]],
        fixed_features: Optional[ObservationFeatures],
        optimization_config: Optional[OptimizationConfig],
        model_gen_options: Optional[TConfig],
    ) -> GenResults:
        """Generate new candidates according to a search_space."""
        # Extract parameter values
        search_space_digest = extract_search_space_digest(search_space, self.parameters)
        # Get fixed features
        fixed_features_dict = get_fixed_features(fixed_features, self.parameters)
        # Extract param constraints
        linear_constraints = extract_parameter_constraints(
            search_space.parameter_constraints, self.parameters
        )
        # Generate the candidates
        X, w = self.model.gen(
            n=n,
            bounds=search_space_digest.bounds,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features_dict,
            model_gen_options=model_gen_options,
            rounding_func=transform_callback(self.parameters, self.transforms),
        )
        observation_features = parse_observation_features(X, self.parameters)
        return GenResults(
            observation_features=observation_features,
            weights=w.tolist(),
        )

    def _predict(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationData]:
        """Apply terminal transform, predict, and reverse terminal transform on
        output.
        """
        raise NotImplementedError("RandomModelBridge does not support prediction.")

    def _cross_validate(
        self,
        search_space: SearchSpace,
        cv_training_data: List[Observation],
        cv_test_points: List[ObservationFeatures],
    ) -> List[ObservationData]:
        raise NotImplementedError

    def _set_status_quo(
        self,
        experiment: Optional[Experiment],
        status_quo_name: Optional[str],
        status_quo_features: Optional[ObservationFeatures],
    ) -> None:
        pass
