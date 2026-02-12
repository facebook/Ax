#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from collections.abc import Mapping, Sequence

from ax.adapter.adapter_utils import (
    extract_parameter_constraints,
    extract_search_space_digest,
    get_fixed_features,
    parse_observation_features,
    transform_callback,
)
from ax.adapter.base import Adapter, DataLoaderConfig, GenResults
from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.generators.random.base import RandomGenerator
from ax.generators.types import TConfig


class RandomAdapter(Adapter):
    """An adapter for using purely random ``RandomGenerator``s.
    Data and optimization configs are not required.

    Please refer to base ``Adapter`` class for documentation of constructor arguments.
    """

    def __init__(
        self,
        *,
        experiment: Experiment,
        generator: RandomGenerator,
        search_space: SearchSpace | None = None,
        data: Data | None = None,
        transforms: Sequence[type[Transform]] | None = None,
        transform_configs: Mapping[str, TConfig] | None = None,
        optimization_config: OptimizationConfig | None = None,
        fit_tracking_metrics: bool = True,
        fit_on_init: bool = True,
        data_loader_config: DataLoaderConfig | None = None,
    ) -> None:
        self.parameters: list[str] = []
        super().__init__(
            search_space=search_space,
            generator=generator,
            transforms=transforms,
            experiment=experiment,
            data=data,
            transform_configs=transform_configs,
            optimization_config=optimization_config,
            expand_model_space=False,
            data_loader_config=data_loader_config,
            fit_tracking_metrics=fit_tracking_metrics,
            fit_on_init=fit_on_init,
        )
        # Re-assign for more precise typing.
        self.generator: RandomGenerator = generator

    def _fit(
        self,
        search_space: SearchSpace,
        experiment_data: ExperimentData,
    ) -> None:
        """Extracts the list of parameters from the search space."""
        self.parameters = list(search_space.parameters.keys())

    def _gen(
        self,
        n: int,
        search_space: SearchSpace,
        pending_observations: dict[str, list[ObservationFeatures]],
        fixed_features: ObservationFeatures | None,
        optimization_config: OptimizationConfig | None,
        model_gen_options: TConfig | None,
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
        X, w = self.generator.gen(
            n=n,
            search_space_digest=search_space_digest,
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

    def _set_status_quo(self, experiment: Experiment) -> None:
        pass
