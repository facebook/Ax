#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from collections.abc import Mapping, Sequence

import numpy as np
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.modelbridge.base import Adapter, DataLoaderConfig, GenResults
from ax.modelbridge.modelbridge_utils import (
    extract_parameter_constraints,
    extract_search_space_digest,
    get_fixed_features,
    parse_observation_features,
    transform_callback,
)
from ax.modelbridge.transforms.base import Transform
from ax.models.random.base import RandomGenerator
from ax.models.types import TConfig


class RandomAdapter(Adapter):
    """An adaptor for using purely random ``RandomGenerator``s.
    Data and optimization configs are not required.

    Please refer to base ``Adapter`` class for documentation of constructor arguments.
    """

    def __init__(
        self,
        *,
        experiment: Experiment,
        model: RandomGenerator,
        search_space: SearchSpace | None = None,
        data: Data | None = None,
        transforms: Sequence[type[Transform]] | None = None,
        transform_configs: Mapping[str, TConfig] | None = None,
        optimization_config: OptimizationConfig | None = None,
        fit_tracking_metrics: bool = True,
        fit_on_init: bool = True,
        data_loader_config: DataLoaderConfig | None = None,
        fit_out_of_design: bool | None = None,
        fit_abandoned: bool | None = None,
    ) -> None:
        self.parameters: list[str] = []
        super().__init__(
            search_space=search_space,
            model=model,
            transforms=transforms,
            experiment=experiment,
            data=data,
            transform_configs=transform_configs,
            optimization_config=optimization_config,
            expand_model_space=False,
            data_loader_config=data_loader_config,
            fit_out_of_design=fit_out_of_design,
            fit_abandoned=fit_abandoned,
            fit_tracking_metrics=fit_tracking_metrics,
            fit_on_init=fit_on_init,
        )
        # Re-assign for more precise typing.
        self.model: RandomGenerator = model

    def _fit(
        self,
        search_space: SearchSpace,
        observations: list[Observation] | None = None,
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
        # Extract generated points to deduplicate against.
        generated_points = None
        if self.model.deduplicate:
            arms_to_deduplicate = self._experiment.arms_by_signature_for_deduplication
            generated_obs = [
                ObservationFeatures.from_arm(arm=arm)
                for arm in arms_to_deduplicate.values()
            ]
            # Transform
            for t in self.transforms.values():
                generated_obs = t.transform_observation_features(generated_obs)
            # Add pending observations -- already transformed.
            generated_obs.extend(
                [obs for obs_list in pending_observations.values() for obs in obs_list]
            )
            if len(generated_obs) > 0:
                # Extract generated points array (n x d).
                generated_points = np.array(
                    [
                        [obs.parameters[p] for p in self.parameters]
                        for obs in generated_obs
                    ]
                )
                # Take unique points only, since there may be duplicates coming
                # from pending observations for different metrics.
                generated_points = np.unique(generated_points, axis=0)

        # Generate the candidates
        X, w = self.model.gen(
            n=n,
            bounds=search_space_digest.bounds,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features_dict,
            model_gen_options=model_gen_options,
            rounding_func=transform_callback(self.parameters, self.transforms),
            generated_points=generated_points,
        )
        observation_features = parse_observation_features(X, self.parameters)
        return GenResults(
            observation_features=observation_features,
            weights=w.tolist(),
        )

    def _predict(
        self,
        observation_features: list[ObservationFeatures],
        use_posterior_predictive: bool = False,
    ) -> list[ObservationData]:
        """Apply terminal transform, predict, and reverse terminal transform on
        output.
        """
        raise NotImplementedError("RandomAdapter does not support prediction.")

    def _cross_validate(
        self,
        search_space: SearchSpace,
        cv_training_data: list[Observation],
        cv_test_points: list[ObservationFeatures],
        use_posterior_predictive: bool = False,
    ) -> list[ObservationData]:
        raise NotImplementedError

    def _set_status_quo(self, experiment: Experiment) -> None:
        pass
