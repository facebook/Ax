#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from collections.abc import Mapping, Sequence

import numpy as np
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
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.generators.random.base import RandomGenerator
from ax.generators.random.in_sample import InSampleUniformGenerator
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
        # Extract generated points.
        # For normal generators these are used to deduplicate against.
        # For in-sample generators (LILO labeling) they are the selection
        # pool from which arms are drawn — not a dedup set.  The two use
        # cases have been shoehorned into the same code path; consider
        # splitting them into separate methods in a future refactor.
        generated_points = None
        is_in_sample = isinstance(self.generator, InSampleUniformGenerator)
        if self.generator.deduplicate:
            # For normal generators, exclude arms from FAILED trials so the
            # model may re-suggest them.  For in-sample generators this
            # exclusion is harmful: LILO labeling trials borrow arms from
            # regular trials, so a FAILED labeling trial would incorrectly
            # remove the original arm from the selection pool.  Use the
            # full arms_by_signature instead.
            arms_to_deduplicate = (
                self._experiment.arms_by_signature
                if is_in_sample
                else self._experiment.arms_by_signature_for_deduplication
            )
            # For in-sample generators, restrict to arms from trials that
            # have or expect observed data (COMPLETED, EARLY_STOPPED,
            # RUNNING). This prevents selecting arms from CANDIDATE/STAGED
            # trials that have never been evaluated.
            if is_in_sample:
                # Also exclude abandoned arms within data-expecting trials:
                # they have no observed data and would cause downstream
                # failures (e.g., LILO source resolution).
                abandoned_arm_names: set[str] = {
                    a.name
                    for t in self._experiment.trials.values()
                    if isinstance(t, BatchTrial)
                    for a in t.abandoned_arms
                }
                expecting_sigs = {
                    arm.signature
                    for trial in self._experiment.trials.values()
                    if trial.status.expecting_data
                    for arm in trial.arms
                    if arm.name not in abandoned_arm_names
                }
                arms_to_deduplicate = {
                    sig: arm
                    for sig, arm in arms_to_deduplicate.items()
                    if sig in expecting_sigs
                }
            generated_obs = [
                ObservationFeatures.from_arm(arm=arm)
                for arm in arms_to_deduplicate.values()
                if self._search_space.check_membership(parameterization=arm.parameters)
            ]
            # Transform
            for t in self.transforms.values():
                generated_obs = t.transform_observation_features(generated_obs)
            # Add pending observations -- already transformed.
            # Skipped for in-sample generators: pending observations include
            # CANDIDATE arms that should not enter the selection pool.
            if not is_in_sample:
                generated_obs.extend(
                    [
                        obs
                        for obs_list in pending_observations.values()
                        for obs in obs_list
                    ]
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
        X, w = self.generator.gen(
            n=n,
            search_space_digest=search_space_digest,
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

    def _set_status_quo(self, experiment: Experiment) -> None:
        pass
