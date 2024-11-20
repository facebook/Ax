#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from typing import Any

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
from ax.modelbridge.transforms.base import Transform
from ax.models.random.base import RandomModel
from ax.models.types import TConfig


FIT_MODEL_ERROR = "Model must be fit before {action}."


class RandomModelBridge(ModelBridge):
    """A model bridge for using purely random 'models'.
    Data and optimization configs are not required.

    This model bridge interfaces with RandomModel.

    Attributes:
        model: A RandomModel used to generate candidates
            (note: this an awkward use of the word 'model').
        parameters: Params found in search space on modelbridge init.

    Args:
        experiment: Is used to get arm parameters. Is not mutated.
        search_space: Search space for fitting the model. Constraints need
            not be the same ones used in gen. RangeParameter bounds are
            considered soft and will be expanded to match the range of the
            data sent in for fitting, if expand_model_space is True.
        data: Ax Data.
        model: Interface will be specified in subclass. If model requires
            initialization, that should be done prior to its use here.
        transforms: List of uninitialized transform classes. Forward
            transforms will be applied in this order, and untransforms in
            the reverse order.
        transform_configs: A dictionary from transform name to the
            transform config dictionary.
        status_quo_name: Name of the status quo arm. Can only be used if
            Data has a single set of ObservationFeatures corresponding to
            that arm.
        status_quo_features: ObservationFeatures to use as status quo.
            Either this or status_quo_name should be specified, not both.
        optimization_config: Optimization config defining how to optimize
            the model.
        fit_out_of_design: If specified, all training data are used.
            Otherwise, only in design points are used.
        fit_abandoned: Whether data for abandoned arms or trials should be
            included in model training data. If ``False``, only
            non-abandoned points are returned.
        fit_tracking_metrics: Whether to fit a model for tracking metrics.
            Setting this to False will improve runtime at the expense of
            models not being available for predicting tracking metrics.
            NOTE: This can only be set to False when the optimization config
            is provided.
        fit_on_init: Whether to fit the model on initialization. This can
            be used to skip model fitting when a fitted model is not needed.
            To fit the model afterwards, use `_process_and_transform_data`
            to get the transformed inputs and call `_fit_if_implemented` with
            the transformed inputs.
    """

    # pyre-fixme[13]: Attribute `model` is never initialized.
    model: RandomModel
    # pyre-fixme[13]: Attribute `parameters` is never initialized.
    parameters: list[str]

    def __init__(
        self,
        search_space: SearchSpace,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        model: Any,
        transforms: list[type[Transform]] | None = None,
        experiment: Experiment | None = None,
        data: Data | None = None,
        transform_configs: dict[str, TConfig] | None = None,
        status_quo_name: str | None = None,
        status_quo_features: ObservationFeatures | None = None,
        optimization_config: OptimizationConfig | None = None,
        fit_out_of_design: bool = False,
        fit_abandoned: bool = False,
        fit_tracking_metrics: bool = True,
        fit_on_init: bool = True,
    ) -> None:
        super().__init__(
            search_space=search_space,
            model=model,
            transforms=transforms,
            experiment=experiment,
            data=data,
            transform_configs=transform_configs,
            status_quo_name=status_quo_name,
            status_quo_features=status_quo_features,
            optimization_config=optimization_config,
            expand_model_space=False,
            fit_out_of_design=fit_out_of_design,
            fit_abandoned=fit_abandoned,
            fit_tracking_metrics=fit_tracking_metrics,
            fit_on_init=fit_on_init,
        )

    def _fit(
        self,
        model: RandomModel,
        search_space: SearchSpace,
        observations: list[Observation] | None = None,
    ) -> None:
        self.model = model
        # Extract and fix parameters from initial search space.
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
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationData]:
        """Apply terminal transform, predict, and reverse terminal transform on
        output.
        """
        raise NotImplementedError("RandomModelBridge does not support prediction.")

    def _cross_validate(
        self,
        search_space: SearchSpace,
        cv_training_data: list[Observation],
        cv_test_points: list[ObservationFeatures],
        use_posterior_predictive: bool = False,
    ) -> list[ObservationData]:
        raise NotImplementedError

    def _set_status_quo(
        self,
        experiment: Experiment | None,
        status_quo_name: str | None,
        status_quo_features: ObservationFeatures | None,
    ) -> None:
        pass
