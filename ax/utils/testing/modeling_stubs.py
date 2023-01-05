#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger
from typing import Any, Dict, List, Optional, Type

import numpy as np
from ax.core.experiment import Experiment
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import FixedParameter, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.completion_criterion import MinimumPreferenceOccurances
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.int_to_float import IntToFloat
from ax.utils.common.logger import get_logger
from ax.utils.testing.core_stubs import (
    get_experiment,
    get_search_space,
    get_search_space_for_value,
)

logger: Logger = get_logger(__name__)


# Observations


def get_observation_features() -> ObservationFeatures:
    return ObservationFeatures(
        parameters={"x": 2.0, "y": 10.0}, trial_index=np.int64(0)
    )


def get_observation(
    first_metric_name: str = "a",
    second_metric_name: str = "b",
) -> Observation:
    return Observation(
        features=ObservationFeatures(
            parameters={"x": 2.0, "y": 10.0}, trial_index=np.int64(0)
        ),
        data=ObservationData(
            means=np.array([2.0, 4.0]),
            covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
            metric_names=[first_metric_name, second_metric_name],
        ),
        arm_name="1_1",
    )


def get_observation1(
    first_metric_name: str = "a",
    second_metric_name: str = "b",
) -> Observation:
    return Observation(
        features=ObservationFeatures(
            parameters={"x": 2.0, "y": 10.0}, trial_index=np.int64(0)
        ),
        data=ObservationData(
            means=np.array([2.0, 4.0]),
            covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
            metric_names=[first_metric_name, second_metric_name],
        ),
        arm_name="1_1",
    )


def get_observation_status_quo0(
    first_metric_name: str = "a",
    second_metric_name: str = "b",
) -> Observation:
    return Observation(
        features=ObservationFeatures(
            parameters={"w": 0.85, "x": 1, "y": "baz", "z": False},
            trial_index=np.int64(0),
        ),
        data=ObservationData(
            means=np.array([2.0, 4.0]),
            covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
            metric_names=[first_metric_name, second_metric_name],
        ),
        arm_name="0_0",
    )


def get_observation_status_quo1(
    first_metric_name: str = "a",
    second_metric_name: str = "b",
) -> Observation:
    return Observation(
        features=ObservationFeatures(
            parameters={"w": 0.85, "x": 1, "y": "baz", "z": False},
            trial_index=np.int64(1),
        ),
        data=ObservationData(
            means=np.array([2.0, 4.0]),
            covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
            metric_names=[first_metric_name, second_metric_name],
        ),
        arm_name="0_0",
    )


def get_observation1trans(
    first_metric_name: str = "a",
    second_metric_name: str = "b",
) -> Observation:
    return Observation(
        features=ObservationFeatures(
            parameters={"x": 9.0, "y": 121.0}, trial_index=np.int64(0)
        ),
        data=ObservationData(
            means=np.array([9.0, 25.0]),
            covariance=np.array([[1.0, 2.0], [3.0, 4.0]]),
            metric_names=[first_metric_name, second_metric_name],
        ),
        arm_name="1_1",
    )


def get_observation2(
    first_metric_name: str = "a",
    second_metric_name: str = "b",
) -> Observation:
    return Observation(
        features=ObservationFeatures(
            parameters={"x": 3.0, "y": 2.0}, trial_index=np.int64(1)
        ),
        data=ObservationData(
            means=np.array([2.0, 1.0]),
            covariance=np.array([[2.0, 3.0], [4.0, 5.0]]),
            metric_names=[first_metric_name, second_metric_name],
        ),
        arm_name="1_1",
    )


def get_observation2trans(
    first_metric_name: str = "a",
    second_metric_name: str = "b",
) -> Observation:
    return Observation(
        features=ObservationFeatures(
            parameters={"x": 16.0, "y": 9.0}, trial_index=np.int64(1)
        ),
        data=ObservationData(
            means=np.array([9.0, 4.0]),
            covariance=np.array([[2.0, 3.0], [4.0, 5.0]]),
            metric_names=[first_metric_name, second_metric_name],
        ),
        arm_name="1_1",
    )


# Modeling layer


def get_generation_strategy(
    with_experiment: bool = False,
    with_callable_model_kwarg: bool = True,
    with_completion_criteria: int = 0,
) -> GenerationStrategy:
    gs = choose_generation_strategy(
        search_space=get_search_space(), should_deduplicate=True
    )
    if with_experiment:
        gs._experiment = get_experiment()
    fake_func = get_experiment
    if with_callable_model_kwarg:
        # pyre-ignore[16]: testing hack to test serialization of callable kwargs
        # in generation steps.
        gs._steps[0].model_kwargs["model_constructor"] = fake_func

    if with_completion_criteria > 0:
        gs._steps[0].num_trials = -1
        gs._steps[0].completion_criteria = [
            MinimumPreferenceOccurances(metric_name="m1", threshold=3)
        ] * with_completion_criteria
    return gs


def get_transform_type() -> Type[Transform]:
    return IntToFloat


def get_experiment_for_value() -> Experiment:
    return Experiment(get_search_space_for_value(), "test")


def get_legacy_list_surrogate_generation_step_as_dict() -> Dict[str, Any]:
    """
    For use ensuring backwards compatibility loading the now deprecated ListSurrogate.
    """

    # Generated via `get_sobol_botorch_modular_saas_fully_bayesian_single_task_gp_qnei`
    # before new multi-Surrogate Model and new Surrogate diffs D42013742
    return {
        "__type": "GenerationStep",
        "model": {"__type": "Models", "name": "BOTORCH_MODULAR"},
        "num_trials": -1,
        "min_trials_observed": 0,
        "completion_criteria": [],
        "max_parallelism": 1,
        "use_update": False,
        "enforce_num_trials": True,
        "model_kwargs": {
            "surrogate": {
                "__type": "ListSurrogate",
                "botorch_submodel_class_per_outcome": {},
                "botorch_submodel_class": {
                    "__type": "Type[Model]",
                    "index": "SaasFullyBayesianSingleTaskGP",
                    "class": "<class 'botorch.models.model.Model'>",
                },
                "submodel_options_per_outcome": {},
                "submodel_options": {},
                "mll_class": {
                    "__type": "Type[MarginalLogLikelihood]",
                    "index": "ExactMarginalLogLikelihood",
                    "class": (
                        "<class 'gpytorch.mlls.marginal_log_likelihood."
                        "MarginalLogLikelihood'>"
                    ),
                },
                "mll_options": {},
                "submodel_outcome_transforms": None,
                "submodel_input_transforms": None,
                "submodel_covar_module_class": None,
                "submodel_covar_module_options": {},
                "submodel_likelihood_class": None,
                "submodel_likelihood_options": {},
            },
            "botorch_acqf_class": {
                "__type": "Type[AcquisitionFunction]",
                "index": "qNoisyExpectedImprovement",
                "class": "<class 'botorch.acquisition.acquisition.AcquisitionFunction'>",  # noqa
            },
        },
        "model_gen_kwargs": {},
        "index": 1,
        "should_deduplicate": False,
    }


class transform_1(Transform):
    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        new_ss = search_space.clone()
        for param in new_ss.parameters.values():
            if isinstance(param, FixedParameter):
                param._value += 1.0
            elif isinstance(param, RangeParameter):
                param._lower += 1.0
                param._upper += 1.0
        return new_ss

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        modelbridge: Optional[ModelBridge],
        fixed_features: Optional[ObservationFeatures],
    ) -> OptimizationConfig:
        return (  # pyre-ignore[7]: pyre is right, this is a hack for testing.
            # pyre-fixme[58]: `+` is not supported for operand types
            #  `OptimizationConfig` and `int`.
            optimization_config + 1
            if isinstance(optimization_config, int)
            else optimization_config
        )

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in obsf.parameters:
                obsf.parameters[p_name] += 1  # pyre-ignore
        return observation_features

    def _transform_observation_data(
        self,
        observation_data: List[ObservationData],
    ) -> List[ObservationData]:
        for obsd in observation_data:
            obsd.means += 1
        return observation_data

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name in obsf.parameters:
                obsf.parameters[p_name] -= 1  # pyre-ignore
        return observation_features

    def _untransform_observation_data(
        self,
        observation_data: List[ObservationData],
    ) -> List[ObservationData]:
        for obsd in observation_data:
            obsd.means -= 1
        return observation_data


class transform_2(Transform):
    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        new_ss = search_space.clone()
        for param in new_ss.parameters.values():
            if isinstance(param, FixedParameter):
                param._value *= 2.0
            elif isinstance(param, RangeParameter):
                param._lower *= 2.0
                param._upper *= 2.0
        return new_ss

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        modelbridge: Optional[ModelBridge],
        fixed_features: Optional[ObservationFeatures],
    ) -> OptimizationConfig:
        return (
            # pyre-fixme[58]: `**` is not supported for operand types
            #  `OptimizationConfig` and `int`.
            optimization_config**2
            if isinstance(optimization_config, int)
            else optimization_config
        )

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for pname in obsf.parameters:
                obsf.parameters[pname] = obsf.parameters[pname] ** 2  # pyre-ignore
        return observation_features

    def _transform_observation_data(
        self,
        observation_data: List[ObservationData],
    ) -> List[ObservationData]:
        for obsd in observation_data:
            obsd.means = obsd.means**2
        return observation_data

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for pname in obsf.parameters:
                obsf.parameters[pname] = np.sqrt(obsf.parameters[pname])
        return observation_features

    def _untransform_observation_data(
        self,
        observation_data: List[ObservationData],
    ) -> List[ObservationData]:
        for obsd in observation_data:
            obsd.means = np.sqrt(obsd.means)
        return observation_data
