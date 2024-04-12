#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger
from typing import Any, Dict, List, Optional, Type

import numpy as np
from ax.core.base_trial import TrialStatus
from ax.core.experiment import Experiment
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import FixedParameter, RangeParameter
from ax.core.search_space import SearchSpace
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.generation_node import GenerationNode
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.model_spec import ModelSpec
from ax.modelbridge.registry import Models
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.int_to_float import IntToFloat
from ax.modelbridge.transition_criterion import (
    MaxGenerationParallelism,
    MaxTrials,
    MinimumPreferenceOccurances,
    MinTrials,
)
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.utils.common.logger import get_logger
from ax.utils.testing.core_stubs import (
    get_experiment,
    get_search_space,
    get_search_space_for_value,
)
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.input import InputTransform, Normalize
from botorch.models.transforms.outcome import OutcomeTransform, Standardize

logger: Logger = get_logger(__name__)


# Observations


def get_observation_features() -> ObservationFeatures:
    return ObservationFeatures(parameters={"x": 2.0, "y": 10.0}, trial_index=0)


def get_observation(
    first_metric_name: str = "a",
    second_metric_name: str = "b",
) -> Observation:
    return Observation(
        features=ObservationFeatures(parameters={"x": 2.0, "y": 10.0}, trial_index=0),
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
        features=ObservationFeatures(parameters={"x": 2.0, "y": 10.0}, trial_index=0),
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
            trial_index=0,
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
            trial_index=1,
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
        features=ObservationFeatures(parameters={"x": 9.0, "y": 121.0}, trial_index=0),
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
        features=ObservationFeatures(parameters={"x": 3.0, "y": 2.0}, trial_index=1),
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
        features=ObservationFeatures(parameters={"x": 16.0, "y": 9.0}, trial_index=1),
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
    with_generation_nodes: bool = False,
) -> GenerationStrategy:
    if with_generation_nodes:
        gs = sobol_gpei_generation_node_gs()
        gs._nodes[0]._model_spec_to_gen_from = ModelSpec(
            model_enum=Models.SOBOL,
            model_kwargs={"init_position": 3},
            model_gen_kwargs={"some_gen_kwarg": "some_value"},
        )
        if with_callable_model_kwarg:
            # pyre-ignore[16]: testing hack to test serialization of callable kwargs
            # in generation steps.
            gs._nodes[0]._model_spec_to_gen_from.model_kwargs[
                "model_constructor"
            ] = get_sobol
    else:
        gs = choose_generation_strategy(
            search_space=get_search_space(), should_deduplicate=True
        )
        if with_callable_model_kwarg:
            # pyre-ignore[16]: testing hack to test serialization of callable kwargs
            # in generation steps.
            gs._steps[0].model_kwargs["model_constructor"] = get_sobol
    if with_experiment:
        gs._experiment = get_experiment()

    if with_completion_criteria > 0:
        gs._steps[0].num_trials = -1
        gs._steps[0].completion_criteria = [
            MinimumPreferenceOccurances(metric_name="m1", threshold=3)
        ] * with_completion_criteria
    return gs


def sobol_gpei_generation_node_gs() -> GenerationStrategy:
    """Returns a basic SOBOL +GPEI GS usecase using GenerationNodes for testing"""
    sobol_criterion = [
        MaxTrials(
            threshold=5,
            transition_to="GPEI_node",
            block_gen_if_met=True,
            only_in_statuses=None,
            not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
        )
    ]
    gpei_criterion = [
        MaxTrials(
            threshold=2,
            transition_to=None,
            block_gen_if_met=True,
            only_in_statuses=None,
            not_in_statuses=[TrialStatus.FAILED, TrialStatus.ABANDONED],
        ),
        # Here MinTrials and MaxParallelism don't enforce anything, but
        # we wanted to have an instance of them to test for storage compatibility.
        MinTrials(
            threshold=0,
            transition_to=None,
            block_gen_if_met=False,
            only_in_statuses=[TrialStatus.CANDIDATE],
            not_in_statuses=None,
        ),
        MaxGenerationParallelism(
            threshold=1000,
            transition_to=None,
            block_gen_if_met=True,
            only_in_statuses=[TrialStatus.RUNNING],
            not_in_statuses=None,
        ),
    ]
    step_model_kwargs = {"silently_filter_kwargs": True}
    sobol_model_spec = ModelSpec(
        model_enum=Models.SOBOL,
        model_kwargs=step_model_kwargs,
        model_gen_kwargs={},
    )
    gpei_model_spec = ModelSpec(
        model_enum=Models.GPEI,
        model_kwargs=step_model_kwargs,
        model_gen_kwargs={},
    )
    sobol_node = GenerationNode(
        node_name="sobol_node",
        transition_criteria=sobol_criterion,
        model_specs=[sobol_model_spec],
    )
    gpei_node = GenerationNode(
        node_name="GPEI_node",
        transition_criteria=gpei_criterion,
        model_specs=[gpei_model_spec],
    )

    sobol_GPEI_GS_nodes = GenerationStrategy(
        name="Sobol+GPEI_Nodes",
        nodes=[sobol_node, gpei_node],
        steps=None,
    )
    return sobol_GPEI_GS_nodes


def get_transform_type() -> Type[Transform]:
    return IntToFloat


def get_input_transform_type() -> Type[InputTransform]:
    return Normalize


def get_outcome_transfrom_type() -> Type[OutcomeTransform]:
    return Standardize


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
                "submodel_outcome_transforms": [
                    {
                        "__type": "Standardize",
                        "index": {
                            "__type": "Type[OutcomeTransform]",
                            "index": "Standardize",
                            "class": (
                                "<class 'botorch.models.transforms.outcome."
                                "OutcomeTransform'>"
                            ),
                        },
                        "class": (
                            "<class 'botorch.models.transforms.outcome.Standardize'>"
                        ),
                        "state_dict": {"m": 1, "outputs": None, "min_stdv": 1e-8},
                    }
                ],
                "submodel_input_transforms": [
                    {
                        "__type": "Normalize",
                        "index": {
                            "__type": "Type[InputTransform]",
                            "index": "Normalize",
                            "class": (
                                "<class 'botorch.models.transforms.input."
                                "InputTransform'>"
                            ),
                        },
                        "class": "<class 'botorch.models.transforms.input.Normalize'>",
                        "state_dict": {
                            "d": 3,
                            "indices": None,
                            "transform_on_train": True,
                            "transform_on_eval": True,
                            "transform_on_fantasize": True,
                            "reverse": False,
                            "min_range": 1e-08,
                            "learn_bounds": False,
                        },
                    }
                ],
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
        "index": -1,
        "should_deduplicate": False,
    }


def get_surrogate_generation_step() -> GenerationStep:
    return GenerationStep(
        model=Models.BOTORCH_MODULAR,
        num_trials=-1,
        max_parallelism=1,
        model_kwargs={
            "surrogate": Surrogate(
                botorch_model_class=SaasFullyBayesianSingleTaskGP,
                input_transform_classes=[Normalize],
                input_transform_options={
                    "Normalize": {
                        "d": 3,
                        "indices": None,
                        "transform_on_train": True,
                        "transform_on_eval": True,
                        "transform_on_fantasize": True,
                        "reverse": False,
                        "min_range": 1e-08,
                        "learn_bounds": False,
                    }
                },
                outcome_transform_classes=[Standardize],
                outcome_transform_options={
                    "Standardize": {"m": 1, "outputs": None, "min_stdv": 1e-8}
                },
            ),
            "botorch_acqf_class": qNoisyExpectedImprovement,
        },
    )


def get_surrogate_as_dict() -> Dict[str, Any]:
    """
    For use ensuring backwards compatibility when loading Surrogate
    with input_transform and outcome_transform kwargs.
    """
    return {
        "__type": "Surrogate",
        "botorch_model_class": None,
        "model_options": {},
        "mll_class": {
            "__type": "Type[MarginalLogLikelihood]",
            "index": "ExactMarginalLogLikelihood",
            "class": (
                "<class 'gpytorch.mlls.marginal_log_likelihood."
                "MarginalLogLikelihood'>"
            ),
        },
        "mll_options": {},
        "outcome_transform": None,
        "input_transform": None,
        "covar_module_class": None,
        "covar_module_options": {},
        "likelihood_class": None,
        "likelihood_options": {},
        "allow_batched_models": False,
    }


def get_surrogate_spec_as_dict(
    model_class: Optional[str] = None, with_legacy_input_transform: bool = False
) -> Dict[str, Any]:
    """
    For use ensuring backwards compatibility when loading SurrogateSpec
    with input_transform and outcome_transform kwargs.
    """
    if model_class is None:
        model_class = "SingleTaskGP"
    if with_legacy_input_transform:
        input_transform = {
            "__type": "Normalize",
            "index": {
                "__type": "Type[InputTransform]",
                "index": "Normalize",
                "class": "<class 'botorch.models.transforms.input.InputTransform'>",
            },
            "class": "<class 'botorch.models.transforms.input.Normalize'>",
            "state_dict": {
                "d": 7,
                "indices": None,
                "bounds": None,
                "batch_shape": {"__type": "torch_Size", "value": "[]"},
                "transform_on_train": True,
                "transform_on_eval": True,
                "transform_on_fantasize": True,
                "reverse": False,
                "min_range": 1e-08,
                "learn_bounds": False,
            },
        }
    else:
        input_transform = None
    return {
        "__type": "SurrogateSpec",
        "botorch_model_class": {
            "__type": "Type[Model]",
            "index": model_class,
            "class": "<class 'botorch.models.model.Model'>",
        },
        "botorch_model_kwargs": {},
        "mll_class": {
            "__type": "Type[MarginalLogLikelihood]",
            "index": "ExactMarginalLogLikelihood",
            "class": (
                "<class 'gpytorch.mlls.marginal_log_likelihood"
                ".MarginalLogLikelihood'>"
            ),
        },
        "mll_kwargs": {},
        "covar_module_class": None,
        "covar_module_kwargs": None,
        "likelihood_class": None,
        "likelihood_kwargs": None,
        "input_transform": input_transform,
        "outcome_transform": None,
        "allow_batched_models": False,
        "outcomes": [],
    }


class transform_1(Transform):
    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        new_ss = search_space.clone()
        for param in new_ss.parameters.values():
            if isinstance(param, FixedParameter):
                if param._value is not None and not isinstance(param._value, str):
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
                if param._value is not None and not isinstance(param._value, str):
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
