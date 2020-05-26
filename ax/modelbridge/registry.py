#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from inspect import isfunction, signature
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type

import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.search_space import SearchSpace
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.convert_metric_names import ConvertMetricNames
from ax.modelbridge.transforms.derelativize import Derelativize
from ax.modelbridge.transforms.int_range_to_choice import IntRangeToChoice
from ax.modelbridge.transforms.int_to_float import IntToFloat
from ax.modelbridge.transforms.ivw import IVW
from ax.modelbridge.transforms.log import Log
from ax.modelbridge.transforms.one_hot import OneHot
from ax.modelbridge.transforms.ordered_choice_encode import OrderedChoiceEncode
from ax.modelbridge.transforms.remove_fixed import RemoveFixed
from ax.modelbridge.transforms.search_space_to_choice import SearchSpaceToChoice
from ax.modelbridge.transforms.standardize_y import StandardizeY
from ax.modelbridge.transforms.stratified_standardize_y import StratifiedStandardizeY
from ax.modelbridge.transforms.task_encode import TaskEncode
from ax.modelbridge.transforms.trial_as_task import TrialAsTask
from ax.modelbridge.transforms.unit_x import UnitX
from ax.models.base import Model
from ax.models.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ax.models.discrete.full_factorial import FullFactorialGenerator
from ax.models.discrete.thompson import ThompsonSampler
from ax.models.random.sobol import SobolGenerator
from ax.models.random.uniform import UniformGenerator
from ax.models.torch.botorch import BotorchModel
from ax.models.torch.botorch_kg import KnowledgeGradient
from ax.models.torch.botorch_mes import MaxValueEntropySearch
from ax.utils.common.kwargs import (
    consolidate_kwargs,
    get_function_argument_names,
    get_function_default_arguments,
    validate_kwarg_typing,
)
from ax.utils.common.logger import get_logger
from ax.utils.common.serialization import callable_from_reference, callable_to_reference
from ax.utils.common.typeutils import checked_cast, not_none


logger = get_logger(__name__)


"""
Module containing a registry of standard models (and generators, samplers etc.)
such as Sobol generator, GP+EI, Thompson sampler, etc.

Use of `Models` enum allows for serialization and reinstantiation of models and
generation strategies from generator runs they produced. To reinstantiate a model
from generator run, use `get_model_from_generator_run` utility from this module.
"""


Cont_X_trans: List[Type[Transform]] = [
    RemoveFixed,
    OrderedChoiceEncode,
    OneHot,
    IntToFloat,
    Log,
    UnitX,
]

Discrete_X_trans: List[Type[Transform]] = [IntRangeToChoice]

Y_trans: List[Type[Transform]] = [IVW, Derelativize, StandardizeY]

# Expected `List[Type[Transform]]` for 2nd anonymous parameter to
# call `list.__add__` but got `List[Type[SearchSpaceToChoice]]`.
TS_trans: List[Type[Transform]] = Y_trans + [SearchSpaceToChoice]

# Multi-type MTGP transforms
MT_MTGP_trans: List[Type[Transform]] = Cont_X_trans + [
    Derelativize,
    ConvertMetricNames,
    TrialAsTask,
    StratifiedStandardizeY,
    TaskEncode,
]

# Single-type MTGP transforms
ST_MTGP_trans: List[Type[Transform]] = Cont_X_trans + [
    Derelativize,
    TrialAsTask,
    StratifiedStandardizeY,
    TaskEncode,
]

# Single-type MTGP transforms
Specified_Task_ST_MTGP_trans: List[Type[Transform]] = (
    Cont_X_trans + [Derelativize, StratifiedStandardizeY, TaskEncode]
)

STANDARD_TORCH_BRIDGE_KWARGS: Dict[str, Any] = {
    "torch_dtype": torch.double,
    "torch_device": torch.device("cpu"),
}


class ModelSetup(NamedTuple):
    """A model setup defines a coupled combination of a model, a model bridge,
    standard set of transforms, and standard model bridge keyword arguments.
    This coupled combination yields a given standard modeling strategy in Ax,
    such as BoTorch GP+EI, a Thompson sampler, or a Sobol quasirandom generator.
    """

    bridge_class: Type[ModelBridge]
    model_class: Type[Model]
    transforms: List[Type[Transform]]
    standard_bridge_kwargs: Optional[Dict[str, Any]] = None


"""A mapping of string keys that indicate a model, to the corresponding
model setup, which defines which model, model bridge, transforms, and
standard arguments a given model requires.
"""
MODEL_KEY_TO_MODEL_SETUP: Dict[str, ModelSetup] = {
    "BO": ModelSetup(
        bridge_class=TorchModelBridge,
        model_class=BotorchModel,
        transforms=Cont_X_trans + Y_trans,
        standard_bridge_kwargs=STANDARD_TORCH_BRIDGE_KWARGS,
    ),
    "GPEI": ModelSetup(
        bridge_class=TorchModelBridge,
        model_class=BotorchModel,
        transforms=Cont_X_trans + Y_trans,
        standard_bridge_kwargs=STANDARD_TORCH_BRIDGE_KWARGS,
    ),
    "GPKG": ModelSetup(
        bridge_class=TorchModelBridge,
        model_class=KnowledgeGradient,
        transforms=Cont_X_trans + Y_trans,
        standard_bridge_kwargs=STANDARD_TORCH_BRIDGE_KWARGS,
    ),
    "GPMES": ModelSetup(
        bridge_class=TorchModelBridge,
        model_class=MaxValueEntropySearch,
        transforms=Cont_X_trans + Y_trans,
        standard_bridge_kwargs=STANDARD_TORCH_BRIDGE_KWARGS,
    ),
    "EB": ModelSetup(
        bridge_class=DiscreteModelBridge,
        model_class=EmpiricalBayesThompsonSampler,
        transforms=TS_trans,
    ),
    "Factorial": ModelSetup(
        bridge_class=DiscreteModelBridge,
        model_class=FullFactorialGenerator,
        transforms=Discrete_X_trans,
    ),
    "Thompson": ModelSetup(
        bridge_class=DiscreteModelBridge,
        model_class=ThompsonSampler,
        transforms=TS_trans,
    ),
    "Sobol": ModelSetup(
        bridge_class=RandomModelBridge,
        model_class=SobolGenerator,
        transforms=Cont_X_trans,
    ),
    "Uniform": ModelSetup(
        bridge_class=RandomModelBridge,
        model_class=UniformGenerator,
        transforms=Cont_X_trans,
    ),
}


class Models(Enum):
    """Registry of available models.

    Uses MODEL_KEY_TO_MODEL_SETUP to retrieve settings for model and model bridge,
    by the key stored in the enum value.

    To instantiate a model in this enum, simply call an enum member like so:
    `Models.SOBOL(search_space=search_space)` or
    `Models.GPEI(experiment=experiment, data=data)`. Keyword arguments
    specified to the call will be passed into the model or the model bridge
    constructors according to their keyword.

    For instance, `Models.SOBOL(search_space=search_space, scramble=False)`
    will instantiate a `RandomModelBridge(search_space=search_space)`
    with a `SobolGenerator(scramble=False)` underlying model.
    """

    SOBOL = "Sobol"
    GPEI = "GPEI"
    GPKG = "GPKG"
    GPMES = "GPMES"
    FACTORIAL = "Factorial"
    THOMPSON = "Thompson"
    BOTORCH = "BO"
    EMPIRICAL_BAYES_THOMPSON = "EB"
    UNIFORM = "Uniform"

    @property
    def model_class(self) -> Type[Model]:
        """Type of `Model` used for the given model+bridge setup."""
        return MODEL_KEY_TO_MODEL_SETUP[self.value].model_class

    @property
    def model_bridge_class(self) -> Type[ModelBridge]:
        """Type of `ModelBridge` used for the given model+bridge setup."""
        return MODEL_KEY_TO_MODEL_SETUP[self.value].bridge_class

    # TODO[T67370152]: Test that none of the `ModelSetup`-s share a kwarg.
    def __call__(
        self,
        search_space: Optional[SearchSpace] = None,
        experiment: Optional[Experiment] = None,
        data: Optional[Data] = None,
        silently_filter_kwargs: bool = True,  # TODO[Lena]: default to False
        **kwargs: Any,
    ) -> ModelBridge:
        assert self.value in MODEL_KEY_TO_MODEL_SETUP, f"Unknown model {self.value}"
        # All model bridges require either a search space or an experiment.
        assert search_space or experiment, "Search space or experiment required."
        model_setup_info = MODEL_KEY_TO_MODEL_SETUP[self.value]
        model_class = model_setup_info.model_class
        bridge_class = model_setup_info.bridge_class
        if not silently_filter_kwargs:
            validate_kwarg_typing(  # TODO[Lena]: T46467254, pragma: no cover
                typed_callables=[model_class, bridge_class],
                search_space=search_space,
                experiment=experiment,
                data=data,
                **kwargs,
            )

        # Create model with consolidated arguments: defaults + passed in kwargs.
        model_kwargs = consolidate_kwargs(
            kwargs_iterable=[get_function_default_arguments(model_class), kwargs],
            keywords=get_function_argument_names(model_class),
        )
        model = model_class(**model_kwargs)

        # Create `ModelBridge`: defaults + standard kwargs + passed in kwargs.
        bridge_kwargs = consolidate_kwargs(
            kwargs_iterable=[
                get_function_default_arguments(bridge_class),
                model_setup_info.standard_bridge_kwargs,
                {"transforms": model_setup_info.transforms},
                kwargs,
            ],
            keywords=get_function_argument_names(
                function=bridge_class, omit=["experiment", "search_space", "data"]
            ),
        )

        # Create model bridge with the consolidated kwargs.
        model_bridge = bridge_class(
            search_space=search_space or not_none(experiment).search_space,
            experiment=experiment,
            data=data,
            model=model,
            **bridge_kwargs,
        )

        # Store all kwargs on model bridge, to be saved on generator run.
        model_bridge._set_kwargs_to_save(
            model_key=self.value,
            model_kwargs=_encode_callables_as_references(model_kwargs),
            bridge_kwargs=_encode_callables_as_references(bridge_kwargs),
        )
        return model_bridge

    def view_defaults(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Obtains the default keyword arguments for the model and the modelbridge
        specified through the Models enum, for ease of use in notebook environment,
        since models and bridges cannot be inspected directly through the enum.

        Returns:
            A tuple of default keyword arguments for the model and the model bridge.
        """
        model_setup_info = not_none(MODEL_KEY_TO_MODEL_SETUP.get(self.value))
        return (
            self._get_model_kwargs(info=model_setup_info),
            self._get_bridge_kwargs(info=model_setup_info),
        )

    def view_kwargs(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Obtains annotated keyword arguments that the model and the modelbridge
        (corresponding to a given member of the Models enum) constructors expect.

        Returns:
            A tuple of annotated keyword arguments for the model and the model bridge.
        """
        model_class = self.model_class
        bridge_class = self.model_bridge_class
        return (
            {kw: p.annotation for kw, p in signature(model_class).parameters.items()},
            {kw: p.annotation for kw, p in signature(bridge_class).parameters.items()},
        )

    @staticmethod
    def _get_model_kwargs(
        info: ModelSetup, kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return consolidate_kwargs(
            [get_function_default_arguments(info.model_class), kwargs],
            keywords=get_function_argument_names(info.model_class),
        )

    @staticmethod
    def _get_bridge_kwargs(
        info: ModelSetup, kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return consolidate_kwargs(
            [
                get_function_default_arguments(info.bridge_class),
                info.standard_bridge_kwargs,
                {"transforms": info.transforms},
                kwargs,
            ],
            keywords=get_function_argument_names(
                info.bridge_class, omit=["experiment", "search_space", "data"]
            ),
        )


def get_model_from_generator_run(
    generator_run: GeneratorRun,
    experiment: Experiment,
    data: Data,
    models_enum: Optional[Type["Models"]] = None,
    after_gen: bool = False,
) -> ModelBridge:
    """Reinstantiate a model from model key and kwargs stored on a given generator
    run, with the given experiment and the data to initialize the model with.

    Note: requires that the model that was used to get the generator run, is part
    of the `Models` registry enum.

    Args:
        generator_run: A `GeneratorRun` created by the model we are looking to
            reinstantiate.
        experiment: The experiment for which the model is reinstantiated.
        data: Data, with which to reinstantiate the model.
        models_enum: Optional subclass of `Models` registry, from which to obtain
            the settings of the model. Useful only if the generator run was
            created via a model that could not be included into the main registry,
            but can still be represented as a `ModelSetup` and was added to a
            registry that extends `Models`.
        after_gen: Whether to reinstantiate the model in the state, in which it
            was after it created this generator run, as opposed to before.
            Defaults to False, useful when reinstantiating the model to resume
            optimization, rather than to recreate its state at the time of
            generation.
    """
    if not generator_run._model_key:  # pragma: no cover
        raise ValueError(
            "Cannot restore model from generator run as no model key was "
            "on the generator run stored."
        )
    model = (models_enum or Models)(generator_run._model_key)
    model_kwargs = generator_run._model_kwargs or {}
    if after_gen and generator_run._model_state_after_gen is not None:
        serialized_model_state = not_none(generator_run._model_state_after_gen)
        model_kwargs.update(model.model_class.deserialize_state(serialized_model_state))
    bridge_kwargs = generator_run._bridge_kwargs or {}
    model_kwargs = _decode_callables_from_references(model_kwargs)
    bridge_kwargs = _decode_callables_from_references(bridge_kwargs)
    model_keywords = list(model_kwargs.keys())
    for key in model_keywords:
        if key in bridge_kwargs:  # pragma: no cover
            logger.info(
                f"Keyword argument `{key}` occurs in both model and model bridge "
                f"kwargs stored in the generator run. Assuming the `{key}` kwarg "
                "is passed into the model by the model bridge and removing its "
                "value from the model kwargs."
            )
            del model_kwargs[key]
    return model(experiment=experiment, data=data, **bridge_kwargs, **model_kwargs)


def _encode_callables_as_references(kwarg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Converts callables to references of form <module>.<qualname>, and returns
    the resulting dictionary.
    """
    return {
        k: {"is_callable_as_path": True, "value": callable_to_reference(v)}
        if isfunction(v)
        else v
        for k, v in kwarg_dict.items()
    }


def _decode_callables_from_references(kwarg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieves callables from references of form <module>.<qualname>, and returns
    the resulting dictionary.
    """
    return {
        k: callable_from_reference(checked_cast(str, v.get("value")))
        if isinstance(v, dict) and v.get("is_callable_as_path", False)
        else v
        for k, v in kwarg_dict.items()
    }
