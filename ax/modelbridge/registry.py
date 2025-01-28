#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Module containing a registry of standard models (and generators, samplers etc.)
such as Sobol generator, GP+EI, Thompson sampler, etc.

Use of `Models` enum allows for serialization and reinstantiation of models and
generation strategies from generator runs they produced. To reinstantiate a model
from generator run, use `get_model_from_generator_run` utility from this module.
"""

from __future__ import annotations

from enum import Enum
from inspect import isfunction, signature
from logging import Logger
from typing import Any, NamedTuple

from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.search_space import SearchSpace
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.choice_encode import (
    ChoiceToNumericChoice,
    OrderedChoiceToIntegerRange,
)
from ax.modelbridge.transforms.derelativize import Derelativize
from ax.modelbridge.transforms.fill_missing_parameters import FillMissingParameters
from ax.modelbridge.transforms.int_range_to_choice import IntRangeToChoice
from ax.modelbridge.transforms.int_to_float import IntToFloat, LogIntToFloat
from ax.modelbridge.transforms.ivw import IVW
from ax.modelbridge.transforms.log import Log
from ax.modelbridge.transforms.logit import Logit
from ax.modelbridge.transforms.merge_repeated_measurements import (
    MergeRepeatedMeasurements,
)
from ax.modelbridge.transforms.one_hot import OneHot
from ax.modelbridge.transforms.relativize import Relativize
from ax.modelbridge.transforms.remove_fixed import RemoveFixed
from ax.modelbridge.transforms.search_space_to_choice import SearchSpaceToChoice
from ax.modelbridge.transforms.standardize_y import StandardizeY
from ax.modelbridge.transforms.stratified_standardize_y import StratifiedStandardizeY
from ax.modelbridge.transforms.task_encode import TaskChoiceToIntTaskChoice
from ax.modelbridge.transforms.transform_to_new_sq import TransformToNewSQ
from ax.modelbridge.transforms.trial_as_task import TrialAsTask
from ax.modelbridge.transforms.unit_x import UnitX
from ax.models.base import Model
from ax.models.discrete.eb_ashr import EBAshr
from ax.models.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ax.models.discrete.full_factorial import FullFactorialGenerator
from ax.models.discrete.thompson import ThompsonSampler
from ax.models.random.sobol import SobolGenerator
from ax.models.random.uniform import UniformGenerator
from ax.models.torch.botorch import BotorchModel
from ax.models.torch.botorch_modular.model import BoTorchModel as ModularBoTorchModel
from ax.models.torch.botorch_modular.surrogate import SurrogateSpec
from ax.models.torch.cbo_sac import SACBO
from ax.utils.common.kwargs import (
    consolidate_kwargs,
    get_function_argument_names,
    get_function_default_arguments,
)
from ax.utils.common.logger import get_logger
from ax.utils.common.serialization import callable_from_reference, callable_to_reference
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.fully_bayesian_multitask import SaasFullyBayesianMultiTaskGP
from pyre_extensions import assert_is_instance, none_throws

logger: Logger = get_logger(__name__)

# This set of transforms uses continuous relaxation to handle discrete parameters.
# All candidate generation is done in the continuous space, and the generated
# candidates are rounded to fit the original search space. This is can be
# suboptimal when there are discrete parameters with a small number of options.
Cont_X_trans: list[type[Transform]] = [
    FillMissingParameters,
    RemoveFixed,
    OrderedChoiceToIntegerRange,
    OneHot,
    IntToFloat,
    Log,
    Logit,
    UnitX,
]

# This is a modification of Cont_X_trans that aims to avoid continuous relaxation
# where possible. It replaces IntToFloat with LogIntToFloat, which is only transforms
# log-scale integer parameters, which still use continuous relaxation. Other discrete
# transforms will remain discrete. When used with MBM, a Normalize input transform
# will be added to replace the UnitX transform. This setup facilitates the use of
# optimize_acqf_mixed_alternating, which is a more efficient acquisition function
# optimizer for mixed discrete/continuous problems.
MBM_X_trans: list[type[Transform]] = [
    FillMissingParameters,
    RemoveFixed,
    OrderedChoiceToIntegerRange,
    OneHot,
    LogIntToFloat,
    Log,
    Logit,
]


Discrete_X_trans: list[type[Transform]] = [IntRangeToChoice]

EB_ashr_trans: list[type[Transform]] = [
    TransformToNewSQ,
    MergeRepeatedMeasurements,
    SearchSpaceToChoice,
]

rel_EB_ashr_trans: list[type[Transform]] = [
    Relativize,
    MergeRepeatedMeasurements,
    SearchSpaceToChoice,
]

# This is a modification of Cont_X_trans that replaces OneHot and
# OrderedChoiceToIntegerRange with ChoiceToNumericChoice. This results in retaining
# all choice parameters as discrete, while using continuous relaxation for integer
# valued RangeParameters.
Mixed_transforms: list[type[Transform]] = [
    FillMissingParameters,
    RemoveFixed,
    ChoiceToNumericChoice,
    IntToFloat,
    Log,
    Logit,
    UnitX,
]

Y_trans: list[type[Transform]] = [IVW, Derelativize, StandardizeY]

# Expected `List[Type[Transform]]` for 2nd anonymous parameter to
# call `list.__add__` but got `List[Type[SearchSpaceToChoice]]`.
TS_trans: list[type[Transform]] = Y_trans + [SearchSpaceToChoice]

# Single-type MTGP transforms
ST_MTGP_trans: list[type[Transform]] = Cont_X_trans + [
    Derelativize,
    TrialAsTask,
    StratifiedStandardizeY,
    TaskChoiceToIntTaskChoice,
]

MBM_MTGP_trans: list[type[Transform]] = MBM_X_trans + [
    Derelativize,
    TrialAsTask,
    StratifiedStandardizeY,
    TaskChoiceToIntTaskChoice,
]


class ModelSetup(NamedTuple):
    """A model setup defines a coupled combination of a model, a model bridge,
    standard set of transforms, and standard model bridge keyword arguments.
    This coupled combination yields a given standard modeling strategy in Ax,
    such as BoTorch GP+EI, a Thompson sampler, or a Sobol quasirandom generator.
    """

    bridge_class: type[ModelBridge]
    model_class: type[Model]
    transforms: list[type[Transform]]
    default_model_kwargs: dict[str, Any] | None = None
    standard_bridge_kwargs: dict[str, Any] | None = None
    not_saved_model_kwargs: list[str] | None = None


"""A mapping of string keys that indicate a model, to the corresponding
model setup, which defines which model, model bridge, transforms, and
standard arguments a given model requires.
"""
MODEL_KEY_TO_MODEL_SETUP: dict[str, ModelSetup] = {
    "BoTorch": ModelSetup(
        bridge_class=TorchModelBridge,
        model_class=ModularBoTorchModel,
        transforms=MBM_X_trans + Y_trans,
    ),
    "Legacy_GPEI": ModelSetup(
        bridge_class=TorchModelBridge,
        model_class=BotorchModel,
        transforms=Cont_X_trans + Y_trans,
    ),
    "EB": ModelSetup(
        bridge_class=DiscreteModelBridge,
        model_class=EmpiricalBayesThompsonSampler,
        transforms=TS_trans,
    ),
    "EB_Ashr": ModelSetup(
        bridge_class=DiscreteModelBridge,
        model_class=EBAshr,
        transforms=EB_ashr_trans,
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
    "ST_MTGP": ModelSetup(
        bridge_class=TorchModelBridge,
        model_class=ModularBoTorchModel,
        transforms=MBM_MTGP_trans,
    ),
    "BO_MIXED": ModelSetup(
        bridge_class=TorchModelBridge,
        model_class=ModularBoTorchModel,
        transforms=Mixed_transforms + Y_trans,
    ),
    "SAASBO": ModelSetup(
        bridge_class=TorchModelBridge,
        model_class=ModularBoTorchModel,
        transforms=MBM_X_trans + Y_trans,
        default_model_kwargs={
            "surrogate_spec": SurrogateSpec(
                botorch_model_class=SaasFullyBayesianSingleTaskGP
            )
        },
    ),
    "SAAS_MTGP": ModelSetup(
        bridge_class=TorchModelBridge,
        model_class=ModularBoTorchModel,
        transforms=MBM_MTGP_trans,
        default_model_kwargs={
            "surrogate_spec": SurrogateSpec(
                botorch_model_class=SaasFullyBayesianMultiTaskGP
            )
        },
    ),
    "Contextual_SACBO": ModelSetup(
        bridge_class=TorchModelBridge,
        model_class=SACBO,
        transforms=Cont_X_trans + Y_trans,
    ),
}


class ModelRegistryBase(Enum):
    """Base enum that provides instrumentation of `__call__` on enum values,
    for enums that link their values to `ModelSetup`-s like `Models`.
    """

    @property
    def model_class(self) -> type[Model]:
        """Type of `Model` used for the given model+bridge setup."""
        return MODEL_KEY_TO_MODEL_SETUP[self.value].model_class

    @property
    def model_bridge_class(self) -> type[ModelBridge]:
        """Type of `ModelBridge` used for the given model+bridge setup."""
        return MODEL_KEY_TO_MODEL_SETUP[self.value].bridge_class

    def __call__(
        self,
        search_space: SearchSpace | None = None,
        experiment: Experiment | None = None,
        data: Data | None = None,
        silently_filter_kwargs: bool = False,
        **kwargs: Any,
    ) -> ModelBridge:
        assert self.value in MODEL_KEY_TO_MODEL_SETUP, f"Unknown model {self.value}"
        # All model bridges require either a search space or an experiment.
        assert search_space or experiment, "Search space or experiment required."
        search_space = search_space or none_throws(experiment).search_space
        model_setup_info = MODEL_KEY_TO_MODEL_SETUP[self.value]
        model_class = model_setup_info.model_class
        bridge_class = model_setup_info.bridge_class

        if not silently_filter_kwargs:
            # Check correct kwargs are present
            callables = (model_class, bridge_class)
            kwargs_to_check = {
                "search_space": search_space,
                "experiment": experiment,
                "data": data,
                **kwargs,
            }
            checked_kwargs = set()
            for fn in callables:
                params = signature(fn).parameters
                for kw in params.keys():
                    if kw in kwargs_to_check:
                        if kw in checked_kwargs:
                            logger.debug(
                                f"`{callables}` have duplicate keyword argument: {kw}."
                            )
                        else:
                            checked_kwargs.add(kw)

            # Check if kwargs contains keywords not exist in any callables
            extra_keywords = [kw for kw in kwargs.keys() if kw not in checked_kwargs]
            if len(extra_keywords) != 0:
                raise ValueError(
                    f"Arguments {extra_keywords} are not expected by any of {callables}"
                )

        # Create model with consolidated arguments: defaults + passed in kwargs.
        model_kwargs = consolidate_kwargs(
            kwargs_iterable=[
                get_function_default_arguments(model_class),
                model_setup_info.default_model_kwargs,
                kwargs,
            ],
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
            search_space=search_space or none_throws(experiment).search_space,
            experiment=experiment,
            data=data,
            model=model,
            **bridge_kwargs,
        )

        if model_setup_info.not_saved_model_kwargs:
            for key in model_setup_info.not_saved_model_kwargs:
                model_kwargs.pop(key, None)

        # Store all kwargs on model bridge, to be saved on generator run.
        model_bridge._set_kwargs_to_save(
            model_key=self.value,
            model_kwargs=_encode_callables_as_references(model_kwargs),
            bridge_kwargs=_encode_callables_as_references(bridge_kwargs),
        )
        return model_bridge

    def view_defaults(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Obtains the default keyword arguments for the model and the modelbridge
        specified through the Models enum, for ease of use in notebook environment,
        since models and bridges cannot be inspected directly through the enum.

        Returns:
            A tuple of default keyword arguments for the model and the model bridge.
        """
        model_setup_info = none_throws(MODEL_KEY_TO_MODEL_SETUP.get(self.value))
        return (
            self._get_model_kwargs(info=model_setup_info),
            self._get_bridge_kwargs(info=model_setup_info),
        )

    def view_kwargs(self) -> tuple[dict[str, Any], dict[str, Any]]:
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
        info: ModelSetup, kwargs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return consolidate_kwargs(
            [get_function_default_arguments(info.model_class), kwargs],
            keywords=get_function_argument_names(info.model_class),
        )

    @staticmethod
    def _get_bridge_kwargs(
        info: ModelSetup, kwargs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
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


class Models(ModelRegistryBase):
    """Registry of available models.

    Uses MODEL_KEY_TO_MODEL_SETUP to retrieve settings for model and model bridge,
    by the key stored in the enum value.

    To instantiate a model in this enum, simply call an enum member like so:
    `Models.SOBOL(search_space=search_space)` or
    `Models.BOTORCH(experiment=experiment, data=data)`. Keyword arguments
    specified to the call will be passed into the model or the model bridge
    constructors according to their keyword.

    For instance, `Models.SOBOL(search_space=search_space, scramble=False)`
    will instantiate a `RandomModelBridge(search_space=search_space)`
    with a `SobolGenerator(scramble=False)` underlying model.

    NOTE: If you deprecate a model, please add its replacement to
    `ax.storage.json_store.decoder._DEPRECATED_MODEL_TO_REPLACEMENT` to ensure
    backwards compatibility of the storage layer.
    """

    SOBOL = "Sobol"
    FACTORIAL = "Factorial"
    SAASBO = "SAASBO"
    SAAS_MTGP = "SAAS_MTGP"
    THOMPSON = "Thompson"
    LEGACY_BOTORCH = "Legacy_GPEI"
    BOTORCH_MODULAR = "BoTorch"
    EMPIRICAL_BAYES_THOMPSON = "EB"
    EB_ASHR = "EB_Ashr"
    UNIFORM = "Uniform"
    ST_MTGP = "ST_MTGP"
    BO_MIXED = "BO_MIXED"
    CONTEXT_SACBO = "Contextual_SACBO"


def get_model_from_generator_run(
    generator_run: GeneratorRun,
    experiment: Experiment,
    data: Data,
    models_enum: type[ModelRegistryBase],
    after_gen: bool = True,
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
        models_enum: Subclass of `Models` registry, from which to obtain
            the settings of the model. Useful only if the generator run was
            created via a model that could not be included into the main registry,
            but can still be represented as a `ModelSetup` and was added to a
            registry that extends `Models`.
        after_gen: Whether to reinstantiate the model in the state, in which it
            was after it created this generator run, as opposed to before.
            Defaults to True, useful when reinstantiating the model to resume
            optimization, rather than to recreate its state at the time of
            generation. TO recreate state at the time of generation, set to `False`.
    """
    if not generator_run._model_key:
        raise ValueError(
            "Cannot restore model from generator run as no model key was "
            "on the generator run stored."
        )
    model = models_enum(generator_run._model_key)
    model_kwargs = generator_run._model_kwargs or {}
    if after_gen:
        model_kwargs = _combine_model_kwargs_and_state(
            generator_run=generator_run, model_class=model.model_class
        )
    bridge_kwargs = generator_run._bridge_kwargs or {}
    model_kwargs = _decode_callables_from_references(model_kwargs)
    bridge_kwargs = _decode_callables_from_references(bridge_kwargs)
    model_keywords = list(model_kwargs.keys())
    for key in model_keywords:
        if key in bridge_kwargs:
            logger.debug(
                f"Keyword argument `{key}` occurs in both model and model bridge "
                f"kwargs stored in the generator run. Assuming the `{key}` kwarg "
                "is passed into the model by the model bridge and removing its "
                "value from the model kwargs."
            )
            del model_kwargs[key]
    return model(experiment=experiment, data=data, **bridge_kwargs, **model_kwargs)


def _combine_model_kwargs_and_state(
    generator_run: GeneratorRun,
    model_class: type[Model],
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Produces a combined dict of model kwargs and model state after gen,
    extracted from generator run. If model kwargs are not specified,
    model kwargs from the generator run will be used.
    """
    model_kwargs = model_kwargs or generator_run._model_kwargs or {}
    if generator_run._model_state_after_gen is None:
        return model_kwargs

    # We don't want to update `model_kwargs` on the `GenerationStep`,
    # just to add to them for the purpose of this function.
    return {
        **model_kwargs,
        **_extract_model_state_after_gen(
            generator_run=generator_run, model_class=model_class
        ),
    }


def _extract_model_state_after_gen(
    generator_run: GeneratorRun, model_class: type[Model]
) -> dict[str, Any]:
    """Extracts serialized post-generation model state from a generator run and
    deserializes it.
    """
    serialized_model_state = generator_run._model_state_after_gen or {}
    return model_class.deserialize_state(serialized_model_state)


def _encode_callables_as_references(kwarg_dict: dict[str, Any]) -> dict[str, Any]:
    """Converts callables to references of form <module>.<qualname>, and returns
    the resulting dictionary.
    """
    return {
        k: (
            {"is_callable_as_path": True, "value": callable_to_reference(v)}
            if isfunction(v)
            else v
        )
        for k, v in kwarg_dict.items()
    }


def _decode_callables_from_references(kwarg_dict: dict[str, Any]) -> dict[str, Any]:
    """Retrieves callables from references of form <module>.<qualname>, and returns
    the resulting dictionary.
    """
    return {
        k: (
            callable_from_reference(assert_is_instance(v.get("value"), str))
            if isinstance(v, dict) and v.get("is_callable_as_path", False)
            else v
        )
        for k, v in kwarg_dict.items()
    }
