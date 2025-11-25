#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Module containing a registry of standard models (and generators, samplers etc.)
such as Sobol generator, GP+EI, Thompson sampler, etc.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from inspect import isfunction, signature
from logging import Logger
from typing import Any, NamedTuple

from ax.adapter.base import Adapter
from ax.adapter.discrete import DiscreteAdapter
from ax.adapter.random import RandomAdapter
from ax.adapter.torch import TorchAdapter
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.bilog_y import BilogY
from ax.adapter.transforms.choice_encode import (
    ChoiceToNumericChoice,
    OrderedChoiceToIntegerRange,
)
from ax.adapter.transforms.derelativize import Derelativize
from ax.adapter.transforms.int_range_to_choice import IntRangeToChoice
from ax.adapter.transforms.int_to_float import IntToFloat
from ax.adapter.transforms.log import Log
from ax.adapter.transforms.logit import Logit
from ax.adapter.transforms.map_key_to_float import MapKeyToFloat
from ax.adapter.transforms.merge_repeated_measurements import MergeRepeatedMeasurements
from ax.adapter.transforms.one_hot import OneHot
from ax.adapter.transforms.relativize import Relativize
from ax.adapter.transforms.remove_fixed import RemoveFixed
from ax.adapter.transforms.search_space_to_choice import SearchSpaceToChoice
from ax.adapter.transforms.standardize_y import StandardizeY
from ax.adapter.transforms.stratified_standardize_y import StratifiedStandardizeY
from ax.adapter.transforms.task_encode import TaskChoiceToIntTaskChoice
from ax.adapter.transforms.transform_to_new_sq import TransformToNewSQ
from ax.adapter.transforms.trial_as_task import TrialAsTask
from ax.adapter.transforms.unit_x import UnitX
from ax.adapter.transforms.winsorize import Winsorize
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.exceptions.core import UserInputError
from ax.generators.base import Generator
from ax.generators.discrete.eb_ashr import EBAshr
from ax.generators.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ax.generators.discrete.full_factorial import FullFactorialGenerator
from ax.generators.discrete.thompson import ThompsonSampler
from ax.generators.random.sobol import SobolGenerator
from ax.generators.random.uniform import UniformGenerator
from ax.generators.torch.botorch_modular.generator import (
    BoTorchGenerator as ModularBoTorchGenerator,
)
from ax.generators.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
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
    RemoveFixed,
    OrderedChoiceToIntegerRange,
    OneHot,
    IntToFloat,
    Log,
    Logit,
    UnitX,
]

# This is a modification of Cont_X_trans that aims to avoid continuous relaxation
# where possible. To this end, the Log transform converts log-scale integer
# RangeParameters to numeric discrete ChoiceParameters. Other discrete transforms
# will remain discrete. When used with MBM, a Normalize input transform will be
# added to replace the UnitX transform. This setup facilitates the use of
# optimize_acqf_mixed_alternating, which is a more efficient acquisition function
# optimizer for mixed discrete/continuous problems.
MBM_X_trans_base: list[type[Transform]] = [
    RemoveFixed,
    OrderedChoiceToIntegerRange,
    OneHot,
    Log,
    Logit,
]
MBM_X_trans: list[type[Transform]] = [MapKeyToFloat, *MBM_X_trans_base]


Discrete_X_trans: list[type[Transform]] = [IntRangeToChoice]

EB_ashr_trans: list[type[Transform]] = [
    Derelativize,  # necessary to support relative constraints
    # scales data from multiple trials since we currently don't filter to single
    # trial data
    TransformToNewSQ,
    # Ensure we pass unique arms to EBAshr. This assumes treatment effects are
    # stationarity, but also should help with estimating the task-task covariance.
    MergeRepeatedMeasurements,
    SearchSpaceToChoice,
]

# TODO: @mgarrard remove this once non-gs methods are reaped
rel_EB_ashr_trans: list[type[Transform]] = [
    Relativize,
    MergeRepeatedMeasurements,
    SearchSpaceToChoice,
]

# This is a modification of MBM_X_trans that replaces OneHot and
# OrderedChoiceToIntegerRange with ChoiceToNumericChoice.
# This retains unordered choice parameters as a single parameter
# and uses MixedSingleTaskGP with CategoricalKernel in MBM to support them.
Mixed_transforms: list[type[Transform]] = [
    RemoveFixed,
    ChoiceToNumericChoice,
    Log,
    Logit,
]

Y_trans: list[type[Transform]] = [Derelativize, Winsorize, BilogY, StandardizeY]

# Expected `List[Type[Transform]]` for 2nd anonymous parameter to
# call `list.__add__` but got `List[Type[SearchSpaceToChoice]]`.
TS_trans: list[type[Transform]] = [
    Derelativize,
    BilogY,
    StandardizeY,
    SearchSpaceToChoice,
]

MTGP_Y_trans: list[type[Transform]] = [
    Derelativize,
    TrialAsTask,
    StratifiedStandardizeY,
    TaskChoiceToIntTaskChoice,
]

# Single-type MTGP transforms
ST_MTGP_trans: list[type[Transform]] = Cont_X_trans + MTGP_Y_trans

MBM_MTGP_trans: list[type[Transform]] = MBM_X_trans + MTGP_Y_trans


class ModelSetup(NamedTuple):
    """A model setup defines a coupled combination of a model, an adapter,
    standard set of transforms, and standard adapter keyword arguments.
    This coupled combination yields a given standard modeling strategy in Ax,
    such as BoTorch GP+EI, a Thompson sampler, or a Sobol quasirandom generator.
    """

    adapter_class: type[Adapter]
    generator_class: type[Generator]
    transforms: Sequence[type[Transform]]
    default_model_kwargs: Mapping[str, Any] | None = None
    standard_bridge_kwargs: Mapping[str, Any] | None = None
    not_saved_model_kwargs: Sequence[str] | None = None


"""A mapping of string keys that indicate a model, to the corresponding
model setup, which defines which model, adapter, transforms, and
standard arguments a given model requires.
"""
MODEL_KEY_TO_MODEL_SETUP: dict[str, ModelSetup] = {
    "BoTorch": ModelSetup(
        adapter_class=TorchAdapter,
        generator_class=ModularBoTorchGenerator,
        transforms=MBM_X_trans + Y_trans,
    ),
    "EB": ModelSetup(
        adapter_class=DiscreteAdapter,
        generator_class=EmpiricalBayesThompsonSampler,
        transforms=TS_trans,
    ),
    "EB_Ashr": ModelSetup(
        adapter_class=DiscreteAdapter,
        generator_class=EBAshr,
        transforms=EB_ashr_trans,
    ),
    "Factorial": ModelSetup(
        adapter_class=DiscreteAdapter,
        generator_class=FullFactorialGenerator,
        transforms=Discrete_X_trans,
    ),
    "Thompson": ModelSetup(
        adapter_class=DiscreteAdapter,
        generator_class=ThompsonSampler,
        transforms=TS_trans,
    ),
    "Sobol": ModelSetup(
        adapter_class=RandomAdapter,
        generator_class=SobolGenerator,
        transforms=Cont_X_trans,
    ),
    "Uniform": ModelSetup(
        adapter_class=RandomAdapter,
        generator_class=UniformGenerator,
        transforms=Cont_X_trans,
    ),
    "ST_MTGP": ModelSetup(
        adapter_class=TorchAdapter,
        generator_class=ModularBoTorchGenerator,
        transforms=MBM_MTGP_trans,
    ),
    "BO_MIXED": ModelSetup(
        adapter_class=TorchAdapter,
        generator_class=ModularBoTorchGenerator,
        transforms=Mixed_transforms + Y_trans,
    ),
    "SAASBO": ModelSetup(
        adapter_class=TorchAdapter,
        generator_class=ModularBoTorchGenerator,
        transforms=MBM_X_trans + Y_trans,
        default_model_kwargs={
            "surrogate_spec": SurrogateSpec(
                model_configs=[
                    ModelConfig(
                        botorch_model_class=SaasFullyBayesianSingleTaskGP, name="SAASBO"
                    )
                ]
            )
        },
    ),
    "SAAS_MTGP": ModelSetup(
        adapter_class=TorchAdapter,
        generator_class=ModularBoTorchGenerator,
        transforms=MBM_MTGP_trans,
        default_model_kwargs={
            "surrogate_spec": SurrogateSpec(
                model_configs=[
                    ModelConfig(
                        botorch_model_class=SaasFullyBayesianMultiTaskGP,
                        name="SAAS_MTGP",
                    )
                ]
            )
        },
    ),
}


class GeneratorRegistryBase(Enum):
    """Base enum that provides instrumentation of `__call__` on enum values,
    for enums that link their values to `ModelSetup`-s like `Generators`.
    """

    @property
    def model_key_to_model_setup(self) -> dict[str, ModelSetup]:
        return MODEL_KEY_TO_MODEL_SETUP

    @property
    def generator_class(self) -> type[Generator]:
        """Type of `Generator` used for the given model+adapter setup."""
        return self.model_key_to_model_setup[self.value].generator_class

    @property
    def adapter_class(self) -> type[Adapter]:
        """Type of `Adapter` used for the given model+adapter setup."""
        return self.model_key_to_model_setup[self.value].adapter_class

    def __call__(
        self,
        experiment: Experiment,
        data: Data | None = None,
        silently_filter_kwargs: bool = False,
        model_key_override: str | None = None,
        **kwargs: Any,
    ) -> Adapter:
        if self.value not in self.model_key_to_model_setup:
            raise UserInputError(f"Unknown model {self.value}")
        model_setup_info = self.model_key_to_model_setup[self.value]
        generator_class = model_setup_info.generator_class
        adapter_class = model_setup_info.adapter_class
        search_space = experiment.search_space

        if not silently_filter_kwargs:
            # Check correct kwargs are present
            callables = (generator_class, adapter_class)
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

        # Create generator with consolidated arguments: defaults + passed in kwargs.
        generator_kwargs = consolidate_kwargs(
            kwargs_iterable=[
                get_function_default_arguments(generator_class),
                model_setup_info.default_model_kwargs,
                kwargs,
            ],
            keywords=get_function_argument_names(generator_class),
        )
        generator = generator_class(**generator_kwargs)

        # Create `Adapter`: defaults + standard kwargs + passed in kwargs.
        adapter_kwargs = consolidate_kwargs(
            kwargs_iterable=[
                get_function_default_arguments(adapter_class),
                model_setup_info.standard_bridge_kwargs,
                {"transforms": model_setup_info.transforms},
                kwargs,
            ],
            keywords=get_function_argument_names(
                function=adapter_class, omit=["experiment", "search_space", "data"]
            ),
        )

        # Create adapter with the consolidated kwargs.
        adapter = adapter_class(
            search_space=search_space,
            experiment=experiment,
            data=data,
            generator=generator,
            **adapter_kwargs,
        )

        if model_setup_info.not_saved_model_kwargs:
            for key in model_setup_info.not_saved_model_kwargs:
                generator_kwargs.pop(key, None)

        # Store all kwargs on adapter, to be saved on generator run.
        model_key = model_key_override if model_key_override else self.value
        adapter._set_kwargs_to_save(
            model_key=model_key,
            generator_kwargs=_encode_callables_as_references(generator_kwargs),
            adapter_kwargs=_encode_callables_as_references(adapter_kwargs),
        )
        return adapter

    def view_defaults(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Obtains the default keyword arguments for the model and the adapter
        specified through the Generators enum, for ease of use in notebook environment,
        since models and adapters cannot be inspected directly through the enum.

        Returns:
            A tuple of default keyword arguments for the model and the adapter.
        """
        model_setup_info = none_throws(self.model_key_to_model_setup.get(self.value))
        return (
            self._get_model_kwargs(info=model_setup_info),
            self._get_bridge_kwargs(info=model_setup_info),
        )

    def view_kwargs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Obtains annotated keyword arguments that the model and the adapter
        (corresponding to a given member of the Generators enum) constructors expect.

        Returns:
            A tuple of annotated keyword arguments for the model and the adapter.
        """
        generator_class = self.generator_class
        adapter_class = self.adapter_class
        return (
            {
                kw: p.annotation
                for kw, p in signature(generator_class).parameters.items()
            },
            {kw: p.annotation for kw, p in signature(adapter_class).parameters.items()},
        )

    @staticmethod
    def _get_model_kwargs(
        info: ModelSetup, kwargs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return consolidate_kwargs(
            [get_function_default_arguments(info.generator_class), kwargs],
            keywords=get_function_argument_names(info.generator_class),
        )

    @staticmethod
    def _get_bridge_kwargs(
        info: ModelSetup, kwargs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return consolidate_kwargs(
            [
                get_function_default_arguments(info.adapter_class),
                info.standard_bridge_kwargs,
                {"transforms": info.transforms},
                kwargs,
            ],
            keywords=get_function_argument_names(
                info.adapter_class, omit=["experiment", "search_space", "data"]
            ),
        )


class Generators(GeneratorRegistryBase):
    """Registry of available models.

    Uses MODEL_KEY_TO_MODEL_SETUP to retrieve settings for model and adapter,
    by the key stored in the enum value.

    To instantiate a model in this enum, simply call an enum member like so:
    `Generators.SOBOL(search_space=search_space)` or
    `Generators.BOTORCH(experiment=experiment, data=data)`. Keyword arguments
    specified to the call will be passed into the model or the adapter
    constructors according to their keyword.

    For instance, `Generators.SOBOL(search_space=search_space, scramble=False)`
    will instantiate a `RandomAdapter(search_space=search_space)`
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
    BOTORCH_MODULAR = "BoTorch"
    EMPIRICAL_BAYES_THOMPSON = "EB"
    EB_ASHR = "EB_Ashr"
    UNIFORM = "Uniform"
    ST_MTGP = "ST_MTGP"
    BO_MIXED = "BO_MIXED"


def _extract_model_state_after_gen(
    generator_run: GeneratorRun, generator_class: type[Generator]
) -> dict[str, Any]:
    """Extracts serialized post-generation model state from a generator run and
    deserializes it.
    """
    serialized_model_state = generator_run._model_state_after_gen or {}
    return generator_class.deserialize_state(serialized_model_state)


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
