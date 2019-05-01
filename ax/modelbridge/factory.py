#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from enum import Enum
from typing import List, Optional, Type

import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.search_space import SearchSpace
from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.convert_metric_names import (
    ConvertMetricNames,
    tconfig_from_mt_experiment,
)
from ax.modelbridge.transforms.derelativize import Derelativize
from ax.modelbridge.transforms.int_range_to_choice import IntRangeToChoice
from ax.modelbridge.transforms.int_to_float import IntToFloat
from ax.modelbridge.transforms.ivw import IVW
from ax.modelbridge.transforms.log import Log
from ax.modelbridge.transforms.one_hot import OneHot
from ax.modelbridge.transforms.ordered_choice_encode import OrderedChoiceEncode
from ax.modelbridge.transforms.out_of_design import OutOfDesign
from ax.modelbridge.transforms.remove_fixed import RemoveFixed
from ax.modelbridge.transforms.search_space_to_choice import SearchSpaceToChoice
from ax.modelbridge.transforms.standardize_y import StandardizeY
from ax.modelbridge.transforms.stratified_standardize_y import StratifiedStandardizeY
from ax.modelbridge.transforms.task_encode import TaskEncode
from ax.modelbridge.transforms.trial_as_task import TrialAsTask
from ax.modelbridge.transforms.unit_x import UnitX
from ax.models.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ax.models.discrete.full_factorial import FullFactorialGenerator
from ax.models.discrete.thompson import ThompsonSampler
from ax.models.random.sobol import SobolGenerator
from ax.models.random.uniform import UniformGenerator
from ax.models.torch.botorch import (
    BotorchModel,
    TAcqfConstructor,
    TModelConstructor,
    TModelPredictor,
    TOptimizer,
)
from ax.models.torch.botorch_defaults import (
    get_and_fit_model,
    get_NEI,
    predict_from_model,
    scipy_optimizer,
)


"""
Module containing functions that generate standard models, such as Sobol,
GP+EI, etc.

Note: a special case here is a composite generator, which requires an
additional ``GenerationStrategy`` and is able to delegate work to multiple models
(for instance, to a random model to generate the first trial, and to an
optimization model for subsequent trials).

"""


Cont_X_trans: List[Type[Transform]] = [
    OutOfDesign,
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
TS_trans: List[Type[Transform]] = Discrete_X_trans + Y_trans + [SearchSpaceToChoice]
MTGP_trans: List[Type[Transform]] = [
    RemoveFixed,
    OrderedChoiceEncode,
    OneHot,
    IntToFloat,
    Log,
    UnitX,
    Derelativize,
    ConvertMetricNames,
    TrialAsTask,
    StratifiedStandardizeY,
    TaskEncode,
]


DEFAULT_TORCH_DEVICE = torch.device("cpu")


def get_sobol(
    search_space: SearchSpace,
    seed: Optional[int] = None,
    deduplicate: bool = False,
    init_position: int = 0,
    scramble: bool = True,
) -> RandomModelBridge:
    """Instantiates a Sobol sequence quasi-random generator.

    Args:
        search_space: Sobol generator search space.
        kwargs: Custom args for sobol generator.

    Returns:
        RandomModelBridge, with SobolGenerator as model.
    """
    return RandomModelBridge(
        search_space=search_space,
        model=SobolGenerator(
            seed=seed,
            deduplicate=deduplicate,
            init_position=init_position,
            scramble=scramble,
        ),
        transforms=Cont_X_trans,
    )


def get_uniform(
    search_space: SearchSpace, deduplicate: bool = False, seed: Optional[int] = None
) -> RandomModelBridge:
    """Instantiate uniform generator.

    Args:
        search_space: Uniform generator search space.
        kwargs: Custom args for uniform generator.

    Returns:
        RandomModelBridge, with UniformGenerator as model.
    """
    return RandomModelBridge(
        search_space=search_space,
        model=UniformGenerator(deduplicate=deduplicate, seed=seed),
        transforms=Cont_X_trans,
    )


def get_botorch(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    dtype: torch.dtype = torch.double,
    device: torch.device = DEFAULT_TORCH_DEVICE,
    transforms: List[Type[Transform]] = Cont_X_trans + Y_trans,
    model_constructor: TModelConstructor = get_and_fit_model,  # pyre-ignore[9]
    model_predictor: TModelPredictor = predict_from_model,
    acqf_constructor: TAcqfConstructor = get_NEI,  # pyre-ignore[9]
    acqf_optimizer: TOptimizer = scipy_optimizer,  # pyre-ignore[9]
    refit_on_cv: bool = False,
    refit_on_update: bool = True,
) -> TorchModelBridge:
    """Instantiates a BotorchModel."""
    if search_space is None:
        search_space = experiment.search_space
    if data.df.empty:  # pragma: no cover
        raise ValueError("BotorchModel requires non-empty data.")
    return TorchModelBridge(
        experiment=experiment,
        search_space=search_space,
        data=data,
        model=BotorchModel(
            model_constructor=model_constructor,
            model_predictor=model_predictor,
            acqf_constructor=acqf_constructor,
            acqf_optimizer=acqf_optimizer,
        ),
        transforms=transforms,
        torch_dtype=dtype,
        torch_device=device,
    )


def get_GPEI(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    dtype: torch.dtype = torch.double,
    device: torch.device = DEFAULT_TORCH_DEVICE,
) -> TorchModelBridge:
    """Instantiates a GP model that generates points with EI."""
    if data.df.empty:  # pragma: no cover
        raise ValueError("GP+EI BotorchModel requires non-empty data.")
    return get_botorch(
        experiment=experiment,
        data=data,
        search_space=search_space,
        dtype=dtype,
        device=device,
    )


def get_MTGP(
    experiment: MultiTypeExperiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
) -> TorchModelBridge:
    """Instantiates a Multi-task GP model that generates points with EI."""
    trial_index_to_type = {t.index: t.trial_type for t in experiment.trials.values()}
    return TorchModelBridge(
        experiment=experiment,
        search_space=search_space or experiment.search_space,
        data=data,
        model=BotorchModel(),
        transforms=MTGP_trans,
        transform_configs={
            "TrialAsTask": {"trial_level_map": {"trial_type": trial_index_to_type}},
            "ConvertMetricNames": tconfig_from_mt_experiment(experiment),
        },
        torch_dtype=torch.double,
        torch_device=DEFAULT_TORCH_DEVICE,
    )


def get_factorial(search_space: SearchSpace) -> DiscreteModelBridge:
    """Instantiates a factorial generator."""
    return DiscreteModelBridge(
        search_space=search_space,
        data=Data(),
        model=FullFactorialGenerator(),
        transforms=Discrete_X_trans,
    )


def get_empirical_bayes_thompson(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    num_samples: int = 10000,
    min_weight: Optional[float] = None,
    uniform_weights: bool = False,
) -> DiscreteModelBridge:
    """Instantiates an empirical Bayes / Thompson sampling model."""
    if data.df.empty:  # pragma: no cover
        raise ValueError("Empirical Bayes Thompson sampler requires non-empty data.")
    model = EmpiricalBayesThompsonSampler(
        num_samples=num_samples, min_weight=min_weight, uniform_weights=uniform_weights
    )
    return DiscreteModelBridge(
        experiment=experiment,
        search_space=search_space
        if search_space is not None
        else experiment.search_space,
        data=data,
        model=model,
        transforms=TS_trans,
    )


def get_thompson(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    num_samples: int = 10000,
    min_weight: Optional[float] = None,
    uniform_weights: bool = False,
) -> DiscreteModelBridge:
    """Instantiates a Thompson sampling model."""
    if data.df.empty:  # pragma: no cover
        raise ValueError("Thompson sampler requires non-empty data.")
    model = ThompsonSampler(
        num_samples=num_samples, min_weight=min_weight, uniform_weights=uniform_weights
    )
    return DiscreteModelBridge(
        experiment=experiment,
        search_space=search_space
        if search_space is not None
        else experiment.search_space,
        data=data,
        model=model,
        transforms=TS_trans,
    )


class Models(Enum):
    """Registry of available factory functions."""

    SOBOL = get_sobol
    GPEI = get_GPEI
    FACTORIAL = get_factorial
    THOMPSON = get_thompson
    BOTORCH = get_botorch
    EMPIRICAL_BAYES_THOMPSON = get_empirical_bayes_thompson
    UNIFORM = get_uniform
