#!/usr/bin/env python3
from typing import Dict, List, Optional, Type, Union

import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.search_space import SearchSpace
from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.base import Transform
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
from ax.modelbridge.transforms.unit_x import UnitX
from ax.models.discrete.ancillary_eb_thompson import AncillaryEBThompsonSampler
from ax.models.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ax.models.discrete.full_factorial import FullFactorialGenerator
from ax.models.discrete.thompson import ThompsonSampler
from ax.models.random.sobol import SobolGenerator
from ax.models.random.uniform import UniformGenerator
from ax.models.torch.botorch import BotorchModel


"""
Module containing functions that generate standard models, such as Sobol,
GP+EI, etc.

Note: a special case here is a composite generator, which requires an
additional ``GenerationStrategy`` and is able to delegate work to multiple models
(for instance, to a random model to generate the first trial, and to an
optimization model for subsequent trials).

"""


Cont_X_trans: List[Type[Transform]] = [
    RemoveFixed,
    OrderedChoiceEncode,
    OneHot,
    IntToFloat,
    Log,
    UnitX,
]
Discrete_X_trans: List[Type[Transform]] = [RemoveFixed, IntRangeToChoice]
Y_trans: List[Type[Transform]] = [IVW, Derelativize, StandardizeY]
# Expected `List[Type[Transform]]` for 2nd anonymous parameter to
# call `list.__add__` but got `List[Type[SearchSpaceToChoice]]`.
TS_trans: List[Type[Transform]] = Discrete_X_trans + Y_trans + [SearchSpaceToChoice]
# Same as TS_trans but omit StandardizeY
Ancillary_EB_trans: List[Type[Transform]] = Discrete_X_trans + [
    IVW,
    Derelativize,
    SearchSpaceToChoice,
]


DEFAULT_TORCH_DEVICE = torch.device("cpu")


def get_sobol(
    search_space: SearchSpace, **kwargs: Union[int, bool]
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
        # pyre-ignore[6]: expected `bool` for the 1st anon. param., got `int`
        model=SobolGenerator(**kwargs),
        transforms=Cont_X_trans,
    )


def get_uniform(
    search_space: SearchSpace, **kwargs: Union[int, bool]
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
        # pyre-ignore[6]: expected `bool` for the 1st anon. param., got `int`
        model=UniformGenerator(**kwargs),
        transforms=Cont_X_trans,
    )


def get_GPEI(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    dtype: torch.dtype = torch.double,
    device: torch.device = DEFAULT_TORCH_DEVICE,
    **kwargs: Union[Dict[str, Union[int, bool]], bool, int, float, str]
) -> TorchModelBridge:
    """Instantiates a GP model that generates points with EI."""
    if search_space is None:
        search_space = experiment.search_space
    return TorchModelBridge(
        experiment=experiment,
        search_space=search_space,
        data=data,
        # Expected `Optional[Dict[str, Union[float, int]]]` for 1st anon. param.
        # to call `ae.lazarus.models.torch.botorch.BotorchModel.__init__`
        # pyre-ignore[6]: but got `Dict[str, Union[bool, int]].
        model=BotorchModel(**kwargs),
        transforms=Cont_X_trans + Y_trans,
        torch_dtype=dtype,
        torch_device=device,
    )


def get_factorial(search_space: SearchSpace) -> DiscreteModelBridge:
    """Instantiates a factorial generator."""
    return DiscreteModelBridge(
        search_space=search_space,
        data=Data(),
        model=FullFactorialGenerator(),
        transforms=Discrete_X_trans,
    )


def get_ancillary_eb_thompson(
    experiment: Experiment,
    data: Data,
    primary_outcome: str,
    secondary_outcome: str,
    search_space: Optional[SearchSpace] = None,
    num_samples: int = 10000,
    min_weight: Optional[float] = None,
    uniform_weights: bool = False,
) -> DiscreteModelBridge:
    """Instantiates an Ancillary EB / Thompson sampling generator."""
    model = AncillaryEBThompsonSampler(
        primary_outcome=primary_outcome,
        secondary_outcome=secondary_outcome,
        num_samples=num_samples,
        min_weight=min_weight,
        uniform_weights=uniform_weights,
    )
    return DiscreteModelBridge(
        experiment=experiment,
        search_space=search_space
        if search_space is not None
        else experiment.search_space,
        data=data,
        model=model,
        transforms=Ancillary_EB_trans,
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
