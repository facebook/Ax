#!/usr/bin/env python3
from typing import Dict, List, Optional, Type, Union

import torch
from ae.lazarus.ae.core.data import Data
from ae.lazarus.ae.core.experiment import Experiment
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.generator.discrete import DiscreteGenerator
from ae.lazarus.ae.generator.numpy import NumpyGenerator
from ae.lazarus.ae.generator.random import RandomGenerator
from ae.lazarus.ae.generator.torch import TorchGenerator
from ae.lazarus.ae.generator.transforms.base import Transform
from ae.lazarus.ae.generator.transforms.derelativize import Derelativize
from ae.lazarus.ae.generator.transforms.int_range_to_choice import IntRangeToChoice
from ae.lazarus.ae.generator.transforms.int_to_float import IntToFloat
from ae.lazarus.ae.generator.transforms.ivw import IVW
from ae.lazarus.ae.generator.transforms.log import Log
from ae.lazarus.ae.generator.transforms.one_hot import OneHot
from ae.lazarus.ae.generator.transforms.ordered_choice_encode import OrderedChoiceEncode
from ae.lazarus.ae.generator.transforms.remove_fixed import RemoveFixed
from ae.lazarus.ae.generator.transforms.search_space_to_choice import (
    SearchSpaceToChoice,
)
from ae.lazarus.ae.generator.transforms.standardize_y import StandardizeY
from ae.lazarus.ae.generator.transforms.unit_x import UnitX
from ae.lazarus.ae.models.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ae.lazarus.ae.models.discrete.full_factorial import FullFactorialGenerator
from ae.lazarus.ae.models.discrete.thompson import ThompsonSampler
from ae.lazarus.ae.models.numpy.gpy import GPyGP
from ae.lazarus.ae.models.random.sobol import SobolGenerator
from ae.lazarus.ae.models.random.uniform import UniformGenerator
from ae.lazarus.ae.models.torch.botorch import BotorchModel


"""
Module containing functions that generate standard generators, such as Sobol,
GP+EI, etc.

Note: a special case here is a composite generator, which requires an
additional ``GenerationStrategy`` and is able to delegate work to multiple generators
(for instance, to an initialization generator for the first trial, and to an
optimization generator for subsequent trials).

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


DEFAULT_TORCH_DEVICE = torch.device("cpu")


def get_sobol(search_space: SearchSpace, **kwargs: Union[int, bool]) -> RandomGenerator:
    """Instantiates a Sobol sequence quasi-random generator.

    Args:
        search_space: Sobol generator search space.
        kwargs: Custom args for sobol generator.

    Returns:
        RandomGenerator, with SobolGenerator as model.
    """
    return RandomGenerator(
        search_space=search_space,
        # pyre-ignore[6]: expected `bool` for the 1st anon. param., got `int`
        model=SobolGenerator(**kwargs),
        transforms=Cont_X_trans,
    )


def get_uniform(
    search_space: SearchSpace, **kwargs: Union[int, bool]
) -> RandomGenerator:
    """Instantiate uniform generator.

    Args:
        search_space: Uniform generator search space.
        kwargs: Custom args for uniform generator.

    Returns:
        RandomGenerator, with UniformGenerator as model.
    """
    return RandomGenerator(
        search_space=search_space,
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
) -> TorchGenerator:
    """Instantiates a GP+EI generator."""
    if search_space is None:
        search_space = experiment.search_space
    return TorchGenerator(
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


def get_GPyGPEI(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    **kwargs: Union[int, bool, str]
) -> NumpyGenerator:
    """Instantiates a GPy-backed GP+EI generator."""
    if search_space is None:
        search_space = experiment.search_space
    return NumpyGenerator(
        experiment=experiment,
        search_space=search_space,
        data=data,
        # Expected `int` for 1st anon. param. to `ae.lazarus.models.numpy.
        # pyre-ignore [6]: gpy.GPyGP.__init__`, but got `str`.
        model=GPyGP(**kwargs),
        transforms=Cont_X_trans + Y_trans,
    )


def get_factorial(search_space: SearchSpace) -> DiscreteGenerator:
    """Instantiates a factorial generator."""
    return DiscreteGenerator(
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
) -> DiscreteGenerator:
    """Instantiates an empirical Bayes / Thompson sampling generator."""
    model = EmpiricalBayesThompsonSampler(
        num_samples=num_samples, min_weight=min_weight, uniform_weights=uniform_weights
    )
    return DiscreteGenerator(
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
) -> DiscreteGenerator:
    """Instantiates a Thompson sampling generator."""
    model = ThompsonSampler(
        num_samples=num_samples, min_weight=min_weight, uniform_weights=uniform_weights
    )
    return DiscreteGenerator(
        experiment=experiment,
        search_space=search_space
        if search_space is not None
        else experiment.search_space,
        data=data,
        model=model,
        transforms=TS_trans,
    )
