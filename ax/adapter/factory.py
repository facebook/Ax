#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from ax.adapter.discrete import DiscreteAdapter
from ax.adapter.random import RandomAdapter
from ax.adapter.registry import Generators
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.search_space import SearchSpace
from pyre_extensions import assert_is_instance


DEFAULT_TORCH_DEVICE = torch.device("cpu")


"""
Module containing functions that generate standard models, such as Sobol,
GP+EI, etc.

Note: a special case here is a composite generator, which requires an
additional ``GenerationStrategy`` and is able to delegate work to multiple models
(for instance, to a random model to generate the first trial, and to an
optimization model for subsequent trials).

"""


def get_sobol(
    search_space: SearchSpace,
    seed: int | None = None,
    deduplicate: bool = False,
    init_position: int = 0,
    scramble: bool = True,
    fallback_to_sample_polytope: bool = False,
) -> RandomAdapter:
    """Instantiates a Sobol sequence quasi-random generator.

    Args:
        search_space: Sobol generator search space.
        kwargs: Custom args for sobol generator.

    Returns:
        RandomAdapter, with SobolGenerator as model.
    """
    return assert_is_instance(
        Generators.SOBOL(
            experiment=Experiment(search_space=search_space),
            seed=seed,
            deduplicate=deduplicate,
            init_position=init_position,
            scramble=scramble,
            fallback_to_sample_polytope=fallback_to_sample_polytope,
        ),
        RandomAdapter,
    )


def get_factorial(search_space: SearchSpace) -> DiscreteAdapter:
    """Instantiates a factorial generator."""
    return assert_is_instance(
        Generators.FACTORIAL(search_space=search_space),
        DiscreteAdapter,
    )


def get_empirical_bayes_thompson(
    experiment: Experiment,
    data: Data,
    search_space: SearchSpace | None = None,
    num_samples: int = 10000,
    min_weight: float | None = None,
    uniform_weights: bool = False,
) -> DiscreteAdapter:
    """Instantiates an empirical Bayes / Thompson sampling model."""
    if data.df.empty:
        raise ValueError("Empirical Bayes Thompson sampler requires non-empty data.")
    return assert_is_instance(
        Generators.EMPIRICAL_BAYES_THOMPSON(
            experiment=experiment,
            data=data,
            search_space=search_space,
            num_samples=num_samples,
            min_weight=min_weight,
            uniform_weights=uniform_weights,
        ),
        DiscreteAdapter,
    )


def get_thompson(
    experiment: Experiment,
    data: Data,
    search_space: SearchSpace | None = None,
    num_samples: int = 10000,
    min_weight: float | None = None,
    uniform_weights: bool = False,
) -> DiscreteAdapter:
    """Instantiates a Thompson sampling model."""
    if data.df.empty:
        raise ValueError("Thompson sampler requires non-empty data.")
    return assert_is_instance(
        Generators.THOMPSON(
            experiment=experiment,
            data=data,
            search_space=search_space,
            num_samples=num_samples,
            min_weight=min_weight,
            uniform_weights=uniform_weights,
        ),
        DiscreteAdapter,
    )
