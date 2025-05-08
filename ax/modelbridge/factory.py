#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Mapping, Sequence

import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.modelbridge.base import DataLoaderConfig
from ax.modelbridge.discrete import DiscreteAdapter
from ax.modelbridge.random import RandomAdapter
from ax.modelbridge.registry import Cont_X_trans, Generators, Y_trans
from ax.modelbridge.torch import TorchAdapter
from ax.modelbridge.transforms.base import Transform
from ax.models.torch.botorch import (
    TAcqfConstructor,
    TModelConstructor,
    TModelPredictor,
    TOptimizer,
)
from ax.models.torch.botorch_defaults import (
    get_and_fit_model,
    get_qLogNEI,
    scipy_optimizer,
)
from ax.models.torch.utils import predict_from_model
from ax.models.types import TConfig
from pyre_extensions import assert_is_instance


DEFAULT_TORCH_DEVICE = torch.device("cpu")
DEFAULT_EHVI_BATCH_LIMIT = 5


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


def get_uniform(
    search_space: SearchSpace, deduplicate: bool = False, seed: int | None = None
) -> RandomAdapter:
    """Instantiate uniform generator.

    Args:
        search_space: Uniform generator search space.
        kwargs: Custom args for uniform generator.

    Returns:
        RandomAdapter, with UniformGenerator as model.
    """
    return assert_is_instance(
        Generators.UNIFORM(
            experiment=Experiment(search_space=search_space),
            seed=seed,
            deduplicate=deduplicate,
        ),
        RandomAdapter,
    )


def get_botorch(
    experiment: Experiment,
    data: Data,
    search_space: SearchSpace | None = None,
    device: torch.device = DEFAULT_TORCH_DEVICE,
    transforms: Sequence[type[Transform]] = Cont_X_trans + Y_trans,
    transform_configs: Mapping[str, TConfig] | None = None,
    model_constructor: TModelConstructor = get_and_fit_model,
    model_predictor: TModelPredictor = predict_from_model,
    acqf_constructor: TAcqfConstructor = get_qLogNEI,
    acqf_optimizer: TOptimizer = scipy_optimizer,  # pyre-ignore[9]
    refit_on_cv: bool = False,
    optimization_config: OptimizationConfig | None = None,
) -> TorchAdapter:
    """Instantiates a LegacyBoTorchGenerator."""
    if data.df.empty:
        raise ValueError("`LegacyBoTorchGenerator` requires non-empty data.")
    return assert_is_instance(
        Generators.LEGACY_BOTORCH(
            experiment=experiment,
            data=data,
            search_space=search_space or experiment.search_space,
            torch_device=device,
            transforms=transforms,
            transform_configs=transform_configs,
            model_constructor=model_constructor,
            model_predictor=model_predictor,
            acqf_constructor=acqf_constructor,
            acqf_optimizer=acqf_optimizer,
            refit_on_cv=refit_on_cv,
            optimization_config=optimization_config,
        ),
        TorchAdapter,
    )


def get_factorial(search_space: SearchSpace) -> DiscreteAdapter:
    """Instantiates a factorial generator."""
    return assert_is_instance(
        Generators.FACTORIAL(
            search_space=search_space,
            data_loader_config=DataLoaderConfig(fit_out_of_design=True),
        ),
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
            search_space=search_space or experiment.search_space,
            num_samples=num_samples,
            min_weight=min_weight,
            uniform_weights=uniform_weights,
            data_loader_config=DataLoaderConfig(fit_out_of_design=True),
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
            search_space=search_space or experiment.search_space,
            num_samples=num_samples,
            min_weight=min_weight,
            uniform_weights=uniform_weights,
            data_loader_config=DataLoaderConfig(fit_out_of_design=True),
        ),
        DiscreteAdapter,
    )
