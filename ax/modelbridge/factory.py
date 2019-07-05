#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import List, Optional, Type

import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.random import RandomModelBridge
from ax.modelbridge.registry import (
    Cont_X_trans,
    Models,
    MT_MTGP_trans,
    ST_MTGP_trans,
    Y_trans,
)
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.convert_metric_names import tconfig_from_mt_experiment
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
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast


logger = get_logger(__name__)


# pyre-fixme[19]: __init__ expects 0 args but got 1.
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
    logger.info(
        "Factory functions (like `get_sobol`) will soon be deprecated. Use "
        "the model registry instead (`Models.SOBOL(...)`)."
    )
    return checked_cast(
        RandomModelBridge,
        Models.SOBOL(
            search_space=search_space,
            seed=seed,
            deduplicate=deduplicate,
            init_position=init_position,
            scramble=scramble,
        ),
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
    logger.info(
        "Factory functions (like `get_uniform`) will soon be deprecated). Use "
        "the model registry instead (`Models.UNIFORM(...)`)."
    )
    return checked_cast(
        RandomModelBridge,
        Models.UNIFORM(search_space=search_space, seed=seed, deduplicate=deduplicate),
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
    optimization_config: Optional[OptimizationConfig] = None,
) -> TorchModelBridge:
    """Instantiates a BotorchModel."""
    if data.df.empty:  # pragma: no cover
        raise ValueError("`BotorchModel` requires non-empty data.")
    logger.info(
        "Factory functions (like `get_botorch`) will soon be deprecated). Use "
        "the model registry instead (`Models.BOTORCH(...)`)."
    )
    return checked_cast(
        TorchModelBridge,
        Models.BOTORCH(
            experiment=experiment,
            data=data,
            search_space=search_space or experiment.search_space,
            torch_dtype=dtype,
            torch_device=device,
            transforms=transforms,
            model_constructor=model_constructor,
            model_predictor=model_predictor,
            acqf_constructor=acqf_constructor,
            acqf_optimizer=acqf_optimizer,
            refit_on_cv=refit_on_cv,
            refit_on_update=refit_on_update,
            optimization_config=optimization_config,
        ),
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
    logger.info(
        "Factory functions (like `get_GPEI`) will soon be deprecated). Use "
        "the model registry instead (`Models.GPEI(...)`)."
    )
    return checked_cast(
        TorchModelBridge,
        Models.BOTORCH(
            experiment=experiment,
            data=data,
            search_space=search_space or experiment.search_space,
            torch_dtype=dtype,
            torch_device=device,
        ),
    )


# TODO[Lena]: how to instantiate MTGP through the enum? The Multi-type MTGP requires
# a MultiTypeExperiment, so we would need validation for that, but more importantly,
# we need to create `trial_index_to_type` as in the factory function below.
# Maybe `MultiTypeExperiment` could have that mapping as a property?
def get_MTGP(
    experiment: Experiment,
    data: Data,
    is_multi_type: bool = True,
    search_space: Optional[SearchSpace] = None,
) -> TorchModelBridge:
    """Instantiates a Multi-task GP model that generates points with EI.

    Args:
        is_multi_type: If is_multi_type is True then experiment should be a
            MultiTypeExperiment and a Multi-type Multi-task GP model will be
            instantiated.
            Otherwise, the model will be a Single-type Multi-task GP.
    """

    if is_multi_type and isinstance(experiment, MultiTypeExperiment):
        trial_index_to_type = {
            t.index: t.trial_type for t in experiment.trials.values()
        }
        transforms = MT_MTGP_trans
        transform_configs = {
            "TrialAsTask": {"trial_level_map": {"trial_type": trial_index_to_type}},
            "ConvertMetricNames": tconfig_from_mt_experiment(experiment),
        }
    elif is_multi_type:
        raise ValueError(
            "If is_multi_type is True, the input experiment type should be "
            "MultiTypeExperiment."
        )
    else:
        transforms = ST_MTGP_trans
        transform_configs = None

    return TorchModelBridge(
        experiment=experiment,
        search_space=search_space or experiment.search_space,
        data=data,
        model=BotorchModel(),
        transforms=transforms,
        transform_configs=transform_configs,
        torch_dtype=torch.double,
        torch_device=DEFAULT_TORCH_DEVICE,
    )


def get_factorial(search_space: SearchSpace) -> DiscreteModelBridge:
    """Instantiates a factorial generator."""
    logger.info(
        "Factory functions (like `get_factorial`) will soon be deprecated). Use "
        "the model registry instead (`Models.FACTORIAL(...)`)."
    )
    return checked_cast(
        DiscreteModelBridge, Models.FACTORIAL(search_space=search_space)
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
    logger.info(
        "Factory functions (like `get_empirical_bayes_thompson`) will soon be "
        "deprecated). Use the model registry instead (`Models.EMPIRICAL_BAYES"
        "(...)`)."
    )
    return checked_cast(
        DiscreteModelBridge,
        Models.EMPIRICAL_BAYES_THOMPSON(
            experiment=experiment,
            data=data,
            search_space=search_space or experiment.search_space,
            num_samples=num_samples,
            min_weight=min_weight,
            uniform_weights=uniform_weights,
        ),
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
    return checked_cast(
        DiscreteModelBridge,
        Models.THOMPSON(
            experiment=experiment,
            data=data,
            search_space=search_space or experiment.search_space,
            num_samples=num_samples,
            min_weight=min_weight,
            uniform_weights=uniform_weights,
        ),
    )
