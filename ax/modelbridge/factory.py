#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger
from typing import Any, Dict, List, Optional, Type

import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
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


logger: Logger = get_logger(__name__)


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
    transform_configs: Optional[Dict[str, TConfig]] = None,
    model_constructor: TModelConstructor = get_and_fit_model,
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
    return checked_cast(
        TorchModelBridge,
        Models.BOTORCH(
            experiment=experiment,
            data=data,
            search_space=search_space or experiment.search_space,
            torch_dtype=dtype,
            torch_device=device,
            transforms=transforms,
            transform_configs=transform_configs,
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


def get_GPKG(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    cost_intercept: float = 0.01,
    dtype: torch.dtype = torch.double,
    device: torch.device = DEFAULT_TORCH_DEVICE,
    transforms: List[Type[Transform]] = Cont_X_trans + Y_trans,
    transform_configs: Optional[Dict[str, TConfig]] = None,
    **kwargs: Any,
) -> TorchModelBridge:
    """Instantiates a GP model that generates points with KG."""
    if search_space is None:
        search_space = experiment.search_space
    if data.df.empty:  # pragma: no cover
        raise ValueError("GP+KG BotorchModel requires non-empty data.")

    inputs = {
        "search_space": search_space,
        "experiment": experiment,
        "data": data,
        "cost_intercept": cost_intercept,
        "torch_dtype": dtype,
        "torch_device": device,
        "transforms": transforms,
        "transform_configs": transform_configs,
    }

    if any(p.is_fidelity for k, p in experiment.parameters.items()):
        inputs["linear_truncated"] = kwargs.get("linear_truncated", True)
    return checked_cast(TorchModelBridge, Models.GPKG(**inputs))  # pyre-ignore: [16]


# TODO[Lena]: how to instantiate MTGP through the enum? The Multi-type MTGP requires
# a MultiTypeExperiment, so we would need validation for that, but more importantly,
# we need to create `trial_index_to_type` as in the factory function below.
# Maybe `MultiTypeExperiment` could have that mapping as a property?
def get_MTGP(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    trial_index: Optional[int] = None,
) -> TorchModelBridge:
    """Instantiates a Multi-task Gaussian Process (MTGP) model that generates
    points with EI.

    If the input experiment is a MultiTypeExperiment then a
    Multi-type Multi-task GP model will be instantiated.
    Otherwise, the model will be a Single-type Multi-task GP.
    """

    if isinstance(experiment, MultiTypeExperiment):
        trial_index_to_type = {
            t.index: t.trial_type for t in experiment.trials.values()
        }
        transforms = MT_MTGP_trans
        transform_configs = {
            "TrialAsTask": {"trial_level_map": {"trial_type": trial_index_to_type}},
            "ConvertMetricNames": tconfig_from_mt_experiment(experiment),
        }
    else:
        # Set transforms for a Single-type MTGP model.
        transforms = ST_MTGP_trans
        transform_configs = None

    # Choose the status quo features for the experiment from the selected trial.
    # If trial_index is None, we will look for a status quo from the last
    # experiment trial to use as a status quo for the experiment.
    if trial_index is None:
        trial_index = len(experiment.trials) - 1
    elif trial_index >= len(experiment.trials):
        raise ValueError("trial_index is bigger than the number of experiment trials")

    # pyre-fixme[16]: `ax.core.base_trial.BaseTrial` has no attribute `status_quo`.
    status_quo = experiment.trials[trial_index].status_quo
    if status_quo is None:
        status_quo_features = None
    else:
        status_quo_features = ObservationFeatures(
            parameters=status_quo.parameters, trial_index=trial_index
        )

    return TorchModelBridge(
        experiment=experiment,
        search_space=search_space or experiment.search_space,
        data=data,
        model=BotorchModel(),
        transforms=transforms,
        # pyre-fixme[6]: Expected `Optional[Dict[str, Dict[str,
        #  typing.Union[botorch.acquisition.acquisition.AcquisitionFunction, float,
        #  int, str]]]]` for 6th param but got `Optional[Dict[str,
        #  typing.Union[Dict[str, Dict[str, Dict[int, Optional[str]]]], Dict[str,
        #  typing.Union[botorch.acquisition.acquisition.AcquisitionFunction, float,
        #  int, str]]]]]`.
        transform_configs=transform_configs,
        torch_dtype=torch.double,
        torch_device=DEFAULT_TORCH_DEVICE,
        status_quo_features=status_quo_features,
    )


def get_factorial(search_space: SearchSpace) -> DiscreteModelBridge:
    """Instantiates a factorial generator."""
    return checked_cast(
        DiscreteModelBridge,
        Models.FACTORIAL(search_space=search_space, fit_out_of_design=True),
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
    return checked_cast(
        DiscreteModelBridge,
        Models.EMPIRICAL_BAYES_THOMPSON(
            experiment=experiment,
            data=data,
            search_space=search_space or experiment.search_space,
            num_samples=num_samples,
            min_weight=min_weight,
            uniform_weights=uniform_weights,
            fit_out_of_design=True,
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
            fit_out_of_design=True,
        ),
    )


def get_GPMES(
    experiment: Experiment,
    data: Data,
    search_space: Optional[SearchSpace] = None,
    cost_intercept: float = 0.01,
    dtype: torch.dtype = torch.double,
    device: torch.device = DEFAULT_TORCH_DEVICE,
    transforms: List[Type[Transform]] = Cont_X_trans + Y_trans,
    transform_configs: Optional[Dict[str, TConfig]] = None,
    **kwargs: Any,
) -> TorchModelBridge:
    """Instantiates a GP model that generates points with MES."""
    if search_space is None:
        search_space = experiment.search_space
    if data.df.empty:  # pragma: no cover
        raise ValueError("GP + MES BotorchModel requires non-empty data.")

    inputs = {
        "search_space": search_space,
        "experiment": experiment,
        "data": data,
        "cost_intercept": cost_intercept,
        "torch_dtype": dtype,
        "torch_device": device,
        "transforms": transforms,
        "transform_configs": transform_configs,
    }

    if any(p.is_fidelity for k, p in experiment.parameters.items()):
        inputs["linear_truncated"] = kwargs.get("linear_truncated", True)
    return checked_cast(TorchModelBridge, Models.GPMES(**inputs))  # pyre-ignore: [16]
