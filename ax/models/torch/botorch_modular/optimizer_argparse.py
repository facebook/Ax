#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict, Optional, Type, TypeVar, Union

import torch
from ax.utils.common.constants import Keys
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from botorch.utils.dispatcher import Dispatcher

T = TypeVar("T")
MaybeType = Union[T, Type[T]]  # Annotation for a type or instance thereof


# pyre-fixme[2]: Parameter annotation cannot be `Any`.
# pyre-fixme[24]: Generic type `type` expects 1 type parameter, use `typing.Type` to
#  avoid runtime subscripting errors.
def _optimizerArgparse_encoder(arg: Any) -> Type:
    """
    Transforms arguments passed to `optimizer_argparse.__call__`
    at runtime to construct the key used for method lookup as
    `tuple(map(arg_transform, args))`.

    This custom arg_transform allow type variables to be passed
    at runtime.
    """
    # Allow type variables to be passed as arguments at runtime
    return arg if isinstance(arg, type) else type(arg)


optimizer_argparse = Dispatcher(
    name="optimizer_argparse", encoder=_optimizerArgparse_encoder
)


@optimizer_argparse.register(AcquisitionFunction)
def _argparse_base(
    acqf: MaybeType[AcquisitionFunction],
    sequential: bool = True,
    num_restarts: int = 20,
    raw_samples: int = 1024,
    init_batch_limit: int = 32,
    batch_limit: int = 5,
    optimizer_options: Optional[Dict[str, Any]] = None,
    optimizer_is_discrete: bool = False,
    **ignore: Any,
) -> Dict[str, Any]:
    """Extract the base optimizer kwargs form the given arguments.

    NOTE: Since `optimizer_options` is how the user would typically pass in these
    options, it takes precedence over other arguments. E.g., if both `num_restarts`
    and `optimizer_options["num_restarts"]` are provided, this will use
    `num_restarts` from `optimizer_options`.

    Args:
        acqf: The acquisition function being optimized.
        sequential: Whether we choose one candidate at a time in a sequential manner.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of samples for initialization.
        init_batch_limit: The size of mini-batches used to evaluate the `raw_samples`.
            This helps reduce peak memory usage.
        batch_limit: The size of mini-batches used while optimizing the `acqf`.
            This helps reduce peak memory usage.
        optimizer_options: An optional dictionary of optimizer options. This may
            include overrides for the above options (some of these under an `options`
            dictionary) or any other option that is accepted by the optimizer. See
            the docstrings in `botorch/optim/optimize.py` for supported options.
            Example:
                >>> optimizer_options = {
                >>>     "num_restarts": 20,
                >>>     "options": {
                >>>         "maxiter": 200,
                >>>         "batch_limit": 5,
                >>>     },
                >>>     "retry_on_optimization_warning": False,
                >>> }
        optimizer_is_discrete: True if the optimizer is `optimizer_acqf_discrete`,
            which supports a limited set of arguments.
    """
    optimizer_options = optimizer_options or {}
    if optimizer_is_discrete:
        return optimizer_options
    return {
        "sequential": sequential,
        "num_restarts": num_restarts,
        "raw_samples": raw_samples,
        "options": {
            "init_batch_limit": init_batch_limit,
            "batch_limit": batch_limit,
            **optimizer_options.get("options", {}),
        },
        **{k: v for k, v in optimizer_options.items() if k != "options"},
    }


@optimizer_argparse.register(qExpectedHypervolumeImprovement)
def _argparse_ehvi(
    acqf: MaybeType[qExpectedHypervolumeImprovement],
    sequential: bool = True,
    init_batch_limit: int = 32,
    batch_limit: int = 5,
    optimizer_options: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return {
        **_argparse_base(
            acqf=acqf,
            init_batch_limit=init_batch_limit,
            batch_limit=batch_limit,
            optimizer_options=optimizer_options,
            **kwargs,
        ),
        "sequential": sequential,
    }


@optimizer_argparse.register(qKnowledgeGradient)
def _argparse_kg(
    acqf: qKnowledgeGradient,
    q: int,
    bounds: torch.Tensor,
    num_restarts: int = 20,
    raw_samples: int = 1024,
    frac_random: float = 0.1,
    optimizer_options: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:

    optimizer_options = optimizer_options or {}
    base_options = _argparse_base(
        acqf,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        optimizer_options=optimizer_options,
        **kwargs,
    )

    initial_conditions = gen_one_shot_kg_initial_conditions(
        acq_function=acqf,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={
            Keys.FRAC_RANDOM: frac_random,
            Keys.NUM_INNER_RESTARTS: num_restarts,
            Keys.RAW_INNER_SAMPLES: raw_samples,
        },
    )

    return {
        **base_options,
        Keys.BATCH_INIT_CONDITIONS: initial_conditions,
    }


@optimizer_argparse.register(qMaxValueEntropy)
def _argparse_mes(
    acqf: AcquisitionFunction,
    sequential: bool = True,
    optimizer_options: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    return {
        **_argparse_base(acqf=acqf, optimizer_options=optimizer_options, **kwargs),
        "sequential": sequential,
    }
