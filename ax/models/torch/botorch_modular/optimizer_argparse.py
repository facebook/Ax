#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Any, Optional, TypeVar, Union

import torch
from ax.utils.common.constants import Keys
from ax.utils.common.typeutils import _argparse_type_encoder
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from botorch.utils.dispatcher import Dispatcher

T = TypeVar("T")
MaybeType = Union[T, type[T]]  # Annotation for a type or instance thereof

# Acquisition defaults
NUM_RESTARTS = 20
RAW_SAMPLES = 1024
INIT_BATCH_LIMIT = 32
BATCH_LIMIT = 5


optimizer_argparse = Dispatcher(
    name="optimizer_argparse", encoder=_argparse_type_encoder
)


@optimizer_argparse.register(AcquisitionFunction)
def _argparse_base(
    acqf: MaybeType[AcquisitionFunction],
    *,
    optimizer: str,
    sequential: bool = True,
    num_restarts: int = NUM_RESTARTS,
    raw_samples: int = RAW_SAMPLES,
    init_batch_limit: int = INIT_BATCH_LIMIT,
    batch_limit: int = BATCH_LIMIT,
    optimizer_options: Optional[dict[str, Any]] = None,
    **ignore: Any,
) -> dict[str, Any]:
    """Extract the kwargs to be passed to a BoTorch optimizer.

    NOTE: Since `optimizer_options` is how the user would typically pass in these
    options, it takes precedence over other arguments. E.g., if both `num_restarts`
    and `optimizer_options["num_restarts"]` are provided, this will use
    `num_restarts` from `optimizer_options`.

    Args:
        acqf: The acquisition function being optimized.
        optimizer: one of "optimize_acqf",
            "optimize_acqf_discrete_local_search", "optimize_acqf_discrete", or
            "optimize_acqf_mixed". This is generally chosen by
            `Acquisition.optimize`.
        sequential: Whether we choose one candidate at a time in a sequential
            manner. Ignored unless the optimizer is `optimize_acqf`.
        num_restarts: The number of starting points for multistart acquisition
            function optimization. Ignored if the optimizer is
            `optimize_acqf_discrete`.
        raw_samples: The number of samples for initialization. Ignored if the
            optimizer is `optimize_acqf_discrete`.
        init_batch_limit: The size of mini-batches used to evaluate the `raw_samples`.
            This helps reduce peak memory usage. Ignored if the optimizer is
            `optimize_acqf_discrete` or `optimize_acqf_discrete_local_search`.
        batch_limit: The size of mini-batches used while optimizing the `acqf`.
            This helps reduce peak memory usage. Ignored if the optimizer is
            `optimize_acqf_discrete` or `optimize_acqf_discrete_local_search`.
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
    """
    supported_optimizers = [
        "optimize_acqf",
        "optimize_acqf_discrete_local_search",
        "optimize_acqf_discrete",
        "optimize_acqf_mixed",
        "optimize_acqf_homotopy",
    ]
    if optimizer not in supported_optimizers:
        raise ValueError(
            f"optimizer=`{optimizer}` is not supported. Accepted options are "
            f"{supported_optimizers}"
        )
    provided_options = optimizer_options if optimizer_options is not None else {}

    # optimize_acqf_discrete only accepts 'choices', 'max_batch_size', 'unique'
    if optimizer == "optimize_acqf_discrete":
        return provided_options

    # construct arguments from options that are not `provided_options`
    options = {
        "num_restarts": num_restarts,
        "raw_samples": raw_samples,
    }
    # if not, 'options' will be silently ignored
    if optimizer in ["optimize_acqf", "optimize_acqf_mixed", "optimize_acqf_homotopy"]:
        options["options"] = {
            "init_batch_limit": init_batch_limit,
            "batch_limit": batch_limit,
            **provided_options.get("options", {}),
        }

    if optimizer == "optimize_acqf":
        options["sequential"] = sequential

    options.update(**{k: v for k, v in provided_options.items() if k != "options"})
    return options


@optimizer_argparse.register(qKnowledgeGradient)
def _argparse_kg(
    acqf: qKnowledgeGradient,
    q: int,
    bounds: torch.Tensor,
    num_restarts: int = NUM_RESTARTS,
    raw_samples: int = RAW_SAMPLES,
    frac_random: float = 0.1,
    optimizer_options: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Argument constructor for optimization with qKG, differing from the
    base case in that it computes and returns initial conditions.

    To do so, it requires specifying additional arguments `q` and `bounds` and
    allows for specifying `frac_random`.
    """
    base_options = _argparse_base(
        acqf,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        optimizer_options=optimizer_options,
        optimizer="optimize_acqf",
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
