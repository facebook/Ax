#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Any

from ax.exceptions.core import UnsupportedError
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient

# Acquisition defaults
NUM_RESTARTS = 20
RAW_SAMPLES = 1024
INIT_BATCH_LIMIT = 32
BATCH_LIMIT = 5


def optimizer_argparse(
    acqf: AcquisitionFunction,
    *,
    optimizer: str,
    optimizer_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract the kwargs to be passed to a BoTorch optimizer.

    Args:
        acqf: The acquisition function being optimized.
        optimizer: The optimizer to parse args for. Typically chosen by
            `Acquisition.optimize`. Must be one of:
            - "optimize_acqf",
            - "optimize_acqf_discrete_local_search",
            - "optimize_acqf_discrete",
            - "optimize_acqf_homotopy",
            - "optimize_acqf_mixed",
            - "optimize_acqf_mixed_alternating".
        optimizer_options: An optional dictionary of optimizer options (some of
            these under an `options` dictionary); default values will be used
            where not specified. See the docstrings in
            `botorch/optim/optimize.py` for supported options.
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
        "optimize_acqf_homotopy",
        "optimize_acqf_mixed",
        "optimize_acqf_mixed_alternating",
    ]
    if optimizer not in supported_optimizers:
        raise ValueError(
            f"optimizer=`{optimizer}` is not supported. Accepted options are "
            f"{supported_optimizers}"
        )
    if (optimizer != "optimize_acqf") and isinstance(acqf, qKnowledgeGradient):
        raise RuntimeError(
            "Ax is attempting to use a discrete or mixed optimizer, "
            f"`{optimizer}`, but this is not compatible with "
            "`qKnowledgeGradient` or its subclasses. To address this, please "
            "either use a different acquisition class or make parameters "
            "continuous using the transform "
            "`ax.modelbridge.registry.Cont_X_trans`."
        )
    provided_options = optimizer_options if optimizer_options is not None else {}

    # Construct arguments from options that are not `provided_options`.
    if optimizer == "optimize_acqf_discrete":
        # `optimize_acqf_discrete` only accepts 'choices', 'max_batch_size', 'unique'.
        options = {}
    else:
        options = {
            "num_restarts": NUM_RESTARTS,
            "raw_samples": RAW_SAMPLES,
        }

    if optimizer in [
        "optimize_acqf",
        "optimize_acqf_homotopy",
        "optimize_acqf_mixed",
        "optimize_acqf_mixed_alternating",
    ]:
        options["options"] = {
            "init_batch_limit": INIT_BATCH_LIMIT,
            "batch_limit": BATCH_LIMIT,
            **provided_options.get("options", {}),
        }
    # Error out if options are specified for an optimizer that does not support the arg.
    elif "options" in provided_options:
        raise UnsupportedError(
            f"`{optimizer=}` does not support the `options` argument."
        )

    if optimizer == "optimize_acqf":
        options["sequential"] = True

    options.update(**{k: v for k, v in provided_options.items() if k != "options"})
    return options
