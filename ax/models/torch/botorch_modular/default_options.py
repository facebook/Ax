#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import warnings
from inspect import isclass
from typing import Dict, Any, Type

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)

# Options dictionary specifying optimizer defaults for acquisition functions.
DEFAULT_OPTIMIZER_OPTIONS: Dict[Type[AcquisitionFunction], Dict[str, Any]] = {}


def mk_generic_default_optimizer_options() -> Dict[str, Any]:
    """Makes a copy of dictionary for generic default optimizer options,
    used when optimizer options for a given acquisition function are not
    registered.

    NOTE: Return of this function is safe to modify without affecting the
    default options returned subsequently.
    """
    return {"num_restarts": 40, "raw_samples": 1024}


def mk_ehvi_default_optimizer_options() -> Dict[str, Any]:
    """Makes a copy of dictionary for generic default optimizer options
    for EHVI-based acquisition function, used when optimizer options
    for a given acquisition function are not registered.

    NOTE: Return of this function is safe to modify without affecting the
    default options returned subsequently.
    """
    return {
        "sequential": True,
        "num_restarts": 40,
        "raw_samples": 1024,
        "options": {
            "init_batch_limit": 128,  # Used in `gen_batch_initial_conditions`.
            "batch_limit": 5,  # Batch limit prevents memory issues in initialization.
        },
    }


def get_default_optimizer_options(
    acqf_class: Type[AcquisitionFunction],
) -> Dict[str, Any]:
    """Returns default options as a dict for a given acquisition function
    class.

    NOTE: Logs a warning and returns ``DEFAULT_OPTIMIZER_OPTIONS`` if acquisition
    function is not registered in ``DEFAULT_OPTIMIZER_OPTIONS``.
    """
    if not isclass(acqf_class):  # pragma: no cover
        raise TypeError(f"{acqf_class} is not an acquisition function class.")
    if acqf_class not in DEFAULT_OPTIMIZER_OPTIONS:
        warnings.warn(
            "No default optimizer options registered for acquisition function "
            f"class {acqf_class.__name__}; using generic default optimizer options"
            f": {mk_generic_default_optimizer_options()}. To register default "
            "optimizer options for an acquisition function, add it via `ax.models."
            "torch.botorch_modular.default_options.register_default_optimizer_options`."
        )
        return mk_generic_default_optimizer_options()
    return DEFAULT_OPTIMIZER_OPTIONS[acqf_class]


def register_default_optimizer_options(
    acqf_class: Type[AcquisitionFunction], default_options: Dict[str, Any]
) -> None:
    """Registers default optimizer options for a given acquisition function."""
    DEFAULT_OPTIMIZER_OPTIONS[acqf_class] = default_options


# ----------- Adding individual acquisition function classes to the registry: ----------


register_default_optimizer_options(
    acqf_class=qNoisyExpectedImprovement,
    default_options=mk_generic_default_optimizer_options(),
)
register_default_optimizer_options(
    acqf_class=qExpectedImprovement,
    default_options=mk_generic_default_optimizer_options(),
)
register_default_optimizer_options(
    acqf_class=qExpectedHypervolumeImprovement,
    default_options=mk_ehvi_default_optimizer_options(),
)
register_default_optimizer_options(
    acqf_class=qNoisyExpectedHypervolumeImprovement,
    default_options=mk_ehvi_default_optimizer_options(),
)
