#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Re-export shim for backwards compatibility.

The canonical location for these symbols is now ``ax.core.data_utils``.
"""

from ax.core.data_utils import (  # noqa: F401
    _extract_arm_data,
    _extract_observation_data,
    _maybe_normalize_map_key,
    _use_object_dtype_for_strings,
    DataLoaderConfig,
    ExperimentData,
    extract_experiment_data,
)

__all__ = [
    "_extract_arm_data",
    "_extract_observation_data",
    "_maybe_normalize_map_key",
    "_use_object_dtype_for_strings",
    "DataLoaderConfig",
    "ExperimentData",
    "extract_experiment_data",
]
