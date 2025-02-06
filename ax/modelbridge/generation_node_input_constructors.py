#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from ax.generation_strategy.generation_node_input_constructors import *  # noqa

warnings.warn(
    "Please import from 'ax.generation_strategy.generation_node_input_constructors'"
    "instead of 'ax.modelbridge.generation_node_input_constructors'. The latter is "
    "deprecated and will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
