#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from ax.generation_strategy.generator_spec import *  # noqa

warnings.warn(
    "Please import from 'ax.generation_strategy.generator_spec' instead of "
    "'ax.adapter.generator_spec'. The latter is deprecated and will be "
    "removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
