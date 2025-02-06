#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from ax.generation_strategy.dispatch_utils import *  # noqa
from ax.generation_strategy.dispatch_utils import (  # noqa
    _get_winsorization_transform_config,  # noqa
    _make_botorch_step,  # noqa
    _make_sobol_step,  # noqa
)

warnings.warn(
    "Please import from 'ax.generation_strategy.dispatch_utils'"
    "instead of 'ax.modelbridge.dispatch_utils'. The latter is "
    "deprecated and will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
