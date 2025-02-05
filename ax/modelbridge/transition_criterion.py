# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from ax.generation_strategy.transition_criterion import *  # noqa

warnings.warn(
    "Please import from 'ax.generation_strategy.transition_criterion' instead of "
    "'ax.modelbridge.transition_criterion'. The latter is deprecated and will be "
    "removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
