# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Models whose generated trails will count towards initialization_trials
from typing import List

from ax.modelbridge.registry import Models

INITIALIZATION_MODELS: List[Models] = [Models.SOBOL, Models.UNIFORM]

# Models whose generated trails will count towards other_trials
OTHER_MODELS: List[Models] = []
