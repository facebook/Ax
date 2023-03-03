# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import numpy as np
import torch
from torch import Tensor


# pyre-fixme[2]: Parameter annotation cannot be `Any`.
def generic_equals(first: Any, second: Any) -> bool:
    if isinstance(first, Tensor):
        return isinstance(second, Tensor) and torch.equal(first, second)
    if isinstance(first, np.ndarray):
        return isinstance(second, np.ndarray) and np.array_equal(
            first, second, equal_nan=True
        )
    if isinstance(first, dict):
        return isinstance(second, dict) and generic_equals(
            sorted(first.items()), sorted(second.items())
        )
    if isinstance(first, (tuple, list)):
        if type(first) != type(second) or len(first) != len(second):
            return False
        for f, s in zip(first, second):
            if not generic_equals(f, s):
                return False
        return True
    return first == second
