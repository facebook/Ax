# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterable

import torch
from torch import Tensor


# pyre-fixme[2]: Parameter annotation cannot be `Any`.
def generic_equals(first: Any, second: Any) -> bool:
    if isinstance(first, Tensor):
        return torch.equal(first, second)
    if isinstance(first, Iterable):
        for f, s in zip(first, second):
            if not generic_equals(f, s):
                return False
        return True
    return first == second
