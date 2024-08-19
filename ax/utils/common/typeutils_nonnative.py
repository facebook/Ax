#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any

import numpy as np


# pyre-fixme[3]: Return annotation cannot be `Any`.
# pyre-fixme[2]: Parameter annotation cannot be `Any`.
def numpy_type_to_python_type(value: Any) -> Any:
    """If `value` is a Numpy int or float, coerce to a Python int or float.
    This is necessary because some of our transforms return Numpy values.
    """
    if isinstance(value, np.integer):
        value = int(value)  # pragma: nocover (covered by generator tests)
    if isinstance(value, np.floating):
        value = float(value)  # pragma: nocover  (covered by generator tests)
    return value
