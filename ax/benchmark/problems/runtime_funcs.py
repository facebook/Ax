# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Mapping

from ax.core.arm import Arm
from ax.core.types import TParamValue


def int_from_params(
    params: Mapping[str, TParamValue], n_possibilities: int = 10
) -> int:
    """
    Get an int between 0 and n_possibilities - 1, using a hash of the parameters.
    """
    arm_hash = Arm.md5hash(parameters=params)
    return int(arm_hash[-1], base=16) % n_possibilities
