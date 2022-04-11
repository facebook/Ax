#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from ax.modelbridge.transforms.unit_x import UnitX


class CenteredUnitX(UnitX):
    """Map X to [-1, 1]^d for RangeParameter of type float and not log scale.

    Transform is done in-place.
    """

    target_lb: float = -1.0
    target_range: float = 2.0
