#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.modelbridge.tests.test_unit_x_transform import UnitXTransformTest
from ax.modelbridge.transforms.centered_unit_x import CenteredUnitX


class CenteredUnitXTransformTest(UnitXTransformTest):

    transform_class = CenteredUnitX
    # pyre-fixme[4]: Attribute must be annotated.
    expected_c_dicts = [{"x": -0.5, "y": 0.5}, {"x": -0.5, "a": 1.0}]
    expected_c_bounds = [0.0, 1.5]
