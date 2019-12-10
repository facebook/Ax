#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from ax.modelbridge.array import ArrayModelBridge
from ax.models.numpy_base import NumpyModel


# pyre-fixme[13]: Attribute `model` is never initialized.
# pyre-fixme[13]: Attribute `outcomes` is never initialized.
# pyre-fixme[13]: Attribute `parameters` is never initialized.
class NumpyModelBridge(ArrayModelBridge):
    """A model bridge for using numpy array-based models.

    This model bridge interfaces with NumpyModel.

    Requires that all parameters have been transformed to RangeParameters
    or FixedParameters with float type and no log scale.
    """

    model: NumpyModel
    outcomes: List[str]
    parameters: List[str]
