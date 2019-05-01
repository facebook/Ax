#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import List

from ax.modelbridge.array import ArrayModelBridge
from ax.models.numpy_base import NumpyModel


class NumpyModelBridge(ArrayModelBridge):
    """A model bridge for using numpy array-based models.

    This model bridge interfaces with NumpyModel.

    Requires that all parameters have been transformed to RangeParameters
    or FixedParameters with float type and no log scale.
    """

    model: NumpyModel
    outcomes: List[str]
    parameters: List[str]
