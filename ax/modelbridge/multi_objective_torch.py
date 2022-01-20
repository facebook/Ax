#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.modelbridge.torch import TorchModelBridge
from ax.utils.common.logger import get_logger


logger = get_logger("MultiObjectiveTorchModelBridge")


# TODO(ehotaj): Remove references to MultiObjectiveTorchModelBridge used throughout the
# code.
class MultiObjectiveTorchModelBridge(TorchModelBridge):
    """A model bridge for using multi-objective torch-based models.

    Specifies an interface that is implemented by MultiObjectiveTorchModel. In
    particular, model should have methods fit, predict, and gen. See
    MultiObjectiveTorchModel for the API for each of these methods.

    Requires that all parameters have been transformed to RangeParameters
    or FixedParameters with float type and no log scale.

    This class converts Ax parameter types to torch tensors before passing
    them to the model.
    """
