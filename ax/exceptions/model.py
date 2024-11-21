#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from ax.exceptions.core import AxError


class ModelError(AxError):
    """Raised when an error occurs during modeling."""

    pass


class CVNotSupportedError(AxError):
    """Raised when cross validation is applied to a model which doesn't
    support it.
    """

    pass


class ModelBridgeMethodNotImplementedError(AxError, NotImplementedError):
    """Raised when a ``ModelBridge`` method is not implemented by subclasses.

    NOTE: ``ModelBridge`` may catch and silently discard this error.
    """

    pass
