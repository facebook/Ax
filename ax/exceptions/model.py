#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ax.exceptions.core import AxError


class ModelError(AxError):
    """Raised when an error occurs during modeling."""

    pass


class CVNotSupportedError(AxError):
    """Raised when cross validation is applied to a model which doesn't
    support it.
    """

    pass
