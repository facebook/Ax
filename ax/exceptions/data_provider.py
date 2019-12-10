#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any


class DataProviderError(Exception):
    """Base Exception for Ax DataProviders.

    The type of the data provider must be included.
    The raw error is stored in the data_provider_error section,
    and an Ax-friendly message is stored as the actual error message.
    """

    def __init__(
        self, message: str, data_provider: str, data_provider_error: Any
    ) -> None:
        self.message = message
        self.data_provider = data_provider
        self.data_provider_error = data_provider_error

    def __str__(self) -> str:
        return (
            "{message}. \n Error thrown by: {dp} data provider \n"
            + "Native {dp} data provider error: {dp_error}"
        ).format(
            dp=self.data_provider,
            message=self.message,
            dp_error=self.data_provider_error,
        )
