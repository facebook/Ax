#!/usr/bin/env python3

from typing import Any

from baldr.ttypes import Error as BaldrAPIError


class DataProviderError(Exception):
    """Base Exception for AE DataProviders.

    The type of the driver must be included.
    The raw error is stored in the data_provider_error section,
    and an AE-friendly message is stored as the actual error message.
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


class BaldrError(DataProviderError):
    "Raised when a error is thrown from the Baldr API."

    def __init__(self, message: str, data_provider_error: BaldrAPIError) -> None:
        super().__init__(
            message=message,
            data_provider="BALDR",
            data_provider_error=data_provider_error,
        )
