#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class AxError(Exception):
    """Base Ax exception.

    All exceptions derived from AxError need to define a custom error message.
    Additionally, exceptions can define a hint property that provides additional
    guidance as to how to remedy the error.

    """

    def __init__(self, message: str, hint: str = "") -> None:
        self.message: str = message
        self.hint: str = hint

    def __str__(self) -> str:
        return " ".join([self.message, getattr(self, "hint", "")]).rstrip()


class UnsupportedError(AxError):
    """Raised when an unsupported request is made.

    UnsupportedError may seem similar to NotImplementedError (NIE).
    It differs in the following ways:

    1. UnsupportedError is not used for abstract methods, which
        is the official NIE use case.
    2. UnsupportedError indicates an intentional and permanent lack of support.
        It should not be used for TODO (another common use case of NIE).
    """

    pass


class ExperimentNotReadyError(AxError):
    """Raised when failing to query data due to immature experiment.

    Useful to distinguish data failure reasons in automated analyses.
    """

    def __init__(
        self, message: str, hint: str = "", exposures_unavailable: bool = False
    ) -> None:
        super().__init__(message=message, hint=hint)
        self.exposures_unavailable = exposures_unavailable


class NoDataError(AxError):
    """Raised when no data is found for experiment in underlying data store.

    Useful to distinguish data failure reasons in automated analyses.
    """

    pass


class DataRequiredError(AxError):
    """Raised when more observed data is needed by the model to continue the
    optimization.

    Useful to distinguish when user needs to wait to request more trials until
    more data is available.
    """

    pass


class MisconfiguredExperiment(AxError):
    """Raised when experiment has incomplete or incorrect information."""

    pass


class AxWarning(Warning):
    """Base Ax warning.

    All warnings derived from AxWarning need to define a custom warning message.
    Additionally, warnings can define a hint property that provides additional
    guidance as to how to remedy the warning.

    """

    def __init__(self, message: str, hint: str = "") -> None:
        self.message: str = message
        self.hint: str = hint

    def __str__(self) -> str:
        return " ".join([self.message, getattr(self, "hint", "")]).rstrip()
