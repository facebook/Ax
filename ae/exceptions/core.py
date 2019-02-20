#!/usr/bin/env python3


class AEError(Exception):
    """Base AE exception.

    All exceptions derived from AEError need to define a custom error message.
    Additionally, exceptions can define a hint property that provides additional
    guidance as to how to remedy the error.

    """

    def __init__(self, message: str, hint: str = "") -> None:
        self.message: str = message
        self.hint: str = hint

    def __str__(self) -> str:
        return " ".join([self.message, getattr(self, "hint", "")])


class UnsupportedError(AEError):
    """Raised when an unsupported request is made.

    UnsupportedError may seem similar to NotImplementedError (NIE).
    It differs in the following ways:
      1. UnsupportedError is not used for abstract methods, which
        is the official NIE use case.
      2. UnsupportedError indicates an intentional and permanent lack of support.
        It should not be used for TODO (another common use case of NIE).
    """

    pass


class ExperimentNotReadyError(AEError):
    """Raised when failing to query data due to immature experiment.

    Useful to distinguish data failure reasons in automated analyses.
    """

    pass


class NoDataError(AEError):
    """Raised when no data is found for experiment in underlying data store.

    Useful to distinguish data failure reasons in automated analyses.
    """

    pass
