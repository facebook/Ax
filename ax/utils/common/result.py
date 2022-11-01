# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty

from typing import Any, Callable, cast, Generic, NoReturn, Optional, TypeVar, Union


T = TypeVar("T", covariant=True)
E = TypeVar("E", covariant=True)

U = TypeVar("U")
F = TypeVar("F")


class Result(Generic[T, E], ABC):
    """
    A minimal implementation of a rusty Result monad.
    See https://doc.rust-lang.org/std/result/enum.Result.html for more information.
    """

    @abstractmethod
    def is_ok(self) -> bool:
        pass

    @abstractmethod
    def is_err(self) -> bool:
        pass

    @abstractproperty
    def ok(self) -> Optional[T]:
        pass

    @abstractproperty
    def err(self) -> Optional[E]:
        pass

    @abstractproperty
    def value(self) -> Union[T, E]:
        pass

    @abstractmethod
    def map(self, op: Callable[[T], U]) -> Result[U, E]:
        """
        Maps a Result[T, E] to Result[U, E] by applying a function to a contained Ok
        value, leaving an Err value untouched. This function can be used to compose
        the results of two functions.
        """

        pass

    @abstractmethod
    def map_err(self, op: Callable[[E], F]) -> Result[T, F]:
        """
        Maps a Result[T, E] to Result[T, F] by applying a function to a contained Err
        value, leaving an Ok value untouched. This function can be used to pass
        through a successful result while handling an error.
        """

        pass

    @abstractmethod
    def map_or(self, default: U, op: Callable[[T], U]) -> U:
        """
        Returns the provided default (if Err), or applies a function to the contained
        value (if Ok).
        """

        pass

    @abstractmethod
    def map_or_else(self, default_op: Callable[[], U], op: Callable[[T], U]) -> U:
        """
        Maps a Result[T, E] to U by applying fallback function default to a contained
        Err value, or function op to a contained Ok value. This function can be used
        to unpack a successful result while handling an error.
        """

        pass

    @abstractmethod
    def unwrap(self) -> T:
        """
        Returns the contained Ok value.

        Because this function may raise an UnwrapError, its use is generally
        discouraged. Instead, prefer to handle the Err case explicitly, or call
        unwrap_or, unwrap_or_else, or unwrap_or_default.
        """

        pass

    @abstractmethod
    def unwrap_err(self) -> E:
        """
        Returns the contained Err value.

        Because this function may raise an UnwrapError, its use is generally
        discouraged. Instead, prefer to handle the Err case explicitly, or call
        unwrap_or, unwrap_or_else, or unwrap_or_default.
        """

        pass

    @abstractmethod
    # pyre-ignore[46]: The type variable `Variable[T](covariant)` is covariant and
    # cannot be a parameter type.
    def unwrap_or(self, default: T) -> T:
        """Returns the contained Ok value or a provided default."""

        pass

    @abstractmethod
    def unwrap_or_else(self, op: Callable[[E], T]) -> T:
        """Returns the contained Ok value or computes it from a Callable."""

        pass


class Ok(Generic[T, E], Result[T, E]):
    """
    Contains the success value.
    """

    _value: T

    def __init__(self, value: T) -> None:
        self._value = value

    def __repr__(self) -> str:
        return f"Ok({self._value})"

    # pyre-ignore[2]: Parameter `other` must have a type other than `Any`.
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Ok) and self._value == other._value

    def __hash__(self) -> int:
        return hash((True, self._value))

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
        return False

    @property
    def ok(self) -> T:
        return self._value

    @property
    def err(self) -> None:
        return None

    @property
    def value(self) -> T:
        return self._value

    def map(self, op: Callable[[T], U]) -> Result[U, E]:
        return Ok(op(self._value))

    def map_err(self, op: Callable[[E], F]) -> Result[T, F]:
        return cast(Result[T, F], self)

    def map_or(self, default: U, op: Callable[[T], U]) -> U:
        return op(self._value)

    def map_or_else(self, default_op: Callable[[], U], op: Callable[[T], U]) -> U:
        return op(self._value)

    def unwrap(self) -> T:
        return self._value

    def unwrap_err(self) -> NoReturn:
        raise UnwrapError(f"Tried to unwrap_err {self}.")

    def unwrap_or(self, default: U) -> T:
        return self._value

    def unwrap_or_else(self, op: Callable[[E], T]) -> T:
        return self._value


class Err(Generic[T, E], Result[T, E]):
    """
    Contains the error value.
    """

    def __init__(self, value: E) -> None:
        self._value = value

    def __repr__(self) -> str:
        return f"Err({self._value})"

    # pyre-ignore[2]: Parameter `other` must have a type other than `Any`.
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Err) and self._value == other._value

    def __hash__(self) -> int:
        return hash((False, self._value))

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    @property
    def ok(self) -> None:
        return None

    @property
    def err(self) -> E:
        return self._value

    @property
    def value(self) -> E:
        return self._value

    def map(self, op: Callable[[T], U]) -> Result[U, E]:
        return cast(Result[U, E], self)

    def map_err(self, op: Callable[[E], F]) -> Result[T, F]:
        return Err(op(self._value))

    def map_or(self, default: U, op: Callable[[T], U]) -> U:
        return default

    def map_or_else(self, default_op: Callable[[], U], op: Callable[[T], U]) -> U:
        return default_op()

    def unwrap(self) -> NoReturn:
        raise UnwrapError(f"Tried to unwrap {self}.")

    def unwrap_err(self) -> E:
        return self._value

    # pyre-ignore[46]: The type variable `Variable[T](covariant)` is covariant and
    # cannot be a parameter type.
    def unwrap_or(self, default: T) -> T:
        return default

    def unwrap_or_else(self, op: Callable[[E], T]) -> T:
        return op(self._value)


class UnwrapError(Exception):
    """
    Exception that indicates something has gone wrong in an unwrap call.

    This should not happen in real world use and indicates a user has impropperly
    or unsafely used the Result abstraction.
    """

    pass
