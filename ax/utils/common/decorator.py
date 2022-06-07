# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class ClassDecorator(ABC):
    """
    Template for making a decorator work as a class level decorator.  That decorator
    should extend `ClassDecorator`.  It must implement `__init__` and
    `decorate_callable`.  See `disable_logger.decorate_callable` for an example.
    `decorate_callable` should call `self._call_func()` instead of directly calling
    `func` to handle static functions.
    Note: `_call_func` is still imperfect and unit tests should be used to ensure
    everything is working properly.  There is a lot of complexity in detecting
    classmethods and staticmethods and removing the self argument in the right
    situations. For best results always use keyword args in the decorated class.

    `DECORATE_PRIVATE` can be set to determine whether private methods should be
    decorated. In the case of a logging decorator, you may only want to decorate things
    the user calls. But in the case of a disable logging decorator, you may want to
    decorate everything to ensure no logs escape.
    """

    DECORATE_PRIVATE = True

    def decorate_class(self, klass: T) -> T:
        for attr in dir(klass):
            if not self.DECORATE_PRIVATE and attr[0] == "_":
                continue

            attr_value = getattr(klass, attr)
            if (
                not callable(attr_value)
                or isinstance(attr_value, type)
                or attr
                in (
                    "__subclasshook__",
                    "__class__",
                    "__repr__",
                    "__str__",
                    "__getattribute__",
                    "__new__",
                    "__call__",
                    "__eq__",
                    "_call_func",
                )
            ):
                continue

            setattr(klass, attr, self.decorate_callable(attr_value))
        return klass

    @abstractmethod
    def decorate_callable(self, func: Callable[..., T]) -> Callable[..., T]:
        pass

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        if isinstance(func, type):
            return self.decorate_class(func)
        return self.decorate_callable(func)

    def _call_func(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except TypeError as e:
            # static functions
            try:
                return func(*args[1:], **kwargs)
            except TypeError:
                # it wasn't that it was a static function
                raise e
