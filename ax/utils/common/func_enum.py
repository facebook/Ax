# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from collections.abc import Callable
from enum import Enum, unique
from importlib import import_module
from typing import Any

from ax.exceptions.core import UnsupportedError


@unique
class FuncEnum(Enum):
    """A base class for all enums with the following structure: string values that
    map to names of functions, which reside in the same module as the enum."""

    def __call__(self, **kwargs: Any) -> Any:
        """Defines a method, by which the members of this enum can be called,
        e.g. ``MyFunctions.F(**kwargs)``, which will call the corresponding
        function registered by the name ``F`` in the enum."""
        return self._get_function_for_value()(**kwargs)

    def _get_function_for_value(self) -> Callable[..., Any]:
        """Retrieve the function in this module, name of which corresponds to the
        value of the enum member."""
        try:
            return getattr(import_module(self.__module__), self.value)
        except AttributeError:
            raise UnsupportedError(
                f"{self.value} is not defined as a method in "
                f"`{self.__module__}`. Please add the method "
                "to the file."
            )
