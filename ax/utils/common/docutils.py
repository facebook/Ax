#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Support functions for sphinx et. al
"""


from typing import TypeVar


_T = TypeVar("_T")


def copy_doc(src: _T) -> _T:
    """A decorator that copies the docstring of another object

    Since ``sphinx`` actually loads the python modules to grab the docstrings
    this works with both ``sphinx`` and the ``help`` function.

    .. code:: python

      class Cat(Mamal):

        @property
        @copy_doc(Mamal.is_feline)
        def is_feline(self) -> true:
            ...
    """
    # It would be tempting to try to get the doc through the class the method
    # is bound to (via __self__) but decorators are called before __self__ is
    # assigned.
    # One other solution would be to use a decorator on classes that would fill
    # all the missing docstrings but we want to be able to detect syntactically
    # when docstrings are copied to keep things nice and simple

    if src.__doc__ is None:
        # pyre-fixme[16]: `_T` has no attribute `__qualname__`.
        raise ValueError(f"{src.__qualname__} has no docstring to copy")

    def copy_doc(dst: _T) -> _T:
        if dst.__doc__ is not None:
            # pyre-fixme[16]: `_T` has no attribute `__qualname__`.
            raise ValueError(f"{dst.__qualname__} already has a docstring")
        dst.__doc__ = src.__doc__
        return dst

    # pyre-fixme[7]: Expected `_T` but got `Callable[[_T], _T]`.
    return copy_doc
