#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from inspect import Parameter, signature
from typing import Any, Callable, Dict, Iterable, List, Optional


def consolidate_kwargs(
    kwargs_iterable: Iterable[Optional[Dict[str, Any]]], keywords: Iterable[str]
) -> Dict[str, Any]:
    """Combine an iterable of kwargs into a single dict of kwargs, where kwargs
    by duplicate keys that appear later in the iterable get priority over the
    ones that appear earlier and only kwargs referenced in keywords will be
    used. This allows to combine somewhat redundant sets of kwargs, where a
    user-set kwarg, for instance, needs to override a default kwarg.

    >>> consolidate_kwargs(
    ...     kwargs_iterable=[{'a': 1, 'b': 2}, {'b': 3, 'c': 4, 'd': 5}],
    ...     keywords=['a', 'b', 'd']
    ... )
    {'a': 1, 'b': 3, 'd': 5}
    """
    all_kwargs = {}
    for kwargs in kwargs_iterable:
        if kwargs is not None:
            all_kwargs.update({kw: p for kw, p in kwargs.items() if kw in keywords})
    return all_kwargs


def get_function_argument_names(
    function: Callable, omit: Optional[List[str]] = None
) -> List[str]:
    """Extract parameter names from function signature."""
    omit = omit or []
    return [p for p in signature(function).parameters.keys() if p not in omit]


def get_function_default_arguments(function: Callable) -> Dict[str, Any]:
    """Extract default arguments from function signature."""
    params = signature(function).parameters
    return {
        kw: p.default for kw, p in params.items() if p.default is not Parameter.empty
    }


def validate_kwarg_typing(typed_callables: List[Callable], **kwargs: Any) -> None:
    """Raises a value error if some of the keyword argument types do not match
    the signatures of the specified typed callables.

    Note: this function expects the typed callables to have unique keywords for
    the arguments and will raise an error if repeat keywords are found.
    """
    # TODO[Lena]: T46467254
    pass
