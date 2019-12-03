#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from inspect import Parameter, Signature, signature
from typing import Any, Callable, Dict, Iterable, List, Optional
from unittest.mock import Mock


def _signature(callable: Any) -> Signature:
    """Utility that wraps `inspect.signature` and makes it return the signature
    of the mocked object when a signature of the mock is requested. (Othewise
    inspecting signatures of mocks yields just `(*args, **kwargs)` as parameters).
    """
    if isinstance(callable, Mock):
        # Mocks that have a spec set, use the spec object class as `__class__`.
        # Ones that do not have a spec have their `__class__` as just `MagicMock`.
        if issubclass(callable.__class__, Mock):
            raise ValueError(f"Cannot get signature of unspecced mock: {callable}.")
        return signature(callable.__class__)
    return signature(callable)


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
    return [p for p in _signature(function).parameters.keys() if p not in omit]


def get_function_default_arguments(function: Callable) -> Dict[str, Any]:
    """Extract default arguments from function signature."""
    params = signature(function).parameters
    return {
        kw: p.default for kw, p in params.items() if p.default is not Parameter.empty
    }


def filter_kwargs(function: Callable, **kwargs: Any) -> Any:
    """Filter out kwargs that are not applicable for a given function.
    Return a copy of given kwargs dict with only the required kwargs."""
    return {k: v for k, v in kwargs.items() if k in _signature(function).parameters}


def validate_kwarg_typing(typed_callables: List[Callable], **kwargs: Any) -> None:
    """Raises a value error if some of the keyword argument types do not match
    the signatures of the specified typed callables.

    Note: this function expects the typed callables to have unique keywords for
    the arguments and will raise an error if repeat keywords are found.
    """
    # TODO[Lena]: T46467254
    pass
