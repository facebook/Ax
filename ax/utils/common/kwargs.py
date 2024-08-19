#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Iterable
from inspect import Parameter, signature

from logging import Logger
from typing import Any, Callable, Optional

from ax.utils.common.logger import get_logger

logger: Logger = get_logger(__name__)

TKwargs = dict[str, Any]


def consolidate_kwargs(
    kwargs_iterable: Iterable[Optional[dict[str, Any]]], keywords: Iterable[str]
) -> dict[str, Any]:
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
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    function: Callable,
    omit: Optional[list[str]] = None,
) -> list[str]:
    """Extract parameter names from function signature."""
    omit = omit or []
    return [p for p in signature(function).parameters.keys() if p not in omit]


# pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
def get_function_default_arguments(function: Callable) -> dict[str, Any]:
    """Extract default arguments from function signature."""
    params = signature(function).parameters
    return {
        kw: p.default for kw, p in params.items() if p.default is not Parameter.empty
    }


# pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
def warn_on_kwargs(callable_with_kwargs: Callable, **kwargs: Any) -> None:
    """Log a warning when a decoder function receives unexpected kwargs.

    NOTE: This mainly caters to the use case where an older version of Ax is
    used to decode objects, serialized to JSON by a newer version of Ax (and
    therefore potentially containing new fields). In that case, the decoding
    function should not fail when encountering those additional fields, but
    rather just ignore them and log a warning using this function.
    """
    if kwargs:
        logger.warning(
            "Found unexpected kwargs: %s while calling %s "
            "from JSON. These kwargs will be ignored.",
            kwargs,
            callable_with_kwargs,
        )


# pyre-fixme[3]: Return annotation cannot be `Any`.
# pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
def filter_kwargs(function: Callable, **kwargs: Any) -> Any:
    """Filter out kwargs that are not applicable for a given function.
    Return a copy of given kwargs dict with only the required kwargs."""
    return {k: v for k, v in kwargs.items() if k in signature(function).parameters}
