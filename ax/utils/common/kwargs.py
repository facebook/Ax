#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from inspect import Parameter, signature
from typing import Any, Callable, Dict, Iterable, List, Optional

from ax.utils.common.logger import get_logger
from typeguard import check_type


logger = get_logger(__name__)

TKwargs = Dict[str, Any]


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
    """Check if keywords in kwargs exist in any of the typed_callables and
    if the type of each keyword value matches the type of corresponding arg in one of
    the callables

    Note: this function expects the typed callables to have unique keywords for
    the arguments and will raise an error if repeat keywords are found.
    """
    checked_kwargs = set()
    for typed_callable in typed_callables:
        params = signature(typed_callable).parameters
        for kw, param in params.items():
            if kw in kwargs:
                if kw in checked_kwargs:
                    logger.debug(
                        f"`{typed_callables}` have duplicate keyword argument: {kw}."
                    )
                else:
                    checked_kwargs.add(kw)
                    kw_val = kwargs.get(kw)
                    # if the keyword is a callable, we only do shallow checks
                    if not (callable(kw_val) and callable(param.annotation)):
                        try:
                            check_type(kw, kw_val, param.annotation)
                        except TypeError:
                            message = (
                                f"`{typed_callable}` expected argument `{kw}` to be of"
                                f" type {param.annotation}. Got {kw_val}"
                                f" (type: {type(kw_val)})."
                            )
                            logger.warning(message)

    # check if kwargs contains keywords not exist in any callables
    extra_keywords = [kw for kw in kwargs.keys() if kw not in checked_kwargs]
    if len(extra_keywords) != 0:
        raise ValueError(
            f"Arguments {extra_keywords} are not expected by any of {typed_callables}."
        )


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
