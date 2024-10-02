#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings


def _validate_force_random_search(
    no_bayesian_optimization: bool | None = None,
    force_random_search: bool = False,
    exception_cls: type[Exception] = ValueError,
) -> None:
    """Helper function to validate interaction between `force_random_search`
    and `no_bayesian_optimization` (supported until deprecation in [T199632397])
    """
    if no_bayesian_optimization is not None:
        # users are effectively permitted to continue using
        # `no_bayesian_optimization` so long as it doesn't
        # conflict with `force_random_search`
        if no_bayesian_optimization != force_random_search:
            raise exception_cls(
                "Conflicting values for `force_random_search` "
                "and `no_bayesian_optimization`! "
                "Please only specify `force_random_search`."
            )
        warnings.warn(
            "`no_bayesian_optimization` is deprecated. Please use "
            "`force_random_search` in the future.",
            DeprecationWarning,
            stacklevel=2,
        )
