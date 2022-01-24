# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack, contextmanager
from functools import wraps
from typing import Callable, Generator
from unittest import mock

from scipy.optimize import minimize


@contextmanager
def fast_botorch_optimize_context_manager() -> Generator[None, None, None]:
    """A context manager to force botorch to speed up optimization. Currently, the
    primary tactic is to force the underlying scipy methods to stop after just one
    iteration.
    """

    def one_iteration_minimize(*args, **kwargs):
        if kwargs["options"] is None:
            kwargs["options"] = {}

        kwargs["options"]["maxiter"] = 1

        return minimize(*args, **kwargs)

    with ExitStack() as es:
        mock_generation = es.enter_context(
            mock.patch(
                "botorch.generation.gen.minimize",
                wraps=one_iteration_minimize,
            )
        )

        mock_fit = es.enter_context(
            mock.patch(
                "botorch.optim.fit.minimize",
                wraps=one_iteration_minimize,
            )
        )

        yield

    if mock_generation.call_count < 1 and mock_fit.call_count < 1:
        raise AssertionError(
            "No mocks were called in the context manager. Please remove unused "
            "fast_botorch_optimize_context_manager()."
        )


def fast_botorch_optimize(f: Callable) -> Callable:
    """Wraps f in the fast_botorch_optimize_context_manager for use as a decorator."""

    @wraps(f)
    def inner(*args, **kwargs):
        with fast_botorch_optimize_context_manager():
            return f(*args, **kwargs)

    return inner
