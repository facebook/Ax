# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack, contextmanager
from functools import wraps
from typing import Callable, Generator
from unittest import mock

from botorch.optim.initializers import (
    gen_one_shot_kg_initial_conditions,
    gen_batch_initial_conditions,
)
from gpytorch import settings as gpt_settings
from scipy.optimize import minimize


@contextmanager
def fast_botorch_context_manager() -> Generator[None, None, None]:
    """A context manager to force botorch to speed up optimization. Currently, the
    primary tactic is to force the underlying scipy methods to stop after just one
    iteration.
    """

    def one_iteration_minimize(*args, **kwargs):
        if kwargs["options"] is None:
            kwargs["options"] = {}

        kwargs["options"]["maxiter"] = 1

        return minimize(*args, **kwargs)

    def minimal_gen_ics(*args, **kwargs):
        kwargs["num_restarts"] = 2
        kwargs["raw_samples"] = 4

        return gen_batch_initial_conditions(*args, **kwargs)

    def minimal_gen_os_ics(*args, **kwargs):
        kwargs["num_restarts"] = 2
        kwargs["raw_samples"] = 4

        return gen_one_shot_kg_initial_conditions(*args, **kwargs)

    with ExitStack() as es:
        es.enter_context(
            mock.patch(
                "botorch.generation.gen.minimize",
                wraps=one_iteration_minimize,
            )
        )

        es.enter_context(
            mock.patch(
                "botorch.optim.fit.minimize",
                wraps=one_iteration_minimize,
            )
        )

        es.enter_context(
            mock.patch(
                "botorch.optim.optimize.gen_batch_initial_conditions",
                wraps=minimal_gen_ics,
            )
        )

        es.enter_context(
            mock.patch(
                "botorch.optim.optimize.gen_one_shot_kg_initial_conditions",
                wraps=minimal_gen_os_ics,
            )
        )

        yield


def fast_botorch(f: Callable) -> Callable:
    """Wraps f in the fast_modelingmanager for use as a decorator."""

    @wraps(f)
    def inner(*args, **kwargs):
        with fast_botorch_context_manager():
            return f(*args, **kwargs)

    return inner


@contextmanager
def fast_gpytorch_context_manager() -> Generator[None, None, None]:
    """A context manager to force GPyTorch to use fast approximations to various
    mathematical functions used in GP inference.
    """
    with ExitStack() as es:
        es.enter_context(gpt_settings.fast_pred_var())
        es.enter_context(gpt_settings.fast_pred_samples())
        es.enter_context(gpt_settings.fast_computations())

        yield


def fast_gpytorch(f: Callable) -> Callable:
    """Wraps f in the fast_gpytorch_context_manager for use as a decorator."""

    @wraps(f)
    def inner(*args, **kwargs):
        with fast_gpytorch_context_manager():
            return f(*args, **kwargs)

    return inner


def fast_modeling(f: Callable) -> Callable:
    """Wraps f in various context managers to speed up modeling"""

    @wraps(f)
    @fast_botorch
    @fast_gpytorch
    def inner(*args, **kwargs):
        return f(*args, **kwargs)

    return inner
