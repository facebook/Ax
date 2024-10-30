# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Callable, Generator
from contextlib import contextmanager, ExitStack
from functools import wraps
from typing import Any
from unittest import mock

from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.optim.optimize_mixed import optimize_acqf_mixed_alternating
from botorch.test_utils.mock import mock_optimize_context_manager
from torch import Tensor


@contextmanager
def mock_botorch_optimize_context_manager(
    force: bool = False,
) -> Generator[None, None, None]:
    """A context manager that uses mocks to speed up optimization for testing.
    Currently, the primary tactic is to force the underlying scipy methods to
    stop after just one iteration.

    This context manager uses BoTorch's `mock_optimize_context_manager`, and
    adds some additional mocks that are not possible to cover in BoTorch due to
    the need to mock the functions where they are used.

    Args:
        force: If True will not raise an AssertionError if no mocks are called.
            USE RESPONSIBLY.
    """

    def minimal_fit_fully_bayesian(*args: Any, **kwargs: Any) -> None:
        fit_fully_bayesian_model_nuts(*args, **_get_minimal_mcmc_kwargs(**kwargs))

    def minimal_mixed_optimizer(*args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
        # BoTorch's `mock_optimize_context_manager` also has some mocks for this,
        # but the full set of mocks applied here cannot be covered by that.
        kwargs["raw_samples"] = 2
        kwargs["num_restarts"] = 1
        kwargs["options"].update(
            {
                "maxiter_alternating": 1,
                "maxiter_continuous": 1,
                "maxiter_init": 1,
                "maxiter_discrete": 1,
            }
        )
        return optimize_acqf_mixed_alternating(*args, **kwargs)

    with ExitStack() as es:
        mock_mcmc_mbm = es.enter_context(
            mock.patch(
                "ax.models.torch.botorch_modular.utils.fit_fully_bayesian_model_nuts",
                wraps=minimal_fit_fully_bayesian,
            )
        )

        mock_mixed_optimizer = es.enter_context(
            mock.patch(
                "ax.models.torch.botorch_modular.acquisition."
                "optimize_acqf_mixed_alternating",
                wraps=minimal_mixed_optimizer,
            )
        )

        es.enter_context(mock_optimize_context_manager())

        yield

        # Only raise if none of the BoTorch or Ax side mocks were called.
        # We do this by catching the error that could be raised by the BoTorch
        # context manager, and combining it with the signals from Ax side mocks.
        try:
            es.close()
        except AssertionError as e:
            # Check if the error is due to no BoTorch mocks being called.
            if "No mocks were called" in str(e):
                botorch_mocks_called = False
            else:
                raise
        else:
            botorch_mocks_called = True

    if (
        not force
        and all(
            mock_.call_count < 1
            for mock_ in [
                mock_mcmc_mbm,
                mock_mixed_optimizer,
            ]
        )
        and botorch_mocks_called is False
    ):
        raise AssertionError(
            "No mocks were called in the context manager. Please remove unused "
            "mock_botorch_optimize_context_manager()."
        )


def mock_botorch_optimize(f: Callable) -> Callable:
    """Wraps `f` in `mock_botorch_optimize_context_manager` for use as a decorator."""

    @wraps(f)
    # pyre-fixme[3]: Return type must be annotated.
    def inner(*args: Any, **kwargs: Any):
        with mock_botorch_optimize_context_manager():
            return f(*args, **kwargs)

    return inner


@contextmanager
def skip_fit_gpytorch_mll_context_manager() -> Generator[None, None, None]:
    """A context manager that makes `fit_gpytorch_mll` a no-op.

    This should only be used to speed up slow tests.
    """
    with mock.patch(
        "botorch.fit.FitGPyTorchMLL", side_effect=lambda *args, **kwargs: args[0]
    ) as mock_fit:
        yield
    if mock_fit.call_count < 1:
        raise AssertionError(
            "No mocks were called in the context manager. Please remove unused "
            "skip_fit_gpytorch_mll_context_manager()."
        )


def skip_fit_gpytorch_mll(f: Callable) -> Callable:
    """Wraps f in the skip_fit_gpytorch_mll_context_manager for use as a decorator."""

    @wraps(f)
    # pyre-fixme[3]: Return type must be annotated.
    def inner(*args: Any, **kwargs: Any):
        with skip_fit_gpytorch_mll_context_manager():
            return f(*args, **kwargs)

    return inner


def _get_minimal_mcmc_kwargs(**kwargs: Any) -> dict[str, Any]:
    kwargs["warmup_steps"] = 0
    # Just get as many samples as otherwise expected.
    kwargs["num_samples"] = kwargs.get("num_samples", 256) // kwargs.get("thinning", 16)
    kwargs["thinning"] = 1
    return kwargs
