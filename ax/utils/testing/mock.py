# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager, ExitStack
from functools import wraps
from typing import Any, Callable, Dict, Generator, Optional
from unittest import mock

from ax.models.torch.fully_bayesian import run_inference
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.generation.gen import minimize_with_timeout
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
)
from scipy.optimize.optimize import OptimizeResult
from torch import Tensor


@contextmanager
def fast_botorch_optimize_context_manager(
    force: bool = False,
) -> Generator[None, None, None]:
    """A context manager to force botorch to speed up optimization. Currently, the
    primary tactic is to force the underlying scipy methods to stop after just one
    iteration.

        force: If True will not raise an AssertionError if no mocks are called.
            USE RESPONSIBLY.
    """

    def one_iteration_minimize(*args: Any, **kwargs: Any) -> OptimizeResult:
        if kwargs["options"] is None:
            kwargs["options"] = {}

        kwargs["options"]["maxiter"] = 1
        return minimize_with_timeout(*args, **kwargs)

    def minimal_gen_ics(*args: Any, **kwargs: Any) -> Tensor:
        kwargs["num_restarts"] = 2
        kwargs["raw_samples"] = 4

        return gen_batch_initial_conditions(*args, **kwargs)

    def minimal_gen_os_ics(*args: Any, **kwargs: Any) -> Optional[Tensor]:
        kwargs["num_restarts"] = 2
        kwargs["raw_samples"] = 4

        return gen_one_shot_kg_initial_conditions(*args, **kwargs)

    def minimal_run_inference(*args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        return run_inference(*args, **_get_minimal_mcmc_kwargs(**kwargs))

    def minimal_fit_fully_bayesian(*args: Any, **kwargs: Any) -> None:
        fit_fully_bayesian_model_nuts(*args, **_get_minimal_mcmc_kwargs(**kwargs))

    with ExitStack() as es:
        mock_generation = es.enter_context(
            mock.patch(
                "botorch.generation.gen.minimize_with_timeout",
                wraps=one_iteration_minimize,
            )
        )

        mock_fit = es.enter_context(
            mock.patch(
                "botorch.optim.core.minimize_with_timeout",
                wraps=one_iteration_minimize,
            )
        )

        mock_gen_ics = es.enter_context(
            mock.patch(
                "botorch.optim.optimize.gen_batch_initial_conditions",
                wraps=minimal_gen_ics,
            )
        )

        mock_gen_os_ics = es.enter_context(
            mock.patch(
                "botorch.optim.optimize.gen_one_shot_kg_initial_conditions",
                wraps=minimal_gen_os_ics,
            )
        )

        mock_mcmc_legacy = es.enter_context(
            mock.patch(
                "ax.models.torch.fully_bayesian.run_inference",
                wraps=minimal_run_inference,
            )
        )

        mock_mcmc_mbm = es.enter_context(
            mock.patch(
                "ax.models.torch.botorch_modular.utils.fit_fully_bayesian_model_nuts",
                wraps=minimal_fit_fully_bayesian,
            )
        )

        yield

    if (not force) and all(
        mock_.call_count < 1
        for mock_ in [
            mock_generation,
            mock_fit,
            mock_gen_ics,
            mock_gen_os_ics,
            mock_mcmc_legacy,
            mock_mcmc_mbm,
        ]
    ):
        raise AssertionError(
            "No mocks were called in the context manager. Please remove unused "
            "fast_botorch_optimize_context_manager()."
        )


# pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
def fast_botorch_optimize(f: Callable) -> Callable:
    """Wraps f in the fast_botorch_optimize_context_manager for use as a decorator."""

    @wraps(f)
    # pyre-fixme[3]: Return type must be annotated.
    def inner(*args: Any, **kwargs: Any):
        with fast_botorch_optimize_context_manager():
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


# pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
def skip_fit_gpytorch_mll(f: Callable) -> Callable:
    """Wraps f in the skip_fit_gpytorch_mll_context_manager for use as a decorator."""

    @wraps(f)
    # pyre-fixme[3]: Return type must be annotated.
    def inner(*args: Any, **kwargs: Any):
        with skip_fit_gpytorch_mll_context_manager():
            return f(*args, **kwargs)

    return inner


def _get_minimal_mcmc_kwargs(**kwargs: Any) -> Dict[str, Any]:
    kwargs["warmup_steps"] = 0
    # Just get as many samples as otherwise expected.
    kwargs["num_samples"] = kwargs.get("num_samples", 256) // kwargs.get("thinning", 16)
    kwargs["thinning"] = 1
    return kwargs
