#!/usr/bin/env python3

import warnings
from typing import Callable, Dict, Optional, Union

import torch
from ae.lazarus.ae.exceptions.model import ModelError
from botorch.acquisition.batch_modules import BatchAcquisitionFunction
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.gen import gen_candidates_scipy, get_best_candidates
from botorch.models import Model, MultiOutputGP
from botorch.models.constant_noise import ConstantNoiseGP
from botorch.models.gp_regression import HeteroskedasticSingleTaskGP, SingleTaskGP
from botorch.optim.initializers import (
    get_similarity_measure,
    initialize_q_batch,
    initialize_q_batch_simple,
)
from botorch.utils import draw_sobol_samples
from torch import Tensor
from torch.nn import Module


NOISELESS_MODELS = {SingleTaskGP}
MIN_INFERRED_NOISE_LEVEL = 1e-5
MIN_OBSERVED_NOISE_LEVEL = 1e-7


def is_noiseless(model: Model) -> bool:
    """Check if a given (single-task) botorch model is noiseless"""
    if isinstance(model, MultiOutputGP):  # pyre-ignore: [16]
        raise ModelError(
            "Checking for noisless models only applies to sub-models of MultiOutputGP"
        )
    return model.__class__ in NOISELESS_MODELS


def _get_model(X: Tensor, Y: Tensor, Yvar: Tensor) -> Model:
    """Instantiate a model of type depending on the input data"""
    Yvar = Yvar.view(-1)  # last dimension is not needed for botorch
    # Determine if we want to treat the noise as constant
    mean_var = Yvar.mean().clamp_min_(MIN_OBSERVED_NOISE_LEVEL)
    # Look at relative variance in noise level
    if Yvar.nelement() > 1:
        Yvar_std = Yvar.std()
    else:
        Yvar_std = torch.tensor(0).to(device=Yvar.device, dtype=Yvar.dtype)
    if Yvar_std / mean_var < 0.1:
        model = ConstantNoiseGP(
            train_X=X, train_Y=Y.view(-1), train_Y_se=mean_var.sqrt()
        )
    else:
        Yvar = Yvar.clamp_min(MIN_OBSERVED_NOISE_LEVEL)
        model = HeteroskedasticSingleTaskGP(
            train_X=X, train_Y=Y.view(-1), train_Y_se=Yvar.view(-1).sqrt()
        )
    model.to(dtype=X.dtype, device=X.device)  # pyre-ignore: [28]
    return model


def _sequential_optimize(
    acq_function: BatchAcquisitionFunction,
    bounds: Tensor,
    n: int,
    num_restarts: int,
    raw_samples: int,
    model: Model,
    options: Dict[str, Union[bool, float, int]],
    fixed_features: Optional[Dict[int, float]] = None,
    rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """
    Returns a set of candidates via sequential multi-start optimization.

    Args:
        acq_function:  An acquisition function Module
        bounds: A `2 x d` tensor of lower and upper bounds for each column of X.
        n: The number of candidates
        num_restarts:  Number of starting points for multistart acquisition
            function optimization.
        raw_samples: number of samples for initialization
        model: the model
        options: options for candidate generation
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value during generation.
        rounding_func: A function that rounds an optimization result
            appropriately (i.e., according to `round-trip` transformations).
    Returns:
        The set of generated candidates
    """
    candidate_list = []
    candidates = torch.tensor([])
    base_X_pending = acq_function.X_pending
    # Needed to clear base_samples
    acq_function._set_X_pending(base_X_pending)
    for _ in range(n):
        candidate = _joint_optimize(
            acq_function=acq_function,
            bounds=bounds,
            n=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            model=model,
            options=options,
            fixed_features=fixed_features,
        )
        if rounding_func is not None:
            candidate_shape = candidate.shape
            candidate = rounding_func(candidate.view(-1)).view(*candidate_shape)
        candidate_list.append(candidate)
        candidates = torch.cat(candidate_list, dim=-2)
        acq_function._set_X_pending(
            torch.cat([base_X_pending, candidates], dim=-2)
            if base_X_pending is not None
            else candidates
        )
    # Reset acq_func to previous X_pending state
    acq_function._set_X_pending(base_X_pending)
    return candidates


def _joint_optimize(
    acq_function: Module,
    bounds: Tensor,
    n: int,
    num_restarts: int,
    raw_samples: int,
    model: Model,
    options: Dict[str, Union[bool, float, int]],
    fixed_features: Optional[Dict[int, float]] = None,
    rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """
    Returns a set of candidates via joint multi-start optimization.

    Args:
        acq_function:  An acquisition function Module
        bounds: A `2 x d` tensor of lower and upper bounds for each column of X.
        n: The number of candidates
        num_restarts:  Number of starting points for multistart acquisition
            function optimization.
        raw_samples: number of samples for initialization
        model: the model
        options: options for candidate generation
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value during generation.
        rounding_func: A function that rounds an optimization result
            appropriately (i.e., according to `round-trip` transformations).
            Note: rounding_func is not used by _joint_optimize and is only included
            to match _sequential_optimize.

    Returns:
        The set of generated candidates
    """
    batch_initial_arms = _gen_batch_initial_arms(
        acq_function=acq_function,
        bounds=bounds,
        n=n,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        model=model,
        options=options,
    )
    # optimize using random restart optimization
    batch_candidates, batch_acq_values = gen_candidates_scipy(
        initial_candidates=batch_initial_arms,
        acquisition_function=acq_function,
        lower_bounds=bounds[0],
        upper_bounds=bounds[1],
        options=None,
        fixed_features=fixed_features,
    )
    return get_best_candidates(
        batch_candidates=batch_candidates, batch_values=batch_acq_values
    )


def _gen_batch_initial_arms(
    acq_function: Module,
    bounds: Tensor,
    n: int,
    num_restarts: int,
    raw_samples: int,
    model: Model,
    options: Dict[str, Union[bool, float, int]],
) -> Tensor:
    seed: Optional[int] = options.get("seed")  # pyre-ignore
    batch_initial_arms: Tensor
    factor, max_factor = 1, 5
    while factor < max_factor:
        with warnings.catch_warnings(record=True) as ws:
            X_rnd = draw_sobol_samples(
                bounds=bounds, n=raw_samples * factor, q=n, seed=seed
            )
            with torch.no_grad():
                Y_rnd = acq_function(X_rnd)
            if options.get("simple_init", True):
                batch_initial_arms = initialize_q_batch_simple(
                    X=X_rnd, Y=Y_rnd, n=num_restarts, options=options
                )
            else:
                sim_measure = get_similarity_measure(model=model)
                batch_initial_arms = initialize_q_batch(
                    X=X_rnd,
                    Y=Y_rnd,
                    n=num_restarts,
                    sim_measure=sim_measure,
                    options=options,
                )

            if not any(
                issubclass(w.category, BadInitialCandidatesWarning)  # pyre-ignore: [16]
                for w in ws  # pyre-ignore: [16]
            ):
                return batch_initial_arms
            if factor < max_factor:
                factor += 1
    warnings.warn(
        "Unable to find non-zero acquistion function values - initial arms"
        "are being selected randomly.",
        BadInitialCandidatesWarning,  # pyre-ignore: [16]
    )
    return batch_initial_arms
