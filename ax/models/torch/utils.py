#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from logging import Logger
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from ax.exceptions.core import UnsupportedError
from ax.exceptions.model import ModelError
from ax.models.model_utils import filter_constraints_and_fixed_features, get_observed
from ax.models.random.sobol import SobolGenerator
from ax.models.types import TConfig
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.acquisition.multi_objective.multi_output_risk_measures import (
    MARS,
    MultiOutputRiskMeasureMCObjective,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    WeightedMCMultiOutputObjective,
)
from botorch.acquisition.objective import (
    ConstrainedMCObjective,
    IdentityMCObjective,
    LinearMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
    ScalarizedPosteriorTransform,
)
from botorch.acquisition.risk_measures import RiskMeasureMCObjective
from botorch.acquisition.utils import get_infeasible_cost
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.model import Model
from botorch.posteriors.fully_bayesian import FullyBayesianPosterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior_list import PosteriorList
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler
from botorch.utils.constraints import get_outcome_constraint_transforms
from botorch.utils.datasets import FixedNoiseDataset, SupervisedDataset
from botorch.utils.objective import get_objective_weights_transform
from botorch.utils.sampling import sample_hypersphere, sample_simplex
from torch import Tensor

logger: Logger = get_logger(__name__)


NOISELESS_MODELS = {SingleTaskGP}


# Distributions
SIMPLEX = "simplex"
HYPERSPHERE = "hypersphere"


@dataclass
class SubsetModelData:
    model: Model
    objective_weights: Tensor
    outcome_constraints: Optional[Tuple[Tensor, Tensor]]
    objective_thresholds: Optional[Tensor]
    indices: Tensor


def is_noiseless(model: Model) -> bool:
    """Check if a given (single-task) botorch model is noiseless"""
    if isinstance(model, ModelListGP):
        raise ModelError(
            "Checking for noisless models only applies to sub-models of ModelListGP"
        )
    return model.__class__ in NOISELESS_MODELS


def _filter_X_observed(
    Xs: List[Tensor],
    objective_weights: Tensor,
    bounds: List[Tuple[float, float]],
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
) -> Optional[Tensor]:
    r"""Filter input points to those appearing in objective or constraints.

    Args:
        Xs: The input tensors of a model.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        bounds: A list of (lower, upper) tuples for each column of X.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b. (Not used by single task models)
        linear_constraints: A tuple of (A, b). For k linear constraints on
            d-dimensional x, A is (k x d) and b is (k x 1) such that
            A x <= b. (Not used by single task models)
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value during generation.

    Returns:
        Tensor: All points that are feasible and appear in the objective or
            the constraints. None if there are no such points.
    """
    # Get points observed for all objective and constraint outcomes
    X_obs = get_observed(
        Xs=Xs,
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
    )
    # Filter to those that satisfy constraints.
    X_obs = filter_constraints_and_fixed_features(
        X=X_obs,
        bounds=bounds,
        linear_constraints=linear_constraints,
        fixed_features=fixed_features,
    )
    if len(X_obs) > 0:
        return torch.as_tensor(X_obs)  # please the linter


def _get_X_pending_and_observed(
    Xs: List[Tensor],
    objective_weights: Tensor,
    bounds: List[Tuple[float, float]],
    pending_observations: Optional[List[Tensor]] = None,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    r"""Get pending and observed points.

    If all points would otherwise be filtered, remove `linear_constraints`
    and `fixed_features` from filter and retry.

    Args:
        Xs: The input tensors of a model.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        bounds: A list of (lower, upper) tuples for each column of X.
        pending_observations:  A list of m (k_i x d) feature tensors X
            for m outcomes and k_i pending observations for outcome i.
            (Only used if n > 1).
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b. (Not used by single task models)
        linear_constraints: A tuple of (A, b). For k linear constraints on
            d-dimensional x, A is (k x d) and b is (k x 1) such that
            A x <= b. (Not used by single task models)
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value during generation.

    Returns:
        Tensor: Pending points that are feasible and appear in the objective or
            the constraints. None if there are no such points.
        Tensor: Observed points that are feasible and appear in the objective or
            the constraints. None if there are no such points.
    """
    if pending_observations is None:
        X_pending = None
    else:
        X_pending = _filter_X_observed(
            Xs=pending_observations,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            bounds=bounds,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
        )
    filtered_X_observed = _filter_X_observed(
        Xs=Xs,
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
        bounds=bounds,
        linear_constraints=linear_constraints,
        fixed_features=fixed_features,
    )
    if filtered_X_observed is not None and len(filtered_X_observed) > 0:
        return X_pending, filtered_X_observed
    else:
        unfiltered_X_observed = _filter_X_observed(
            Xs=Xs,
            objective_weights=objective_weights,
            bounds=bounds,
            outcome_constraints=outcome_constraints,
        )
        return X_pending, unfiltered_X_observed


def _generate_sobol_points(
    n_sobol: int,
    bounds: List[Tuple[float, float]],
    device: torch.device,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
    model_gen_options: Optional[TConfig] = None,
) -> Tensor:
    linear_constraints_array = None

    if linear_constraints is not None:
        linear_constraints_array = (
            linear_constraints[0].detach().cpu().numpy(),
            linear_constraints[1].detach().cpu().numpy(),
        )

    array_rounding_func = None
    if rounding_func is not None:
        array_rounding_func = tensor_callable_to_array_callable(
            tensor_func=rounding_func, device=device
        )

    sobol = SobolGenerator(deduplicate=False, seed=np.random.randint(10000))
    array_X, _ = sobol.gen(
        n=n_sobol,
        bounds=bounds,
        linear_constraints=linear_constraints_array,
        fixed_features=fixed_features,
        rounding_func=array_rounding_func,
        model_gen_options=model_gen_options,
    )
    return torch.from_numpy(array_X).to(device)


def normalize_indices(indices: List[int], d: int) -> List[int]:
    r"""Normalize a list of indices to ensure that they are positive.

    Args:
        indices: A list of indices (may contain negative indices for indexing
            "from the back").
        d: The dimension of the tensor to index.

    Returns:
        A normalized list of indices such that each index is between `0` and `d-1`.
    """
    normalized_indices = []
    for i in indices:
        if i < 0:
            i = i + d
        if i < 0 or i > d - 1:
            raise ValueError(f"Index {i} out of bounds for tensor or length {d}.")
        normalized_indices.append(i)
    return normalized_indices


def subset_model(
    model: Model,
    objective_weights: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    objective_thresholds: Optional[Tensor] = None,
) -> SubsetModelData:
    """Subset a botorch model to the outputs used in the optimization.

    Args:
        model: A BoTorch Model. If the model does not implement the
            `subset_outputs` method, this function is a null-op and returns the
            input arguments.
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        objective_thresholds:  The `m`-dim tensor of objective thresholds. There
            is one for each modeled metric.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b. (Not used by single task models)

    Returns:
        A SubsetModelData dataclass containing the model, objective_weights,
        outcome_constraints, objective thresholds, all subset to only those
        outputs that appear in either the objective weights or the outcome
        constraints, along with the indices of the outputs.
    """
    nonzero = objective_weights != 0
    if outcome_constraints is not None:
        A, _ = outcome_constraints
        nonzero = nonzero | torch.any(A != 0, dim=0)
    idcs_t = torch.arange(nonzero.size(0), device=objective_weights.device)[nonzero]
    idcs = idcs_t.tolist()
    if len(idcs) == model.num_outputs:
        # if we use all model outputs, just return the inputs
        return SubsetModelData(
            model=model,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            objective_thresholds=objective_thresholds,
            indices=torch.arange(
                model.num_outputs,
                device=objective_weights.device,
            ),
        )
    elif len(idcs) > model.num_outputs:
        raise RuntimeError(
            "Model size inconsistency. Tryting to subset a model with "
            f"{model.num_outputs} outputs to {len(idcs)} outputs"
        )
    try:
        model = model.subset_output(idcs=idcs)
        objective_weights = objective_weights[nonzero]
        if outcome_constraints is not None:
            A, b = outcome_constraints
            outcome_constraints = A[:, nonzero], b
        if objective_thresholds is not None:
            objective_thresholds = objective_thresholds[nonzero]
    except NotImplementedError:
        idcs_t = torch.arange(
            model.num_outputs,
            device=objective_weights.device,
        )
    return SubsetModelData(
        model=model,
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
        objective_thresholds=objective_thresholds,
        indices=idcs_t,
    )


def _to_inequality_constraints(
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None
) -> Optional[List[Tuple[Tensor, Tensor, float]]]:
    if linear_constraints is not None:
        A, b = linear_constraints
        inequality_constraints = []
        k, d = A.shape
        for i in range(k):
            indicies = A[i, :].nonzero(as_tuple=False).squeeze()
            coefficients = -A[i, indicies]
            rhs = -b[i, 0]
            inequality_constraints.append((indicies, coefficients, rhs))
    else:
        inequality_constraints = None
    return inequality_constraints


def tensor_callable_to_array_callable(
    tensor_func: Callable[[Tensor], Tensor], device: torch.device
) -> Callable[[np.ndarray], np.ndarray]:
    """transfer a tensor callable to an array callable"""

    def array_func(x: np.ndarray) -> np.ndarray:
        return tensor_func(torch.from_numpy(x).to(device)).detach().cpu().numpy()

    return array_func


def _get_weighted_mo_objective(
    objective_weights: Tensor,
) -> WeightedMCMultiOutputObjective:
    """Constructs the `WeightedMCMultiOutputObjective` for the given
    objective weights.
    """
    nonzero_idcs = torch.nonzero(objective_weights).view(-1)
    objective_weights = objective_weights[nonzero_idcs]
    return WeightedMCMultiOutputObjective(
        weights=objective_weights, outcomes=nonzero_idcs.tolist()
    )


def _get_risk_measure(
    model: Model,
    objective_weights: Tensor,
    risk_measure: RiskMeasureMCObjective,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    X_observed: Optional[Tensor] = None,
) -> RiskMeasureMCObjective:
    r"""Processes the risk measure for `get_botorch_objective_and_transform`.
    See the docstring of `get_botorch_objective_and_transform` for the arguments.
    """
    if outcome_constraints is not None:
        # TODO[T131759270]: Handle the constraints via feasibility weighting.
        # See `FeasibilityWeightedMCMultiOutputObjective`.
        raise NotImplementedError(
            "Outcome constraints are not supported with risk measures."
        )
    # Isinstance doesn't work since it covers subclasses as well.
    if risk_measure.preprocessing_function.__class__ not in (
        IdentityMCObjective,
        IdentityMCMultiOutputObjective,
    ) or hasattr(risk_measure.preprocessing_function, "outcomes"):
        raise UnsupportedError(
            "User supplied preprocessing functions for the risk measures are not "
            "supported. We construct a new one based on `objective_weights` instead."
        )
    if isinstance(risk_measure, MultiOutputRiskMeasureMCObjective):
        risk_measure.preprocessing_function = _get_weighted_mo_objective(
            objective_weights=objective_weights
        )
        if isinstance(risk_measure, MARS):
            risk_measure.chebyshev_weights = sample_simplex(
                len(objective_weights.nonzero())
            ).squeeze()
            if X_observed is None:
                raise UnsupportedError("X_observed is required when using MARS.")
            risk_measure.set_baseline_Y(model=model, X_baseline=X_observed)
    else:
        risk_measure.preprocessing_function = LinearMCObjective(
            weights=objective_weights
        )
    return risk_measure


def get_botorch_objective_and_transform(
    model: Model,
    objective_weights: Tensor,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    objective_thresholds: Optional[Tensor] = None,
    X_observed: Optional[Tensor] = None,
    risk_measure: Optional[RiskMeasureMCObjective] = None,
) -> Tuple[Optional[MCAcquisitionObjective], Optional[PosteriorTransform]]:
    """Constructs a BoTorch `AcquisitionObjective` object.

    Args:
        model: A BoTorch Model
        objective_weights: The objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b. (Not used by single task models)
        objective_thresholds: A tensor containing thresholds forming a reference point
            from which to calculate pareto frontier hypervolume. Points that do not
            dominate the objective_thresholds contribute nothing to hypervolume.
        X_observed: Observed points that are feasible and appear in the
            objective or the constraints. None if there are no such points.
        risk_measure: An optional risk measure for robust optimization.

    Returns:
        A two-tuple containing (optionally) an `MCAcquisitionObjective` and
        (optionally) a `PosteriorTransform`.
    """
    if risk_measure is not None:
        risk_measure = _get_risk_measure(
            model=model,
            objective_weights=objective_weights,
            risk_measure=risk_measure,
            outcome_constraints=outcome_constraints,
            X_observed=X_observed,
        )
        return risk_measure, None

    if objective_thresholds is not None:
        # We are doing multi-objective optimization.
        return _get_weighted_mo_objective(objective_weights=objective_weights), None
    if outcome_constraints:
        if X_observed is None:
            raise UnsupportedError(
                "X_observed is required to construct a constrained BoTorch objective."
            )
        # If there are outcome constraints, we use MC Acquisition functions.
        obj_tf: Callable[
            [Tensor, Optional[Tensor]], Tensor
        ] = get_objective_weights_transform(objective_weights)

        def objective(samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
            return obj_tf(samples, X)

        con_tfs = get_outcome_constraint_transforms(outcome_constraints)
        inf_cost = get_infeasible_cost(X=X_observed, model=model, objective=obj_tf)
        objective = ConstrainedMCObjective(
            objective=objective, constraints=con_tfs or [], infeasible_cost=inf_cost
        )
        return objective, None
    # Case of linear weights - use ScalarizedPosteriorTransform
    transform = ScalarizedPosteriorTransform(weights=objective_weights)
    return None, transform


def get_out_of_sample_best_point_acqf(
    model: Model,
    Xs: List[Tensor],
    X_observed: Tensor,
    objective_weights: Tensor,
    mc_samples: int = 512,
    fixed_features: Optional[Dict[int, float]] = None,
    fidelity_features: Optional[List[int]] = None,
    target_fidelities: Optional[Dict[int, float]] = None,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    seed_inner: Optional[int] = None,
    qmc: bool = True,
    risk_measure: Optional[RiskMeasureMCObjective] = None,
    **kwargs: Any,
) -> Tuple[AcquisitionFunction, Optional[List[int]]]:
    """Picks an appropriate acquisition function to find the best
    out-of-sample (predicted by the given surrogate model) point
    and instantiates it.

    NOTE: Typically the appropriate function is the posterior mean,
    but can differ to account for fidelities etc.
    """
    model = model

    # subset model only to the outcomes we need for the optimization
    if kwargs.get(Keys.SUBSET_MODEL, True):
        subset_model_results = subset_model(
            model=model,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
        )
        model = subset_model_results.model
        objective_weights = subset_model_results.objective_weights
        outcome_constraints = subset_model_results.outcome_constraints

    fixed_features = fixed_features or {}
    target_fidelities = target_fidelities or {}

    if fidelity_features:
        # we need to optimize at the target fidelities
        if any(f in fidelity_features for f in fixed_features):
            raise RuntimeError("Fixed features cannot also be fidelity features.")
        elif set(fidelity_features) != set(target_fidelities):
            raise RuntimeError(
                "Must provide a target fidelity for every fidelity feature."
            )
        # make sure to not modify fixed_features in-place
        fixed_features = {**fixed_features, **target_fidelities}
    elif target_fidelities:
        raise RuntimeError(
            "Must specify fidelity_features in fit() when using target fidelities."
        )

    acqf_class, acqf_options = pick_best_out_of_sample_point_acqf_class(
        outcome_constraints=outcome_constraints,
        mc_samples=mc_samples,
        qmc=qmc,
        seed_inner=seed_inner,
        risk_measure=risk_measure,
    )
    objective, posterior_transform = get_botorch_objective_and_transform(
        model=model,
        objective_weights=objective_weights,
        outcome_constraints=outcome_constraints,
        X_observed=X_observed,
        risk_measure=risk_measure,
    )

    if objective is not None:
        if not isinstance(objective, MCAcquisitionObjective):
            raise UnsupportedError(
                f"Unknown objective type: {objective.__class__}"  # pragma: nocover
            )
        acqf_options = {"objective": objective, **acqf_options}
    if posterior_transform is not None:
        acqf_options = {"posterior_transform": posterior_transform, **acqf_options}

    acqf = acqf_class(model=model, **acqf_options)  # pyre-ignore [45]

    if fixed_features:
        acqf = FixedFeatureAcquisitionFunction(
            acq_function=acqf,
            d=X_observed.size(-1),
            columns=list(fixed_features.keys()),
            values=list(fixed_features.values()),
        )
        non_fixed_idcs = [i for i in range(Xs[0].size(-1)) if i not in fixed_features]

    else:
        non_fixed_idcs = None

    return acqf, non_fixed_idcs


def pick_best_out_of_sample_point_acqf_class(
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    mc_samples: int = 512,
    qmc: bool = True,
    seed_inner: Optional[int] = None,
    risk_measure: Optional[RiskMeasureMCObjective] = None,
) -> Tuple[Type[AcquisitionFunction], Dict[str, Any]]:
    if outcome_constraints is None and risk_measure is None:
        acqf_class = PosteriorMean
        acqf_options = {}
    else:
        acqf_class = qSimpleRegret
        sampler_class = SobolQMCNormalSampler if qmc else IIDNormalSampler
        acqf_options = {
            Keys.SAMPLER.value: sampler_class(
                sample_shape=torch.Size([mc_samples]), seed=seed_inner
            )
        }

    return cast(Type[AcquisitionFunction], acqf_class), acqf_options


def predict_from_model(model: Model, X: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Predicts outcomes given a model and input tensor.

    For a `FullyBayesianPosterior` we currently use a Gaussian approximation where we
    compute the mean and variance of the Gaussian mixture. This should ideally be
    changed to compute quantiles instead when Ax supports non-Gaussian distributions.

    Args:
        model: A botorch Model.
        X: A `n x d` tensor of input parameters.

    Returns:
        Tensor: The predicted posterior mean as an `n x o`-dim tensor.
        Tensor: The predicted posterior covariance as a `n x o x o`-dim tensor.
    """
    with torch.no_grad():
        # TODO: Allow Posterior to (optionally) return the full covariance matrix
        posterior = model.posterior(X)
        if isinstance(posterior, FullyBayesianPosterior):
            mean = posterior.mixture_mean.cpu().detach()
            var = posterior.mixture_variance.cpu().detach().clamp_min(0)
        elif isinstance(posterior, (GPyTorchPosterior, PosteriorList)):
            mean = posterior.mean.cpu().detach()
            var = posterior.variance.cpu().detach().clamp_min(0)
        else:  # pragma: no cover
            raise UnsupportedError(
                "Non-Gaussian posteriors are currently not supported."
            )
    cov = torch.diag_embed(var)
    return mean, cov


# TODO(jej): Possibly refactor to use "objective_directions".
def randomize_objective_weights(
    objective_weights: Tensor, **acquisition_function_kwargs: Any
) -> Tensor:
    """Generate a random weighting based on acquisition function settings.

    Args:
        objective_weights: Base weights to multiply by random values..
        **acquisition_function_kwargs: Kwargs containing weight generation algorithm
            options.

    Returns:
        A normalized list of indices such that each index is between `0` and `d-1`.
    """
    # Set distribution and sample weights.
    distribution = acquisition_function_kwargs.get(
        "random_scalarization_distribution", SIMPLEX
    )
    dtype = objective_weights.dtype
    device = objective_weights.device
    if distribution == SIMPLEX:
        random_weights = sample_simplex(
            len(objective_weights), dtype=dtype, device=device
        ).squeeze()
    elif distribution == HYPERSPHERE:
        random_weights = torch.abs(
            sample_hypersphere(
                len(objective_weights), dtype=dtype, device=device
            ).squeeze()
        )
    # pyre-fixme[61]: `random_weights` may not be initialized here.
    objective_weights = torch.mul(objective_weights, random_weights)
    return objective_weights


def _datasets_to_legacy_inputs(
    datasets: List[SupervisedDataset],
) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    """Convert a dictionary of dataset containers to legacy X, Y, Yvar inputs"""
    Xs, Ys, Yvars = [], [], []
    for dataset in datasets:
        if not isinstance(dataset, SupervisedDataset):
            raise UnsupportedError("Legacy setup only supports `SupervisedDataset`s")
        Xs.append(dataset.X())
        Ys.append(dataset.Y())
        if isinstance(dataset, FixedNoiseDataset):
            Yvars.append(dataset.Yvar())
        else:
            Yvars.append(torch.full_like(Ys[-1], float("nan")))
    return Xs, Ys, Yvars
