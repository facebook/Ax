#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from collections import OrderedDict
from collections.abc import Sequence
from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxError, AxWarning, UnsupportedError
from ax.models.types import TConfig
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel, GPyTorchModel
from botorch.models.model import Model, ModelList
from botorch.models.multitask import MultiTaskGP
from botorch.models.pairwise_gp import PairwiseGP
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.transforms import is_fully_bayesian
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from torch import Tensor

MIN_OBSERVED_NOISE_LEVEL = 1e-7
logger: Logger = get_logger(__name__)


def use_model_list(
    datasets: Sequence[SupervisedDataset],
    botorch_model_class: Type[Model],
    allow_batched_models: bool = True,
) -> bool:

    if issubclass(botorch_model_class, MultiTaskGP):
        # We currently always wrap multi-task models into `ModelListGP`.
        return True
    elif issubclass(botorch_model_class, SaasFullyBayesianSingleTaskGP):
        # SAAS models do not support multiple outcomes.
        # Use model list if there are multiple outcomes.
        return len(datasets) > 1 or datasets[0].Y.shape[-1] > 1
    elif len(datasets) == 1:
        # Just one outcome, can use single model.
        return False
    elif issubclass(botorch_model_class, BatchedMultiOutputGPyTorchModel) and all(
        torch.equal(datasets[0].X, ds.X) for ds in datasets[1:]
    ):
        # Use batch models if allowed
        return not allow_batched_models
    # If there are multiple Xs and they are not all equal, we
    # use `ListSurrogate` and `ModelListGP`.
    return True


def choose_model_class(
    datasets: Sequence[SupervisedDataset],
    search_space_digest: SearchSpaceDigest,
) -> Type[Model]:
    """Chooses a BoTorch `Model` using the given data (currently just Yvars)
    and its properties (information about task and fidelity features).

    Args:
        Yvars: List of tensors, each representing observation noise for a
            given outcome, where outcomes are in the same order as in Xs.
        task_features: List of columns of X that are tasks.
        fidelity_features: List of columns of X that are fidelity parameters.

    Returns:
        A BoTorch `Model` class.
    """
    if len(search_space_digest.fidelity_features) > 1:
        raise NotImplementedError(
            "Only a single fidelity feature supported "
            f"(got: {search_space_digest.fidelity_features})."
        )
    if len(search_space_digest.task_features) > 1:
        raise NotImplementedError(
            f"Only a single task feature supported "
            f"(got: {search_space_digest.task_features})."
        )
    if search_space_digest.task_features and search_space_digest.fidelity_features:
        raise NotImplementedError(
            "Multi-task multi-fidelity optimization not yet supported."
        )

    is_fixed_noise = [ds.Yvar is not None for ds in datasets]
    all_inferred = not any(is_fixed_noise)
    if not all_inferred and not all(is_fixed_noise):
        raise ValueError(
            "Mix of known and unknown variances indicates valuation function "
            "errors. Variances should all be specified, or none should be."
        )

    # Multi-task case (when `task_features` is specified).
    if search_space_digest.task_features:
        model_class = MultiTaskGP

    # Single-task multi-fidelity cases.
    elif search_space_digest.fidelity_features:
        model_class = SingleTaskMultiFidelityGP

    # Mixed optimization case. Note that presence of categorical
    # features in search space digest indicates that downstream in the
    # stack we chose not to perform continuous relaxation on those
    # features.
    elif search_space_digest.categorical_features:
        model_class = MixedSingleTaskGP

    # Single-task single-fidelity cases.
    else:
        model_class = SingleTaskGP

    logger.debug(f"Chose BoTorch model class: {model_class}.")
    return model_class


def choose_botorch_acqf_class(
    pending_observations: Optional[List[Tensor]] = None,
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    objective_thresholds: Optional[Tensor] = None,
    objective_weights: Optional[Tensor] = None,
) -> Type[AcquisitionFunction]:
    """Chooses a BoTorch `AcquisitionFunction` class."""
    if objective_thresholds is not None or (
        # using objective_weights is a less-than-ideal fix given its ambiguity,
        # the real fix would be to revisit the infomration passed down via
        # the modelbridge (and be explicit about whether we scalarize or perform MOO)
        objective_weights is not None
        and objective_weights.nonzero().numel() > 1
    ):
        acqf_class = qLogNoisyExpectedHypervolumeImprovement
    else:
        acqf_class = qLogNoisyExpectedImprovement

    logger.debug(f"Chose BoTorch acquisition function class: {acqf_class}.")
    return acqf_class


def construct_acquisition_and_optimizer_options(
    acqf_options: TConfig, model_gen_options: Optional[TConfig] = None
) -> Tuple[TConfig, TConfig]:
    """Extract acquisition and optimizer options from `model_gen_options`."""
    acq_options = acqf_options.copy()
    opt_options = {}

    if model_gen_options:
        acq_options.update(
            checked_cast(dict, model_gen_options.get(Keys.ACQF_KWARGS, {}))
        )
        # TODO: Add this if all acq. functions accept the `subset_model`
        # kwarg or opt for kwarg filtering.
        # acq_options[SUBSET_MODEL] = model_gen_options.get(SUBSET_MODEL)
        opt_options = checked_cast(
            dict, model_gen_options.get(Keys.OPTIMIZER_KWARGS, {})
        ).copy()
    return acq_options, opt_options


def convert_to_block_design(
    datasets: Sequence[SupervisedDataset],
    force: bool = False,
) -> List[SupervisedDataset]:
    # Convert data to "block design". TODO: Figure out a better
    # solution for this using the data containers (pass outcome
    # names as properties of the data containers)
    is_fixed = [ds.Yvar is not None for ds in datasets]
    if any(is_fixed) and not all(is_fixed):
        raise UnsupportedError(
            "Cannot convert mixed data with and without variance "
            "observations to `block design`."
        )
    is_fixed = all(is_fixed)
    Xs = [dataset.X for dataset in datasets]
    for dset in datasets[1:]:
        if dset.feature_names != datasets[0].feature_names:
            raise ValueError(
                "Feature names must be the same across all datasets, "
                f"got {dset.feature_names} and {datasets[0].feature_names}"
            )

    # Join the outcome names of datasets.
    outcome_names = sum([ds.outcome_names for ds in datasets], [])

    if len({X.shape for X in Xs}) != 1 or not all(
        torch.equal(X, Xs[0]) for X in Xs[1:]
    ):
        if not force:
            raise UnsupportedError(
                "Cannot convert data to non-block design data. "
                "To force this and drop data not shared between "
                "outcomes use `force=True`."
            )
        warnings.warn(
            "Forcing converion of data not complying to a block design "
            "to block design by dropping observations that are not shared "
            "between outcomes.",
            AxWarning,
            stacklevel=3,
        )
        X_shared, idcs_shared = _get_shared_rows(Xs=Xs)
        Y = torch.cat([ds.Y[i] for ds, i in zip(datasets, idcs_shared)], dim=-1)
        if is_fixed:
            Yvar = torch.cat(
                # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
                [ds.Yvar[i] for ds, i in zip(datasets, idcs_shared)],
                dim=-1,
            )
        else:
            Yvar = None
        datasets = [
            SupervisedDataset(
                X=X_shared,
                Y=Y,
                Yvar=Yvar,
                feature_names=datasets[0].feature_names,
                outcome_names=outcome_names,
            )
        ]
        return datasets

    # data complies to block design, can concat with impunity
    Y = torch.cat([ds.Y for ds in datasets], dim=-1)
    if is_fixed:
        Yvar = torch.cat([not_none(ds.Yvar) for ds in datasets], dim=-1)
    else:
        Yvar = None
    datasets = [
        SupervisedDataset(
            X=Xs[0],
            Y=Y,
            Yvar=Yvar,
            feature_names=datasets[0].feature_names,
            outcome_names=outcome_names,
        )
    ]
    return datasets


def _get_shared_rows(Xs: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    """Extract shared rows from a list of tensors

    Args:
        Xs: A list of m two-dimensional tensors with shapes
            `(n_1 x d), ..., (n_m x d)`. It is not required that
            the `n_i` are the same.

    Returns:
        A two-tuple containing (i) a Tensor with the rows that are
        shared between all the Tensors in `Xs`, and (ii) a list of
        index tensors that indicate the location of these rows
        in the respective elements of `Xs`.
    """
    idcs_shared = []
    Xs_sorted = sorted(Xs, key=len)
    X_shared = Xs_sorted[0].clone()
    for X in Xs_sorted[1:]:
        X_shared = X_shared[(X_shared == X.unsqueeze(-2)).all(dim=-1).any(dim=-2)]
    # get indices
    for X in Xs:
        same = (X_shared == X.unsqueeze(-2)).all(dim=-1).any(dim=-1)
        idcs_shared.append(torch.arange(same.shape[-1], device=X_shared.device)[same])
    return X_shared, idcs_shared


def fit_botorch_model(
    model: Model,
    mll_class: Type[MarginalLogLikelihood],
    mll_options: Optional[Dict[str, Any]] = None,
) -> None:
    """Fit a BoTorch model."""
    mll_options = mll_options or {}
    models = model.models if isinstance(model, ModelList) else [model]
    for m in models:
        # TODO: Support deterministic models when we support `ModelList`
        if is_fully_bayesian(m):
            fit_fully_bayesian_model_nuts(
                m,
                disable_progbar=True,
                **mll_options,
            )
        elif isinstance(m, (GPyTorchModel, PairwiseGP)):
            mll_options = mll_options or {}
            mll = mll_class(likelihood=m.likelihood, model=m, **mll_options)
            fit_gpytorch_mll(mll)
        else:
            raise NotImplementedError(
                f"Model of type {m.__class__.__name__} is currently not supported."
            )


def _tensor_difference(A: Tensor, B: Tensor) -> Tensor:
    """Used to return B sans any Xs that also appear in A"""
    C = torch.cat((A, B), dim=0)
    D, inverse_ind = torch.unique(C, return_inverse=True, dim=0)
    n = A.shape[0]
    A_indices = inverse_ind[:n].tolist()
    B_indices = inverse_ind[n:].tolist()
    Bi_set = set(B_indices) - set(A_indices)
    return D[list(Bi_set)]


def get_post_processing_func(
    rounding_func: Optional[Callable[[Tensor], Tensor]],
    optimizer_options: Dict[str, Any],
) -> Optional[Callable[[Tensor], Tensor]]:
    """Get the post processing function by combining the rounding function
    with the post processing function provided as part of the optimizer
    options. If both are given, the post processing function is applied before
    applying the rounding function. If only one of them is given, then
    it is used as the post processing function.
    """
    if "post_processing_func" in optimizer_options:
        provided_func: Callable[[Tensor], Tensor] = optimizer_options.pop(
            "post_processing_func"
        )
        if rounding_func is None:
            # No rounding function is given. We can use the post processing
            # function directly.
            return provided_func
        else:
            # Both post processing and rounding functions are given. We need
            # to chain them and apply the post processing function first.
            base_rounding_func: Callable[[Tensor], Tensor] = rounding_func

            def combined_func(x: Tensor) -> Tensor:
                return base_rounding_func(provided_func(x))

            return combined_func

    else:
        return rounding_func


def check_outcome_dataset_match(
    outcome_names: Sequence[str],
    datasets: Sequence[SupervisedDataset],
    exact_match: bool,
) -> None:
    """Check that the given outcome names match those of datasets.

    Based on `exact_match` we either require that outcome names are
    a subset of all outcomes or require the them to be the same.

    Also checks that there are no duplicates in outcome names.

    Args:
        outcome_names: A list of outcome names.
        datasets: A list of `SupervisedDataset` objects.
        exact_match: If True, outcome_names must be the same as the union of
            outcome names of the datasets. Otherwise, we check that the
            outcome_names are a subset of all outcomes.

    Raises:
        ValueError: If there is no match.
    """
    all_outcomes = sum((ds.outcome_names for ds in datasets), [])
    set_all_outcomes = set(all_outcomes)
    set_all_spec_outcomes = set(outcome_names)
    if len(set_all_outcomes) != len(all_outcomes):
        raise AxError("Found duplicate outcomes in the datasets.")
    if len(set_all_spec_outcomes) != len(outcome_names):
        raise AxError("Found duplicate outcome names.")

    if not exact_match:
        if not set_all_spec_outcomes.issubset(set_all_outcomes):
            raise AxError(
                "Outcome names must be a subset of the outcome names of the datasets."
                f"Got {outcome_names=} but the datasets model {set_all_outcomes}."
            )
    elif set_all_spec_outcomes != set_all_outcomes:
        raise AxError(
            "Each outcome name must correspond to an outcome in the datasets. "
            f"Got {outcome_names=} but the datasets model {set_all_outcomes}."
        )


def get_subset_datasets(
    datasets: Sequence[SupervisedDataset],
    subset_outcome_names: Sequence[str],
) -> List[SupervisedDataset]:
    """Get the list of datasets corresponding to the given subset of
    outcome names. This is used to separate out datasets that are
    used by one surrogate.

    Args:
        datasets: A list of `SupervisedDataset` objects.
        subset_outcome_names: A list of outcome names to get datasets for.

    Returns:
        A list of `SupervisedDataset` objects corresponding to the given
        subset of outcome names.
    """
    check_outcome_dataset_match(
        outcome_names=subset_outcome_names, datasets=datasets, exact_match=False
    )
    single_outcome_datasets = {
        ds.outcome_names[0]: ds for ds in datasets if len(ds.outcome_names) == 1
    }
    multi_outcome_datasets = {
        tuple(ds.outcome_names): ds for ds in datasets if len(ds.outcome_names) > 1
    }
    subset_datasets = []
    outcomes_processed = []
    for outcome_name in subset_outcome_names:
        if outcome_name in outcomes_processed:
            # This can happen if the outcome appears in a multi-outcome
            # dataset that is already processed.
            continue
        if outcome_name in single_outcome_datasets:
            # The default case of outcome with a corresponding dataset.
            ds = single_outcome_datasets[outcome_name]
        else:
            # The case of outcome being part of a multi-outcome dataset.
            for outcome_names in multi_outcome_datasets.keys():
                if outcome_name in outcome_names:
                    ds = multi_outcome_datasets[outcome_names]
                    if not set(ds.outcome_names).issubset(subset_outcome_names):
                        raise UnsupportedError(
                            "Breaking up a multi-outcome dataset between "
                            "surrogates is not supported."
                        )
                    break
        # Pyre-ignore [61]: `ds` may not be defined but it is guaranteed to be defined.
        subset_datasets.append(ds)
        outcomes_processed.extend(ds.outcome_names)
    return subset_datasets


def subset_state_dict(
    state_dict: OrderedDict[str, Tensor],
    submodel_index: int,
) -> OrderedDict[str, Tensor]:
    """Get the state dict for a submodel from the state dict of a model list.

    Args:
        state_dict: A state dict.
        submodel_index: The index of the submodel to extract.

    Returns:
        The state dict for the submodel.
    """
    expected_substring = f"models.{submodel_index}."
    len_substring = len(expected_substring)
    new_items = [
        (k[len_substring:], v)
        for k, v in state_dict.items()
        if k.startswith(expected_substring)
    ]
    return OrderedDict(new_items)
