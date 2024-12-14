#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import warnings
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from logging import Logger
from typing import Any

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.exceptions.core import AxWarning
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.optimizer_argparse import optimizer_argparse
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.logger import get_logger
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.penalized import L0Approximation
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import ModelList
from botorch.optim import (
    gen_batch_initial_conditions,
    Homotopy,
    HomotopyParameter,
    LogLinearHomotopySchedule,
    optimize_acqf_homotopy,
)
from botorch.utils.datasets import SupervisedDataset
from pyre_extensions import assert_is_instance, none_throws
from torch import Tensor

CLAMP_TOL = 1e-2
logger: Logger = get_logger(__name__)


class SEBOAcquisition(Acquisition):
    """
    Implement the acquisition function of Sparsity Exploring Bayesian
    Optimization (SEBO).

    The SEBO is a hyperparameter-free method to simultaneously maximize a target
    objective and sparsity. When L0 norm is used, SEBO uses a novel differentiable
    relaxation based on homotopy continuation to efficiently optimize for sparsity.
    """

    def __init__(
        self,
        surrogate: Surrogate,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        botorch_acqf_class: type[AcquisitionFunction],
        options: dict[str, Any] | None = None,
    ) -> None:
        tkwargs: dict[str, Any] = {"dtype": surrogate.dtype, "device": surrogate.device}
        options = {} if options is None else options
        self.penalty_name: str = options.pop("penalty", "L0_norm")
        self.target_point: Tensor = options.pop("target_point", None)
        if self.target_point is None:
            raise ValueError("please provide target point.")
        self.target_point.to(**tkwargs)
        self.sparsity_threshold: int = options.pop(
            "sparsity_threshold", surrogate.Xs[0].shape[-1]
        )
        # construct determinsitic model for penalty term
        # pyre-fixme[4]: Attribute must be annotated.
        self.deterministic_model = self._construct_penalty()
        surrogate_f = deepcopy(surrogate)
        # we need to clamp the training data to the target point here as it may
        # be slightly off due to numerical issues.
        X_sparse = clamp_to_target(
            X=surrogate_f.Xs[0].clone(),
            target_point=self.target_point,
            clamp_tol=CLAMP_TOL,
        )
        # update the training data in new surrogate
        none_throws(surrogate_f._training_data).append(
            SupervisedDataset(
                X=X_sparse,
                Y=self.deterministic_model(X_sparse),
                Yvar=torch.zeros(X_sparse.shape[0], 1, **tkwargs),  # noiseless
                feature_names=surrogate_f.training_data[0].feature_names,
                outcome_names=[self.penalty_name],
            )
        )
        # update the model in new surrogate
        surrogate_f._model = ModelList(surrogate.model, self.deterministic_model)
        # update objective weights and thresholds in the torch config
        torch_opt_config_sebo = self._transform_torch_config(
            torch_opt_config=torch_opt_config, **tkwargs
        )

        # Change some options (note: we do not want to do this in-place)
        if options.get("cache_root", False):
            warnings.warn(
                "SEBO doesn't support `cache_root=True`. Changing it to `False`.",
                AxWarning,
                stacklevel=3,
            )
            options = {**options, "cache_root": False}

        # Instantiate the `botorch_acqf_class`. We need to modify `a` before doing this
        # (as it controls the L0 norm approximation) since the baseline will be pruned
        # when the acquisition function is created. With a=1e-6 the deterministic model
        # will be numerically close to the true L0 norm and we will select the
        # baseline according to the last homotopy step.
        if self.penalty_name == "L0_norm":
            self.deterministic_model._f.a.fill_(1e-6)
        super().__init__(
            surrogate=surrogate_f,
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config_sebo,
            botorch_acqf_class=qLogNoisyExpectedHypervolumeImprovement,
            options=options,
        )

        # update objective threshold for deterministic model (penalty term)
        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, slice[Any, A...
        self.acqf.ref_point[-1] = self.sparsity_threshold * -1
        self._objective_thresholds[-1] = self.sparsity_threshold  # pyre-ignore

    def _construct_penalty(self) -> GenericDeterministicModel:
        """Construct a penalty term as deterministic model to be included in
        SEBO acqusition function. Currently only L0 and L1 penalty are supported.
        """
        if self.penalty_name == "L0_norm":
            L0 = L0Approximation(target_point=self.target_point)
            return GenericDeterministicModel(f=L0)
        elif self.penalty_name == "L1_norm":
            L1 = functools.partial(
                L1_norm_func,
                init_point=self.target_point,
            )
            return GenericDeterministicModel(f=L1)
        else:
            raise NotImplementedError(
                f"{self.penalty_name} is not currently implemented."
            )

    def _transform_torch_config(
        self,
        torch_opt_config: TorchOptConfig,
        **tkwargs: Any,
    ) -> TorchOptConfig:
        """Transform torch config to include penalty term (deterministic model) as
        an additional outcomes in BoTorch model.
        """
        # update objective weights by appending the weight -1 for sparsity objective.
        objective_weights_sebo = torch.cat(
            [torch_opt_config.objective_weights, -torch.ones(1, **tkwargs)]
        )
        if torch_opt_config.outcome_constraints is not None:
            # update the shape of A matrix in outcome_constraints
            A, b = none_throws(torch_opt_config.outcome_constraints)
            outcome_constraints_sebo = (
                torch.cat([A, torch.zeros(A.shape[0], 1, **tkwargs)], dim=1),
                b,
            )
        else:
            outcome_constraints_sebo = None
        if torch_opt_config.objective_thresholds is not None:
            objective_thresholds_sebo = torch.cat(
                [
                    torch_opt_config.objective_thresholds,
                    torch.tensor([self.sparsity_threshold], **tkwargs),
                ]
            )
        else:
            # NOTE: The reference point will be inferred in the base class.
            objective_thresholds_sebo = None

        # update pending observations (if not none) by appending an obs for
        # the new penalty outcome
        pending_observations = torch_opt_config.pending_observations
        if torch_opt_config.pending_observations is not None:
            pending_observations = torch_opt_config.pending_observations + [
                torch_opt_config.pending_observations[0]
            ]

        return TorchOptConfig(
            objective_weights=objective_weights_sebo,
            outcome_constraints=outcome_constraints_sebo,
            objective_thresholds=objective_thresholds_sebo,
            linear_constraints=torch_opt_config.linear_constraints,
            fixed_features=torch_opt_config.fixed_features,
            pending_observations=pending_observations,
            model_gen_options=torch_opt_config.model_gen_options,
            rounding_func=torch_opt_config.rounding_func,
            opt_config_metrics=torch_opt_config.opt_config_metrics,
            is_moo=True,  # SEBO adds an objective, so it'll always be MOO.
        )

    def optimize(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
        fixed_features: dict[int, float] | None = None,
        rounding_func: Callable[[Tensor], Tensor] | None = None,
        optimizer_options: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate a set of candidates via multi-start optimization. Obtains
        candidates and their associated acquisition function values.

        Args:
            n: The number of candidates to generate.
            search_space_digest: A ``SearchSpaceDigest`` object containing search space
                properties, e.g. ``bounds`` for optimization.
            inequality_constraints: A list of tuples (indices, coefficients, rhs),
                with each tuple encoding an inequality constraint of the form
                ``sum_i (X[indices[i]] * coefficients[i]) >= rhs``.
            fixed_features: A map `{feature_index: value}` for features that
                should be fixed to a particular value during generation.
            rounding_func: A function that post-processes an optimization
                result appropriately (i.e., according to `round-trip`
                transformations).
            optimizer_options: Options for the optimizer function, e.g. ``sequential``
                or ``raw_samples``.

        Returns:
            A three-element tuple containing an `n x d`-dim tensor of generated
            candidates, a tensor with the associated acquisition values, and a tensor
            with the weight for each candidate.
        """
        if self.penalty_name == "L0_norm":
            candidates, expected_acquisition_value, weights = (
                self._optimize_with_homotopy(
                    n=n,
                    search_space_digest=search_space_digest,
                    inequality_constraints=inequality_constraints,
                    fixed_features=fixed_features,
                    rounding_func=rounding_func,
                    optimizer_options=optimizer_options,
                )
            )
        else:
            # if L1 norm use standard moo-opt
            candidates, expected_acquisition_value, weights = super().optimize(
                n=n,
                search_space_digest=search_space_digest,
                inequality_constraints=inequality_constraints,
                fixed_features=fixed_features,
                rounding_func=rounding_func,
                optimizer_options=optimizer_options,
            )

        # similar, make sure if applies to sparse dimensions only
        candidates = clamp_to_target(
            X=candidates, target_point=self.target_point, clamp_tol=CLAMP_TOL
        )
        return candidates, expected_acquisition_value, weights

    def _optimize_with_homotopy(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
        fixed_features: dict[int, float] | None = None,
        rounding_func: Callable[[Tensor], Tensor] | None = None,
        optimizer_options: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Optimize SEBO ACQF with L0 norm using homotopy."""
        optimizer_options = optimizer_options or {}
        # extend to fixed a no homotopy_schedule schedule
        _tensorize = partial(torch.tensor, dtype=self.dtype, device=self.device)
        bounds = _tensorize(search_space_digest.bounds).t()
        homotopy_schedule = LogLinearHomotopySchedule(
            start=0.2,
            end=1e-3,
            num_steps=optimizer_options.get("num_homotopy_steps", 30),
        )

        # Prepare arguments for optimizer
        optimizer_options_with_defaults = optimizer_argparse(
            self.acqf,
            optimizer_options=optimizer_options,
            optimizer="optimize_acqf_homotopy",
        )

        homotopy = Homotopy(
            homotopy_parameters=[
                HomotopyParameter(
                    parameter=self.deterministic_model._f.a,
                    schedule=homotopy_schedule,
                )
            ],
        )

        if "batch_initial_conditions" not in optimizer_options_with_defaults:
            optimizer_options_with_defaults["batch_initial_conditions"] = (
                get_batch_initial_conditions(
                    acq_function=self.acqf,
                    raw_samples=optimizer_options_with_defaults["raw_samples"],
                    inequality_constraints=inequality_constraints,
                    fixed_features=fixed_features,
                    X_pareto=assert_is_instance(self.acqf.X_baseline, Tensor),
                    target_point=self.target_point,
                    bounds=bounds,
                    num_restarts=optimizer_options_with_defaults["num_restarts"],
                )
            )

        candidates, expected_acquisition_value = optimize_acqf_homotopy(
            q=n,
            acq_function=self.acqf,
            bounds=bounds,
            homotopy=homotopy,
            num_restarts=optimizer_options_with_defaults["num_restarts"],
            raw_samples=optimizer_options_with_defaults["raw_samples"],
            inequality_constraints=inequality_constraints,
            post_processing_func=rounding_func,
            fixed_features=fixed_features,
            batch_initial_conditions=optimizer_options_with_defaults[
                "batch_initial_conditions"
            ],
        )
        return (
            candidates,
            expected_acquisition_value,
            torch.ones(n, device=candidates.device, dtype=candidates.dtype),
        )


def L1_norm_func(X: Tensor, init_point: Tensor) -> Tensor:
    r"""L1_norm takes in a a `batch_shape x n x d`-dim input tensor `X`
    to a `batch_shape x n x 1`-dimensional L1 norm tensor. To be used
    for constructing a GenericDeterministicModel.
    """
    return torch.linalg.norm((X - init_point), ord=1, dim=-1, keepdim=True)


def clamp_to_target(X: Tensor, target_point: Tensor, clamp_tol: float) -> Tensor:
    """Clamp generated candidates within the given ranges to the target point.

    Args:
        X: A `batch_shape x n x d`-dim input tensor `X`.
        target_point: A tensor of size `d` corresponding to the target point.
        clamp_tol: The clamping tolerance. Any value within `clamp_tol` of the
            `target_point` will be clamped to the `target_point`.
    """
    clamp_mask = (X - target_point).abs() <= clamp_tol
    X[clamp_mask] = target_point.clone().repeat(*X.shape[:-1], 1)[clamp_mask]
    return X


def get_batch_initial_conditions(
    acq_function: AcquisitionFunction,
    raw_samples: int,
    X_pareto: Tensor,
    target_point: Tensor,
    bounds: Tensor,
    num_restarts: int = 20,
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
    fixed_features: dict[int, float] | None = None,
) -> Tensor:
    """Generate starting points for the SEBO acquisition function optimization."""
    tkwargs: dict[str, Any] = {"device": X_pareto.device, "dtype": X_pareto.dtype}
    num_rand = num_restarts if len(X_pareto) == 0 else num_restarts // 2
    num_local = num_restarts - num_rand

    # (1) Random points (Sobol if no constraints, otherwise uses hit-and-run)
    X_cand_rand = gen_batch_initial_conditions(
        acq_function=acq_function,
        bounds=bounds,
        q=1,
        raw_samples=raw_samples,
        num_restarts=num_rand,
        options={"topn": True},
        fixed_features=fixed_features,
        inequality_constraints=inequality_constraints,
    ).to(**tkwargs)

    if num_local == 0:
        return X_cand_rand

    # (2) Perturbations of points on the Pareto frontier (done by TuRBO/Spearmint)
    X_cand_local = X_pareto.clone()[
        torch.randint(high=len(X_pareto), size=(raw_samples,))
    ]
    mask = X_cand_local != target_point
    X_cand_local[mask] += (
        0.2 * ((bounds[1] - bounds[0]) * torch.randn_like(X_cand_local))[mask]
    )
    X_cand_local = torch.clamp(X_cand_local.unsqueeze(1), min=bounds[0], max=bounds[1])
    X_cand_local = X_cand_local[acq_function(X_cand_local).topk(num_local).indices]
    return torch.cat((X_cand_rand, X_cand_local), dim=0)
