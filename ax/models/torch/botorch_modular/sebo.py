#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.optimizer_argparse import optimizer_argparse
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.models.torch_base import TorchOptConfig
from ax.utils.common.constants import Keys
from ax.utils.common.typeutils import not_none
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.acquisition.penalized import L0Approximation
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import ModelList
from botorch.optim import (
    Homotopy,
    HomotopyParameter,
    LogLinearHomotopySchedule,
    optimize_acqf_homotopy,
)
from botorch.utils.datasets import SupervisedDataset
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor
from torch.quasirandom import SobolEngine

CLAMP_TOL = 0.01


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
        surrogates: Dict[str, Surrogate],
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        botorch_acqf_class: Type[AcquisitionFunction],
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(surrogates) > 1:
            raise ValueError("SEBO does not support support multiple surrogates.")
        surrogate = surrogates[Keys.ONLY_SURROGATE]

        tkwargs: Dict[str, Any] = {"dtype": surrogate.dtype, "device": surrogate.device}
        options = options or {}
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
        # update the training data in new surrogate
        not_none(surrogate_f._training_data).append(
            SupervisedDataset(
                surrogate_f.Xs[0],
                self.deterministic_model(surrogate_f.Xs[0]),
                # append Yvar as zero for penalty term
                Yvar=torch.zeros(surrogate_f.Xs[0].shape[0], 1, **tkwargs),
                feature_names=surrogate_f.training_data[0].feature_names,
                outcome_names=[self.penalty_name],
            )
        )
        # update the model in new surrogate
        surrogate_f._model = ModelList(surrogate.model, self.deterministic_model)
        self.det_metric_indx = -1

        # update objective weights and  thresholds in the torch config
        torch_opt_config_sebo = self._transform_torch_config(
            torch_opt_config, **tkwargs
        )

        # instantiate botorch_acqf_class
        if not issubclass(botorch_acqf_class, qExpectedHypervolumeImprovement):
            raise ValueError("botorch_acqf_class must be qEHVI to use SEBO")
        super().__init__(
            surrogates={"sebo": surrogate_f},
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config_sebo,
            botorch_acqf_class=botorch_acqf_class,
            options=options,
        )
        if not isinstance(self.acqf, qExpectedHypervolumeImprovement):
            raise ValueError("botorch_acqf_class must be qEHVI to use SEBO")

        # update objective threshold for deterministic model (penalty term)
        self.acqf.ref_point[-1] = self.sparsity_threshold * -1
        # pyre-ignore
        self._objective_thresholds[-1] = self.sparsity_threshold

        Y_pareto = torch.cat(
            [d.Y for d in self.surrogates["sebo"].training_data],
            dim=-1,
        )
        ind_pareto = is_non_dominated(Y_pareto * self._full_objective_weights)
        # pyre-ignore
        self.X_pareto = self.surrogates["sebo"].Xs[0][ind_pareto].clone()

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
        # update objective weights by appending the weight (-1) of penalty term
        # at the end
        ow_sebo = torch.cat(
            [torch_opt_config.objective_weights, torch.tensor([-1], **tkwargs)]
        )
        if torch_opt_config.outcome_constraints is not None:
            # update the shape of A matrix in outcome_constraints
            oc_sebo = (
                torch.cat(
                    [
                        torch_opt_config.outcome_constraints[0],
                        torch.zeros(
                            # pyre-ignore
                            torch_opt_config.outcome_constraints[0].shape[0],
                            1,
                            **tkwargs,
                        ),
                    ],
                    dim=1,
                ),
                torch_opt_config.outcome_constraints[1],
            )
        else:
            oc_sebo = None
        if torch_opt_config.objective_thresholds is not None:
            # append the sparsity threshold at the end if objective_thresholds
            # is not None
            ot_sebo = torch.cat(
                [
                    torch_opt_config.objective_thresholds,
                    torch.tensor([self.sparsity_threshold], **tkwargs),
                ]
            )
        else:
            ot_sebo = None

        # update pending observations (if not none) by appending an obs for
        # the new penalty outcome
        pending_observations = torch_opt_config.pending_observations
        if torch_opt_config.pending_observations is not None:
            pending_observations = torch_opt_config.pending_observations + [
                torch_opt_config.pending_observations[0]
            ]

        return TorchOptConfig(
            objective_weights=ow_sebo,
            outcome_constraints=oc_sebo,
            objective_thresholds=ot_sebo,
            linear_constraints=torch_opt_config.linear_constraints,
            fixed_features=torch_opt_config.fixed_features,
            pending_observations=pending_observations,
            model_gen_options=torch_opt_config.model_gen_options,
            rounding_func=torch_opt_config.rounding_func,
            opt_config_metrics=torch_opt_config.opt_config_metrics,
            is_moo=torch_opt_config.is_moo,
        )

    def optimize(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
            if inequality_constraints is not None:
                raise NotImplementedError(
                    "Homotopy does not support optimization with inequality "
                    + "constraints. Use L1 penalty norm instead."
                )
            candidates, expected_acquisition_value, weights = (
                self._optimize_with_homotopy(
                    n=n,
                    search_space_digest=search_space_digest,
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
        candidates = clamp_candidates(
            X=candidates,
            target_point=self.target_point,
            clamp_tol=CLAMP_TOL,
            device=self.device,
            dtype=self.dtype,
        )
        return candidates, expected_acquisition_value, weights

    def _optimize_with_homotopy(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        fixed_features: Optional[Dict[int, float]] = None,
        rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Optimize SEBO ACQF with L0 norm using homotopy."""
        # extend to fixed a no homotopy_schedule schedule
        _tensorize = partial(torch.tensor, dtype=self.dtype, device=self.device)
        ssd = search_space_digest
        bounds = _tensorize(ssd.bounds).t()

        homotopy_schedule = LogLinearHomotopySchedule(start=0.1, end=1e-3, num_steps=30)

        # Prepare arguments for optimizer
        optimizer_options_with_defaults = optimizer_argparse(
            self.acqf,
            bounds=bounds,
            q=n,
            optimizer_options=optimizer_options,
        )

        def callback():  # pyre-ignore
            if (
                self.acqf.cache_pending
            ):  # If true, pending points are concatenated with X_baseline
                if self.acqf._max_iep != 0:
                    raise ValueError(
                        "The maximum number of pending points (max_iep) must be 0"
                    )
                X_baseline = self.acqf._X_baseline_and_pending.clone()
                self.acqf.__init__(  # pyre-ignore
                    X_baseline=X_baseline,
                    model=self.surrogates["sebo"].model,
                    ref_point=self.acqf.ref_point,
                    objective=self.acqf.objective,
                )
            else:  # We can directly get the pending points here
                X_pending = self.acqf.X_pending
                self.acqf.__init__(  # pyre-ignore
                    X_baseline=self.X_observed,
                    model=self.surrogates["sebo"].model,
                    ref_point=self.acqf.ref_point,
                    objective=self.acqf.objective,
                )
                self.acqf.set_X_pending(X_pending)

        homotopy = Homotopy(
            homotopy_parameters=[
                HomotopyParameter(
                    parameter=self.deterministic_model._f.a,
                    schedule=homotopy_schedule,
                )
            ],
            callbacks=[callback],
        )
        # need to know sparse dimensions
        batch_initial_conditions = get_batch_initial_conditions(
            acq_function=self.acqf,
            raw_samples=optimizer_options_with_defaults["raw_samples"],
            X_pareto=self.X_pareto,
            target_point=self.target_point,
            num_restarts=optimizer_options_with_defaults["num_restarts"],
            **{"device": self.device, "dtype": self.dtype},
        )
        candidates, expected_acquisition_value = optimize_acqf_homotopy(
            q=n,
            acq_function=self.acqf,
            bounds=bounds,
            homotopy=homotopy,
            num_restarts=optimizer_options_with_defaults["num_restarts"],
            raw_samples=optimizer_options_with_defaults["raw_samples"],
            post_processing_func=rounding_func,
            fixed_features=fixed_features,
            batch_initial_conditions=batch_initial_conditions,
        )

        return (
            candidates,
            expected_acquisition_value,
            torch.ones(n, dtype=candidates.dtype),
        )


def L1_norm_func(X: Tensor, init_point: Tensor) -> Tensor:
    r"""L1_norm takes in a a `batch_shape x n x d`-dim input tensor `X`
    to a `batch_shape x n x 1`-dimensional L1 norm tensor. To be used
    for constructing a GenericDeterministicModel.
    """
    return torch.linalg.norm((X - init_point), ord=1, dim=-1, keepdim=True)


def clamp_candidates(
    X: Tensor, target_point: Tensor, clamp_tol: float, **tkwargs: Any
) -> Tensor:
    """Clamp generated candidates within the given ranges to the target point."""
    clamp_mask = (X - target_point).abs() < clamp_tol
    clamp_mask = clamp_mask
    X[clamp_mask] = (
        target_point.clone().repeat(*X.shape[:-1], 1).to(**tkwargs)[clamp_mask]
    )
    return X


def get_batch_initial_conditions(
    acq_function: AcquisitionFunction,
    raw_samples: int,
    X_pareto: Tensor,
    target_point: Tensor,
    num_restarts: int = 20,
    **tkwargs: Any,
) -> Tensor:
    """Generate starting points for the SEBO acquisition function optimization."""
    dim = X_pareto.shape[-1]  # dimension
    # (1) Global Sobol points
    X_cand1 = SobolEngine(dimension=dim, scramble=True).draw(raw_samples).to(**tkwargs)
    X_cand1 = X_cand1[
        acq_function(X_cand1.unsqueeze(1)).topk(num_restarts // 5).indices
    ]
    # (2) Global Sobol points with a Bernoulli mask
    X_cand2 = SobolEngine(dimension=dim, scramble=True).draw(raw_samples).to(**tkwargs)
    mask = torch.rand(X_cand2.shape, **tkwargs) < 0.5
    X_cand2[mask] = target_point.repeat(len(X_cand2), 1).to(**tkwargs)[mask]
    X_cand2 = X_cand2[
        acq_function(X_cand2.unsqueeze(1)).topk(num_restarts // 5).indices
    ]
    # (3) Perturbations of points on the Pareto frontier (done by TuRBO and Spearmint)
    X_cand3 = X_pareto.clone()[torch.randint(high=len(X_pareto), size=(raw_samples,))]
    mask = X_cand3 != target_point
    X_cand3[mask] += 0.2 * torch.randn(*X_cand3.shape, **tkwargs)[mask]
    X_cand3 = torch.clamp(X_cand3, min=0.0, max=1.0)
    X_cand3 = X_cand3[
        acq_function(X_cand3.unsqueeze(1)).topk(num_restarts // 5).indices
    ]
    # (4) Apply a Bernoulli mask to points on the Pareto frontier
    X_cand4 = X_pareto.clone()[torch.randint(high=len(X_pareto), size=(raw_samples,))]
    mask = torch.rand(X_cand4.shape, **tkwargs) < 0.5
    X_cand4[mask] = target_point.repeat(len(X_cand4), 1).to(**tkwargs)[mask].clone()
    X_cand4 = X_cand4[
        acq_function(X_cand4.unsqueeze(1)).topk(num_restarts // 5).indices
    ]
    return torch.cat((X_cand1, X_cand2, X_cand3, X_cand4), dim=0).unsqueeze(1)
