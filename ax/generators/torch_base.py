#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import torch
from ax.core.metric import Metric
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata
from ax.generators.base import Generator as BaseGenerator
from ax.generators.types import TConfig
from botorch.utils.datasets import SupervisedDataset
from torch import Tensor


@dataclass
class TorchOptConfig:
    """Container for lightweight representation of optimization arguments.

    This is used for communicating between adapter and models. This is
    an ephemeral object and not meant to be stored / serialized.

    Attributes:
        objective_weights: A 2D tensor of shape ``(n_objectives, n_outcomes)``.
            Each row corresponds to one objective, each column to one modeled
            outcome.  For single-objective optimization the tensor has one row.
            For multi-objective optimization there is one row per objective.
            The nonzero entries indicate which outcomes contribute to each
            objective and their weights / signs (positive = maximize,
            negative = minimize).
        outcome_constraints: A tuple of (A, b). For k outcome constraints
            and m outputs at f(x), A is (k x m) and b is (k x 1) such that
            A f(x) <= b.
        objective_thresholds:  A tensor containing thresholds forming a
            reference point from which to calculate pareto frontier hypervolume.
            Points that do not dominate the objective_thresholds contribute
            nothing to hypervolume.
        linear_constraints: A tuple of (A, b). For k linear constraints on
            d-dimensional x, A is (k x d) and b is (k x 1) such that
            A x <= b for feasible x.
        fixed_features: A map {feature_index: value} for features that
            should be fixed to a particular value during generation.
        pending_observations:  A list of m (k_i x d) feature tensors X
            for m outcomes and k_i pending observations for outcome i.
        model_gen_options: A config dictionary that can contain
            model-specific options. This commonly includes `optimizer_kwargs`,
            which often specifies the optimizer options to be passed to the
            optimizer while optimizing the acquisition function. These are
            generally expected to mimic the signature of `optimize_acqf`,
            though not all models may support all possible arguments and some
            models may support additional arguments that are not passed to the
            optimizer. While constructing a generation strategy, these options
            can be passed in as follows:
            >>> generator_gen_kwargs = {
            >>>     "model_gen_options": {
            >>>         "optimizer_kwargs": {
            >>>             "num_restarts": 20,
            >>>             "sequential": False,
            >>>             "options": {
            >>>                 "batch_limit: 5,
            >>>                 "maxiter": 200,
            >>>             },
            >>>         },
            >>>     },
            >>> }
        rounding_func: A function that rounds an optimization result
            appropriately (i.e., according to `round-trip` transformations).
        opt_config_metrics: A dictionary of metrics that are included in
            the optimization config.
        is_moo: Whether this is a multi-objective optimization problem.
            Inferred from ``objective_weights.shape[0] > 1``.
        outcome_mask: A 1D boolean tensor indicating which outcomes are used
            by any objective.
        pruning_target_point: A `d`-dim tensor that specifies values that irrelevant
            parameters should be set to.
    """

    objective_weights: Tensor
    outcome_constraints: tuple[Tensor, Tensor] | None = None
    objective_thresholds: Tensor | None = None
    linear_constraints: tuple[Tensor, Tensor] | None = None
    fixed_features: dict[int, float] | None = None
    pending_observations: list[Tensor] | None = None
    model_gen_options: TConfig = field(default_factory=dict)
    rounding_func: Callable[[Tensor], Tensor] | None = None
    opt_config_metrics: dict[str, Metric] = field(default_factory=dict)
    use_learned_objective: bool = False
    pruning_target_point: Tensor | None = None

    @cached_property
    def is_moo(self) -> bool:
        """Whether this is a multi-objective optimization problem.

        Inferred from the number of rows in ``objective_weights``.
        """
        return self.objective_weights.shape[0] > 1

    @cached_property
    def outcome_mask(self) -> Tensor:
        """A 1D boolean tensor of shape ``(n_outcomes,)`` that is ``True``
        where any objective has a nonzero weight."""
        return (self.objective_weights != 0).any(dim=0)


@dataclass(frozen=True)
class TorchGenResults:
    """
    points: (n x d) Tensor of generated points.
    weights: n-tensor of weights for each point.
    gen_metadata: Generation metadata
    Dictionary of model-specific metadata for the given
                generation candidates

    """

    points: Tensor  # (n x d)-dim
    weights: Tensor  # n-dim
    gen_metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_metadata: Sequence[TCandidateMetadata] | None = None


class TorchGenerator(BaseGenerator):
    """This class specifies the interface for a torch-based model.

    These methods should be implemented to have access to all of the features
    of Ax.
    """

    dtype: torch.dtype | None = None
    device: torch.device | None = None

    @property
    def can_predict(self) -> bool:
        """Whether this generator can predict outcomes for new parameterizations."""
        return True

    @property
    def can_model_in_sample(self) -> bool:
        """Whether this generator can model (e.g. apply shrinkage) on observed
        parameterizations (in this case, it needs to support calling `predict`()
        on points in the training data / provided during `fit()`)."""
        return True

    def fit(
        self,
        datasets: Sequence[SupervisedDataset],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: list[list[TCandidateMetadata]] | None = None,
    ) -> None:
        """Fit model to m outcomes.

        Args:
            datasets: A list of ``SupervisedDataset`` containers, each
                corresponding to the data of one metric (outcome).
            search_space_digest: A ``SearchSpaceDigest`` object containing
                metadata on the features in the datasets.
            candidate_metadata: Model-produced metadata for candidates, in
                the order corresponding to the Xs.
        """
        pass

    def predict(
        self, X: Tensor, use_posterior_predictive: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Predict

        Args:
            X: (j x d) tensor of the j points at which to make predictions.
            use_posterior_predictive: A boolean indicating if the predictions
                should be from the posterior predictive (i.e. including
                observation noise).
                This option is only supported by the ``BoTorchGenerator``.

        Returns:
            2-element tuple containing

            - (j x m) tensor of outcome predictions at X.
            - (j x m x m) tensor of predictive covariances at X.
              cov[j, m1, m2] is Cov[m1@j, m2@j].
        """
        raise NotImplementedError

    def gen(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
    ) -> TorchGenResults:
        """
        Generate new candidates.

        Args:
            n: Number of candidates to generate.
            search_space_digest: A SearchSpaceDigest object containing metadata
                about the search space (e.g. bounds, parameter types).
            torch_opt_config: A TorchOptConfig object containing optimization
                arguments (e.g., objective weights, constraints).

        Returns:
            A TorchGenResult container.
        """
        raise NotImplementedError

    def best_point(
        self,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
    ) -> Tensor | None:
        """
        Identify the current best point, satisfying the constraints in the same
        format as to gen.

        Return None if no such point can be identified.

        Args:
            search_space_digest: A SearchSpaceDigest object containing metadata
                about the search space (e.g. bounds, parameter types).
            torch_opt_config: A TorchOptConfig object containing optimization
                arguments (e.g., objective weights, constraints).

        Returns:
            d-tensor of the best point.
        """
        return None

    def cross_validate(
        self,
        datasets: Sequence[SupervisedDataset],
        X_test: Tensor,
        search_space_digest: SearchSpaceDigest,
        use_posterior_predictive: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Do cross validation with the given training and test sets.

        Training set is given in the same format as to fit. Test set is given
        in the same format as to predict.

        Args:
            datasets: A list of ``SupervisedDataset`` containers, each
                corresponding to the data of one metric (outcome).
            X_test: (j x d) tensor of the j points at which to make predictions.
            search_space_digest: A SearchSpaceDigest object containing
                metadata on the features in X.
            use_posterior_predictive: A boolean indicating if the predictions
                should be from the posterior predictive (i.e. including
                observation noise).

        Returns:
            2-element tuple containing

            - (j x m) tensor of outcome predictions at X.
            - (j x m x m) tensor of predictive covariances at X.
              cov[j, m1, m2] is Cov[m1@j, m2@j].
        """
        raise NotImplementedError

    def evaluate_acquisition_function(
        self,
        X: Tensor,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        acq_options: dict[str, Any] | None = None,
    ) -> Tensor:
        """Evaluate the acquisition function on the candidate set `X`.

        Args:
            X: (j x d) tensor of the j points at which to evaluate the acquisition
                function.
            search_space_digest: A dataclass used to compactly represent a search space.
            torch_opt_config: A TorchOptConfig object containing optimization
                arguments (e.g., objective weights, constraints).
            acq_options: Keyword arguments used to construct the acquisition function.

        Returns:
            A single-element tensor with the acquisition value for these points.
        """
        raise NotImplementedError  # pragma: nocover
