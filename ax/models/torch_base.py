#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from ax.core.metric import Metric
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata
from ax.models.base import Model as BaseModel
from ax.models.types import TConfig
from botorch.acquisition.risk_measures import RiskMeasureMCObjective
from botorch.utils.datasets import SupervisedDataset
from torch import Tensor


@dataclass
class TorchOptConfig:
    """Container for lightweight representation of optimization arguments.

    This is used for communicating between modelbridge and models. This is
    an ephemeral object and not meant to be stored / serialized.

    Attributes:
        objective_weights: If doing multi-objective optimization, these denote
            which objectives should be maximized and which should be minimized.
            Otherwise, the objective is to maximize a weighted sum of
            the columns of f(x). These are the weights.
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
            model-specific options.
        rounding_func: A function that rounds an optimization result
            appropriately (i.e., according to `round-trip` transformations).
        opt_config_metrics: A dictionary of metrics that are included in
            the optimization config.
        is_moo: A boolean denoting whether this is for an MOO problem.
        risk_measure: An optional risk measure, used for robust optimization.
    """

    objective_weights: Tensor
    outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None
    objective_thresholds: Optional[Tensor] = None
    linear_constraints: Optional[Tuple[Tensor, Tensor]] = None
    fixed_features: Optional[Dict[int, float]] = None
    pending_observations: Optional[List[Tensor]] = None
    model_gen_options: TConfig = field(default_factory=dict)
    rounding_func: Optional[Callable[[Tensor], Tensor]] = None
    opt_config_metrics: Optional[Dict[str, Metric]] = None
    is_moo: bool = False
    risk_measure: Optional[RiskMeasureMCObjective] = None


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
    gen_metadata: Dict[str, Any] = field(default_factory=dict)
    candidate_metadata: Optional[List[TCandidateMetadata]] = None


class TorchModel(BaseModel):
    """This class specifies the interface for a torch-based model.

    These methods should be implemented to have access to all of the features
    of Ax.
    """

    dtype: Optional[torch.dtype] = None
    device: Optional[torch.device] = None
    _supports_robust_optimization: bool = False

    def fit(
        self,
        datasets: List[SupervisedDataset],
        metric_names: List[str],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
    ) -> None:
        """Fit model to m outcomes.

        Args:
            datasets: A list of ``SupervisedDataset`` containers, each
                corresponding to the data of one metric (outcome).
            metric_names: A list of metric names, with the i-th metric
                corresponding to the i-th dataset.
            search_space_digest: A ``SearchSpaceDigest`` object containing
                metadata on the features in the datasets.
            candidate_metadata: Model-produced metadata for candidates, in
                the order corresponding to the Xs.
        """
        pass

    def predict(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict

        Args:
            X: (j x d) tensor of the j points at which to make predictions.

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
    ) -> Optional[Tensor]:
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
        datasets: List[SupervisedDataset],
        metric_names: List[str],
        X_test: Tensor,
        search_space_digest: SearchSpaceDigest,
    ) -> Tuple[Tensor, Tensor]:
        """Do cross validation with the given training and test sets.

        Training set is given in the same format as to fit. Test set is given
        in the same format as to predict.

        Args:
            datasets: A list of ``SupervisedDataset`` containers, each
                corresponding to the data of one metric (outcome).
            metric_names: A list of metric names, with the i-th metric
                corresponding to the i-th dataset.
            X_test: (j x d) tensor of the j points at which to make predictions.
            search_space_digest: A SearchSpaceDigest object containing
                metadata on the features in X.

        Returns:
            2-element tuple containing

            - (j x m) tensor of outcome predictions at X.
            - (j x m x m) tensor of predictive covariances at X.
              cov[j, m1, m2] is Cov[m1@j, m2@j].
        """
        raise NotImplementedError

    def update(
        self,
        datasets: List[Optional[SupervisedDataset]],
        metric_names: List[str],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
    ) -> None:
        """Update the model.

        Updating the model requires both existing and additional data.
        The data passed into this method will become the new training data.

        Args:
            datasets: A list of ``SupervisedDataset`` containers, each
                corresponding to the data of one metric (outcome). `None`
                means that there is no additional data for the corresponding
                outcome.
            metric_names: A list of metric names, with the i-th metric
                corresponding to the i-th dataset.
            search_space_digest: A SearchSpaceDigest object containing
                metadata on the features in X.
            candidate_metadata: Model-produced metadata for candidates, in
                the order corresponding to the Xs.
        """
        raise NotImplementedError

    def evaluate_acquisition_function(
        self,
        X: Tensor,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        acq_options: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """Evaluate the acquisition function on the candidate set `X`.

        Args:
            X: (j x d) tensor of the j points at which to evaluate the acquisition
                function.
            search_space_digest: A dataclass used to compactly represent a search space.
            torch_opt_config: A TorchOptConfig object containing optimization
                arguments (e.g., objective weights, constraints).
            acq_options: Keyword arguments used to contruct the acquisition function.

        Returns:
            A single-element tensor with the acquisition value for these points.
        """
        raise NotImplementedError  # pragma: nocover
