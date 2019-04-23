#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Dict, List, Optional, Tuple

import torch
from ax.core.types import TConfig
from torch import Tensor


class TorchModel:
    """This class specifies the interface for a torch-based model.

    These methods should be implemented to have access to all of the features
    of Ax.
    """

    dtype: Optional[torch.dtype]
    device: Optional[torch.device]

    def fit(
        self,
        Xs: List[Tensor],
        Ys: List[Tensor],
        Yvars: List[Tensor],
        bounds: List[Tuple[float, float]],
        task_features: List[int],
        feature_names: List[str],
    ) -> None:
        """Fit model to m outcomes.

        Args:
            Xs: A list of m (k_i x d) feature tensors X. Number of rows k_i can
                vary from i=1,...,m.
            Ys: The corresponding list of m (k_i x 1) outcome tensors Y, for
                each outcome.
            Yvars: The variances of each entry in Ys, same shape.
            bounds: A list of d (lower, upper) tuples for each column of X.
            task_features: Columns of X that take integer values and should be
                treated as task parameters.
            feature_names: Names of each column of X.
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
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        pending_observations: Optional[List[Tensor]] = None,
        model_gen_options: Optional[TConfig] = None,
        rounding_func: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate new candidates.

        Args:
            n: Number of candidates to generate.
            bounds: A list of (lower, upper) tuples for each column of X.
            objective_weights: The objective is to maximize a weighted sum of
                the columns of f(x). These are the weights.
            outcome_constraints: A tuple of (A, b). For k outcome constraints
                and m outputs at f(x), A is (k x m) and b is (k x 1) such that
                A f(x) <= b.
            linear_constraints: A tuple of (A, b). For k linear constraints on
                d-dimensional x, A is (k x d) and b is (k x 1) such that
                A x <= b.
            fixed_features: A map {feature_index: value} for features that
                should be fixed to a particular value during generation.
            pending_observations:  A list of m (k_i x d) feature tensors X
                for m outcomes and k_i pending observations for outcome i.
            model_gen_options: A config dictionary that can contain
                model-specific options.

        Returns:
            2-element tuple containing

            - (n x d) tensor of generated points.
            - n-tensor of weights for each point.
        """
        raise NotImplementedError

    def best_point(
        self,
        bounds: List[Tuple[float, float]],
        objective_weights: Tensor,
        outcome_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        linear_constraints: Optional[Tuple[Tensor, Tensor]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        model_gen_options: Optional[TConfig] = None,
    ) -> Optional[Tensor]:
        """
        Identify the current best point, satisfying the constraints in the same
        format as to gen.

        Return None if no such point can be identified.

        Args:
            bounds: A list of (lower, upper) tuples for each column of X.
            objective_weights: The objective is to maximize a weighted sum of
                the columns of f(x). These are the weights.
            outcome_constraints: A tuple of (A, b). For k outcome constraints
                and m outputs at f(x), A is (k x m) and b is (k x 1) such that
                A f(x) <= b.
            linear_constraints: A tuple of (A, b). For k linear constraints on
                d-dimensional x, A is (k x d) and b is (k x 1) such that
                A x <= b.
            fixed_features: A map {feature_index: value} for features that
                should be fixed to a particular value in the best point.
            model_gen_options: A config dictionary that can contain
                model-specific options.

        Returns:
            d-tensor of the best point.
        """
        return None

    def cross_validate(
        self,
        Xs_train: List[Tensor],
        Ys_train: List[Tensor],
        Yvars_train: List[Tensor],
        X_test: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Do cross validation with the given training and test sets.

        Training set is given in the same format as to fit. Test set is given
        in the same format as to predict.

        Args:
            Xs_train: A list of m (k_i x d) feature tensors X. Number of rows
                k_i can vary from i=1,...,m.
            Ys_train: The corresponding list of m (k_i x 1) outcome tensors Y,
                for each outcome.
            Yvars_train: The variances of each entry in Ys, same shape.
            X_test: (j x d) tensor of the j points at which to make predictions.

        Returns:
            2-element tuple containing

            - (j x m) tensor of outcome predictions at X.
            - (j x m x m) tensor of predictive covariances at X.
              cov[j, m1, m2] is Cov[m1@j, m2@j].
        """
        raise NotImplementedError

    def update(self, Xs: List[Tensor], Ys: List[Tensor], Yvars: List[Tensor]) -> None:
        """Update the model with additional data.

        Args:
            Xs: Additional data for the model, in the same format as for `fit`.
            Ys: Additional data for the model, in the same format as for `fit`.
            Yvars: Additional data for the model, in the same format as for `fit`.
        """
        raise NotImplementedError
