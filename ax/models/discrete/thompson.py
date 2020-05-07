#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from ax.core.types import TConfig, TGenMetadata, TParamValue, TParamValueList
from ax.exceptions.constants import TS_MIN_WEIGHT_ERROR
from ax.exceptions.model import ModelError
from ax.models.discrete_base import DiscreteModel
from ax.utils.common.docutils import copy_doc


class ThompsonSampler(DiscreteModel):
    """Generator for Thompson sampling.

    The generator performs Thompson sampling on the data passed in via `fit`.
    Arms are given weight proportional to the probability that they are
    winners, according to Monte Carlo simulations.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        min_weight: Optional[float] = None,
        uniform_weights: bool = False,
    ) -> None:
        """
        Args:
            num_samples: The number of samples to draw from the posterior.
            min_weight: The minimum weight a arm must be
                given in order for it to be returned from the gernerator. If not
                specified, will be set to 2 / (number of arms).
            uniform_weights: If True, the arms returned from the
                generator will each be given a weight of 1 / (number of arms).
        """
        self.num_samples = num_samples
        self.min_weight = min_weight
        self.uniform_weights = uniform_weights

        self.X = None
        self.Ys = None
        self.Yvars = None
        self.X_to_Ys_and_Yvars = None

    @copy_doc(DiscreteModel.fit)
    def fit(
        self,
        Xs: List[List[TParamValueList]],
        Ys: List[List[float]],
        Yvars: List[List[float]],
        parameter_values: List[TParamValueList],
        outcome_names: List[str],
    ) -> None:
        self.X = self._fit_X(Xs=Xs)
        self.Ys, self.Yvars = self._fit_Ys_and_Yvars(
            Ys=Ys, Yvars=Yvars, outcome_names=outcome_names
        )
        self.X_to_Ys_and_Yvars = self._fit_X_to_Ys_and_Yvars(
            X=self.X, Ys=self.Ys, Yvars=self.Yvars
        )

    @copy_doc(DiscreteModel.gen)
    def gen(
        self,
        n: int,
        parameter_values: List[TParamValueList],
        objective_weights: Optional[np.ndarray],
        outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fixed_features: Optional[Dict[int, TParamValue]] = None,
        pending_observations: Optional[List[List[TParamValueList]]] = None,
        model_gen_options: Optional[TConfig] = None,
    ) -> Tuple[List[TParamValueList], List[float], TGenMetadata]:
        if objective_weights is None:
            raise ValueError("ThompsonSampler requires objective weights.")

        arms = self.X
        k = len(arms)

        weights = self._generate_weights(
            objective_weights=objective_weights, outcome_constraints=outcome_constraints
        )
        min_weight = self.min_weight if self.min_weight is not None else 2.0 / k

        # Second entry is used for tie-breaking
        weighted_arms = [
            (weights[i], np.random.random(), arms[i])
            for i in range(k)
            # pyre-fixme[6]: Expected `float` for 1st param but got `Optional[float]`.
            if weights[i] > min_weight
        ]

        if len(weighted_arms) == 0:
            raise ModelError(
                TS_MIN_WEIGHT_ERROR.format(
                    min_weight=min_weight, max_weight=max(weights)
                )
            )

        weighted_arms.sort(reverse=True)
        top_weighted_arms = weighted_arms[:n] if n > 0 else weighted_arms
        top_arms = [arm for _, _, arm in top_weighted_arms]
        top_weights = [weight for weight, _, _ in top_weighted_arms]

        if self.uniform_weights:
            top_weights = [1 / len(top_arms) for _ in top_arms]

        return top_arms, [x / sum(top_weights) for x in top_weights], {}

    @copy_doc(DiscreteModel.predict)
    def predict(self, X: List[TParamValueList]) -> Tuple[np.ndarray, np.ndarray]:
        n = len(X)  # number of parameterizations at which to make predictions
        m = len(self.Ys)  # number of outcomes
        f = np.zeros((n, m))  # array of outcome predictions
        cov = np.zeros((n, m, m))  # array of predictive covariances
        predictX = [self._hash_TParamValueList(x) for x in X]
        for i, X_to_Y_and_Yvar in enumerate(self.X_to_Ys_and_Yvars):
            # iterate through outcomes
            for j, x in enumerate(predictX):
                # iterate through parameterizations at which to make predictions
                if x not in X_to_Y_and_Yvar:
                    raise ValueError(
                        "ThompsonSampler does not support out-of-sample prediction."
                    )
                f[j, i], cov[j, i, i] = X_to_Y_and_Yvar[x]
        return f, cov

    def _generate_weights(
        self,
        objective_weights: np.ndarray,
        outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> List[float]:
        samples, fraction_all_infeasible = self._produce_samples(
            num_samples=self.num_samples,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
        )
        if fraction_all_infeasible > 0.99:
            raise ValueError(
                "Less than 1% of samples have a feasible arm. "
                "Check your outcome constraints."
            )

        num_valid_samples = samples.shape[1]
        while num_valid_samples < self.num_samples:
            num_additional_samples = (self.num_samples - num_valid_samples) / (
                1 - fraction_all_infeasible
            )
            num_additional_samples = int(np.maximum(num_additional_samples, 100))
            new_samples, _ = self._produce_samples(
                num_samples=num_additional_samples,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
            )
            samples = np.concatenate([samples, new_samples], axis=1)
            num_valid_samples = samples.shape[1]

        winner_indices = np.argmax(samples, axis=0)  # (num_samples,)
        winner_counts = np.zeros(len(self.X))  # (k,)
        for index in winner_indices:
            winner_counts[index] += 1
        weights = winner_counts / winner_counts.sum()
        return weights.tolist()

    def _produce_samples(
        self,
        num_samples: int,
        objective_weights: np.ndarray,
        outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, float]:
        k = len(self.X)
        samples_per_metric = np.zeros(
            (k, num_samples, len(self.Ys))
        )  # k x num_samples x m
        for i, Y in enumerate(self.Ys):  # (k x 1)
            Yvar = self.Yvars[i]  # (k x 1)
            cov = np.diag(Yvar)  # (k x k)
            samples = np.random.multivariate_normal(
                Y, cov, num_samples
            ).T  # (k x num_samples)
            samples_per_metric[:, :, i] = samples

        any_violation = np.zeros((k, num_samples), dtype=bool)  # (k x num_samples)
        if outcome_constraints:
            # A is (num_constraints x m)
            # b is (num_constraints x 1)
            A, b = outcome_constraints

            # (k x num_samples x m) dot (num_constraints x m)^T
            # = (k x num_samples x m) dot (m x num_constraints)
            # ==> (k x num_samples x num_constraints)
            constraint_values = np.dot(samples_per_metric, A.T)
            violations = constraint_values > b.T
            any_violation = np.max(violations, axis=2)  # (k x num_samples)

        objective_values = np.dot(
            samples_per_metric, objective_weights
        )  # (k x num_samples)
        objective_values[any_violation] = -np.Inf
        best_arm = objective_values.max(axis=0)  # (num_samples,)
        all_arms_infeasible = best_arm == -np.Inf  # (num_samples,)
        fraction_all_infeasible = all_arms_infeasible.mean()
        filtered_objective = objective_values[:, ~all_arms_infeasible]  # (k x ?)
        return filtered_objective, fraction_all_infeasible

    def _validate_Xs(self, Xs: List[List[TParamValueList]]) -> None:
        """
        1. Require that all Xs have the same arms, i.e. we have observed
        all arms for all metrics. If so, we can safely use Xs[0] exclusively.
        2. Require that all rows of X are unique, i.e. only one observation
        per parameterization.
        """
        if not all(x == Xs[0] for x in Xs[1:]):
            raise ValueError(
                "ThompsonSampler requires that all elements of Xs are identical; "
                "i.e. that we have observed all arms for all metrics."
            )

        X = Xs[0]
        uniqueX = {self._hash_TParamValueList(x) for x in X}
        if len(uniqueX) != len(X):
            raise ValueError(
                "ThompsonSampler requires all rows of X to be unique; "
                "i.e. that there is only one observation per parameterization."
            )

    def _fit_X(self, Xs: List[List[TParamValueList]]) -> List[TParamValueList]:
        """After validation has been performed, it's safe to use Xs[0]."""
        self._validate_Xs(Xs=Xs)
        return Xs[0]

    def _fit_Ys_and_Yvars(
        self, Ys: List[List[float]], Yvars: List[List[float]], outcome_names: List[str]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """For plain Thompson Sampling, there's nothing to be done here.
        EmpiricalBayesThompsonSampler will overwrite this method to perform
        shrinkage.
        """
        return Ys, Yvars

    def _fit_X_to_Ys_and_Yvars(
        self, X: List[TParamValueList], Ys: List[List[float]], Yvars: List[List[float]]
    ) -> List[Dict[TParamValueList, Tuple[float, float]]]:
        """Construct lists of mappings, one per outcome, of parameterizations
        to the a tuple of their mean and variance.
        """
        X_to_Ys_and_Yvars = []
        hashableX = [self._hash_TParamValueList(x) for x in X]
        for (Y, Yvar) in zip(Ys, Yvars):
            X_to_Ys_and_Yvars.append(dict(zip(hashableX, zip(Y, Yvar))))
        return X_to_Ys_and_Yvars

    def _hash_TParamValueList(self, x: TParamValueList) -> str:
        """Hash a list of parameter values. This is safer than converting the
        list to a tuple because of int/floats.
        """
        param_values_str = json.dumps(x)
        return hashlib.md5(param_values_str.encode("utf-8")).hexdigest()
