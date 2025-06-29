#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import hashlib
import json
import warnings
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from ax.core.types import TGenMetadata, TParamValue, TParamValueList
from ax.exceptions.constants import TS_MIN_WEIGHT_ERROR, TS_NO_FEASIBLE_ARMS_ERROR
from ax.exceptions.core import AxWarning, UnsupportedError
from ax.exceptions.model import ModelError
from ax.generators.discrete_base import DiscreteGenerator
from ax.generators.types import TConfig
from ax.utils.common.docutils import copy_doc
from pyre_extensions import assert_is_instance, none_throws


class ThompsonSampler(DiscreteGenerator):
    """Generator for Thompson sampling.

    The generator performs Thompson sampling on the data passed in via `fit`.
    Arms are given weight proportional to the probability that they are
    winners, according to Monte Carlo simulations.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        min_weight: float | None = None,
        uniform_weights: bool = False,
        topk: int = 1,
    ) -> None:
        """
        Args:
            num_samples: The number of samples to draw from the posterior.
            min_weight: The minimum weight a arm must be
                given in order for it to be returned from the gernerator. If not
                specified, will be set to 2 / (number of arms).
            uniform_weights: If True, the arms returned from the
                generator will each be given a weight of 1 / (number of arms).
            topk : Number of “top” arms to count in each posterior sample when
                estimating selection probabilities.

                - `topk=1` yields standard Thompson sampling: each draw
                contributes 1 count to only the single best arm.
                - `topk=2` approximates the top-two Thompson sampling (TTTS)
                batch strategy with mixing parameter beta=0.5, by counting both
                the best and runner-up in each draw and then normalizing.
                - More generally, `topk=k` counts the k highest-valued arms per
                draw and divides by k, which corresponds to an implicit
                assumption that we mix equally across the first through
                k-th best positions. Implicit assumption is beta=1/k.

                See Russo (2016) "Simple Bayesian Algorithms for Best Arm
                Identification" for details on TTTS.
        """
        self.num_samples = num_samples
        self.min_weight = min_weight
        self.uniform_weights = uniform_weights
        self.topk = topk

        self.X: Sequence[Sequence[TParamValue]] | None = None
        self.Ys: Sequence[Sequence[float]] | None = None
        self.Yvars: Sequence[Sequence[float]] | None = None
        self.X_to_Ys_and_Yvars: (
            list[dict[TParamValueList, tuple[float, float]]] | None
        ) = None

    @copy_doc(DiscreteGenerator.fit)
    def fit(
        self,
        Xs: Sequence[Sequence[Sequence[TParamValue]]],
        Ys: Sequence[Sequence[float]],
        Yvars: Sequence[Sequence[float]],
        parameter_values: Sequence[Sequence[TParamValue]],
        outcome_names: Sequence[str],
    ) -> None:
        self.X = self._fit_X(Xs=Xs)
        self.Ys, self.Yvars = self._fit_Ys_and_Yvars(
            Ys=Ys, Yvars=Yvars, outcome_names=outcome_names
        )
        self.X_to_Ys_and_Yvars = self._fit_X_to_Ys_and_Yvars(
            X=none_throws(self.X),
            Ys=none_throws(self.Ys),
            Yvars=none_throws(self.Yvars),
        )

    @copy_doc(DiscreteGenerator.gen)
    def gen(
        self,
        n: int,
        parameter_values: Sequence[Sequence[TParamValue]],
        objective_weights: npt.NDArray | None,
        outcome_constraints: tuple[npt.NDArray, npt.NDArray] | None = None,
        fixed_features: Mapping[int, TParamValue] | None = None,
        pending_observations: Sequence[Sequence[Sequence[TParamValue]]] | None = None,
        model_gen_options: TConfig | None = None,
    ) -> tuple[list[Sequence[TParamValue]], list[float], TGenMetadata]:
        if n <= 0:
            # TODO: use more informative error types
            raise ValueError("ThompsonSampler requires n > 0.")
        if objective_weights is None:
            raise ValueError("ThompsonSampler requires objective weights.")

        if np.sum(abs(objective_weights) > 0) > 1:
            warnings.warn(
                "In case of multi-objective adding metric values together might"
                " not lead to a meaningful result.",
                stacklevel=2,
            )

        arms = none_throws(self.X)
        k = len(arms)

        weights = self._generate_weights(
            objective_weights=objective_weights, outcome_constraints=outcome_constraints
        )
        min_weight = self.min_weight if self.min_weight is not None else 2.0 / k
        # Second entry is used for tie-breaking
        weighted_arms = [
            (weights[i], np.random.random(), arms[i])
            for i in range(k)
            if weights[i] > min_weight
        ]

        if len(weighted_arms) == 0:
            raise ModelError(
                TS_MIN_WEIGHT_ERROR.format(
                    min_weight=min_weight, max_weight=max(weights)
                )
            )

        weighted_arms.sort(reverse=True)
        top_weighted_arms = weighted_arms[:n]
        top_arms = [arm for _, _, arm in top_weighted_arms]
        top_weights = [weight for weight, _, _ in top_weighted_arms]

        # N TS arms should have total weight N
        if self.uniform_weights:
            top_weights = [1.0 for _ in top_weights]
        else:
            top_weights = [(x * n) / sum(top_weights) for x in top_weights]
        return (
            top_arms,
            top_weights,
            {
                "arms_to_weights": list(zip(arms, weights)),
                "best_x": weighted_arms[0][2],
            },
        )

    @copy_doc(DiscreteGenerator.predict)
    def predict(
        self, X: Sequence[Sequence[TParamValue]], use_posterior_predictive: bool = False
    ) -> tuple[npt.NDArray, npt.NDArray]:
        if use_posterior_predictive:
            warnings.warn(
                f"{self.__class__.__name__} does not support posterior-predictive "
                "predictions. Ignoring `use_posterior_predictive`. ",
                AxWarning,
                stacklevel=2,
            )
        n = len(X)  # number of parameterizations at which to make predictions
        m = len(none_throws(self.Ys))  # number of outcomes
        f = np.zeros((n, m))  # array of outcome predictions
        cov = np.zeros((n, m, m))  # array of predictive covariances
        predictX = [self._hash_TParamValueList(x) for x in X]
        for i, X_to_Y_and_Yvar in enumerate(none_throws(self.X_to_Ys_and_Yvars)):
            # iterate through outcomes
            for j, x in enumerate(predictX):
                # iterate through parameterizations at which to make predictions
                if x not in X_to_Y_and_Yvar:
                    raise UnsupportedError(
                        "ThompsonSampler does not support out-of-sample prediction. "
                        f"(X: {X[j]} - note that this is post-transform application)."
                    )
                f[j, i], cov[j, i, i] = X_to_Y_and_Yvar[
                    assert_is_instance(x, TParamValue)
                ]
        return f, cov

    def _generate_weights(
        self,
        objective_weights: npt.NDArray,
        outcome_constraints: tuple[npt.NDArray, npt.NDArray] | None = None,
    ) -> list[float]:
        samples, fraction_all_infeasible = self._produce_samples(
            num_samples=self.num_samples,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
        )
        if fraction_all_infeasible > 0.99:
            raise ModelError(TS_NO_FEASIBLE_ARMS_ERROR)

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

        winner_counts = np.zeros(len(none_throws(self.X)))  # (k,)
        # sort each sample, take top k, and count
        sorted_idxs = np.argsort(samples, axis=0)  # shape (num_arms, samples)
        topk_idxs = sorted_idxs[-self.topk :, :]  # shape (topk, samples)
        winner_arms, counts = np.unique(topk_idxs.flatten(), return_counts=True)
        winner_counts[winner_arms] = counts

        weights = winner_counts / winner_counts.sum()
        return weights.tolist()

    def _generate_samples_per_metric(self, num_samples: int) -> npt.NDArray:
        k = len(none_throws(self.X))
        samples_per_metric = np.zeros(
            (k, num_samples, len(none_throws(self.Ys)))
        )  # k x num_samples x m
        for i, Y in enumerate(none_throws(self.Ys)):  # (k)
            Yvar = none_throws(self.Yvars)[i]  # (k)
            cov = np.diag(Yvar)  # (k x k)
            samples = np.random.multivariate_normal(
                mean=Y, cov=cov, size=num_samples
            ).T  # (k x num_samples)
            samples_per_metric[:, :, i] = samples
        return samples_per_metric

    def _produce_samples(
        self,
        num_samples: int,
        objective_weights: npt.NDArray,
        outcome_constraints: tuple[npt.NDArray, npt.NDArray] | None,
    ) -> tuple[npt.NDArray, float]:
        k = len(none_throws(self.X))
        samples_per_metric = self._generate_samples_per_metric(num_samples=num_samples)

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
        objective_values[any_violation] = -np.inf
        best_arm = objective_values.max(axis=0)  # (num_samples,)
        all_arms_infeasible = best_arm == -np.inf  # (num_samples,)
        fraction_all_infeasible = all_arms_infeasible.mean()
        filtered_objective = objective_values[:, ~all_arms_infeasible]  # (k x ?)
        return filtered_objective, fraction_all_infeasible

    def _validate_Xs(self, Xs: Sequence[Sequence[Sequence[TParamValue]]]) -> None:
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

        if getattr(self, "topk", 1) > len(X):
            raise ModelError(
                f"ThompsonSampler `topk={self.topk}` exceeds number of arms ({len(X)})"
            )

    def _fit_X(
        self, Xs: Sequence[Sequence[Sequence[TParamValue]]]
    ) -> Sequence[Sequence[TParamValue]]:
        """After validation has been performed, it's safe to use Xs[0]."""
        self._validate_Xs(Xs=Xs)
        return Xs[0]

    def _fit_Ys_and_Yvars(
        self,
        Ys: Sequence[Sequence[float]],
        Yvars: Sequence[Sequence[float]],
        outcome_names: Sequence[str],
    ) -> tuple[Sequence[Sequence[float]], Sequence[Sequence[float]]]:
        """For plain Thompson Sampling, there's nothing to be done here.
        EmpiricalBayesThompsonSampler will overwrite this method to perform
        shrinkage.
        """
        return Ys, Yvars

    def _fit_X_to_Ys_and_Yvars(
        self,
        X: Sequence[Sequence[TParamValue]],
        Ys: Sequence[Sequence[float]],
        Yvars: Sequence[Sequence[float]],
    ) -> list[dict[TParamValueList, tuple[float, float]]]:
        """Construct lists of mappings, one per outcome, of parameterizations
        to the a tuple of their mean and variance.
        """
        X_to_Ys_and_Yvars = []
        hashableX = [self._hash_TParamValueList(x) for x in X]
        for Y, Yvar in zip(Ys, Yvars):
            X_to_Ys_and_Yvars.append(dict(zip(hashableX, zip(Y, Yvar))))
        return X_to_Ys_and_Yvars

    def _hash_TParamValueList(self, x: Iterable[TParamValue]) -> str:
        """Hash a list of parameter values. This is safer than converting the
        list to a tuple because of int/floats.
        """
        param_values_str = json.dumps(x)
        return hashlib.md5(param_values_str.encode("utf-8")).hexdigest()
