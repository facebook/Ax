#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import warnings
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from ax.core.types import TGenMetadata, TParamValue, TParamValueList
from ax.models.discrete.ashr_utils import Ashr, GaussianMixture
from ax.models.discrete.thompson import ThompsonSampler
from ax.models.discrete_base import DiscreteModel
from ax.models.types import TConfig
from ax.utils.common.docutils import copy_doc
from pyre_extensions import none_throws


class EBAshr(ThompsonSampler):
    """
    Determines feasible arms and sorts them by the objective
    improvements. Feasibile arms are required to have probability
    of feasibility with respect to all metrics in objective and constraints
    above 1-regression_prob_threshold, otherwise they are deemed infeasible.
    These probabilities are computed via the discrete emprirical Bayes Ashr model.
    Shrinkage: Ashr model provides shrinkage to metric outcomes across arms
    with model fitted separately for each metric.
    Prior: Ashr model uses a prior consisting of mixture of Gaussian
    distributions, all of which are centered at zero. The number of Gaussians
    in the prior and their variances are chosen based on observed data to cover
    the whole data range. One of the variances is chosen to be zero so that
    one of the Gaussians in the prior mixture is a point mass at zero.
    The mixture proportions in the prior are learned based on observed outcome
    data (Ys) and their standard errors (Yvars) via standard empirical Bayes
    methodology. The method operates in the outcome space and it doesn't use
    the space of features (Xs).
    """

    def __init__(
        self,
        eb_penalty_param: float = 1.0,
        eb_nsteps: int = 1000,
        eb_threshold: float = 10e-4,
        eb_grid_param: float = 2,
        min_variance_threshold: float = 10e-16,
    ) -> None:
        r"""
        Args:
            eb_penalty_param: Penalty parameter, part of Ashr model log-likelihood,
                represents the amount of shrinkage towards zero. Penalty parameter takes
                real values no smaller than 1. The default 1 means there will be regular
                shrinkage from the Bayes model and the prior but not additional
                shrinkage. Parameter value above one encourages overestimation of the
                null (zero) class so there is additional shrinkage. Recommended
                parameter range is [1, 10].
            eb_nsteps: The number of steps in Ashr fitting EM algorithm.
            eb_threshold: The precision threshold in Ashr fitting EM algorithm.
            eb_grid_param: The grid parameter affecting the number of Gaaussians in the
                prior mixture for Ashr prior distribution.
            min_variance_thresold: All observed test groups/arms with variance below
                this threshold will be considered fixed variables with zero variance.
        """
        self.X: Sequence[Sequence[TParamValue]] | None = None
        self.Ys: Sequence[Sequence[float]] | None = None
        self.Yvars: Sequence[Sequence[float]] | None = None
        self.X_to_Ys_and_Yvars: (
            list[dict[TParamValueList, tuple[float, float]]] | None
        ) = None
        # for each metric, posterior_feasibility gives
        # posterior probabilities of feasibility across all arms
        self.posterior_feasibility: (
            list[Callable[[float, float], npt.NDArray]] | None
        ) = None
        self.eb_penalty_param: float = eb_penalty_param
        self.eb_nsteps: int = eb_nsteps
        self.eb_threshold: float = eb_threshold
        self.eb_grid_param: float = eb_grid_param
        self.min_variance_threshold: float = min_variance_threshold

    def _fit_Ys_and_Yvars(
        self,
        Ys: Sequence[Sequence[float]],
        Yvars: Sequence[Sequence[float]],
        outcome_names: Sequence[str],
    ) -> tuple[list[list[float]], list[list[float]]]:
        r"""
        Args:
            Ys: List of length m of measured metrics across k arms. The outcomes
                are assumed to be relativized with respect to status quo.
            Yvars: List of length m of measured metrics variances across arms.
            outcome_names: List of m metric names.

        Returns: A tuple containing
            - shrinken outcome estimates and
            - their variances.
        """
        newYs = []
        newYvars = []
        self.posterior_feasibility = []

        for Y, Yvar in zip(Ys, Yvars, strict=True):
            newY: npt.NDArray = np.array(Y, dtype=float)
            newYvar = np.array(Yvar, dtype=float)

            # excluding arms with zero variance, e.g. status quo
            nonconstant_rvs = np.abs(newYvar) > self.min_variance_threshold
            # Case where the standard deviations are not infinitesimal, so we do
            # shrinkage.
            if nonconstant_rvs.any():
                # Ashr model applied to arms with non-zero variance
                stats = newY[nonconstant_rvs]
                variances = newYvar[nonconstant_rvs]

                # setting up the Ashr model
                model = Ashr(Y=stats, Yvar=variances, eb_grid_param=self.eb_grid_param)
                k = len(model.prior_vars)
                # run Ashr fitting procedure
                model_fit = model.fit(
                    lambdas=np.array([self.eb_penalty_param] + [1.0] * (k - 1)),
                    nsteps=self.eb_nsteps,
                    threshold=self.eb_threshold,
                )

                # Ashr model posterior
                posterior = model.posterior(model_fit["weights"])
                newY[nonconstant_rvs] = posterior.means
                newYvar[nonconstant_rvs] = posterior.vars

                f_posterior_feas: Callable[[float, float], npt.NDArray] = (
                    functools.partial(
                        posterior_feasibility_util,
                        posterior=posterior,
                        nonconstant_rvs=nonconstant_rvs,
                        Y=newY,
                    )
                )
            else:
                f_posterior_feas: Callable[[float, float], npt.NDArray] = (
                    functools.partial(no_model_feasibility_util, Y=newY)
                )

            none_throws(self.posterior_feasibility).append(f_posterior_feas)

            newYs.append(list(newY))
            newYvars.append(list(newYvar))

        return newYs, newYvars

    def _check_metric_direction(
        self,
        objective_weights: npt.NDArray,
        outcome_constraints: tuple[npt.NDArray, npt.NDArray] | None = None,
    ) -> npt.NDArray:
        r"""
        Args:
            objective_weights: A length m array corresponding to the metrics weights in
                a maximization problem. A positive weight means the corresponding metric
                succeeding in the positive direction. A negative weight means
                the corresponding metric succeeding in the negative direction.
                Outcomes that are modeled but not part of the objective get objective
                weight 0.
            outcome_constraints: A tuple of (A, b), where A is of size
                (num of constraints x m), m is the number of outputs/metrics,
                and b is of size (num of constraints x 1).

        Returns:
            A boolean array of length m indicating the direction of success
            for each metric.
        """
        # metrics in the objective
        upper_is_better = objective_weights > 0

        # constraints
        if outcome_constraints:
            A, b = outcome_constraints
            upper_is_better_constraints = -np.apply_along_axis(
                np.sum, 0, A
            )  # column-wise (arm-wise)
            objectives = objective_weights != 0
            upper_is_better[~objectives] = upper_is_better_constraints[~objectives]

        return upper_is_better

    def _get_regression_indicator(
        self,
        objective_weights: npt.NDArray,
        outcome_constraints: tuple[npt.NDArray, npt.NDArray] | None = None,
        regression_prob_threshold: float = 0.90,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        r"""
        Args:
            regression_prob_threshold: Arms having a regression probability with respect
                to any metric (in either objective or constraints) above a given
                threshold are marked as regressions.

        Returns: A tuple containing:
            - A length k array containing regression indicators for each of the k arms.
            - (num of arms, num of metrics) matrix of arm, metric pairs, (i, j)-entry
            corresponds to the probability of infeasibility or i-th arm and j-th metric.
        """
        num_metrics = len(none_throws(self.Ys))
        num_arms = len(none_throws(self.Ys)[0])
        prob_infeasibility = np.zeros(
            (num_arms, num_metrics), dtype=float
        )  # (num of arms) x (num of metrics)

        upper_is_better = self._check_metric_direction(
            objective_weights=objective_weights, outcome_constraints=outcome_constraints
        )
        if np.sum(abs(objective_weights) > 0) > 1:
            warnings.warn(
                "In case of multi-objective adding metric values together might"
                " not lead to a meaningful result.",
                stacklevel=2,
            )
        # compute probabilities of feasibility for metrics in the objective
        # equals regression probabilities in this case
        for i in range(num_metrics):
            if objective_weights[i] != 0.0:
                lb = 0.0 if upper_is_better[i] else -np.inf
                ub = np.inf if upper_is_better[i] else 0.0
                prob_infeasibility[:, i] = 1.0 - none_throws(
                    self.posterior_feasibility
                )[i](lb, ub)  # per i-th metric, across arms

        # compute probabilities of feasibility for constraints
        if outcome_constraints:
            A, b = outcome_constraints

            if np.any(
                np.apply_along_axis(lambda x: np.linalg.norm(x, ord=0) > 1, 1, A)
            ):
                raise ValueError(
                    "Only one metric per constraint allowed. Scalarized "
                    " OutcomeConstraint with multiple metrics per constraint not "
                    "supported."
                )

            for i in range(A.shape[1]):
                if np.linalg.norm(A[:, i], ord=1) == 0:
                    continue
                upper = A[:, i] > 0
                lower = A[:, i] < 0
                ub = np.min(b[upper] / A[upper, i]) if np.any(upper) else np.inf
                lb = np.max(b[lower] / A[lower, i]) if np.any(lower) else -np.inf
                prob_infeasibility[:, i] = 1.0 - none_throws(
                    self.posterior_feasibility
                )[i](
                    lb, ub
                )  # per i-th metric, across arms; probability a metric strictly
                # smaller than lb or strictly larger than ub

        # for each arm check if it is infeasible with respect to any metric
        regressions = np.apply_along_axis(
            np.any, 1, prob_infeasibility >= regression_prob_threshold
        )

        return regressions, prob_infeasibility

    def _get_success_measurement(self, objective_weights: npt.NDArray) -> npt.NDArray:
        r"""
        Returns:
            A length k array returning a measure of success for each of k arms.
        """

        posterior_means = np.array(none_throws(self.Ys)).T
        success = np.dot(posterior_means, objective_weights)
        return success

    @copy_doc(DiscreteModel.gen)
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
        if objective_weights is None:
            raise ValueError("EB Ashr model requires objective weights.")

        model_gen_options = model_gen_options or {}
        regression_prob_threshold = model_gen_options.get(
            "regression_prob_threshold", 0.90
        )
        if not isinstance(regression_prob_threshold, float):
            raise TypeError(
                "`regression_prob_threshold` is required among `model_gen_kwargs` \
                and must be set to a float."
            )

        # prob_infeasibility: matrix containing probabilities of infeasibilities
        # for each arm, metric pair
        regression, prob_infeasibility = self._get_regression_indicator(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            regression_prob_threshold=regression_prob_threshold,
        )
        success = self._get_success_measurement(objective_weights=objective_weights)

        arms = none_throws(self.X)
        sorted_arms = sorted(
            zip(arms, list(success), list(regression), np.arange(len(arms))),
            key=lambda x: x[1] if not x[2] else -1.0,
            reverse=True,
        )

        top_arms = [sorted_arms[i][0] for i in range(n)]

        return (
            top_arms,
            [1.0] * len(top_arms),  # uniform weights
            {
                "regression": regression,
                "success": success,
                "prob_infeasibility": prob_infeasibility,
                "best_x": sorted_arms[0][0],
            },
        )


def posterior_feasibility_util(
    lb: float,
    ub: float,
    posterior: GaussianMixture,
    nonconstant_rvs: npt.NDArray,
    Y: npt.NDArray,
) -> npt.NDArray:
    r"""
    Probabilities of a posterior rv being inside the given interval [lb, up]
    with bounds included per single metric across arms.

    Args:
        lb: Lower bound on the random variable.
        ub: Upper bound on the random variable.
        posterior: Posterior classs computing posterior distribution for all
            non-degenerate (non-constant) arms.
        nonconstant_rvs: Length k boolean vector indicating which variables
            are constants.
        Y: Length k outcome vector.

    Returns:
        Length k array of probabilities of feasibility across k arms.
    """
    # Pr(m>ub)
    # TODO: jmarkovic simplify this directly in Ashr
    upper_tail = (
        posterior.tail_probabilities(left_tail=False, threshold=ub)
        + posterior.weights[:, 0] * (ub < 0)
        if ub < np.inf
        else 0
    )
    # Pr(m<lb)
    lower_tail = (
        posterior.tail_probabilities(left_tail=True, threshold=lb)
        + posterior.weights[:, 0] * (lb > 0)
        if lb > -np.inf
        else 0
    )

    prob_feas = np.zeros_like(nonconstant_rvs, dtype=float)
    # pyre-fixme[58]: `-` is not supported for operand types `float` and
    #  `Union[np.ndarray[typing.Any, np.dtype[typing.Any]], int]`.
    prob_feas[nonconstant_rvs] = 1.0 - upper_tail - lower_tail
    prob_feas[~nonconstant_rvs] = (Y[~nonconstant_rvs] >= lb) & (
        Y[~nonconstant_rvs] <= ub
    )
    return prob_feas


def no_model_feasibility_util(lb: float, ub: float, Y: npt.NDArray) -> npt.NDArray:
    r"""
    Probabilities (0 or 1) of whether Y is inside [lb, ub].

    This covers the degenerate case when all arms have zero variance, and
    `posterior_feasibility_util` is not usable because there is no posterior.
    """
    return (Y >= lb) & (Y <= ub)
