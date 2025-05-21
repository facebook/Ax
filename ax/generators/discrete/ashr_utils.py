#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
from ax.exceptions.core import UserInputError
from scipy.stats import norm


class Ashr:
    r"""
    An empirical Bayes model for estimating the effect sizes. Given the observations
    Y_i and their variances Yvar_i, Ashr estimates the underlying effect sizes mu_i
    by placing a mixture of Gaussians prior on the effect sizes. The prior consists of
    a point mass at zero and a set of centered Gaussians with specified variaces.
    The mixture proportions in the prior as well as the variances of the Gaussians
    in the mixture are learned based on observed outcome data.
    Methodology is based on the paper: False discovery rates: a new deal by
    M. Stephens https://academic.oup.com/biostatistics/article/18/2/275/2557030.
    """

    def __init__(
        self,
        Y: npt.NDArray,
        Yvar: npt.NDArray,
        prior_vars: npt.NDArray | None = None,
        eb_grid_param: float = 2.0,
    ) -> None:
        r"""
        Args:
            Y: A length n array denoting the observed treatment effects.
            Yvar: A length n array denoting the variances
                of the observed values.
            prior_vars: A length k array denoting the variances
                of normal distributions in the mixture of Gaussians prior.
                In case of None, the variances are estimated based on data
                using the provided grid parameter.
            eb_grid_param: A grid parameter for estimating the prior variances
                based on data in case none were given.
        """
        self.Y: npt.NDArray = Y
        self.Yvar: npt.NDArray = Yvar
        if prior_vars is None:
            prior_stds = prior_grid(Y=Y, Yvar=Yvar, grid_param=eb_grid_param)
            prior_vars = prior_stds**2
        self.prior_vars: npt.NDArray = prior_vars
        self.ll: npt.NDArray = marginal_densities(Y=Y, Yvar=Yvar, prior_vars=prior_vars)

    def posterior(self, w: npt.NDArray) -> GaussianMixture:
        r"""
        The posterior for mu_i can be calculated via the following rules.
        For the normal prior mu_i~N(0, sigma_k^2) and
        likelihood hat{mu}_i~N(mu_i, s_i^2),
        posterior for mu_i is N(sigma_1^2*hat{mu}_i/s_i*2, sigma_1^2),
        where sigma_1^2=sigma_k^2*s_i^2/(sigma_k^2+s_i^2).

        Args:
            w: (n,k) dim matrix containing the weights of the posterior mixture.

        Returns:
            Mixutre of Gaussians distribution.
        """

        # (n, k) matrix of normal variances
        # a variance per mixture component and per observation
        normal_vars_by_class = np.divide(
            np.multiply.outer(self.Yvar, self.prior_vars),
            np.add.outer(self.Yvar, self.prior_vars),
        )
        # (n, k) matrix of normal standard deviations
        # a mean per mixture component and per observation
        normal_means_by_class = normal_vars_by_class * (self.Y / self.Yvar)[:, None]

        return GaussianMixture(
            normal_means_by_class=normal_means_by_class,
            normal_vars_by_class=normal_vars_by_class,
            weights=w,
        )

    def fit(
        self,
        lambdas: npt.NDArray | None = None,
        threshold: float = 10e-4,
        nsteps: int = 1000,
    ) -> dict[str, npt.NDArray]:
        r"""
        Fit Ashr to estimate prior proportions pi, posterior weights and lfdr.

        Args:
            lambdas: A length k array of penalty levels corresponding to each of
                k classes with entry equal to one meaning no penalty
                for the corresponding class.
            thereshold: The threshold used in the EM stoping rule.
                If the difference between two consecutive estimates of
                weights in the EM algorithm is smaller than the threshold,
                the algorithm stops.
            nsteps: The maximum number of steps in the EM algorithm.

        Returns:
            A dict containing
            - weights: (n, k)-dim estimated weight matrix for computing posteriors,
            - pi: parameter estimates: (n, k) dim matrix of proportions, and
            - lfdr: length n local False Discovery Rate (FDR) sequence. lfdr equals
                the posterior probability of true parameter being zero.
        """
        k = len(self.prior_vars)  # total number of classes
        if lambdas is None:
            lambdas = np.ones(k)  # no penalty
        if len(lambdas) != k:
            raise ValueError(
                "The length of the penalty sequence should be the number of "
                "prior classes."
            )
        lambdas = lambdas.astype(np.float64)

        results = fit_ashr_em(
            ll=self.ll, lambdas=lambdas, threshold=threshold, nsteps=nsteps
        )

        return results


class GaussianMixture:
    r"""
    A weighted mixure of Gaussian distributions. The class computes the mean,
    standard errors and tails of each of n random variables
    from a mixture of k Gaussians.
    The Gaussians in the mixture are allowed to be degenerate,
    i.e., have zero variance. This is used in the Ashr model since one of the
    distributions in the prior mixture is a point mass at zero.
    """

    def __init__(
        self,
        normal_means_by_class: npt.NDArray,
        normal_vars_by_class: npt.NDArray,
        weights: npt.NDArray,
    ) -> None:
        r"""
        Args:
            normal_means_by_class: (n, k)-dim matrix of normal means for
                each of the classes. Each of the n random variables comes from
                a mixture of k normal distributions.
            normal_var_by_class: (n, k)-dim matrix of normal variances
                for each of the k classes. The first column is all zeros,
                the variance of the null class.
            weights: (n, k)-dim matrix of weights on individual distributions
                per prior class.
        """
        self.normal_means_by_class = normal_means_by_class
        self.normal_vars_by_class = normal_vars_by_class
        self.weights = weights

    def tail_probabilities(
        self, left_tail: bool = True, threshold: float = 0.0
    ) -> npt.NDArray:
        r"""
        Args:
            left_tail: An indicator for the tail probability to calculate.
                Note that neither tail includes null class.

            threshold: For left tail, the returned value measures probability of the
                effect being less than the threshold. For right tail, it is the
                probability of the effect being larger than the threshold.

        Returns:
            Length n array of tail probabilities for each of n rvs.
        """
        tails_by_class = np.zeros_like(self.normal_means_by_class)

        # normal left tails
        for i in range(tails_by_class.shape[0]):
            for j in range(tails_by_class.shape[1]):
                tails_by_class[i, j] = (
                    norm.cdf(
                        -np.divide(
                            self.normal_means_by_class[i, j] - threshold,
                            np.sqrt(self.normal_vars_by_class[i, j]),
                        )
                    )
                    if self.normal_vars_by_class[i, j] > 0
                    else 0.0
                )
                # correcting the normal tails in case of a right tail
                if left_tail is False and self.normal_vars_by_class[i, j] > 0:
                    tails_by_class[i, j] = 1.0 - tails_by_class[i, j]

        return np.multiply(tails_by_class, self.weights).sum(axis=1)

    @property
    def means(self) -> npt.NDArray:
        r"""
        Returns:
            Length n array of final means for each rv.
        """
        return np.multiply(self.weights, self.normal_means_by_class).sum(axis=1)

    @property
    def vars(self) -> npt.NDArray:
        r""" "
        Returns:
            Length n array of final standard deviations for each effect.
        """
        # standard errors of the mixture distributions
        # https://en.wikipedia.org/wiki/Mixture_distribution#Moments
        return (
            np.multiply(
                self.weights,
                self.normal_means_by_class**2 + self.normal_vars_by_class,
            ).sum(axis=1)
            - self.means**2
        )


def prior_grid(
    Y: npt.NDArray, Yvar: npt.NDArray, grid_param: float = 2.0
) -> npt.NDArray:
    r"""
    Produces the grid of standard deviations for each of the Gaussians in the prior
    mixture based on the observed data.

    Args:
        Y: A length n array of the observed treatment effects.
        Yvar: A length n array of observed variances of the above observations.
        grid_param: A grid parameter. Default 2.0 recommended in the paper to control
            the number of Gaussians in the mixture.

    Returns:
        A length n array of the standard deviations of the centered Gaussians in the
        mixture of Gaussians prior.
    """
    m = np.sqrt(grid_param)
    sigma_lower = np.min(np.sqrt(Yvar)) / 10.0
    sigma_upper = np.max(Y**2 - Yvar)
    sigma_upper = 2 * np.sqrt(sigma_upper) if sigma_upper > 0 else 8 * sigma_lower
    max_power = int(np.ceil(np.log(sigma_upper / sigma_lower) / np.log(m)))
    return np.array([0] + [sigma_lower * (m**power) for power in range(max_power + 1)])


def marginal_densities(
    Y: npt.NDArray,
    Yvar: npt.NDArray,
    prior_vars: npt.NDArray,
) -> npt.NDArray:
    r"""
    Evaluates marginal densities for each observed statistics and each prior class.

    Args:
        Y: A length n array denoting the observed treatment effects.
        Yvar: A length n array denoting the standard variances
            of the observed values.
        prior_vars: A length k array denoting the variances
            of prior classes.

    Returns:
        (n, k) dim matrix ll, where ll_{jk} is marginal density of j-th statistics
        eval at its observed value, assuming prior is coming from k-th class.
    """
    k = len(prior_vars)  # total number of classes
    n = len(Y)  # total number of observations
    if prior_vars[0] != 0:
        raise UserInputError(
            "Ashr prior consists of a mixture of Gaussians where the "
            "first Gaussian in the prior mixture should be a point mass at zero. \
            This degenerate Gaussian represents the prior on the effects being null."
        )
    ll = np.zeros((n, k), dtype=np.float64)

    # marginal densities when prior is mass at zero
    ll[:, 0] = norm.pdf(Y, loc=0, scale=np.sqrt(Yvar))

    for i in range(1, k):
        ll[:, i] = norm.pdf(
            Y,
            loc=0.0,
            scale=np.sqrt(prior_vars[i] + Yvar),
        )
    return ll


def compute_weights(ll: npt.NDArray, pi: npt.NDArray) -> npt.NDArray:
    r"""
    Compute posterior weights based on marginal densities and prior probabilities.

    Args:
        ll: (n,k)-dim matrix of marginal densities eval at each observation,
        pi: length k vector of prior mixture proportions.

    Returns:
        (n,k)-dim matrix of weights.
    """
    # multiply each row of ll with the corresponding element of pi vector
    w = np.multiply(ll, pi)
    # divide weights by the sum across each row
    w = w / w.sum(axis=1, keepdims=True)
    return w


def fit_ashr_em(
    ll: npt.NDArray,
    lambdas: npt.NDArray,
    threshold: float = 10e-4,
    nsteps: int = 1000,
) -> dict[str, npt.NDArray]:
    r"""
    Estimating proportions and posterior weights via an
    Expectation Maximization (EM) algorithm.

    Args:
        ll: (n,k)-dim matrix of marginal densities eval at each observation,
            marginalizing over the true effects.
        lambdas: A length k array of penalty levels.
        thereshold: If the difference between two consecutive estimates of
            weights is smaller than the threshold, the algorithm stops.
        nsteps: The maximum number of steps in the EM algorithm.

    Returns:
        A dictionary containing:
        - weights: (n,k)-dim matrix of weights,
        - pi: length k vector of estimates of prior mixture proportions, and
       - lfdr: length n local False Discovery Rate (FDR) sequence. lfdr equals
            the posterior probability of true parameter being zero.
    """
    n, k = ll.shape
    # initializing pi vector
    if k - 1 < n:
        pi = np.ones(k) / n
        pi[0] = 1 - (k - 1) / n
    else:
        pi = np.ones(k) / k

    w = np.zeros_like(ll)

    for _ in range(nsteps):
        # E-step: compute weight matrix w; size of w: (n, k)
        w = compute_weights(ll=ll, pi=pi)
        # M-step: update pi
        ns = w.sum(axis=0).squeeze() + lambdas - 1.0  # length k
        pi_new = ns / ns.sum()  # length k
        if sum(abs(pi - pi_new)) <= threshold:
            w = compute_weights(ll=ll, pi=pi_new)
            return {"weights": w, "pi": pi_new, "lfdr": w[:, 0]}
        pi = pi_new

    warnings.warn("EM did not converge.", stacklevel=2)
    return {"weights": w, "pi": pi, "lfdr": w[:, 0]}
