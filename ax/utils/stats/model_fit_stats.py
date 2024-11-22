# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections.abc import Mapping
from enum import Enum
from logging import Logger
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ax.utils.common.logger import get_logger
from scipy.stats import fisher_exact, norm, pearsonr, spearmanr
from sklearn.neighbors import KernelDensity


logger: Logger = get_logger(__name__)


DEFAULT_KDE_BANDWIDTH = 0.1  # default bandwidth for kernel density estimators
MEAN_PREDICTION_CI = "Mean prediction CI"
MAPE = "MAPE"
wMAPE = "wMAPE"
TOTAL_RAW_EFFECT = "Total raw effect"
CORRELATION_COEFFICIENT = "Correlation coefficient"
RANK_CORRELATION = "Rank correlation"
FISHER_EXACT_TEST_P = "Fisher exact test p"
LOG_LIKELIHOOD = "Log likelihood"
MSE = "MSE"


class ModelFitMetricDirection(Enum):
    """Model fit metric directions."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


"""
################################ Model Fit Metrics ###############################
"""


class ModelFitMetricProtocol(Protocol):
    """Structural type for model fit metrics."""

    def __call__(
        self,
        y_obs: npt.NDArray,
        y_pred: npt.NDArray,
        se_pred: npt.NDArray,
    ) -> float: ...


def compute_model_fit_metrics(
    y_obs: Mapping[str, npt.NDArray],
    y_pred: Mapping[str, npt.NDArray],
    se_pred: Mapping[str, npt.NDArray],
    fit_metrics_dict: Mapping[str, ModelFitMetricProtocol],
) -> dict[str, dict[str, float]]:
    """Computes the model fit metrics for each experimental metric in the input dicts.

    Args:
        y_obs: A dictionary mapping from experimental metric name to observed values.
        y_pred: A dictionary mapping from experimental metric name to predicted values.
        se_pred: A dictionary mapping from experimental metric name to predicted
            standard errors.
        fit_metrics_dict: A dictionary mapping from *model fit* metric name to a
            ModelFitMetricProtocol function that evaluates a model fit metric.

    Returns:
        A nested dictionary mapping from *model fit* and *experimental* metric names
        to their corresponding model fit metrics values.
    """
    metric_names = list(y_obs.keys())
    return {
        fit_metric_name: {
            exp_metric_name: fit_metric(
                y_obs=y_obs[exp_metric_name],
                y_pred=y_pred[exp_metric_name],
                se_pred=se_pred[exp_metric_name],
            )
            for exp_metric_name in metric_names
        }
        for fit_metric_name, fit_metric in fit_metrics_dict.items()
    }


def coefficient_of_determination(
    y_obs: npt.NDArray,
    y_pred: npt.NDArray,
    se_pred: npt.NDArray | None = None,
    eps: float = 1e-12,
) -> float:
    """Computes coefficient of determination, the proportion of variance in `y_obs`
    accounted for by predictions `y_pred`.

    Args:
        y_obs: An array of observations for a single metric.
        y_pred: An array of the predicted values corresponding to y_obs.
        se_pred: Not used, kept for API compatibility.
        eps: A small constant to add to the denominator for numerical stability.

    Returns:
        The scalar coefficient of determination, "R squared".
    """
    ss_res = ((y_obs - y_pred) ** 2).sum()
    ss_tot = ((y_obs - y_obs.mean()) ** 2).sum()
    return 1 - (ss_res / (ss_tot + eps))


def mean_of_the_standardized_error(  # i.e. standardized bias
    y_obs: npt.NDArray,
    y_pred: npt.NDArray,
    se_pred: npt.NDArray,
) -> float:
    """Computes the mean of the error standardized by the predictive standard deviation
    of the model `se_pred`. If the model makes good predictions and its uncertainty is
    quantified well, should be close to 0 and be normally distributed.

    NOTE: This assumes that `se_pred` is the predictive standard deviation of the
    *observations* of the objective `y`, not the predictive standard deviation of the
    objective `f` itself. In practice, this will matter for very noisy observations.

    Args:
        y_obs: An array of observations for a single metric.
        y_pred: An array of the predicted values corresponding to y_obs.
        se_pred: An array of the standard errors of the predicted values.

    Returns:
        The scalar mean of the standardized error.
    """
    return ((y_obs - y_pred) / se_pred).mean()


def std_of_the_standardized_error(
    y_obs: npt.NDArray,
    y_pred: npt.NDArray,
    se_pred: npt.NDArray,
) -> float:
    """Standard deviation of the error standardized by the predictive standard deviation
    of the model `se_pred`. If the uncertainty is quantified well, should be close to 1.

    NOTE: This assumes that `se_pred` is the predictive standard deviation of the
    *observations* of the objective `y`, not the predictive standard deviation of the
    objective `f` itself. In practice, this will matter for very noisy observations.

    Args:
        y_obs: An array of observations for a single metric.
        y_pred: An array of the predicted values corresponding to y_obs.
        se_pred: An array of the standard errors of the predicted values.

    Returns:
        The scalar standard deviation of the standardized error.
    """
    return ((y_obs - y_pred) / se_pred).std()


def entropy_of_observations(
    y_obs: npt.NDArray,
    y_pred: npt.NDArray,
    se_pred: npt.NDArray,
    bandwidth: float = DEFAULT_KDE_BANDWIDTH,
) -> float:
    """Computes the entropy of the observations y_obs using a kernel density estimator.
    This can be used to quantify how "clustered" the outcomes are. NOTE: y_pred and
    se_pred are not used, but are required for the API.

    Args:
        y_obs: An array of observations for a single metric.
        y_pred: Unused.
        se_pred: Unused.
        bandwidth: The kernel bandwidth. Defaults to 0.1, which is a reasonable value
            for standardized outcomes y_obs. The rank ordering of the results on a set
            of y_obs data sets is not generally sensitive to the bandwidth, if it is
            held fixed across the data sets. The absolute value of the results however
            changes significantly with the bandwidth.

    Returns:
        The scalar entropy of the observations.
    """
    if y_obs.ndim == 1:
        y_obs = y_obs[:, np.newaxis]

    # Check if standardization was applied to the observations.
    if bandwidth == DEFAULT_KDE_BANDWIDTH:
        y_std = np.std(y_obs, axis=0, ddof=1)
        if np.any(y_std < 0.5) or np.any(2.0 < y_std):  # allowing a fudge factor of 2.
            logger.warning(
                "Standardization of observations was not applied. "
                f"The default bandwidth of {DEFAULT_KDE_BANDWIDTH} is a reasonable "
                "choice if observations are standardize, but may not be otherwise."
            )
    return _entropy_via_kde(y_obs, bandwidth=bandwidth)


def _entropy_via_kde(y: npt.NDArray, bandwidth: float = DEFAULT_KDE_BANDWIDTH) -> float:
    """Computes the entropy of the kernel density estimate of the input data.

    Args:
        y: An (n x m) array of observations.
        bandwidth: The kernel bandwidth.

    Returns:
        The scalar entropy of the kernel density estimate.
    """
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(y)
    log_p = kde.score_samples(y)  # computes the log probability of each data point
    return -np.sum(np.exp(log_p) * log_p)  # compute entropy, the negated sum of p log p


def _mean_prediction_ci(
    y_obs: npt.NDArray,
    y_pred: npt.NDArray,
    se_pred: npt.NDArray,
) -> float:
    # Pyre does not allow float * np.ndarray.
    return float(np.mean(1.96 * 2 * se_pred / np.abs(y_obs)))


def _log_likelihood(
    y_obs: npt.NDArray,
    y_pred: npt.NDArray,
    se_pred: npt.NDArray,
) -> float:
    return float(np.sum(norm.logpdf(y_obs, loc=y_pred, scale=se_pred)))


def _mape(y_obs: npt.NDArray, y_pred: npt.NDArray, se_pred: npt.NDArray) -> float:
    """Mean absolute predictive error"""
    eps = np.finfo(y_obs.dtype).eps
    return float(np.mean(np.abs(y_pred - y_obs) / np.abs(y_obs).clip(min=eps)))


def _mse(y_obs: npt.NDArray, y_pred: npt.NDArray, se_pred: npt.NDArray) -> float:
    """Mean squared error"""
    return float(np.mean((y_pred - y_obs) ** 2))


def _wmape(y_obs: npt.NDArray, y_pred: npt.NDArray, se_pred: npt.NDArray) -> float:
    """Weighted mean absolute predictive error"""
    eps = np.finfo(y_obs.dtype).eps
    return float(np.sum(np.abs(y_pred - y_obs)) / np.sum(np.abs(y_obs)).clip(min=eps))


def _total_raw_effect(
    y_obs: npt.NDArray,
    y_pred: npt.NDArray,
    se_pred: npt.NDArray,
) -> float:
    min_y_obs = np.min(y_obs)
    return float((np.max(y_obs) - min_y_obs) / min_y_obs)


def _correlation_coefficient(
    y_obs: npt.NDArray,
    y_pred: npt.NDArray,
    se_pred: npt.NDArray,
) -> float:
    with np.errstate(invalid="ignore"):
        rho, _ = pearsonr(y_pred, y_obs)
    return float(rho)


def _rank_correlation(
    y_obs: npt.NDArray,
    y_pred: npt.NDArray,
    se_pred: npt.NDArray,
) -> float:
    with np.errstate(invalid="ignore"):
        rho, _ = spearmanr(y_pred, y_obs)
    return float(rho)


def _fisher_exact_test_p(
    y_obs: npt.NDArray,
    y_pred: npt.NDArray,
    se_pred: npt.NDArray,
) -> float:
    """Perform a Fisher exact test on the contingency table constructed from
    agreement/disagreement between the predicted and observed data.

    Null hypothesis: Agreement between the predicted and observed data arises consistent
    with random chance

    P-value < 0.05 indicates that the null hypothesis may be rejected at the 5% level of
    significance i.e. the model's predictive performance would be unlikely to arise
    through random chance.

    Args:
        y_obs: NumPy array of observed data of shape (n_obs,)
        y_pred: NumPy array of predicted data of shape (n_obs,)
        se_pred: NumPy array of standard errors of shape (n_obs,), not used by the
            calculation but required for interface compatbility.
    Returns:
        The p-value of the Fisher exact test on the contingency table.
    """

    if y_obs.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_obs and y_pred must be 1-dimensional.")

    n_half = len(y_obs) // 2
    top_obs = y_obs.argsort(axis=0)[-n_half:]
    top_est = y_pred.argsort(axis=0)[-n_half:]
    # Construct contingency table
    tp = len(set(top_est).intersection(top_obs))
    fp = n_half - tp
    fn = n_half - tp
    tn = (len(y_obs) - n_half) - (n_half - tp)
    table = np.array([[tp, fp], [fn, tn]])
    # Compute the test statistic
    _, p = fisher_exact(table, alternative="greater")
    return float(p)


DIAGNOSTIC_FNS: dict[str, ModelFitMetricProtocol] = {
    MEAN_PREDICTION_CI: _mean_prediction_ci,
    MAPE: _mape,
    wMAPE: _wmape,
    TOTAL_RAW_EFFECT: _total_raw_effect,
    CORRELATION_COEFFICIENT: _correlation_coefficient,
    RANK_CORRELATION: _rank_correlation,
    FISHER_EXACT_TEST_P: _fisher_exact_test_p,
    LOG_LIKELIHOOD: _log_likelihood,
    MSE: _mse,
}

DIAGNOSTIC_FN_DIRECTIONS: dict[str, ModelFitMetricDirection] = {
    MEAN_PREDICTION_CI: ModelFitMetricDirection.MINIMIZE,
    MAPE: ModelFitMetricDirection.MINIMIZE,
    wMAPE: ModelFitMetricDirection.MINIMIZE,
    TOTAL_RAW_EFFECT: ModelFitMetricDirection.MAXIMIZE,
    CORRELATION_COEFFICIENT: ModelFitMetricDirection.MAXIMIZE,
    RANK_CORRELATION: ModelFitMetricDirection.MAXIMIZE,
    FISHER_EXACT_TEST_P: ModelFitMetricDirection.MINIMIZE,
    LOG_LIKELIHOOD: ModelFitMetricDirection.MAXIMIZE,
    MSE: ModelFitMetricDirection.MINIMIZE,
}
