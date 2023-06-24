# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Mapping, Optional, Protocol

import numpy as np
from scipy.stats import fisher_exact, norm, pearsonr, spearmanr

"""
################################ Model Fit Metrics ###############################
"""


class ModelFitMetricProtocol(Protocol):
    """Structural type for model fit metrics."""

    @staticmethod
    def __call__(y_obs: np.ndarray, y_pred: np.ndarray, se_pred: np.ndarray) -> float:
        pass  # pragma: no cover  # pyre-ignore[7]


def compute_model_fit_metrics(
    y_obs: Mapping[str, np.ndarray],
    y_pred: Mapping[str, np.ndarray],
    se_pred: Mapping[str, np.ndarray],
    fit_metrics_dict: Mapping[str, ModelFitMetricProtocol],
) -> Dict[str, Dict[str, float]]:
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
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    se_pred: Optional[np.ndarray] = None,
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
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    se_pred: np.ndarray,
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
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    se_pred: np.ndarray,
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


def _mean_prediction_ci(
    y_obs: np.ndarray, y_pred: np.ndarray, se_pred: np.ndarray
) -> float:
    # Pyre does not allow float * np.ndarray.
    return float(np.mean(1.96 * 2 * se_pred / np.abs(y_obs)))


def _log_likelihood(
    y_obs: np.ndarray, y_pred: np.ndarray, se_pred: np.ndarray
) -> float:
    return float(np.sum(norm.logpdf(y_obs, loc=y_pred, scale=se_pred)))


def _mape(y_obs: np.ndarray, y_pred: np.ndarray, se_pred: np.ndarray) -> float:
    """Mean absolute predictive error"""
    return float(np.mean(np.abs((y_pred - y_obs) / y_obs)))


def _total_raw_effect(
    y_obs: np.ndarray, y_pred: np.ndarray, se_pred: np.ndarray
) -> float:
    min_y_obs = np.min(y_obs)
    return float((np.max(y_obs) - min_y_obs) / min_y_obs)


def _correlation_coefficient(
    y_obs: np.ndarray, y_pred: np.ndarray, se_pred: np.ndarray
) -> float:
    with np.errstate(invalid="ignore"):
        rho, _ = pearsonr(y_pred, y_obs)
    return float(rho)


def _rank_correlation(
    y_obs: np.ndarray, y_pred: np.ndarray, se_pred: np.ndarray
) -> float:
    with np.errstate(invalid="ignore"):
        rho, _ = spearmanr(y_pred, y_obs)
    return float(rho)


def _fisher_exact_test_p(
    y_obs: np.ndarray, y_pred: np.ndarray, se_pred: np.ndarray
) -> float:
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
