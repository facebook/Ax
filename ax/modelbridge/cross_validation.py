#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from functools import partial

from logging import Logger
from numbers import Number
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import numpy as np
from ax.core.observation import Observation, ObservationData
from ax.core.optimization_config import OptimizationConfig
from ax.modelbridge.base import ModelBridge
from ax.utils.common.logger import get_logger
from scipy.stats import fisher_exact, norm, pearsonr, spearmanr

logger: Logger = get_logger(__name__)

CVDiagnostics = Dict[str, Dict[str, float]]

MEAN_PREDICTION_CI = "Mean prediction CI"
MAPE = "MAPE"
TOTAL_RAW_EFFECT = "Total raw effect"
CORRELATION_COEFFICIENT = "Correlation coefficient"
RANK_CORRELATION = "Rank correlation"
FISHER_EXACT_TEST_P = "Fisher exact test p"
LOG_LIKELIHOOD = "Log likelihood"


class CVResult(NamedTuple):
    """Container for cross validation results."""

    observed: Observation
    predicted: ObservationData


class AssessModelFitResult(NamedTuple):
    """Container for model fit assessment results"""

    good_fit_metrics_to_fisher_score: Dict[str, float]
    bad_fit_metrics_to_fisher_score: Dict[str, float]


def cross_validate(
    model: ModelBridge,
    folds: int = -1,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    test_selector: Optional[Callable] = None,
) -> List[CVResult]:
    """Cross validation for model predictions.

    Splits the model's training data into train/test folds and makes
    out-of-sample predictions on the test folds.

    Train/test splits are made based on arm names, so that repeated
    observations of a arm will always be in the train or test set
    together.

    The test set can be limited to a specific set of observations by passing in
    a test_selector callable. This function should take in an Observation
    and return a boolean indiciating if it should be used in the test set or
    not. For example, we can limit the test set to arms with trial 0 with
    test_selector = lambda obs: obs.features.trial_index == 0
    If not provided, all observations will be available for the test set.

    Args:
        model: Fitted model (ModelBridge) to cross validate.
        folds: Number of folds. Use -1 for leave-one-out, otherwise will be
            k-fold.
        test_selector: Function for selecting observations for the test set.

    Returns:
        A CVResult for each observation in the training data.
    """
    # Get in-design training points
    training_data = [
        obs
        for i, obs in enumerate(model.get_training_data())
        if model.training_in_design[i]
    ]
    arm_names = {obs.arm_name for obs in training_data}
    n = len(arm_names)
    if folds > n:
        raise ValueError(f"Training data only has {n} arms, which is less than folds")
    elif n == 0:
        raise ValueError(
            f"{model.__class__.__name__} has no training data.  Either it has been "
            "incorrectly initialized or should not be cross validated."
        )
    elif folds < 2 and folds != -1:
        raise ValueError("Folds must be -1 for LOO, or > 1.")
    elif folds == -1:
        folds = n

    arm_names_rnd = np.array(list(arm_names))
    np.random.shuffle(arm_names_rnd)
    result = []
    for train_names, test_names in _gen_train_test_split(
        folds=folds, arm_names=arm_names_rnd
    ):
        # Construct train/test data
        cv_training_data = []
        cv_test_data = []
        cv_test_points = []
        for obs in training_data:
            if obs.arm_name in train_names:
                cv_training_data.append(obs)
            elif obs.arm_name in test_names and (
                test_selector is None or test_selector(obs)
            ):
                cv_test_points.append(obs.features)
                cv_test_data.append(obs)
        if len(cv_test_points) == 0:
            continue
        # Make the prediction
        cv_test_predictions = model.cross_validate(
            cv_training_data=cv_training_data, cv_test_points=cv_test_points
        )
        # Form CVResult objects
        for i, obs in enumerate(cv_test_data):
            result.append(CVResult(observed=obs, predicted=cv_test_predictions[i]))
    return result


def cross_validate_by_trial(model: ModelBridge, trial: int = -1) -> List[CVResult]:
    """Cross validation for model predictions on a particular trial.

    Uses all of the data up until the specified trial to predict each of the
    arms that was launched in that trial. Defaults to the last trial.

    Args:
        model: Fitted model (ModelBridge) to cross validate.
        trial: Trial for which predictions are evaluated.

    Returns:
        A CVResult for each observation in the training data.
    """
    # Get in-design training points
    training_data = [
        obs
        for i, obs in enumerate(model.get_training_data())
        if model.training_in_design[i]
    ]
    all_trials = {
        int(d.features.trial_index)
        for d in training_data
        if d.features.trial_index is not None
    }
    if len(all_trials) < 2:
        raise ValueError(f"Training data has fewer than 2 trials ({all_trials})")
    if trial < 0:
        trial = max(all_trials)
    elif trial not in all_trials:
        raise ValueError(f"Trial {trial} not found in training data")
    # Construct train/test data
    cv_training_data = []
    cv_test_data = []
    cv_test_points = []
    for obs in training_data:
        if obs.features.trial_index is None:
            continue
        elif obs.features.trial_index < trial:
            cv_training_data.append(obs)
        elif obs.features.trial_index == trial:
            cv_test_points.append(obs.features)
            cv_test_data.append(obs)
    # Make the prediction
    cv_test_predictions = model.cross_validate(
        cv_training_data=cv_training_data, cv_test_points=cv_test_points
    )
    # Form CVResult objects
    result = [
        CVResult(observed=obs, predicted=cv_test_predictions[i])
        for i, obs in enumerate(cv_test_data)
    ]
    return result


def compute_diagnostics(result: List[CVResult]) -> CVDiagnostics:
    """Computes diagnostics for given cross validation results.

    It provides a dictionary with values for the following diagnostics, for
    each metric:

    - 'Mean prediction CI': the average width of the CIs at each of the CV
      predictions, relative to the observed mean.
    - 'MAPE': mean absolute percentage error of the estimated mean relative
      to the observed mean.
    - 'Total raw effect': the multiple change from the smallest observed
      mean to the largest observed mean, i.e. `(max - min) / min`.
    - 'Correlation coefficient': the Pearson correlation of the estimated
      and observed means.
    - 'Rank correlation': the Spearman correlation of the estimated
      and observed means.
    - 'Fisher exact test p': we test if the model is able to distinguish the
      bottom half of the observations from the top half, using Fisher's
      exact test and the observed/estimated means. A low p value indicates
      that the model has some ability to identify good arms. A high p value
      indicates that the model cannot identify arms better than chance, or
      that the observations are too noisy to be able to tell.

    Each of these is returned as a dictionary from metric name to value for
    that metric.

    Args:
        result: Output of cross_validate

    Returns:
        A dictionary keyed by diagnostic name with results as described above.
    """
    # Extract per-metric outcomes from CVResults.
    y_obs = defaultdict(list)
    y_pred = defaultdict(list)
    se_pred = defaultdict(list)
    for res in result:
        for j, metric_name in enumerate(res.observed.data.metric_names):
            y_obs[metric_name].append(res.observed.data.means[j])
            # Find the matching prediction
            k = res.predicted.metric_names.index(metric_name)
            y_pred[metric_name].append(res.predicted.means[k])
            se_pred[metric_name].append(np.sqrt(res.predicted.covariance[k, k]))

    diagnostic_fns = {
        MEAN_PREDICTION_CI: _mean_prediction_ci,
        MAPE: _mape,
        TOTAL_RAW_EFFECT: _total_raw_effect,
        CORRELATION_COEFFICIENT: _correlation_coefficient,
        RANK_CORRELATION: _rank_correlation,
        FISHER_EXACT_TEST_P: _fisher_exact_test_p,
        LOG_LIKELIHOOD: _log_likelihood,
    }

    diagnostics: Dict[str, Dict[str, float]] = defaultdict(dict)
    # Get all per-metric diagnostics.
    for metric_name in y_obs:
        for name, fn in diagnostic_fns.items():
            diagnostics[name][metric_name] = fn(
                y_obs=np.array(y_obs[metric_name]),
                y_pred=np.array(y_pred[metric_name]),
                se_pred=np.array(se_pred[metric_name]),
            )
    return diagnostics


def assess_model_fit(
    diagnostics: CVDiagnostics,
    significance_level: float = 0.1,
) -> AssessModelFitResult:
    """Assess model fit for given diagnostics results.

    It determines if a model fit is good or bad based on Fisher exact test p

    Args:
        diagnostics: Output of compute_diagnostics

    Returns:
        Two dictionaries, one for good metrics, one for bad metrics, each
        mapping metric name to p-value
    """

    good_fit_metrics_to_fisher_score: Dict[str, float] = {}
    bad_fit_metrics_to_fisher_score: Dict[str, float] = {}

    for metric, score in diagnostics[FISHER_EXACT_TEST_P].items():
        if score > significance_level:
            bad_fit_metrics_to_fisher_score[metric] = score
        else:
            good_fit_metrics_to_fisher_score[metric] = score
    if len(bad_fit_metrics_to_fisher_score) > 0:
        logger.warning(
            "{} {} {} unable to be reliably fit.".format(
                ("Metrics" if len(bad_fit_metrics_to_fisher_score) > 1 else "Metric"),
                (" , ".join(bad_fit_metrics_to_fisher_score.keys())),
                ("were" if len(bad_fit_metrics_to_fisher_score) > 1 else "was"),
            )
        )
    return AssessModelFitResult(
        good_fit_metrics_to_fisher_score=good_fit_metrics_to_fisher_score,
        bad_fit_metrics_to_fisher_score=bad_fit_metrics_to_fisher_score,
    )


def has_good_opt_config_model_fit(
    optimization_config: OptimizationConfig,
    assess_model_fit_result: AssessModelFitResult,
) -> bool:
    """Assess model fit for given diagnostics results across the optimization
    config metrics

    Bad fit criteria: Any objective metrics are poorly fit based on
    the Fisher exact test p (see assess_model_fit())

    TODO[]: Incl. outcome constraints in assessment

    Args:
        optimization_config: Objective/Outcome constraint metrics to assess
        diagnostics: Output of compute_diagnostics

    Returns:
        Two dictionaries, one for good metrics, one for bad metrics, each
        mapping metric name to p-value
    """

    # Bad fit criteria: Any objective metrics are poorly fit
    # TODO[]: Incl. outcome constraints in assessment
    has_good_opt_config_fit = all(
        (m.name in assess_model_fit_result.good_fit_metrics_to_fisher_score)
        for m in optimization_config.objective.metrics
    )
    return has_good_opt_config_fit


def _gen_train_test_split(
    folds: int, arm_names: np.ndarray
) -> Iterable[Tuple[Set[str], Set[str]]]:
    """Return train/test splits of arm names.

    Args:
        folds: Number of folds to return
        arm_names: Array of arm names

    Returns:
        Yields (train, test) tuple of arm names.
    """
    n = len(arm_names)
    test_size = n // folds  # The size of all test sets but the last
    final_size = test_size + (n - folds * test_size)  # Grab the leftovers
    for fold in range(folds):
        # We will take the test set from the back of the array.
        # Roll the list of arm names to get a fresh test set
        arm_names = np.roll(arm_names, test_size)
        n_test = test_size if fold < folds - 1 else final_size
        yield set(arm_names[:-n_test]), set(arm_names[-n_test:])


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


class BestModelSelector(ABC):
    @abstractmethod
    def best_diagnostic(self, diagnostics: List[CVDiagnostics]) -> int:
        """
        Return the index of the best diagnostic.
        """
        pass  # pragma: no cover


class CallableEnum(Enum):
    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def __call__(self, *args: Optional[Any], **kwargs: Optional[Any]) -> Any:
        return self.value(*args, **kwargs)


class MetricAggregation(CallableEnum):
    MEAN: Callable[[Iterable[Number]], Number] = partial(np.mean)


class DiagnosticCriterion(CallableEnum):
    MIN: Callable[[Iterable[Number]], Number] = partial(np.amin)


class SingleDiagnosticBestModelSelector(BestModelSelector):
    """Choose the best model using a single cross-validation diagnostic.

    The input is a list of CVDiagnostics, each corresponding to one model.
    The specified diagnostic is extracted from each of the CVDiagnostics,
    its values (each of which corresponds to a separate metric) are
    aggregated with the aggregation function, the best one is determined
    with the criterion, and the index of the best diagnostic result is returned.


     Example:

     ::
        s = SingleDiagnosticBestModelSelector(
            diagnostic = 'Fisher exact test p',
            criterion = DiagnosticCriterion.MIN,
            metric_aggregation = MetricAggregation.MEAN,
        )
        best_diagnostic_index = s.best_diagnostic(diagnostics)

    Args:
         diagnostic (str): The name of the diagnostic to use, which should be
             a key in CVDiagnostic.
         metric_aggregation (MetricAggregation): Callable
            applied to the values of the diagnostic for a single model to
            produce a single number.
         criterion (DiagnosticCriterion): Callable used
            to determine which of the (aggregated) diagnostics is the best.


     Returns:
         int: index of the selected best diagnostic.

    """

    def __init__(
        self,
        diagnostic: str,
        metric_aggregation: MetricAggregation,
        criterion: DiagnosticCriterion,
    ) -> None:
        self.diagnostic = diagnostic
        self.metric_aggregation = metric_aggregation
        self.criterion = criterion

    def best_diagnostic(self, diagnostics: List[CVDiagnostics]) -> int:
        aggregated_diagnostic_values = [
            self.metric_aggregation(list(d[self.diagnostic].values()))
            for d in diagnostics
        ]
        best_diagnostic = self.criterion(aggregated_diagnostic_values)
        return aggregated_diagnostic_values.index(best_diagnostic)
