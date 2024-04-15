#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from functools import partial

from logging import Logger
from numbers import Number
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import numpy as np
from ax.core.experiment import Experiment
from ax.core.observation import Observation, ObservationData, recombine_observations
from ax.core.optimization_config import OptimizationConfig
from ax.modelbridge.base import ModelBridge, unwrap_observation_data
from ax.utils.common.logger import get_logger

from ax.utils.stats.model_fit_stats import (
    _correlation_coefficient,
    _fisher_exact_test_p,
    _log_likelihood,
    _mape,
    _mean_prediction_ci,
    _mse,
    _rank_correlation,
    _total_raw_effect,
    _wmape,
    coefficient_of_determination,
    compute_model_fit_metrics,
    mean_of_the_standardized_error,
    ModelFitMetricProtocol,
    std_of_the_standardized_error,
)
from botorch.exceptions.warnings import InputDataWarning

logger: Logger = get_logger(__name__)

CVDiagnostics = Dict[str, Dict[str, float]]

MEAN_PREDICTION_CI = "Mean prediction CI"
MAPE = "MAPE"
wMAPE = "wMAPE"
TOTAL_RAW_EFFECT = "Total raw effect"
CORRELATION_COEFFICIENT = "Correlation coefficient"
RANK_CORRELATION = "Rank correlation"
FISHER_EXACT_TEST_P = "Fisher exact test p"
LOG_LIKELIHOOD = "Log likelihood"
MSE = "MSE"


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
    untransform: bool = True,
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
        untransform: Whether to untransform the model predictions before
            cross validating.
            Models are trained on transformed data, and candidate generation
            is performed in the transformed space. Computing the model
            quality metric based on the cross-validation results in the
            untransformed space may not be representative of the model that
            is actually used for candidate generation in case of non-invertible
            transforms, e.g., Winsorize or LogY.
            While the model in the transformed space may not be representative
            of the original data in regions where outliers have been removed,
            we have found it to better reflect the how good the model used
            for candidate generation actually is.
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
    # Not necessary to shuffle when using LOO, avoids differences in floating point
    # computations making equality tests brittle.
    if folds != -1:
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
        if untransform:
            cv_test_predictions = model.cross_validate(
                cv_training_data=cv_training_data, cv_test_points=cv_test_points
            )
        else:
            # Get test predictions in transformed space
            (
                cv_training_data,
                cv_test_points,
                search_space,
            ) = model._transform_inputs_for_cv(
                cv_training_data=cv_training_data, cv_test_points=cv_test_points
            )
            with warnings.catch_warnings():
                # Since each CV fold removes points from the training data, the
                # remaining observations will not pass the standardization test.
                # To avoid confusing users with this warning, we filter it out.
                warnings.filterwarnings(
                    "ignore",
                    message="Input data is not standardized.",
                    category=InputDataWarning,
                )
                cv_test_predictions = model._cross_validate(
                    search_space=search_space,
                    cv_training_data=cv_training_data,
                    cv_test_points=cv_test_points,
                )
            # Get test observations in transformed space
            cv_test_data = deepcopy(cv_test_data)
            for t in model.transforms.values():
                cv_test_data = t.transform_observations(cv_test_data)
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
    - 'wMAPE': Weighted mean absolute percentage error.
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
    y_obs = _arrayify_dict_values(y_obs)
    y_pred = _arrayify_dict_values(y_pred)
    se_pred = _arrayify_dict_values(se_pred)

    diagnostic_fns = {
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
    diagnostics = compute_model_fit_metrics(
        y_obs=y_obs, y_pred=y_pred, se_pred=se_pred, fit_metrics_dict=diagnostic_fns
    )
    return diagnostics


def _arrayify_dict_values(d: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
    """Helper to convert dictionary values to numpy arrays."""
    return {k: np.array(v) for k, v in d.items()}


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


class BestModelSelector(ABC):
    @abstractmethod
    def best_diagnostic(self, diagnostics: List[CVDiagnostics]) -> int:
        """
        Return the index of the best diagnostic.
        """
        pass


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


"""
############################## Model Fit Metrics Utils ##############################
"""


def compute_model_fit_metrics_from_modelbridge(
    model_bridge: ModelBridge,
    experiment: Experiment,
    fit_metrics_dict: Optional[Dict[str, ModelFitMetricProtocol]] = None,
    generalization: bool = False,
    untransform: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Computes the model fit metrics given a ModelBridge and an Experiment.

    Args:
        model_bridge: The ModelBridge for which to compute the model fit metrics.
        experiment: The experiment with whose data to compute the metrics if
            generalization == False. Otherwise, the data is taken from the ModelBridge.
        fit_metrics_dict: An optional dictionary with model fit metric functions,
            i.e. a ModelFitMetricProtocol, as values and their names as keys.
        generalization: Boolean indicating whether to compute the generalization
            metrics on cross-validation data or on the training data. The latter
            helps diagnose problems with model training, rather than generalization.
        untransform: Boolean indicating whether to untransform model predictions
            before calcualting the model fit metrics. False by default as models
            are trained in transformed space and model fit should be
            evaluated in transformed space.

    Returns:
        A nested dictionary mapping from the *model fit* metric names and the
        *experimental metric* names to the values of the model fit metrics.

        Example for an imaginary AutoML experiment that seeks to minimize the test
        error after training an expensive model, with respect to hyper-parameters:

        ```
        model_fit_dict = compute_model_fit_metrics_from_modelbridge(model_bridge, exp)
        model_fit_dict["coefficient_of_determination"]["test error"] =
            `coefficient of determination of the test error predictions`
        ```
    """
    predict_func = (
        _predict_on_cross_validation_data
        if generalization
        else _predict_on_training_data
    )
    y_obs, y_pred, se_pred = predict_func(
        model_bridge=model_bridge, untransform=untransform
    )
    if fit_metrics_dict is None:
        fit_metrics_dict = {
            "coefficient_of_determination": coefficient_of_determination,
            "mean_of_the_standardized_error": mean_of_the_standardized_error,
            "std_of_the_standardized_error": std_of_the_standardized_error,
        }

    return compute_model_fit_metrics(
        y_obs=y_obs,
        y_pred=y_pred,
        se_pred=se_pred,
        fit_metrics_dict=fit_metrics_dict,
    )


def _predict_on_training_data(
    model_bridge: ModelBridge,
    untransform: bool = False,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    """Makes predictions on the training data of a given experiment using a ModelBridge
    and returning the observed values, and the corresponding predictive means and
    predictive standard deviations of the model, in transformed space.

    NOTE: This is a helper function for `compute_model_fit_metrics_from_modelbridge`.

    Args:
        model_bridge: A ModelBridge object with which to make predictions.
        untransform: Boolean indicating whether to untransform model predictions.

    Returns:
        A tuple containing three dictionaries for 1) observed metric values, and the
        model's associated 2) predictive means and 3) predictive standard deviations.
    """
    observations = model_bridge.get_training_data()  # List[Observation]

    # NOTE: the following up to the end of the untransform block could be replaced
    # with model_bridge's public predict / private _batch_predict method, if the
    # latter had a boolean untransform flag.

    # Transform observations -- this will transform both obs data and features
    for t in model_bridge.transforms.values():
        observations = t.transform_observations(observations)

    observation_features = [obs.features for obs in observations]

    # Make predictions in transformed space
    observation_data_pred = model_bridge._predict(observation_features)

    if untransform:
        # Apply reverse transforms, in reverse order
        pred_observations = recombine_observations(
            observation_features=observation_features,
            observation_data=observation_data_pred,
        )
        for t in reversed(list(model_bridge.transforms.values())):
            pred_observations = t.untransform_observations(pred_observations)

        observation_data_pred = [obs.data for obs in pred_observations]

    mean_predicted, cov_predicted = unwrap_observation_data(observation_data_pred)
    mean_observed = [
        obs.data.means_dict for obs in observations
    ]  # List[Dict[str, float]]

    metric_names = observations[0].data.metric_names
    mean_observed = _list_of_dicts_to_dict_of_lists(
        list_of_dicts=mean_observed, keys=metric_names
    )
    # converting dictionary values to arrays
    mean_observed = {k: np.array(v) for k, v in mean_observed.items()}
    mean_predicted = {k: np.array(v) for k, v in mean_predicted.items()}
    std_predicted = {m: np.sqrt(np.array(cov_predicted[m][m])) for m in cov_predicted}
    return mean_observed, mean_predicted, std_predicted


def _predict_on_cross_validation_data(
    model_bridge: ModelBridge,
    untransform: bool = False,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    """Makes leave-one-out cross-validation predictions on the training data of the
    ModelBridge and returns the observed values, and the corresponding predictive means
    and predictive standard deviations of the model as numpy arrays,
    in transformed space.

    NOTE: This is a helper function for `compute_model_fit_metrics_from_modelbridge`.

    Args:
        model_bridge: A ModelBridge object with which to make predictions.
        untransform: Boolean indicating whether to untransform model predictions
            before cross validating. False by default.

    Returns:
        A tuple containing three dictionaries, each mapping metric_name to:
            1. observed metric values,
            2. LOOCV predicted mean at each observed point, and
            3. LOOCV predicted standard deviation at each observed point.
    """
    # IDEA: could use cross_validate_by_trial on the last few trials, since the
    # cross-validation performance on these is likely more correlated with the
    # performance of the model on an upcoming trial due to the sequential nature
    # of the data generated by BO.
    cv = cross_validate(model=model_bridge, untransform=untransform)

    metric_names = cv[0].observed.data.metric_names
    mean_observed = {k: [] for k in metric_names}
    mean_predicted = {k: [] for k in metric_names}
    std_predicted = {k: [] for k in metric_names}

    for cvi in cv:
        obs = cvi.observed.data
        for k, v in zip(obs.metric_names, obs.means):
            mean_observed[k].append(v)

        pred = cvi.predicted
        for k, v in zip(pred.metric_names, pred.means):
            mean_predicted[k].append(v)

        pred_se = np.sqrt(pred.covariance.diagonal().clip(0))
        for k, v in zip(pred.metric_names, pred_se):
            std_predicted[k].append(v)

    mean_observed = {k: np.array(v) for k, v in mean_observed.items()}
    mean_predicted = {k: np.array(v) for k, v in mean_predicted.items()}
    std_predicted = {k: np.array(v) for k, v in std_predicted.items()}
    return mean_observed, mean_predicted, std_predicted


def _list_of_dicts_to_dict_of_lists(
    list_of_dicts: List[Dict[str, float]], keys: List[str]
) -> Dict[str, List[float]]:
    """Converts a list of dicts indexed by a string to a dict of lists."""
    return {key: [d[key] for d in list_of_dicts] for key in keys}
