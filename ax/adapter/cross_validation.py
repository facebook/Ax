#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from logging import Logger
from typing import NamedTuple
from warnings import warn

import numpy as np
import numpy.typing as npt
from ax.adapter.base import Adapter, unwrap_observation_data
from ax.adapter.data_utils import ExperimentData
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.exceptions.core import UnsupportedError
from ax.exceptions.model import CrossValidationError
from ax.utils.common.logger import get_logger
from ax.utils.stats.model_fit_stats import (
    coefficient_of_determination,
    compute_model_fit_metrics,
    DIAGNOSTIC_FNS,
    FISHER_EXACT_TEST_P,
    mean_of_the_standardized_error,
    ModelFitMetricProtocol,
    std_of_the_standardized_error,
)
from botorch.settings import validate_input_scaling
from pyre_extensions import assert_is_instance

logger: Logger = get_logger(__name__)

CVDiagnostics = dict[str, dict[str, float]]


class CVResult(NamedTuple):
    """Container for cross validation results."""

    observed: Observation
    predicted: ObservationData


class CVData(NamedTuple):
    """Data for cross validation."""

    training_data: ExperimentData
    test_data: ExperimentData


class AssessModelFitResult(NamedTuple):
    """Container for model fit assessment results"""

    good_fit_metrics_to_fisher_score: dict[str, float]
    bad_fit_metrics_to_fisher_score: dict[str, float]


def cross_validate(
    model: Adapter,
    folds: int = -1,
    test_selector: Callable[[Observation], bool] | None = None,
    untransform: bool = True,
    use_posterior_predictive: bool = False,
    fold_generator: Callable[[ExperimentData], Iterable[CVData]] | None = None,
) -> list[CVResult]:
    """Cross validation for model predictions.

    Splits the model's training data into train/test folds and makes
    out-of-sample predictions on the test folds.

    By default, train/test splits are made based on arm names, so that
    repeated observations of a arm will always be in the train or test set
    together. Different behavior can be achieved by passing in a custom
    fold_generator.

    The test set can be limited to a specific set of observations by passing in
    a test_selector callable. This function should take in an Observation
    and return a boolean indicating if it should be used in the test set or
    not. For example, we can limit the test set to arms with trial 0 with
    test_selector = lambda obs: obs.features.trial_index == 0
    If not provided, all observations will be available for the test set.

    Args:
        model: Fitted model (Adapter) to cross validate.
        folds: Number of folds. Use -1 for leave-one-out, otherwise will be
            k-fold. Unless fold_generator is used to specify different behavior.
        test_selector: Function for selecting observations for the test set.
        untransform: Whether to untransform the model predictions before
            cross validating.
            Generators are trained on transformed data, and candidate generation
            is performed in the transformed space. Computing the model
            quality metric based on the cross-validation results in the
            untransformed space may not be representative of the model that
            is actually used for candidate generation in case of non-invertible
            transforms, e.g., Winsorize or LogY.
            While the model in the transformed space may not be representative
            of the original data in regions where outliers have been removed,
            we have found it to better reflect the how good the model used
            for candidate generation actually is.
        use_posterior_predictive: A boolean indicating if the predictions
            should be from the posterior predictive (i.e. including
            observation noise). Note: we should reconsider how we compute
            cross-validation and model fit metrics where there is non-
            Gaussian noise.
        fold_generator: A function that generates train/test folds in the form
            of CVData objects. Defaults to k-fold CV.

    Returns:
        A CVResult for each observation in the training data.
    """
    # Get in-design training data.
    training_data = model.get_training_data(filter_in_design=True)
    if fold_generator is None:

        def fold_generator(training_data: ExperimentData) -> Iterable[CVData]:
            return _kfold_train_test_split(folds=folds, training_data=training_data)

    result = []
    for cv_data in fold_generator(training_data):
        cv_training_data = cv_data.training_data
        cv_test_data = cv_data.test_data
        cv_test_observations = [
            obs
            for obs in cv_test_data.convert_to_list_of_observations()
            if test_selector is None or test_selector(obs)
        ]
        cv_test_points = [obs.features for obs in cv_test_observations]
        if len(cv_test_points) == 0:
            continue

        # Make the prediction
        if untransform:
            cv_test_predictions = model.cross_validate(
                cv_training_data=cv_training_data,
                cv_test_points=cv_test_points,
                use_posterior_predictive=use_posterior_predictive,
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
            # Since each CV fold removes points from the training data, the
            # remaining observations will not pass the input scaling checks.
            # To avoid confusing users with warnings, we disable these checks.
            with validate_input_scaling(False):
                cv_test_predictions = model._cross_validate(
                    search_space=search_space,
                    cv_training_data=cv_training_data,
                    cv_test_points=cv_test_points,
                    use_posterior_predictive=use_posterior_predictive,
                )
            # Get test observations in transformed space.
            for t in model.transforms.values():
                cv_test_data = t.transform_experiment_data(experiment_data=cv_test_data)
            # Re-construct the test observations with the transformed data.
            cv_test_observations = [
                obs
                for obs in cv_test_data.convert_to_list_of_observations()
                if test_selector is None or test_selector(obs)
            ]
        # Form CVResult objects
        if len(cv_test_observations) < len(cv_test_predictions):
            msg = (
                "There are fewer test observations than predictions. "
                "This can happen when transforms that reduce the number of "
                "observations are used in the Adapter used in cross validation. "
            )
            if folds == -1:
                msg += (
                    "Since this is leave-one-out cross validation, all observations "
                    "correspond to the same arm and we can utilize the first "
                    "observation in CV results."
                )
                logger.warning(msg)
            else:
                msg += (
                    "Since this is not leave-one-out cross validation, we cannot "
                    "guarantee consistency of predictions and observations. "
                    "Use leave-one-out cross validation with data reducing transforms, "
                    "or use cross validation with `untransform=True`."
                )
                raise CrossValidationError(msg)

        for observed, prediction in zip(
            cv_test_observations, cv_test_predictions, strict=False
        ):
            result.append(CVResult(observed=observed, predicted=prediction))
    return result


def compute_diagnostics(result: list[CVResult]) -> CVDiagnostics:
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
        for j, metric_signature in enumerate(res.observed.data.metric_signatures):
            y_obs[metric_signature].append(res.observed.data.means[j])
            # Find the matching prediction
            k = res.predicted.metric_signatures.index(metric_signature)
            y_pred[metric_signature].append(res.predicted.means[k])
            se_pred[metric_signature].append(np.sqrt(res.predicted.covariance[k, k]))
    y_obs = _arrayify_dict_values(y_obs)
    y_pred = _arrayify_dict_values(y_pred)
    se_pred = _arrayify_dict_values(se_pred)

    diagnostics = compute_model_fit_metrics(
        y_obs=y_obs,
        y_pred=y_pred,
        se_pred=se_pred,
        fit_metrics_dict=DIAGNOSTIC_FNS,
    )
    return diagnostics


def _arrayify_dict_values(d: dict[str, list[float]]) -> dict[str, npt.NDArray]:
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

    good_fit_metrics_to_fisher_score: dict[str, float] = {}
    bad_fit_metrics_to_fisher_score: dict[str, float] = {}

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
        (m.signature in assess_model_fit_result.good_fit_metrics_to_fisher_score)
        for m in optimization_config.objective.metrics
    )
    return has_good_opt_config_fit


def _kfold_train_test_split(
    folds: int,
    training_data: ExperimentData,
) -> Iterable[CVData]:
    """Return train/test CV splits based on arm names.

    Args:
        folds: Number of folds to return
        training_data: Training data to split

    Returns:
        Yields CVData object of train/test data.
    """
    arm_name_vals = set(training_data.arm_data.index.unique(level="arm_name"))
    n = len(arm_name_vals)
    if n < 2:
        raise UnsupportedError(
            "Cross validation requires at least two in-design arms in the training "
            f"data. Only {n} in-design arms were found."
        )
    elif folds > n:
        raise ValueError(
            f"Training data only has {n} arms, which is less than {folds} folds."
        )
    elif folds < 2 and folds != -1:
        raise ValueError("Folds must be -1 for LOO, or > 1.")
    elif folds == -1:
        folds = n

    arm_names = np.array(list(arm_name_vals))
    # Not necessary to shuffle when using LOO, avoids differences in floating point
    # computations making equality tests brittle.
    if folds != n:
        np.random.shuffle(arm_names)
    test_size = n // folds  # The size of all test sets but the last
    final_size = test_size + (n - folds * test_size)  # Grab the leftovers
    for fold in range(folds):
        # We will take the test set from the back of the array.
        # Roll the list of arm names to get a fresh test set
        arm_names = np.roll(arm_names, test_size)
        n_test = test_size if fold < folds - 1 else final_size
        train_names = set(arm_names[:-n_test])
        test_names = set(arm_names[-n_test:])
        yield CVData(
            training_data=training_data.filter_by_arm_names(arm_names=train_names),
            test_data=training_data.filter_by_arm_names(arm_names=test_names),
        )


def gen_trial_split(
    training_data: ExperimentData,
    test_trials: list[int],
    train_trials: list[int] | None = None,
) -> Iterable[CVData]:
    """Return a single train/test CV split based on trial index.

    Args:
        training_data: Training data to split
        test_trials: List of trial indices to use as test data

    Returns:
        A single CVData object of train/test data.
    """
    if len(test_trials) == 0:
        raise ValueError("No test trials provided.")
    all_trials = training_data.arm_data.index.get_level_values("trial_index")
    if set(test_trials) - set(all_trials):
        raise ValueError(
            f"Trials {test_trials} not all in training data trials {all_trials}."
        )
    if train_trials is None:
        train_trials = list(set(all_trials) - set(test_trials))
    else:
        if set(train_trials).intersection(set(test_trials)):
            raise ValueError("Test and train trials overlap.")
    if len(train_trials) == 0:
        raise ValueError(f"All trials in data, {all_trials}, specified as test trials.")
    logger.debug(f"Using trials {train_trials} for training.")
    train_data = training_data.filter_by_trial_index(trial_indices=train_trials)
    test_data = training_data.filter_by_trial_index(trial_indices=test_trials)
    return [CVData(training_data=train_data, test_data=test_data)]


"""
############################## Model Fit Metrics Utils ##############################
"""


def get_fit_and_std_quality_and_generalization_dict(
    fitted_adapter: Adapter,
) -> dict[str, float | None]:
    """
    Get stats and gen from a fitted Adapter for analytics purposes.
    """
    try:
        model_fit_dict = compute_model_fit_metrics_from_adapter(
            adapter=fitted_adapter,
            generalization=False,
            untransform=False,
        )
        # similar for uncertainty quantification, but distance from 1 matters
        std = list(model_fit_dict["std_of_the_standardized_error"].values())

        # generalization metrics
        model_gen_dict = compute_model_fit_metrics_from_adapter(
            adapter=fitted_adapter,
            generalization=True,
            untransform=False,
        )
        gen_std = list(model_gen_dict["std_of_the_standardized_error"].values())
        return {
            "model_fit_quality": _model_fit_metric(model_fit_dict),
            "model_std_quality": _model_std_quality(np.array(std)),
            "model_fit_generalization": _model_fit_metric(model_gen_dict),
            "model_std_generalization": _model_std_quality(np.array(gen_std)),
        }

    # Do not warn if the Adapter does not implement a predict method
    # (ex. RandomAdapter).
    except NotImplementedError:
        return {
            "model_fit_quality": None,
            "model_std_quality": None,
            "model_fit_generalization": None,
            "model_std_generalization": None,
        }

    except Exception as e:
        warn("Encountered exception in computing model fit quality: " + str(e))
        return {
            "model_fit_quality": None,
            "model_std_quality": None,
            "model_fit_generalization": None,
            "model_std_generalization": None,
        }


def compute_model_fit_metrics_from_adapter(
    adapter: Adapter,
    fit_metrics_dict: dict[str, ModelFitMetricProtocol] | None = None,
    generalization: bool = False,
    untransform: bool = False,
) -> dict[str, dict[str, float]]:
    """Computes the model fit metrics given a Adapter and an Experiment.

    Args:
        adapter: The Adapter for which to compute the model fit metrics.
        experiment: The experiment with whose data to compute the metrics if
            generalization == False. Otherwise, the data is taken from the Adapter.
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
        model_fit_dict = compute_model_fit_metrics_from_adapter(adapter, exp)
        model_fit_dict["coefficient_of_determination"]["test error"] =
            `coefficient of determination of the test error predictions`
        ```
    """
    predict_func = (
        _predict_on_cross_validation_data
        if generalization
        else _predict_on_training_data
    )
    y_obs, y_pred, se_pred = predict_func(adapter=adapter, untransform=untransform)
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


def _model_fit_metric(metric_dict: dict[str, dict[str, float]]) -> float:
    # We'd ideally log the entire `model_fit_dict` as a single model fit metric
    # can't capture the nuances of multiple experimental metrics, but this might
    # lead to database performance issues. So instead, we take the worst
    # coefficient of determination as model fit quality and store the full data
    # in Manifold (TODO).
    return min(metric_dict["coefficient_of_determination"].values())


def _model_std_quality(std: npt.NDArray) -> float:
    """Quantifies quality of the model uncertainty. A value of one means the
    uncertainty is perfectly predictive of the true standard deviation of the error.
    Values larger than one indicate over-estimation and negative values indicate
    under-estimation of the true standard deviation of the error. In particular, a value
    of 2 (resp. 1 / 2) represents an over-estimation (resp. under-estimation) of the
    true standard deviation of the error by a factor of 2.

    Args:
        std: The standard deviation of the standardized error.

    Returns:
        The factor corresponding to the worst over- or under-estimation factor of the
        standard deviation of the error among all experimentally observed metrics.
    """
    max_std, min_std = np.max(std), np.min(std)
    # comparing worst over-estimation factor with worst under-estimation factor
    inv_model_std_quality = max_std if max_std > 1 / min_std else min_std
    # reciprocal so that values greater than one indicate over-estimation and
    # values smaller than indicate underestimation of the uncertainty.
    return 1 / inv_model_std_quality


def _predict_on_training_data(
    adapter: Adapter,
    untransform: bool = False,
) -> tuple[
    dict[str, npt.NDArray],
    dict[str, npt.NDArray],
    dict[str, npt.NDArray],
]:
    """Makes predictions on the training data of a given experiment using a Adapter
    and returning the observed values, and the corresponding predictive means and
    predictive standard deviations of the model, in transformed space.

    NOTE: This is a helper function for `compute_model_fit_metrics_from_adapter`.

    Args:
        adapter: A Adapter object with which to make predictions.
        untransform: Boolean indicating whether to untransform model predictions.

    Returns:
        A tuple containing three dictionaries for 1) observed metric values, and the
        model's associated 2) predictive means and 3) predictive standard deviations.
    """
    training_data = adapter.get_training_data()
    observation_features = [
        ObservationFeatures(
            # NOTE: It is crucial to pop metadata first here.
            # Otherwise, it'd end up in parameters.
            metadata=row.pop("metadata"),
            parameters=row.to_dict(),
            trial_index=assert_is_instance(index, tuple)[0],
        )
        for index, row in training_data.arm_data.iterrows()
    ]
    observation_data_pred = adapter._predict_observation_data(
        observation_features=observation_features,
        untransform=untransform,
    )

    mean_predicted, cov_predicted = unwrap_observation_data(observation_data_pred)
    mean_observed = {
        name: col.to_numpy()
        for name, col in training_data.observation_data["mean"].items()
    }
    # Converting dictionary values to arrays.
    mean_predicted = {k: np.array(v) for k, v in mean_predicted.items()}
    std_predicted = {m: np.sqrt(np.array(cov_predicted[m][m])) for m in cov_predicted}
    return mean_observed, mean_predicted, std_predicted


def _predict_on_cross_validation_data(
    adapter: Adapter,
    untransform: bool = False,
) -> tuple[
    dict[str, npt.NDArray],
    dict[str, npt.NDArray],
    dict[str, npt.NDArray],
]:
    """Makes leave-one-out cross-validation predictions on the training data of the
    Adapter and returns the observed values, and the corresponding predictive means
    and predictive standard deviations of the model as numpy arrays,
    in transformed space.

    NOTE: This is a helper function for `compute_model_fit_metrics_from_adapter`.

    Args:
        adapter: A Adapter object with which to make predictions.
        untransform: Boolean indicating whether to untransform model predictions
            before cross validating. False by default.

    Returns:
        A tuple containing three dictionaries, each mapping metric signatures to:
            1. observed metric values,
            2. LOOCV predicted mean at each observed point, and
            3. LOOCV predicted standard deviation at each observed point.
    """
    cv = cross_validate(model=adapter, untransform=untransform)

    metric_signatures = cv[0].observed.data.metric_signatures
    mean_observed = {k: [] for k in metric_signatures}
    mean_predicted = {k: [] for k in metric_signatures}
    std_predicted = {k: [] for k in metric_signatures}

    for cvi in cv:
        obs = cvi.observed.data
        for k, v in zip(obs.metric_signatures, obs.means):
            mean_observed[k].append(v)

        pred = cvi.predicted
        for k, v in zip(pred.metric_signatures, pred.means):
            mean_predicted[k].append(v)

        pred_se = np.sqrt(pred.covariance.diagonal().clip(0))
        for k, v in zip(pred.metric_signatures, pred_se):
            std_predicted[k].append(v)

    mean_observed = {k: np.array(v) for k, v in mean_observed.items()}
    mean_predicted = {k: np.array(v) for k, v in mean_predicted.items()}
    std_predicted = {k: np.array(v) for k, v in std_predicted.items()}
    return mean_observed, mean_predicted, std_predicted
