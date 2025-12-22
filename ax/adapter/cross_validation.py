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
from typing import cast, NamedTuple
from warnings import warn

import numpy as np
import numpy.typing as npt
from ax.adapter.adapter_utils import array_to_observation_data
from ax.adapter.base import Adapter
from ax.adapter.data_utils import ExperimentData
from ax.adapter.observation_utils import unwrap_observation_data
from ax.adapter.torch import TorchAdapter
from ax.core.observation import Observation, ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.exceptions.core import UnsupportedError
from ax.exceptions.model import CrossValidationError
from ax.generators.torch.botorch_modular.generator import BoTorchGenerator
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
from botorch.cross_validation import loo_cv
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors.fully_bayesian import GaussianMixturePosterior
from botorch.settings import validate_input_scaling
from pyre_extensions import assert_is_instance, none_throws


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


FoldGenerator = Callable[[ExperimentData], Iterable[CVData]]


class AssessModelFitResult(NamedTuple):
    """Container for model fit assessment results"""

    good_fit_metrics_to_fisher_score: dict[str, float]
    bad_fit_metrics_to_fisher_score: dict[str, float]


def cross_validate(
    adapter: Adapter,
    folds: int = -1,
    test_selector: Callable[[Observation], bool] | None = None,
    untransform: bool = True,
    use_posterior_predictive: bool = False,
    fold_generator: FoldGenerator | None = None,
) -> list[CVResult]:
    """Cross validation for model.

    Splits the model's training data into train/test folds and makes
    out-of-sample predictions on the test folds.

    Train/test splits are made based on arm_name, so that repeated
    observations of a point are always together in the train or test
    set.

    Args:
        adapter: Fitted Adapter to cross validate.
        folds: Number of folds. Use -1 for leave-one-out (LOO) CV, which will
            use efficient O(n^3) computation when the model supports it and
            is configured with refit_on_cv=False.
        test_selector: Function to filter observations for cross validation.
            If provided, only observations for which this function returns
            True will be used as test points. Other observations will always
            be in training set for every fold.
        untransform: If True (default), predictions are untransformed before
            being returned.
        use_posterior_predictive: If True, return the posterior predictive
            (i.e., include observation noise in the predicted variance). If False,
            return the posterior (i.e., predict the noise-free latent function).
        fold_generator: Optional custom fold generator for custom splitting.

    Returns:
        A CVResult for each point held out as the "test" set during CV.
    """
    # Check if model is a TorchAdapter with a BoTorchGenerator
    is_torch_adapter = isinstance(adapter, TorchAdapter)
    has_botorch_generator = is_torch_adapter and isinstance(
        adapter.generator, BoTorchGenerator
    )

    # Check if the experiment has auxiliary experiments (which are not supported
    # by efficient LOO CV because they lead to tuple train_inputs)
    has_auxiliary_experiments = (
        is_torch_adapter
        and adapter._experiment is not None
        and bool(adapter._experiment.auxiliary_experiments_by_purpose)
    )

    # Try to use efficient LOO CV when applicable
    # Only use efficient LOO CV when:
    # 1. LOO CV is requested (folds == -1)
    # 2. No custom test_selector or fold_generator
    # 3. The model is a TorchAdapter with a BoTorchGenerator
    # 4. The generator has refit_on_cv=False (so hyperparameters are fixed)
    # 5. The experiment has no auxiliary experiments (they cause tuple train_inputs)
    use_efficient_loo = (
        folds == -1
        and test_selector is None
        and fold_generator is None
        and has_botorch_generator
        and not cast(BoTorchGenerator, adapter.generator).refit_on_cv
        and not has_auxiliary_experiments
    )

    if use_efficient_loo:
        try:
            return _efficient_loo_cross_validate(
                adapter=adapter,
                untransform=untransform,
                use_posterior_predictive=use_posterior_predictive,
            )
        except Exception as e:
            # Fall back to the standard approach if efficient LOO is not supported
            logger.debug(
                f"Efficient LOO CV failed, falling back to fold-by-fold CV: {e}"
            )

    # Standard fold-by-fold cross-validation
    return _fold_cross_validate(
        adapter=adapter,
        folds=folds,
        test_selector=test_selector,
        untransform=untransform,
        use_posterior_predictive=use_posterior_predictive,
        fold_generator=fold_generator,
    )


def _efficient_loo_cross_validate(
    adapter: Adapter,
    untransform: bool = True,
    use_posterior_predictive: bool = False,
) -> list[CVResult]:
    """Use efficient O(n^3) LOO CV when the model supports it.

    This avoids the O(n^4) cost of naive LOO CV by computing all n
    leave-one-out predictions at once using the efficient formulas
    from GPyTorch's LeaveOneOutPseudoLikelihood.

    Args:
        model: Fitted model (Adapter) to cross validate.
        untransform: If True (default), predictions are untransformed.
        use_posterior_predictive: If True, include observation noise in variance.

    Returns:
        A CVResult for each training point.

    Raises:
        ValueError: If the model doesn't support efficient LOO CV.
    """
    # Check if model is a TorchAdapter with a BoTorchGenerator
    if not isinstance(adapter, TorchAdapter):
        raise ValueError(
            "Efficient LOO cross-validation requires a TorchAdapter. "
            f"Got {type(adapter).__name__}."
        )

    if not isinstance(adapter.generator, BoTorchGenerator):
        raise ValueError(
            "Efficient LOO cross-validation requires a BoTorchGenerator. "
            f"Got {type(adapter.generator).__name__}."
        )

    surrogate = adapter.generator.surrogate
    if surrogate is None or surrogate.model is None:
        raise ValueError(
            "Efficient LOO cross-validation requires a fitted surrogate model."
        )

    botorch_model = none_throws(surrogate.model)

    # Check model is a GPyTorchModel (required for efficient LOO CV)
    if not isinstance(botorch_model, GPyTorchModel):
        raise ValueError(
            "Model must be a GPyTorchModel for efficient LOO CV. "
            f"Got {type(botorch_model).__name__}."
        )

    # Get training observations for observed values (filter to in-design data
    # for consistency with naive CV path in _fold_cross_validate)
    training_data = adapter.get_training_data(filter_in_design=True)
    training_observations = training_data.convert_to_list_of_observations()

    # Check for sufficient training data
    arm_name_vals = set(training_data.arm_data.index.unique(level="arm_name"))
    if len(arm_name_vals) < 2:
        raise ValueError(
            f"Efficient LOO CV requires at least two in-design arms. "
            f"Only {len(arm_name_vals)} in-design arms were found."
        )

    # Use loo_cv which automatically dispatches based on model type
    # observation_noise parameter corresponds to use_posterior_predictive
    loo_results = loo_cv(botorch_model, observation_noise=use_posterior_predictive)
    posterior = loo_results.posterior

    # Extract predictions for all training points
    if isinstance(posterior, GaussianMixturePosterior):
        # Use mixture statistics for ensemble models.
        # NOTE: This can be misleading when the mixture posterior is far from
        # Gaussian (e.g., if the posterior is bimodal/multimodal). The mixture
        # mean and variance don't fully capture the uncertainty in such cases,
        # which can lead to counterintuitive cross-validation results and plots.
        # This is a known limitation for fully Bayesian models and models with
        # non-Gaussian posteriors (such as PFNs).
        # Shape: n x 1 x m
        loo_means = posterior.mixture_mean.detach().cpu().numpy()
        loo_vars = posterior.mixture_variance.detach().cpu().numpy()
    else:
        # Shape: n x 1 x m
        loo_means = posterior.mean.detach().cpu().numpy()
        loo_vars = posterior.variance.detach().cpu().numpy()

    # Squeeze out the q dimension: n x 1 x m -> n x m
    loo_means = loo_means.squeeze(1)
    loo_vars = loo_vars.squeeze(1)

    # Handle the case where there's only one outcome (1D array after squeeze)
    if loo_means.ndim == 1:
        loo_means = loo_means[:, np.newaxis]
        loo_vars = loo_vars[:, np.newaxis]

    n_predictions = loo_means.shape[0]
    n_observations = len(training_observations)

    # Check that the number of observations matches the number of predictions
    # If not, raise an error (some transforms may have changed the data)
    if n_observations != n_predictions:
        raise ValueError(
            f"Number of training observations ({n_observations}) does not match "
            f"number of LOO predictions ({n_predictions}). This can happen when "
            "transforms modify the training data. Falling back to naive CV."
        )

    # Build CVResult objects from LOO predictions and training observations
    return _build_cv_results(
        loo_means=loo_means,
        loo_vars=loo_vars,
        training_observations=training_observations,
        adapter=adapter,
        untransform=untransform,
    )


def _build_cv_results(
    loo_means: npt.NDArray,
    loo_vars: npt.NDArray,
    training_observations: list[Observation],
    adapter: Adapter,
    untransform: bool,
) -> list[CVResult]:
    """Build CVResult objects from LOO predictions and training observations.

    Args:
        loo_means: LOO mean predictions with shape (n, num_outcomes).
        loo_vars: LOO variance predictions with shape (n, num_outcomes).
        training_observations: List of training observations.
        adapter: The adapter used for training.
        untransform: Whether to untransform predictions back to original space.

    Returns:
        List of CVResult objects, one per training observation.
    """
    n_obs, num_outcomes = loo_means.shape

    # Build diagonal covariance matrices for all observations efficiently
    # loo_covs has shape (n_obs, num_outcomes, num_outcomes) with variances on diagonal
    loo_covs = np.zeros((n_obs, num_outcomes, num_outcomes))
    diag_idx = np.arange(num_outcomes)
    loo_covs[:, diag_idx, diag_idx] = loo_vars

    # Convert all predictions to ObservationData at once
    all_predictions = array_to_observation_data(
        f=loo_means,
        cov=loo_covs,
        outcomes=adapter.outcomes,
    )

    # Create Observation objects for predictions, pairing with training features
    pred_observations = [
        Observation(features=obs.features, data=pred)
        for obs, pred in zip(training_observations, all_predictions, strict=True)
    ]

    # Transform observations and predictions to the appropriate space
    observations = list(training_observations)
    if untransform:
        # Untransform predictions to original space (observations are already there)
        for t in reversed(list(adapter.transforms.values())):
            pred_observations = t.untransform_observations(pred_observations)
    else:
        # Transform observations to model space (predictions are already there)
        for t in adapter.transforms.values():
            observations = t.transform_observations(observations)

    # Build CVResult objects from paired observations and predictions
    return [
        CVResult(observed=obs, predicted=pred_obs.data)
        for obs, pred_obs in zip(observations, pred_observations, strict=True)
    ]


def _fold_cross_validate(
    adapter: Adapter,
    folds: int = -1,
    test_selector: Callable[[Observation], bool] | None = None,
    untransform: bool = True,
    use_posterior_predictive: bool = False,
    fold_generator: Callable[[ExperimentData], Iterable[CVData]] | None = None,
) -> list[CVResult]:
    """Cross validation for model predictions using fold-by-fold approach.

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
        adapter: Fitted model (Adapter) to cross validate.
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
    training_data = adapter.get_training_data(filter_in_design=True)
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
            cv_test_predictions = adapter.cross_validate(
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
            ) = adapter._transform_inputs_for_cv(
                cv_training_data=cv_training_data, cv_test_points=cv_test_points
            )
            # Since each CV fold removes points from the training data, the
            # remaining observations will not pass the input scaling checks.
            # To avoid confusing users with warnings, we disable these checks.
            with validate_input_scaling(False):
                cv_test_predictions = adapter._cross_validate(
                    search_space=search_space,
                    cv_training_data=cv_training_data,
                    cv_test_points=cv_test_points,
                    use_posterior_predictive=use_posterior_predictive,
                )
            # Get test observations in transformed space.
            for t in adapter.transforms.values():
                cv_test_observations = t.transform_observations(
                    observations=cv_test_observations
                )
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
    cv = cross_validate(adapter=adapter, untransform=untransform)

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
