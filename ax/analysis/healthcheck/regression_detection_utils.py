# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import defaultdict

import numpy as np
import numpy.typing as npt

from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.observation import observations_from_data

from ax.exceptions.core import DataRequiredError, UserInputError
from ax.modelbridge.discrete import DiscreteModelBridge
from ax.modelbridge.registry import rel_EB_ashr_trans
from ax.models.discrete.eb_ashr import EBAshr
from pyre_extensions import assert_is_instance


def detect_regressions_by_trial(
    experiment: Experiment,
    thresholds: dict[str, tuple[float, float]],
    data: Data | None = None,
) -> dict[int, dict[str, dict[str, float]]]:
    r"""
    Identifies all regressing arms across trial along with the metrics they regress.

    Args:
        experiment: Ax experiment.
        thresholds: A dictionary mapping metric names a tuple of
            (threshold size, threshold probability).
        data: Experiment data. If None, use the attached experiment data.

    Returns: A dictionary mapping trial indices to
        dictionaries containing regressing arm names as keys and a dictionary
        with the corresponding regressed metrics and a tuple of regression sizes and
        corresponding regression probabilities as values.

    """
    data = data if data is not None else experiment.lookup_data()

    regressing_arms_metrics_by_trial = {}

    for trial_index, trial_df in data.df.groupby("trial_index"):
        regressing_arms_metrics_by_trial[trial_index] = detect_regressions_single_trial(
            experiment=experiment,
            data=Data(df=trial_df),
            thresholds=thresholds,
        )

    return regressing_arms_metrics_by_trial


def compute_regression_probabilities_single_trial(
    experiment: Experiment,
    size_thresholds: dict[str, float],
    data: Data | None = None,
) -> tuple[list[str | None], list[str], npt.NDArray]:
    r"""
    Computes the probabilities of regression for all arm metric pairs in a single trial.
    If a metric has lower_is_better indicator set to True
    (regresses in positive direction), regression probability is defined as
    Pr(metric value>=(positive) threshold).
    If a metric has lower_is_better indicator set to False
    (regresses in negative direction), regression probability is defined as
    Pr(metric value<=(negative) threshold).

    Args:
        experiment: Ax experiment.
        size_thresholds: A dictionary mapping metric names to threshold sizes.
        data: Experiment data. If None, use the attached experiment data.

    Returns: A tuple containing
        - arm names,
        - metric names, and corresponding
        - an array of size n x m where n is the number of arms and m is
            the number of metrics. Each entry contains the probability of
            regression for that arm-metric pair.

    """
    data = data if data is not None else experiment.lookup_data()

    if data.df.empty:
        raise DataRequiredError("Data must be non-empty.")

    if len(set(data.df["trial_index"])) > 1:
        raise UserInputError("The input data should contain only one trial.")

    # metric names available in the data and also in the given size thresholds dict
    metric_names = set(size_thresholds.keys()).intersection(set(data.df["metric_name"]))

    if len(metric_names) == 0:
        raise ValueError(
            "No common metrics between the provided data and the size thresholds."
            "Need to provide both data and the size thresholds for metrics of interest."
        )

    target_data = Data(df=data.df[data.df["metric_name"].isin(metric_names)])

    modelbridge = DiscreteModelBridge(
        experiment=experiment,
        search_space=experiment.search_space,
        data=target_data,
        model=EBAshr(),
        transforms=rel_EB_ashr_trans,
        optimization_config=experiment.optimization_config,
    )

    metric_names = modelbridge.outcomes

    lower_is_better_indicators = np.array(
        [-1 if experiment.metrics[m].lower_is_better else 1 for m in metric_names]
    )

    # if lower is better: outcome constraints: metric <= (positive) threshold
    # if upper is better: outcome constraints: metric >= (negative) threshold
    A = -np.diag(lower_is_better_indicators)
    b = np.array([abs(size_thresholds[metric]) for metric in metric_names])

    _, regression_probabilities = assert_is_instance(
        modelbridge.model, EBAshr
    )._get_regression_indicator(
        objective_weights=np.zeros(len(metric_names)), outcome_constraints=(A, b)
    )

    observations = observations_from_data(
        experiment=experiment,
        data=target_data,
    )

    arm_names = [
        observations[i].arm_name for i in range(regression_probabilities.shape[0])
    ]

    return arm_names, metric_names, regression_probabilities


def detect_regressions_single_trial(
    experiment: Experiment,
    thresholds: dict[str, tuple[float, float]],
    data: Data | None = None,
) -> dict[str | None, dict[str, float]]:
    r"""
    Identifies all regressing arms along with the metrics they regress for
    a single trial.

    Args:
        experiment: Ax experiment.
        thresholds: A dictionary mapping metric names a tuple of
            (threshold size, threshold probability).

    Returns: A dictionary with regressing arms as keys and another dictionary of
        corresponding regressed metrics and their corresponding regression probabilities
        as values.

    """
    data = data if data is not None else experiment.lookup_data()

    if len(set(data.df["trial_index"])) > 1:
        raise UserInputError("The input data should contain only one trial.")

    # metric names available in the data and also in the given thresholds
    metric_names = set(thresholds.keys()).intersection(set(data.df["metric_name"]))

    if len(metric_names) == 0:
        raise ValueError(
            "No common metrics between the provided data and the thresholds."
            "Need to provide both data and the size thresholds for metrics of interest."
        )

    size_thresholds = {metric: thresholds[metric][0] for metric in metric_names}

    (
        arm_names,
        metric_names,
        regression_probabilities,
    ) = compute_regression_probabilities_single_trial(
        experiment=experiment,
        data=Data(df=data.df[data.df["metric_name"].isin(metric_names)]),
        size_thresholds=size_thresholds,
    )

    # Regressing arms along with the metrics they regresss
    regressing_arms_metrics: dict[str | None, dict[str, float]] = defaultdict(dict)

    for i, arm_name in enumerate(arm_names):
        for j, metric_name in enumerate(metric_names):
            if regression_probabilities[i, j] >= thresholds[metric_name][1]:
                regressing_arms_metrics[arm_name].update(
                    {metric_name: regression_probabilities[i, j]}
                )

    return regressing_arms_metrics
