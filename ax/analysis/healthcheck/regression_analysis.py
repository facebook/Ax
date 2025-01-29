# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
from typing import Tuple

import pandas as pd
from ax.analysis.analysis import AnalysisCardLevel
from ax.analysis.healthcheck.healthcheck_analysis import (
    HealthcheckAnalysis,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.analysis.healthcheck.regression_detection_utils import (
    detect_regressions_by_trial,
)
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.exceptions.core import UserInputError
from pyre_extensions import none_throws


class RegressionAnalysis(HealthcheckAnalysis):
    r"""
    Analysis for detecting the regressing arm, metric pairs across all trials with data.
    For each metric, the regressions are defined as the arms that have a probability of
    regression above the threshold. The regression probabilities are calculated as
    posterior probabilities of the metric being above or below zero (depending if the
    metric is improving in the negative or positive direction), where the
    posteriors are with respect to the EBAshr
    (empirical Bayes Adaptive Shrinkage Model).
    """

    def __init__(self, prob_threshold: float = 0.95) -> None:
        r"""
        Args:
            prob_theshold: The threshold for the probability of metric regression.
                Regressions are defined as the arms that have a probability of
                regression above this threshold.

        Returns None
        """
        self.prob_threshold = prob_threshold

    def compute(
        self,
        experiment: Experiment | None,
        generation_strategy: GenerationStrategyInterface | None = None,
    ) -> HealthcheckAnalysisCard:
        r"""
        Detect the regressing arms for all trials that have data.

        Args:
            experiment: Ax experiment.
            generation_strategy: Ax generation strategy.

        Returns:
            A HealthcheckAnalysisCard object with the information on regressing arms
            and the corresponding metrics the arms regress.
        """
        if experiment is None:
            raise UserInputError("Experiment cannot be None.")

        if experiment.status_quo is None:
            raise UserInputError(
                "Experiment must have a status quo arm to run the regression analysis "
                "since the regressions are relative to the status quo arm."
            )

        data = none_throws(experiment).lookup_data()
        thresholds = {
            metric: (0.0, self.prob_threshold) for metric in experiment.metrics.keys()
        }
        regressions_by_trial = detect_regressions_by_trial(
            experiment=experiment,
            thresholds=thresholds,
            data=data,
        )
        regressions_by_trial_df, regressions_msg = process_regression_dict(
            regressions_by_trial=regressions_by_trial
        )

        if regressions_by_trial_df.shape[0] > 0:
            df = regressions_by_trial_df
            status = HealthcheckStatus.WARNING
            df["status"] = status
            subtitle = (
                "The following arms are regressing the following metrics "
                f"for the respective trials: \n {regressions_msg}"
            )
            title_status = "Warning"
        else:
            status = HealthcheckStatus.PASS
            df = pd.DataFrame({"status": [status]})
            subtitle = "No metric regessions detected."
            title_status = "Success"

        return HealthcheckAnalysisCard(
            name="RegressionAnalysis",
            title=f"Ax Regression Analysis {title_status}",
            blob=json.dumps({"status": status}),
            subtitle=subtitle,
            df=df,
            level=AnalysisCardLevel.LOW,
        )


def process_regression_dict(
    regressions_by_trial: dict[int, dict[str, dict[str, float]]],
) -> Tuple[pd.DataFrame, str]:
    r"""
    Process the dictionary of trial indices, regressing arms and metrics into
        a dataframe and a string.

    Args:
        regressions_by_trial: A dictionary of the form
            {trial_index: {arm_name: {metric_name: probability}}}.

    Returns: A tuple containing
        - A dataFrame with columns ["trial_index", "arm_name", "metric_name",
            "probability"] and
        - A string of the form containing trial indices, regressing arms and metrics.
    """
    trial_indices = []
    arm_names = []
    metric_names = []
    probabilities = []

    msg = ""

    for trial_index, arm_metric_probs in regressions_by_trial.items():
        if arm_metric_probs is None or len(arm_metric_probs) == 0:
            continue
        msg += f"Trial {trial_index}: \n"

        for arm_name, metrics_probs in arm_metric_probs.items():
            regressing_metrics = list(metrics_probs.keys())
            metric_names.extend(regressing_metrics)
            trial_indices.extend([trial_index] * len(regressing_metrics))
            probabilities.extend(list(metrics_probs.values()))
            arm_names.extend([arm_name] * len(regressing_metrics))

            msg += f" - Arm {arm_name}: \n"
            for metric in regressing_metrics:
                msg += f"{metric}, "
            msg = msg[:-2] + " \n"

    regressions_by_trial_df = pd.DataFrame(
        {
            "trial_index": trial_indices,
            "arm_name": arm_names,
            "metric_name": metric_names,
            "probability": probabilities,
        }
    )
    if "trial_index" in regressions_by_trial_df.columns:
        regressions_by_trial_df["trial_index"] = regressions_by_trial_df[
            "trial_index"
        ].astype(int)
    return regressions_by_trial_df, msg
