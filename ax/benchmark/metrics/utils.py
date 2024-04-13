# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

import pandas as pd
from ax.benchmark.metrics.base import GroundTruthMetricMixin
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import MetricFetchE, MetricFetchResult
from ax.exceptions.core import UnsupportedError
from ax.utils.common.result import Err, Ok


def _fetch_trial_data(
    trial: BaseTrial,
    metric_name: str,
    outcome_index: Optional[int] = None,
    include_noise_sd: bool = True,
    ground_truth: bool = False,
) -> MetricFetchResult:
    """
    Args:
        trial: The trial from which to fetch data.
        metric_name: Name of the metric to fetch. If `metric_index` is not specified,
            this is used to retrieve the index (of the outcomes) from the
            `outcome_names` dict in a trial's `run_metadata`. If `metric_index` is
            specified, this is simply the name of the metric.
        outcome_index: The index (in the last dimension) of the `Ys`, `Ys_true`, and
            `Ystds` lists of outcomes stored by the respective runner in the trial's
            `run_metadata`. If omitted, `run_metadata` must contain a `outcome_names`
            list of names in the same order as the outcomes that will be used to
            determine the index.
        include_noise_sd: Whether to include noise standard deviation in the returned
            data. Must be `False` if `ground_truth` is set to `True`.
        ground_truth: If True, return the ground truth values instead of the actual
            (noisy) observations. In this case, the noise standard deviations will
            be reported as zero.

    Returns:
        A MetricFetchResult containing the data for the requested metric.
    """
    if include_noise_sd and ground_truth:
        raise UnsupportedError(
            "Cannot include noise standard deviation when extracting ground truth "
            "data. Will be set to zero for ground truth observations."
        )

    if outcome_index is None:
        # Look up the index based on the outcome name under which we track the data
        # as part of `run_metadata`.
        outcome_names = trial.run_metadata.get("outcome_names")
        if outcome_names is None:
            raise RuntimeError(
                "Trials' `run_metadata` must contain `outcome_names` if "
                "no `outcome_index` is provided."
            )
        outcome_index = outcome_names.index(
            GroundTruthMetricMixin.get_original_name(metric_name)
            if ground_truth
            else metric_name
        )

    try:
        arm_names = list(trial.arms_by_name.keys())
        all_Ys = trial.run_metadata["Ys_true" if ground_truth else "Ys"]
        Ys = [all_Ys[arm_name][outcome_index] for arm_name in arm_names]

        if include_noise_sd:
            stdvs = [
                trial.run_metadata["Ystds"][arm_name][outcome_index]
                for arm_name in arm_names
            ]
        elif ground_truth:
            # Ground truth observations are noiseless (note that at least currently
            # this information is not being used as we only use the ground truth
            # observations for analysis but not for modeling).
            stdvs = [0.0] * len(Ys)
        else:
            stdvs = [float("nan")] * len(Ys)

        df = pd.DataFrame(
            {
                "arm_name": arm_names,
                "metric_name": metric_name,
                "mean": Ys,
                "sem": stdvs,
                "trial_index": trial.index,
            }
        )
        return Ok(value=Data(df=df))

    except Exception as e:
        return Err(
            MetricFetchE(
                message=f"Failed to obtain data for trial {trial.index}", exception=e
            )
        )
