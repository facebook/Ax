# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import MetricFetchE, MetricFetchResult
from ax.utils.common.result import Err, Ok


def _fetch_trial_data(
    trial: BaseTrial,
    metric_name: str,
    outcome_index: Optional[int] = None,
    include_noise_sd: bool = True,
) -> MetricFetchResult:
    """
    Args:
        trial: The trial from which to fetch data.
        metric_name: Name of the metric to fetch. If `metric_index` is not specified,
            this is used to retrieve the index (of the outcomes) from the
            `outcome_names` dict in a trial's `run_metadata`. If `metric_index` is
            specified, this is simply the name of the metric.
        outcome_index: The index (in the last dimension) of the `Ys` and
            `Ystds` lists of outcomes stored by the respective runner in the trial's
            `run_metadata`. If omitted, `run_metadata` must contain a `outcome_names`
            list of names in the same order as the outcomes that will be used to
            determine the index.
        include_noise_sd: Whether to include noise standard deviation in the returned
            data.

    Returns:
        A MetricFetchResult containing the data for the requested metric.
    """
    if outcome_index is None:
        # Look up the index based on the outcome name under which we track the data
        # as part of `run_metadata`.
        outcome_names = trial.run_metadata.get("outcome_names")
        if outcome_names is None:
            raise RuntimeError(
                "Trials' `run_metadata` must contain `outcome_names` if "
                "no `outcome_index` is provided."
            )
        outcome_index = outcome_names.index(metric_name)

    try:
        arm_names = list(trial.arms_by_name.keys())
        all_Ys = trial.run_metadata["Ys"]
        Ys = [all_Ys[arm_name][outcome_index] for arm_name in arm_names]

        if include_noise_sd:
            stdvs = [
                trial.run_metadata["Ystds"][arm_name][outcome_index]
                for arm_name in arm_names
            ]
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
