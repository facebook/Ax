#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Metrics that allow to retrieve curves of partial results.
Typically used to retrieve partial learning curves of ML training jobs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from logging import Logger
from typing import Any, Optional, Union

import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.map_data import MapData, MapKeyInfo
from ax.core.map_metric import MapMetric
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.exceptions.core import UnsupportedError
from ax.utils.common.logger import get_logger
from ax.utils.common.result import Err, Ok

logger: Logger = get_logger(__name__)


class AbstractCurveMetric(MapMetric, ABC):
    """Metric representing (partial) learning curves of ML model training jobs."""

    map_key_info: MapKeyInfo[float] = MapKeyInfo(key="training_rows", default_value=0.0)

    def __init__(
        self,
        name: str,
        curve_name: str,
        lower_is_better: bool = True,
        cumulative_best: bool = False,
        smoothing_window: Optional[int] = None,
    ) -> None:
        """Inits Metric.

        Args:
            name: The name of the metric.
            curve_name: The name of the learning curve in the training output
                (there may be multiple outputs e.g. for MTML models).
            lower_is_better: If True, lower curve values are considered better.
            cumulative_best: If True, for each trial, apply cumulative best to
                the curve (i.e., if lower is better, then we return a curve
                representing the cumulative min of the raw curve).
            smoothing_window: If not None, specifies the window size used for a
                rolling mean applied to the raw curve data. This can be helpful
                if the underlying data is expected to be very noisy.
        """
        super().__init__(name=name, lower_is_better=lower_is_better)
        self.curve_name = curve_name
        self.cumulative_best = cumulative_best
        self.smoothing_window = smoothing_window

    @classmethod
    def is_available_while_running(cls) -> bool:
        return True

    @abstractmethod
    def get_ids_from_trials(
        self, trials: Iterable[BaseTrial]
    ) -> dict[int, Union[int, str]]:
        """Get backend run ids associated with trials.

        Args:
            trials: The trials for which to retrieve the associated
                ids that can be used to to identify the corresponding
                runs on the backend.

        Returns:
            A dictionary mapping the trial indices to the identifiers
            (ints or strings) corresponding to the backend runs associated
            with the trials. Trials whose corresponding ids could not be
            found should be omitted.
        """
        ...  # pragma: nocover

    @abstractmethod
    def get_curves_from_ids(
        self,
        ids: Iterable[Union[int, str]],
        names: Optional[set[str]] = None,
    ) -> dict[Union[int, str], dict[str, pd.Series]]:
        """Get partial result curves from backend ids.

        Args:
            ids: The ids of the backend runs for which to fetch the
                partial result curves.
            names: The names of the curves to fetch (for each of the runs).
                If omitted, fetch data for all available curves (this may be slow).

        Returns:
            A dictionary mapping the backend id to the partial result
            curves, each of which is represented as a mapping from
            the metric name to a pandas Series indexed by the progression
            (which will be mapped to the `map_key_info.key` of the metric class).
            E.g. if `curve_name=loss` and `map_key_info.key = training_rows`,
            then a Series should look like:

                 training_rows (index) | loss
                -----------------------|------
                                   100 | 0.5
                                   200 | 0.2
        """
        ...  # pragma: nocover

    @property
    def curve_names(self) -> set[str]:
        return {self.curve_name}

    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> MetricFetchResult:
        """Fetch data for one trial."""
        return self.bulk_fetch_trial_data(trial=trial, metrics=[self], **kwargs)[
            self.name
        ]

    def bulk_fetch_trial_data(
        self, trial: BaseTrial, metrics: Iterable[Metric], **kwargs: Any
    ) -> dict[str, MetricFetchResult]:
        """Fetch multiple metrics data for one trial."""
        return self.bulk_fetch_experiment_data(
            experiment=trial.experiment, metrics=metrics, trials=[trial], **kwargs
        )[trial.index]

    def bulk_fetch_experiment_data(
        self,
        experiment: Experiment,
        metrics: Iterable[Metric],
        trials: Optional[Iterable[BaseTrial]] = None,
        **kwargs: Any,
    ) -> dict[int, dict[str, MetricFetchResult]]:
        """Fetch multiple metrics data for an experiment."""
        trials = list(experiment.trials.values() if trials is None else trials)
        trials = [trial for trial in trials if trial.status.expecting_data]
        if any(isinstance(trial, BatchTrial) for trial in trials):
            raise UnsupportedError(
                f"Only (non-batch) Trials are supported by {self.__class__.__name__}"
            )
        metrics = list(metrics)
        if any(not isinstance(metric, AbstractCurveMetric) for metric in metrics):
            raise UnsupportedError(
                "Only metrics that subclass AbstractCurveMetric are supported by "
                + self.__class__.__name__
            )
        try:
            trial_idx_to_id = self.get_ids_from_trials(trials=trials)
            if len(trial_idx_to_id) == 0:
                logger.debug("Could not get ids from trials. Returning MetricFetchE.")
                return {
                    trial.index: {
                        metric.name: Err(
                            value=MetricFetchE(
                                message=(f"Could not get ids from trials: {trials}"),
                                exception=None,
                            )
                        )
                        for metric in metrics
                    }
                    for trial in (trials if trials is not None else [])
                }

            curve_names = set.union(
                *(m.curve_names for m in metrics)  # pyre-ignore[16]
            )
            all_curve_series = self.get_curves_from_ids(
                ids=trial_idx_to_id.values(),
                names=curve_names,
            )

            if all(id_ not in all_curve_series for id_ in trial_idx_to_id.values()):
                logger.debug("Could not get curves from ids. Returning Errs.")
                return {
                    trial.index: {
                        metric.name: Err(
                            value=MetricFetchE(
                                message=(
                                    f"Could not get curves from ids: {trial_idx_to_id}"
                                ),
                                exception=None,
                            )
                        )
                        for metric in metrics
                    }
                    for trial in (trials if trials is not None else [])
                }

            df = self.get_df_from_curve_series(
                experiment=experiment,
                all_curve_series=all_curve_series,
                metrics=metrics,
                trial_idx_to_id=trial_idx_to_id,
            )

            if df is None:
                return {
                    trial.index: {
                        metric.name: Err(
                            value=MetricFetchE(
                                message=("DataFrame from curve series is empty"),
                                exception=None,
                            )
                        )
                        for metric in metrics
                    }
                    for trial in (trials if trials is not None else [])
                }

            return {
                trial.index: {
                    metric.name: Ok(
                        value=MapData(
                            df=(
                                df.loc[
                                    (df["metric_name"] == metric.name)
                                    & (df["trial_index"] == trial.index)
                                ]
                            ),
                            map_key_infos=[self.map_key_info],
                        )
                    )
                    for metric in metrics
                }
                for trial in (trials if trials is not None else [])
            }
        except Exception as e:
            return {
                trial.index: {
                    metric.name: Err(
                        value=MetricFetchE(
                            message=f"Failed to fetch {self}", exception=e
                        )
                    )
                    for metric in metrics
                }
                for trial in (trials if trials is not None else [])
            }

    # TODO: Deduplicate this with get_df_from_scalarized_curve_series
    def get_df_from_curve_series(
        self,
        experiment: Experiment,
        all_curve_series: dict[Union[int, str], dict[str, pd.Series]],
        metrics: Iterable[Metric],
        trial_idx_to_id: dict[int, Union[int, str]],
    ) -> Optional[pd.DataFrame]:
        return get_df_from_curve_series(
            experiment=experiment,
            all_curve_series=all_curve_series,
            metrics=metrics,
            trial_idx_to_id=trial_idx_to_id,
            map_key=self.map_key_info.key,
        )


def get_df_from_curve_series(
    experiment: Experiment,
    all_curve_series: dict[Union[int, str], dict[str, pd.Series]],
    metrics: Iterable[Metric],
    trial_idx_to_id: dict[int, Union[int, str]],
    map_key: str,
) -> Optional[pd.DataFrame]:
    """Convert a `all_curve_series` dict (from `get_curves_from_ids`) into
    a dataframe. For each metric, we get one curve (of name `curve_name`).

    Args:
        experiment: The experiment.
        all_curve_series: A dict containing curve data, as output from
            `get_curves_from_ids`.
        metrics: The metrics from which data is being fetched.
        trial_idx_to_id: A dict mapping trial index to ids.

    Returns:
        A dataframe containing curve data or None if no curve data could be found.
    """
    dfs = []
    for trial_idx, id_ in trial_idx_to_id.items():
        if id_ not in all_curve_series:
            logger.info(f"Could not get curve data for id {id_}. Ignoring.")
            continue
        curve_series = all_curve_series[id_]
        for m in metrics:
            if m.curve_name in curve_series:  # pyre-ignore[16]
                dfi = _get_single_curve(
                    curve_series=curve_series,
                    curve_name=m.curve_name,
                    metric_name=m.name,
                    map_key=map_key,
                    trial=experiment.trials[trial_idx],
                    cumulative_best=m.cumulative_best,  # pyre-ignore[16]
                    lower_is_better=m.lower_is_better,  # pyre-ignore[6]
                    smoothing_window=m.smoothing_window,  # pyre-ignore[16]
                )
                dfs.append(dfi)
            else:
                logger.info(
                    f"{m.curve_name} not yet present in curves from {id_}. "
                    "Returning without this metric."
                )
    if len(dfs) == 0:
        return None
    return pd.concat(dfs, axis=0, ignore_index=True)


def _get_single_curve(
    curve_series: dict[str, pd.Series],
    curve_name: str,
    map_key: str,
    trial: BaseTrial,
    cumulative_best: bool,
    lower_is_better: bool,
    smoothing_window: Optional[int],
    metric_name: Optional[str] = None,
) -> pd.DataFrame:
    """Get a single curve from `curve_series` and return as a dataframe.
    By default, the `metric_name` is set to be the `curve_name`, but if
    an additional `metric_name` is passed, it will be used instead.
    """
    if metric_name is None:
        metric_name = curve_name
    cs = curve_series[curve_name].rename("mean")
    dfi = cs.reset_index().rename(columns={"index": map_key})
    dfi["trial_index"] = trial.index
    dfi["arm_name"] = trial.arm.name  # pyre-ignore [16]
    dfi["metric_name"] = metric_name
    dfi["sem"] = float("nan")
    if smoothing_window is not None and len(dfi["mean"]) >= smoothing_window:
        dfi["mean"] = dfi["mean"].rolling(window=smoothing_window).mean()
        first_smoothed = dfi["mean"].iloc[smoothing_window - 1]
        dfi.iloc[: smoothing_window - 1, dfi.columns.get_loc("mean")] = first_smoothed
    if cumulative_best:
        dfi["mean"] = dfi["mean"].cummin() if lower_is_better else dfi["mean"].cummax()
    # pyre-fixme[7]: Expected `DataFrame` but got `Optional[DataFrame]`.
    return dfi.drop_duplicates()
