#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Metrics that allow to retrieve curves of partial results.
Typically used to retrieve partial learning curves of ML training jobs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from logging import Logger
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.experiment import Experiment
from ax.core.map_data import MapData, MapKeyInfo
from ax.core.map_metric import MapMetric
from ax.core.metric import Metric, MetricFetchResult
from ax.core.trial import Trial
from ax.early_stopping.utils import align_partial_results
from ax.utils.common.logger import get_logger
from ax.utils.common.result import Ok
from ax.utils.common.typeutils import checked_cast

logger: Logger = get_logger(__name__)


class AbstractCurveMetric(MapMetric, ABC):
    """Metric representing (partial) learning curves of ML model training jobs."""

    # pyre-fixme[4]: Attribute must be annotated.
    MAP_KEY = MapKeyInfo(key="training_rows", default_value=0.0)

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

    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> MetricFetchResult:
        """Fetch data for one trial."""
        return self.fetch_trial_data_multi(trial=trial, metrics=[self], **kwargs)[
            self.name
        ]

    @classmethod
    def fetch_trial_data_multi(
        cls, trial: BaseTrial, metrics: Iterable[Metric], **kwargs: Any
    ) -> Dict[str, MetricFetchResult]:
        """Fetch multiple metrics data for one trial."""
        return cls.fetch_experiment_data_multi(
            experiment=trial.experiment, metrics=metrics, trials=[trial], **kwargs
        )[trial.index]

    @classmethod
    def fetch_experiment_data_multi(
        cls,
        experiment: Experiment,
        metrics: Iterable[Metric],
        trials: Optional[Iterable[BaseTrial]] = None,
        **kwargs: Any,
    ) -> Dict[int, Dict[str, MetricFetchResult]]:
        """Fetch multiple metrics data for an experiment."""
        if trials is None:
            trials = list(experiment.trials.values())
        trials = [trial for trial in trials if trial.status.expecting_data]
        if any(not isinstance(trial, Trial) for trial in trials):
            raise RuntimeError(
                f"Only (non-batch) Trials are supported by {cls.__name__}"
            )

        trial_idx_to_id = cls.get_ids_from_trials(trials=trials)
        if len(trial_idx_to_id) == 0:
            logger.debug("Could not get ids from trials. Returning empty data.")
            # TODO[mpolson64] Do we want to return Errs here?
            return {
                trial.index: {
                    metric.name: Ok(value=MapData(map_key_infos=[cls.MAP_KEY]))
                    for metric in metrics
                }
                for trial in (trials if trials is not None else [])
            }

        all_curve_series = cls.get_curves_from_ids(ids=trial_idx_to_id.values())
        if all(id_ not in all_curve_series for id_ in trial_idx_to_id.values()):
            logger.debug("Could not get curves from ids. Returning empty data.")
            # TODO[mpolson64] Do we want to return Errs here?
            return {
                trial.index: {
                    metric.name: Ok(value=MapData(map_key_infos=[cls.MAP_KEY]))
                    for metric in metrics
                }
                for trial in (trials if trials is not None else [])
            }

        df = cls.get_df_from_curve_series(
            experiment=experiment,
            all_curve_series=all_curve_series,
            metrics=metrics,
            trial_idx_to_id=trial_idx_to_id,
        )

        if df is None:
            return {
                trial.index: {
                    metric.name: Ok(value=MapData(map_key_infos=[cls.MAP_KEY]))
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
                        map_key_infos=[cls.MAP_KEY],
                    )
                )
                for metric in metrics
            }
            for trial in (trials if trials is not None else [])
        }

        return MapData(df=df, map_key_infos=[cls.MAP_KEY])

    @classmethod
    def get_df_from_curve_series(
        cls,
        experiment: Experiment,
        all_curve_series: Dict[Union[int, str], Dict[str, pd.Series]],
        metrics: Iterable[Metric],
        trial_idx_to_id: Dict[int, Union[int, str]],
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
                        map_key=cls.MAP_KEY.key,
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

    @classmethod
    @abstractmethod
    def get_ids_from_trials(
        cls, trials: Iterable[BaseTrial]
    ) -> Dict[int, Union[int, str]]:
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

    @classmethod
    @abstractmethod
    def get_curves_from_ids(
        cls, ids: Iterable[Union[int, str]]
    ) -> Dict[Union[int, str], Dict[str, pd.Series]]:
        """Get partial result curves from backend ids.

        Args:
            ids: The ids of the backend runs for which to fetch the
                partial result curves.

        Returns:
            A dictionary mapping the backend id to the partial result
            curves, each of which is represented as a mapping from
            the metric name to a pandas Series indexed by the progression
            (which will be mapped to the `MAP_KEY` of the metric class).
            E.g. if `curve_name=loss` and `MAP_KEY=training_rows`, then a
            Series should look like:

                 training_rows (index) | loss
                -----------------------|------
                                   100 | 0.5
                                   200 | 0.2
        """
        ...  # pragma: nocover


class AbstractScalarizedCurveMetric(AbstractCurveMetric):
    """A linear scalarization of (partial) learning curves of ML model training jobs:

        scalarized_curve = offset + sum_i(coefficients[i] * curve[i]).

    It is assumed that the output of `get_curves_from_ids` contains all of the curves
    necessary for performing the scalarization.
    """

    def __init__(
        self,
        name: str,
        coefficients: Dict[str, float],
        offset: float = 0.0,
        lower_is_better: bool = True,
        cumulative_best: bool = False,
        smoothing_window: Optional[int] = None,
    ) -> None:
        """Construct a AbstractScalarizedCurveMetric.

        Args:
            name: Name of metric.
            coefficients: A mapping from learning curve names to their
                scalarization coefficients.
            offset: The offset of the affine scalarization.
            lower_is_better: If True, lower values (of the scalarized metric) are
                considered better.
            cumulative_best: If True, for each trial, apply cumulative best to
                the curve (i.e., if lower is better, then we return a curve
                representing the cumulative min of the raw curve).
            smoothing_window: If not None, specifies the window size used for a
                rolling mean applied to the raw curve data. This can be helpful
                if the underlying data is expected to be very noisy.
        """
        MapMetric.__init__(self, name=name, lower_is_better=lower_is_better)
        self.coefficients = coefficients
        self.offset = offset
        self.cumulative_best = cumulative_best
        self.smoothing_window = smoothing_window

    @classmethod
    def get_df_from_curve_series(
        cls,
        experiment: Experiment,
        all_curve_series: Dict[Union[int, str], Dict[str, pd.Series]],
        metrics: Iterable[Metric],
        trial_idx_to_id: Dict[int, Union[int, str]],
    ) -> Optional[pd.DataFrame]:
        """Convert a `all_curve_series` dict (from `get_curves_from_ids`) into
        a dataframe. For each metric, we first get all curves represented in
        `coefficients` and then perform scalarization.

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
        complete_metrics_by_trial = {
            trial_idx: [] for trial_idx in trial_idx_to_id.keys()
        }
        for trial_idx, id_ in trial_idx_to_id.items():
            if id_ not in all_curve_series:
                logger.info(f"Could not get curve data for id {id_}. Ignoring.")
                continue
            curve_series = all_curve_series[id_]
            for m in metrics:
                curve_dfs = []
                for curve_name in m.coefficients.keys():  # pyre-ignore[16]
                    if curve_name in curve_series:
                        curve_df = _get_single_curve(
                            curve_series=curve_series,
                            curve_name=curve_name,
                            map_key=cls.MAP_KEY.key,
                            trial=experiment.trials[trial_idx],
                            cumulative_best=m.cumulative_best,  # pyre-ignore[16]
                            lower_is_better=m.lower_is_better,  # pyre-ignore[6]
                            smoothing_window=m.smoothing_window,  # pyre-ignore[16]
                        )
                        curve_dfs.append(curve_df)
                    else:
                        logger.info(
                            f"{curve_name} not present in curves from {id_}, so the "
                            f"scalarization for {m.name} cannot be computed. Returning "
                            "without this metric."
                        )
                        break
                if len(curve_dfs) == len(m.coefficients):
                    # only keep if all curves needed by the metric are available
                    dfs.extend(curve_dfs)
                    # mark metrics who have all underlying curves
                    complete_metrics_by_trial[trial_idx].append(m)

        if len(dfs) == 0:
            return None

        all_data_df = pd.concat(dfs, axis=0, ignore_index=True)
        sub_dfs = []
        # Do not create a common index across trials, only across the curves
        # involved in the scalarized metric.
        for trial_idx, dfi in all_data_df.groupby("trial_index"):
            # the `do_forward_fill = True` pads with the latest
            # observation to handle situations where learning curves
            # report different amounts of data.
            trial_curves = dfi["metric_name"].unique().tolist()
            dfs_mean, dfs_sem = align_partial_results(
                dfi,
                progr_key=cls.MAP_KEY.key,
                metrics=trial_curves,
                do_forward_fill=True,
            )
            for metric in complete_metrics_by_trial[trial_idx]:
                sub_df = _get_scalarized_curve_metric_sub_df(
                    dfs_mean=dfs_mean,
                    dfs_sem=dfs_sem,
                    metric=metric,
                    trial=checked_cast(Trial, experiment.trials[trial_idx]),
                )
                sub_dfs.append(sub_df)
        return pd.concat(sub_dfs, axis=0, ignore_index=True)


def _get_single_curve(
    curve_series: Dict[str, pd.Series],
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
    cs = curve_series[curve_name].rename("mean")  # pyre-ignore [6]
    dfi = cs.reset_index().rename(columns={"index": map_key})  # pyre-ignore [16]
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
    return dfi.drop_duplicates()


def _get_scalarized_curve_metric_sub_df(
    dfs_mean: Dict[str, pd.DataFrame],
    dfs_sem: Dict[str, pd.DataFrame],
    metric: AbstractScalarizedCurveMetric,
    trial: Trial,
) -> pd.DataFrame:
    """Helper to construct sub-dfs for a ScalarizedCurveMetric.

    Args:
        df_mean: A mapping from Curve metric names to a dataframe
            containing the means of the respective metric. The progression
            indices are assumed to be aliged across metrics (e.g. as
            obtained via `align_partial_results`).
        df_sem: A mapping from Curve metric names to a dataframe
            containing the sems of the respective metric. If empty,
            assume the metrics are subject to noise of unknown magnitude.
        metric: The ScalarizedCurveMetric to perform the aggregation for.
        trial: The trial associated with the data in `df_mean` and `df_sem`.

    Returns:
        A dataframe with the scalarized mean and sem in `mean` and `sem`
        columns, respectively.
    """
    sub_df = metric.offset + sum(
        coeff * dfs_mean[metric]  # pyre-ignore [58]
        for metric, coeff in metric.coefficients.items()
    )
    sub_df = sub_df.rename(columns={trial.index: "mean"})  # pyre-ignore [16]
    if dfs_sem:
        var_df = sum(
            (coeff * dfs_sem[metric]) ** 2  # pyre-ignore [58]
            for metric, coeff in metric.coefficients.items()
        )
        sem_df = var_df.apply(np.sqrt).rename(  # pyre-ignore [16]
            columns={trial.index: "sem"}
        )
        sub_df = pd.concat([sub_df, sem_df], axis=1)
    else:
        sub_df["sem"] = float("nan")
    sub_df = sub_df.reset_index()
    sub_df["trial_index"] = trial.index
    sub_df["arm_name"] = trial.arm.name  # pyre-ignore [16]
    sub_df["metric_name"] = metric.name
    # When scalarizing curves, sometimes the last progression will be different
    # across curves, even for the same trial. This dropna() will only keep the
    # progressions that are available for all curves.
    return sub_df.dropna(subset=["mean"])
