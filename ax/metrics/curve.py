#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Metrics that allow to retrieve curves of partial results.
Typically used to retrieve partial learning curves of ML training jobs.
"""

from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.map_data import MapKeyInfo, MapData
from ax.core.map_metric import MapMetric
from ax.core.metric import Metric
from ax.core.trial import Trial
from ax.utils.common.logger import get_logger

logger = get_logger(__name__)


class AbstractCurveMetric(MapMetric, ABC):
    """Metric representing (partial) learning curves of ML model training jobs."""

    MAP_KEY = MapKeyInfo(key="training_rows", default_value=0.0)

    def __init__(
        self,
        name: str,
        curve_name: str,
        lower_is_better: bool = True,
    ) -> None:
        """Inits Metric.

        Args:
            name: The name of the metric.
            curve_name: The name of the learning curve in the training output
                (there may be multiple outputs e.g. for MTML models).
            lower_is_better: If True, lower curve values are considered better.
        """
        super().__init__(name=name, lower_is_better=lower_is_better)
        self.curve_name = curve_name

    @classmethod
    def is_available_while_running(cls) -> bool:
        return True

    @classmethod
    def overwrite_existing_data(cls) -> bool:
        return True

    def fetch_trial_data(self, trial: BaseTrial, **kwargs: Any) -> Data:
        """Fetch data for one trial."""
        return self.fetch_trial_data_multi(trial=trial, metrics=[self], **kwargs)

    @classmethod
    def fetch_trial_data_multi(
        cls, trial: BaseTrial, metrics: Iterable[Metric], **kwargs: Any
    ) -> Data:
        """Fetch multiple metrics data for one trial."""
        return cls.fetch_experiment_data_multi(
            experiment=trial.experiment, metrics=metrics, trials=[trial], **kwargs
        )

    @classmethod
    def fetch_experiment_data_multi(
        cls,
        experiment: Experiment,
        metrics: Iterable[Metric],
        trials: Optional[Iterable[BaseTrial]] = None,
        **kwargs: Any,
    ) -> Data:
        """Fetch multiple metrics data for an experiment."""
        if trials is None:
            trials = list(experiment.trials.values())
        if any(not isinstance(trial, Trial) for trial in trials):
            raise RuntimeError(
                f"Only (non-batch) Trials are supported by {cls.__name__}"
            )
        ids = cls.get_ids_from_trials(trials=trials)

        trials_filtered, ids_filtered = [], []
        for trial, id_ in zip(trials, ids):
            if id_ is None:
                logger.info(f"Could not get id for Trial {trial.index}. Ignoring.")
            else:
                trials_filtered.append(trial)
                ids_filtered.append(id_)

        if len(ids_filtered) == 0:
            logger.info("Could not get ids from trials. Returning empty data.")
            return MapData(map_key_infos=[cls.MAP_KEY])

        all_curve_series = cls.get_curves_from_ids(ids=ids_filtered)
        if all(id_ not in all_curve_series for id_ in ids_filtered):
            logger.info("Could not get curves from ids. Returning empty data.")
            return MapData(map_key_infos=[cls.MAP_KEY])

        for id_, curve_series in all_curve_series.items():
            for m in metrics:
                if m.curve_name not in curve_series:  # pyre-ignore [16]
                    logger.info(
                        f"{m.curve_name} not yet present in curves from {id_}. "
                        "Returning empty data."
                    )
                    return MapData(map_key_infos=[cls.MAP_KEY])

        dfs = []
        for trial, id_ in zip(trials_filtered, ids_filtered):
            if id_ not in all_curve_series:
                logger.info(f"Could not get curve data for id {id_}. Ignoring.")
                continue
            curve_series = all_curve_series[id_]
            for m in metrics:
                cs = curve_series[m.curve_name].rename("mean")  # pyre-ignore [6]
                dfi = cs.reset_index().rename(  # pyre-ignore [16]
                    columns={"index": cls.MAP_KEY.key}
                )
                dfi["trial_index"] = trial.index
                dfi["arm_name"] = trial.arm.name
                dfi["metric_name"] = m.name
                dfi["sem"] = float("nan")
                dfs.append(dfi.drop_duplicates())
        df = pd.concat(dfs, axis=0, ignore_index=True)
        return MapData(df=df, map_key_infos=[cls.MAP_KEY])

    @classmethod
    @abstractmethod
    def get_ids_from_trials(
        cls, trials: Iterable[BaseTrial]
    ) -> List[Optional[Union[int, str]]]:
        """Get backend run ids associated with trials.

        Args:
            trials: The trials for which to retrieve the associated
                ids that can be used to to identify the corresponding
                runs on the backend.

        Returns:
            A list of identifiers (ints or strings) corresponding to
            the backend runs associated with the trials, in the same
            order as the `trials` input.
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
