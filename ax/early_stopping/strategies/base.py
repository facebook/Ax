#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.observation import observations_from_map_data
from ax.modelbridge.modelbridge_utils import (
    _unpack_observations,
    observation_data_to_array,
    observation_features_to_array,
)
from ax.utils.common.base import Base
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none

logger = get_logger(__name__)


@dataclass
class EarlyStoppingTrainingData:
    """Dataclass for keeping data arrays related to model training and
    arm names together."""

    X: np.ndarray
    Y: np.ndarray
    Yvar: np.ndarray
    arm_names: List[Optional[str]]


class BaseEarlyStoppingStrategy(ABC, Base):
    """Interface for heuristics that halt trials early, typically based on early
    results from that trial."""

    def __init__(
        self,
        metric_names: Optional[Iterable[str]] = None,
        seconds_between_polls: int = 60,
        true_objective_metric_name: Optional[str] = None,
    ) -> None:
        """A BaseEarlyStoppingStrategy class.

        Args:
            metric_names: The names of the metrics the strategy will interact with.
                If no metric names are provided the objective metric is assumed.
            seconds_between_polls: How often to poll the early stopping metric to
                evaluate whether or not the trial should be early stopped.
            true_objective_metric_name: The actual objective to be optimized; used in
                situations where early stopping uses a proxy objective (such as training
                loss instead of eval loss) for stopping decisions.
        """
        if seconds_between_polls < 0:
            raise ValueError("`seconds_between_polls may not be less than 0.")
        self.metric_names = metric_names
        self.seconds_between_polls = seconds_between_polls
        self.true_objective_metric_name = true_objective_metric_name

    @abstractmethod
    def should_stop_trials_early(
        self,
        trial_indices: Set[int],
        experiment: Experiment,
        **kwargs: Dict[str, Any],
    ) -> Dict[int, Optional[str]]:
        """Decide whether to complete trials before evaluation is fully concluded.

        Typical examples include stopping a machine learning model's training, or
        halting the gathering of samples before some planned number are collected.

        Args:
            trial_indices: Indices of candidate trials to stop early.
            experiment: Experiment that contains the trials and other contextual data.

        Returns:
            A dictionary mapping trial indices that should be early stopped to
            (optional) messages with the associated reason.
        """
        pass  # pragma: nocover

    def _check_validity_and_get_data(self, experiment: Experiment) -> Optional[MapData]:
        """Validity checks and returns the `MapData` used for early stopping."""
        if self.metric_names is None:
            optimization_config = not_none(experiment.optimization_config)
            metric_names = [optimization_config.objective.metric.name]
        else:
            metric_names = self.metric_names

        data = experiment.lookup_data()
        if data.df.empty:
            logger.info(
                f"{self.__class__.__name__} received empty data. "
                "Not stopping any trials."
            )
            return None
        for metric_name in metric_names:
            if metric_name not in set(data.df["metric_name"]):
                logger.info(
                    f"{self.__class__.__name__} did not receive data "
                    "from the objective metric. Not stopping any trials."
                )
                return None

        if not isinstance(data, MapData):
            logger.info(
                f"{self.__class__.__name__} expects MapData, but the "
                f"data attached to experiment is of type {type(data)}. "
                "Not stopping any trials."
            )
            return None

        data = checked_cast(MapData, data)
        map_keys = data.map_keys
        if len(list(map_keys)) > 1:
            logger.info(
                f"{self.__class__.__name__} expects MapData with a single "
                "map key, but the data attached to the experiment has multiple: "
                f"{data.map_keys}. Not stopping any trials."
            )
            return None
        return data

    @staticmethod
    def _log_and_return_trial_ignored(
        logger: logging.Logger, trial_index: int
    ) -> Tuple[bool, str]:
        """Helper function for logging/constructing a reason when a trial
        should be ignored."""
        logger.info(
            f"Trial {trial_index} should be ignored and not considered "
            "for early stopping."
        )
        return False, "Specified as a trial to be ignored for early stopping."

    @staticmethod
    def _log_and_return_no_data(
        logger: logging.Logger, trial_index: int
    ) -> Tuple[bool, str]:
        """Helper function for logging/constructing a reason when there is no data."""
        logger.info(
            f"There is not yet any data associated with trial {trial_index}. "
            "Not early stopping this trial."
        )
        return False, "No data available to make an early stopping decision."

    @staticmethod
    def _log_and_return_min_progression(
        logger: logging.Logger,
        trial_index: int,
        trial_last_progression: float,
        min_progression: float,
    ) -> Tuple[bool, str]:
        """Helper function for logging/constructing a reason when min progression
        is not yet reached."""
        reason = (
            f"Most recent progression ({trial_last_progression}) is less than "
            "the specified minimum progression for early stopping "
            f"({min_progression}). "
        )
        logger.info(
            f"Trial {trial_index}'s m{reason[1:]} Not early stopping this trial."
        )
        return False, reason

    @staticmethod
    def _log_and_return_completed_trials(
        logger: logging.Logger, num_completed: int, min_curves: float
    ) -> Tuple[bool, str]:
        """Helper function for logging/constructing a reason when min number of
        completed trials is not yet reached."""
        logger.info(
            f"The number of completed trials ({num_completed}) is less than "
            "the minimum number of curves needed for early stopping "
            f"({min_curves}). Not early stopping this trial."
        )
        reason = (
            f"Need {min_curves} completed trials, but only {num_completed} "
            "completed trials so far."
        )
        return False, reason

    @staticmethod
    def _log_and_return_num_trials_with_data(
        logger: logging.Logger,
        trial_index: int,
        trial_last_progression: float,
        num_trials_with_data: int,
        min_curves: float,
    ) -> Tuple[bool, str]:
        """Helper function for logging/constructing a reason when min number of
        trials with data is not yet reached."""
        logger.info(
            f"The number of trials with data ({num_trials_with_data}) "
            f"at trial {trial_index}'s last progression ({trial_last_progression}) "
            "is less than the specified minimum number for early stopping "
            f"({min_curves}). Not early stopping this trial."
        )
        reason = (
            f"Number of trials with data ({num_trials_with_data}) at "
            f"last progression ({trial_last_progression}) is less than the "
            f"specified minimum number for early stopping ({min_curves})."
        )
        return False, reason


class ModelBasedEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
    """A base class for model based early stopping strategies. Includes
    a helper function for processing MapData into arrays."""

    def get_training_data(
        self,
        experiment: Experiment,
        map_data: MapData,
        keep_every_k_per_arm: Optional[int] = None,
    ) -> EarlyStoppingTrainingData:
        """Processes the raw (untransformed) training data into arrays for
        use in modeling.

        Args:
            experiment: Experiment that contains the data.
            map_data: The MapData from the experiment, as can be obtained by
                via `_check_validity_and_get_data`.
            keep_every_k_per_arm Subsample the learning curve by keeping every
                kth entry. Useful for limiting training data for modeling.

        Returns:
            An `EarlyStoppingTrainingData` that contains training data arrays X, Y,
                and Yvar + a list of arm names.
        """
        if keep_every_k_per_arm is not None:
            map_data = _subsample_map_data(
                map_data=map_data, keep_every_k_per_arm=keep_every_k_per_arm
            )
        observations = observations_from_map_data(
            experiment=experiment, map_data=map_data, map_keys_as_parameters=True
        )
        obs_features, obs_data, arm_names = _unpack_observations(observations)
        parameters = list(experiment.search_space.parameters.keys())
        outcome = not_none(experiment.optimization_config).objective.metric_names[0]
        X = observation_features_to_array(
            parameters=parameters + list(map_data.map_keys), obsf=obs_features
        )
        Y, Yvar = observation_data_to_array(
            outcomes=[outcome], observation_data=obs_data
        )
        return EarlyStoppingTrainingData(X=X, Y=Y, Yvar=Yvar, arm_names=arm_names)


def _subsample_map_data(map_data: MapData, keep_every_k_per_arm: int) -> MapData:
    """Helper function for keeping every kth row for each arm."""
    map_df = map_data.map_df
    # count the rows for each arm name and keep every n
    keep = map_df.groupby(["arm_name"]).cumcount()
    keep = (keep % keep_every_k_per_arm) == 0
    map_df_filtered = map_df[keep]
    return MapData(
        df=map_df_filtered,  # pyre-ignore[6]
        map_key_infos=map_data.map_key_infos,
        description=map_data.description,
    )
