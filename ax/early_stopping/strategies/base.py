#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from logging import Logger

import numpy.typing as npt
import pandas as pd
from ax.core.base_trial import TrialStatus
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.map_metric import MapMetric
from ax.core.objective import MultiObjective

from ax.early_stopping.utils import estimate_early_stopping_savings
from ax.modelbridge.map_torch import MapTorchModelBridge
from ax.modelbridge.modelbridge_utils import (
    _unpack_observations,
    observation_data_to_array,
    observation_features_to_array,
)
from ax.modelbridge.registry import Cont_X_trans, Y_trans
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.map_unit_x import MapUnitX

from ax.models.torch_base import TorchModel
from ax.utils.common.base import Base
from ax.utils.common.logger import get_logger
from pyre_extensions import assert_is_instance, none_throws

logger: Logger = get_logger(__name__)


@dataclass
class EarlyStoppingTrainingData:
    """Dataclass for keeping data arrays related to model training and
    arm names together.

    Args:
        X: An `n x d'` array of training features. `d' = d + m`, where `d`
            is the dimension of the design space and `m` are the number of map
            keys. For the case of learning curves, `m = 1` since we have only
            the number of steps as the map key.
        Y: An `n x 1` array of training observations.
        Yvar: An `n x 1` observed measurement noise.
        arm_names: A list of length `n` of arm names. Useful for understanding
            which data come from the same arm.
    """

    X: npt.NDArray
    Y: npt.NDArray
    Yvar: npt.NDArray
    arm_names: list[str | None]


class BaseEarlyStoppingStrategy(ABC, Base):
    """Interface for heuristics that halt trials early, typically based on early
    results from that trial."""

    def __init__(
        self,
        metric_names: Iterable[str] | None = None,
        min_progression: float | None = None,
        max_progression: float | None = None,
        min_curves: int | None = None,
        trial_indices_to_ignore: list[int] | None = None,
        normalize_progressions: bool = False,
    ) -> None:
        """A BaseEarlyStoppingStrategy class.

        Args:
            metric_names: The names of the metrics the strategy will interact with.
                If no metric names are provided the objective metric is assumed.
            min_progression: Only stop trials if the latest progression value
                (e.g. timestamp, epochs, training data used) is greater than this
                threshold. Prevents stopping prematurely before enough data is gathered
                to make a decision.
            max_progression: Do not stop trials that have passed `max_progression`.
                Useful if we prefer finishing a trial that are already near completion.
            min_curves: Trials will not be stopped until a number of trials
                `min_curves` have completed with curve data attached. That is, if
                `min_curves` trials are completed but their curve data was not
                successfully retrieved, further trials may not be early-stopped.
            trial_indices_to_ignore: Trial indices that should not be early stopped.
            normalize_progressions: Normalizes the progression column of the MapData df
                by dividing by the max. If the values were originally in [0, `prog_max`]
                (as we would expect), the transformed values will be in [0, 1]. Useful
                for inferring the max progression and allows `min_progression` to be
                specified in the transformed space. IMPORTANT: Typically, `min_curves`
                should be > 0 to ensure that at least one trial has completed and that
                we have a reliable approximation for `prog_max`.
        """
        self.metric_names = metric_names
        self.min_progression = min_progression
        self.max_progression = max_progression
        self.min_curves = min_curves
        self.trial_indices_to_ignore = trial_indices_to_ignore
        self.normalize_progressions = normalize_progressions

    @abstractmethod
    def should_stop_trials_early(
        self,
        trial_indices: set[int],
        experiment: Experiment,
    ) -> dict[int, str | None]:
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

    def estimate_early_stopping_savings(
        self, experiment: Experiment, map_key: str | None = None
    ) -> float:
        """Estimate early stopping savings using progressions of the MapMetric present
        on the EarlyStoppingConfig as a proxy for resource usage.

        Args:
            experiment: The experiment containing the trials and metrics used
                to estimate early stopping savings.
            map_key: The name of the map_key by which to estimate early stopping
                savings, usually steps. If none is specified use some arbitrary map_key
                in the experiment's MapData.

        Returns:
            The estimated resource savings as a fraction of total resource usage (i.e.
            0.11 estimated savings indicates we would expect the experiment to have used
            11% more resources without early stopping present)
        """
        if experiment.default_data_constructor is not MapData:
            return 0

        if map_key is None:
            if self.metric_names:
                first_metric_name = next(iter(self.metric_names))
                first_metric = experiment.metrics[first_metric_name]
                map_key = assert_is_instance(first_metric, MapMetric).map_key_info.key

        return estimate_early_stopping_savings(
            experiment=experiment,
            map_key=map_key,
        )

    def _check_validity_and_get_data(
        self, experiment: Experiment, metric_names: list[str]
    ) -> MapData | None:
        """Validity checks and returns the `MapData` used for early stopping that
        is associated with `metric_names`. This function also handles normalizing
        progressions.
        """
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
                    f"{self.__class__.__name__} did not receive data from the "
                    f"objective metric `{metric_name}`. Not stopping any trials."
                )
                return None

        if not isinstance(data, MapData):
            logger.info(
                f"{self.__class__.__name__} expects MapData, but the "
                f"data attached to experiment is of type {type(data)}. "
                "Not stopping any trials."
            )
            return None

        data = assert_is_instance(data, MapData)
        map_keys = data.map_keys
        if len(list(map_keys)) > 1:
            logger.info(
                f"{self.__class__.__name__} expects MapData with a single "
                "map key, but the data attached to the experiment has multiple: "
                f"{data.map_keys}. Not stopping any trials."
            )
            return None
        map_df = data.map_df
        # keep only relevant metrics
        map_df = map_df[map_df["metric_name"].isin(metric_names)].copy()
        if self.normalize_progressions:
            for map_key in map_keys:
                values = map_df[map_key].astype(float)
                map_df[map_key] = values / values.abs().max()
        return MapData(
            df=map_df,
            map_key_infos=data.map_key_infos,
            description=data.description,
        )

    @staticmethod
    def _log_and_return_trial_ignored(
        logger: logging.Logger, trial_index: int
    ) -> tuple[bool, str]:
        """Helper function for logging/constructing a reason when a trial
        should be ignored."""
        logger.info(
            f"Trial {trial_index} should be ignored and not considered "
            "for early stopping."
        )
        return False, "Specified as a trial to be ignored for early stopping."

    @staticmethod
    def _log_and_return_no_data(
        logger: logging.Logger, trial_index: int, metric_name: str
    ) -> tuple[bool, str]:
        """Helper function for logging/constructing a reason when there is no data."""
        logger.info(
            f"There is not yet any data associated with trial {trial_index} and "
            f"metric {metric_name}. Not early stopping this trial."
        )
        return False, "No data available to make an early stopping decision."

    @staticmethod
    def _log_and_return_progression_range(
        logger: logging.Logger,
        trial_index: int,
        trial_last_progression: float,
        min_progression: float | None,
        max_progression: float | None,
        metric_name: str,
    ) -> tuple[bool, str]:
        """Helper function for logging/constructing a reason when min progression
        is not yet reached."""
        reason = (
            f"Most recent progression ({trial_last_progression}) that is available for "
            f"metric {metric_name} falls out of the "
            f"min/max_progression range ({min_progression}, {max_progression})."
        )
        logger.info(
            f"Trial {trial_index}'s m{reason[1:]} Not early stopping this trial."
        )
        return False, reason

    @staticmethod
    def _log_and_return_completed_trials(
        logger: logging.Logger, num_completed: int, min_curves: float
    ) -> tuple[bool, str]:
        """Helper function for logging/constructing a reason when min number of
        completed trials is not yet reached."""
        logger.info(
            f"The number of completed trials ({num_completed}) is less than "
            "the minimum number of curves needed for early stopping "
            f"({min_curves}). Not early stopping."
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
        min_curves: int,
    ) -> tuple[bool, str]:
        """Helper function for logging/constructing a reason when min number of
        trials with data is not yet reached."""
        logger.info(
            f"The number of trials with data ({num_trials_with_data}) "
            f"at trial {trial_index}'s last progression ({trial_last_progression}) "
            "is less than the specified minimum number for early stopping "
            f"({min_curves}). Not early stopping."
        )
        reason = (
            f"Number of trials with data ({num_trials_with_data}) at "
            f"last progression ({trial_last_progression}) is less than the "
            f"specified minimum number for early stopping ({min_curves})."
        )
        return False, reason

    def is_eligible_any(
        self,
        trial_indices: set[int],
        experiment: Experiment,
        df: pd.DataFrame,
        map_key: str | None = None,
    ) -> bool:
        """Perform a series of default checks for a set of trials `trial_indices` and
        determine if at least one of them is eligible for further stopping logic:
            1. Check that at least `self.min_curves` trials have completed`
            2. Check that at least one trial has reached `self.min_progression`
        Returns a boolean indicating if all checks are passed.

        This is useful for some situations where if no trials are eligible for stopping,
        then we can skip costly steps, such as model fitting, that occur before
        individual trials are considered for stopping.
        """
        # check that there are sufficient completed trials
        num_completed = len(experiment.trial_indices_by_status[TrialStatus.COMPLETED])
        if self.min_curves is not None and num_completed < self.min_curves:
            self._log_and_return_completed_trials(
                logger=logger,
                num_completed=num_completed,
                min_curves=none_throws(self.min_curves),
            )
            return False

        # check that at least one trial has reached `self.min_progression`
        df_trials = df[df["trial_index"].isin(trial_indices)].dropna(subset=["mean"])
        any_last_prog = 0
        if not df_trials[map_key].empty:
            any_last_prog = df_trials[map_key].max()
        logger.info(
            f"Last progression of any candidate for trial stopping is {any_last_prog}."
        )
        if self.min_progression is not None and any_last_prog < self.min_progression:
            logger.info(
                f"No trials have reached {self.min_progression}. "
                "Not stopping any trials."
            )
            return False
        return True

    def is_eligible(
        self,
        trial_index: int,
        experiment: Experiment,
        df: pd.DataFrame,
        map_key: str,
    ) -> tuple[bool, str | None]:
        """Perform a series of default checks for a specific trial `trial_index` and
        determines whether it is eligible for further stopping logic:
            1. Check for ignored indices based on `self.trial_indices_to_ignore`
            2. Check that `df` contains data for the trial `trial_index`
            3. Check that the trial has reached `self.min_progression`
            4. Check that the trial hasn't surpassed `self.max_progression`
        Returns two elements: a boolean indicating if all checks are passed and a
        str indicating the reason that early stopping is not applied (None if all
        checks pass).

        Args:
            trial_index: The index of the trial to check.
            experiment: The experiment containing the trial.
            df: A dataframe containing the time-dependent metrics for the trial.
                NOTE: `df` should only contain data with `metric_name` fields that are
                associated with the early stopping strategy. This is usually done
                automatically in `_check_validity_and_get_data`. `is_eligible` might
                otherwise return False even though the trial is eligible, if there are
                secondary tracking metrics that are in `df` but shouldn't be considered
                in the early stopping decision.
            map_key: The name of the column containing the progression (e.g. time).

        Returns:
            A tuple of two elements: a boolean indicating if the trial is eligible and
                an optional string indicating any reason for ineligiblity.
        """
        # check for ignored indices
        if (
            self.trial_indices_to_ignore is not None
            and trial_index in self.trial_indices_to_ignore
        ):
            return self._log_and_return_trial_ignored(
                logger=logger, trial_index=trial_index
            )

        # Check eligibility of each metric.
        for metric_name, metric_df in df.groupby("metric_name"):
            # check for no data
            df_trial = metric_df[metric_df["trial_index"] == trial_index]
            df_trial = df_trial.dropna(subset=["mean"])
            if df_trial.empty:
                return self._log_and_return_no_data(
                    logger=logger, trial_index=trial_index, metric_name=metric_name
                )

            # check for min/max progression
            trial_last_prog = df_trial[map_key].max()
            logger.info(
                f"Last progression of Trial {trial_index} is {trial_last_prog}."
            )
            if (
                self.min_progression is not None
                and trial_last_prog < self.min_progression
            ) or (
                self.max_progression is not None
                and trial_last_prog > self.max_progression
            ):
                return self._log_and_return_progression_range(
                    logger=logger,
                    trial_index=trial_index,
                    trial_last_progression=trial_last_prog,
                    min_progression=self.min_progression,
                    max_progression=self.max_progression,
                    metric_name=metric_name,
                )
        return True, None

    def _default_objective_and_direction(
        self, experiment: Experiment
    ) -> tuple[str, bool]:
        if self.metric_names is None:
            optimization_config = none_throws(experiment.optimization_config)
            objective = optimization_config.objective

            # if multi-objective optimization, infer as first objective
            # although it is recommended to specify a metric name(s) explicitly.
            if isinstance(objective, MultiObjective):
                objective = objective.objectives[0]

            metric_name = objective.metric.name
            minimize = objective.minimize
        else:
            metric_name = list(self.metric_names)[0]
            minimize = experiment.metrics[metric_name].lower_is_better or False
        return metric_name, minimize


class ModelBasedEarlyStoppingStrategy(BaseEarlyStoppingStrategy):
    """A base class for model based early stopping strategies. Includes
    a helper function for processing MapData into arrays."""

    def __init__(
        self,
        metric_names: Iterable[str] | None = None,
        min_progression: float | None = None,
        max_progression: float | None = None,
        min_curves: int | None = None,
        trial_indices_to_ignore: list[int] | None = None,
        normalize_progressions: bool = False,
        min_progression_modeling: float | None = None,
    ) -> None:
        """A ModelBasedEarlyStoppingStrategy class.

        Args:
            metric_names: The names of the metrics the strategy will interact with.
                If no metric names are provided the objective metric is assumed.
            min_progression: Only stop trials if the latest progression value
                (e.g. timestamp, epochs, training data used) is greater than this
                threshold. Prevents stopping prematurely before enough data is gathered
                to make a decision.
            max_progression: Do not stop trials that have passed `max_progression`.
                Useful if we prefer finishing a trial that are already near completion.
            min_curves: Trials will not be stopped until a number of trials
                `min_curves` have completed with curve data attached. That is, if
                `min_curves` trials are completed but their curve data was not
                successfully retrieved, further trials may not be early-stopped.
            trial_indices_to_ignore: Trial indices that should not be early stopped.
            normalize_progressions: Normalizes the progression column of the MapData df
                by dividing by the max. If the values were originally in [0, `prog_max`]
                (as we would expect), the transformed values will be in [0, 1]. Useful
                for inferring the max progression and allows `min_progression` to be
                specified in the transformed space. IMPORTANT: Typically, `min_curves`
                should be > 0 to ensure that at least one trial has completed and that
                we have a reliable approximation for `prog_max`.
            min_progression_modeling: If set, this will exclude progressions that are
                below this threshold from being used or modeling. Useful when early data
                is not reliable or excessively noisy.
        """
        super().__init__(
            metric_names=metric_names,
            min_progression=min_progression,
            max_progression=max_progression,
            min_curves=min_curves,
            trial_indices_to_ignore=trial_indices_to_ignore,
            normalize_progressions=normalize_progressions,
        )
        self.min_progression_modeling = min_progression_modeling

    def _check_validity_and_get_data(
        self, experiment: Experiment, metric_names: list[str]
    ) -> MapData | None:
        """Validity checks and returns the `MapData` used for early stopping that
        is associated with `metric_names`. This function also handles normalizing
        progressions.
        """
        map_data = super()._check_validity_and_get_data(
            experiment=experiment, metric_names=metric_names
        )
        if map_data is not None and self.min_progression_modeling is not None:
            map_df = map_data.map_df
            map_df = map_df[
                map_df[map_data.map_keys[0]] >= self.min_progression_modeling
            ]
            map_data = MapData(
                df=map_df,
                map_key_infos=map_data.map_key_infos,
                description=map_data.description,
            )
        return map_data

    def get_training_data(
        self,
        experiment: Experiment,
        map_data: MapData,
        max_training_size: int | None = None,
        outcomes: Sequence[str] | None = None,
        parameters: list[str] | None = None,
    ) -> EarlyStoppingTrainingData:
        """Processes the raw (untransformed) training data into arrays for
        use in modeling. The trailing dimensions of `X` are the map keys, in
        their originally specified order from `map_data`.

        Args:
            experiment: Experiment that contains the data.
            map_data: The MapData from the experiment, as can be obtained by
                via `_check_validity_and_get_data`.
            max_training_size: Subsample the learning curve to keep the total
                number of data points less than this threshold. Passed to
                MapData's `subsample` method as `limit_rows_per_metric`.

        Returns:
            An `EarlyStoppingTrainingData` that contains training data arrays X, Y,
                and Yvar + a list of arm names.
        """
        if max_training_size is not None:
            map_data = map_data.subsample(
                map_key=map_data.map_keys[0],
                limit_rows_per_metric=max_training_size,
            )
        if outcomes is None:
            # default to the default objective
            metric_name, _ = self._default_objective_and_direction(
                experiment=experiment
            )
            outcomes = [metric_name]

        if parameters is None:
            parameters = list(experiment.search_space.parameters.keys())
            parameters = parameters + list(map_data.map_keys)

        # helper Ax model to make use of its transform functionality
        transform_model = get_transform_helper_model(
            experiment=experiment, data=map_data
        )
        observations_raw = transform_model.get_training_data()
        observations, _ = transform_model._transform_data(
            observations=observations_raw,
            search_space=transform_model._model_space,
            transforms=transform_model._raw_transforms,
            transform_configs=None,
        )
        obs_features, obs_data, arm_names = _unpack_observations(observations)
        X = observation_features_to_array(parameters=parameters, obsf=obs_features)
        Y, Yvar = observation_data_to_array(
            outcomes=list(outcomes), observation_data=obs_data
        )
        return EarlyStoppingTrainingData(X=X, Y=Y, Yvar=Yvar, arm_names=arm_names)


def get_transform_helper_model(
    experiment: Experiment,
    data: Data,
    transforms: list[type[Transform]] | None = None,
) -> MapTorchModelBridge:
    """
    Constructs a TorchModelBridge, to be used as a helper for transforming parameters.
    We perform the default `Cont_X_trans` for parameters but do not perform any
    transforms on the observations.

    Args:
        experiment: Experiment.
        data: Data for fitting the model.

    Returns: A torch modelbridge.
    """
    if transforms is None:
        transforms = Cont_X_trans + [MapUnitX] + Y_trans
    return MapTorchModelBridge(
        experiment=experiment,
        search_space=experiment.search_space,
        data=data,
        model=TorchModel(),
        transforms=transforms,
        fit_out_of_design=True,
    )
