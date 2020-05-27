#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional

from ax.core.base import Base
from ax.core.data import Data


if TYPE_CHECKING:  # pragma: no cover
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401


class Metric(Base):
    """Base class for representing metrics.

    Attributes:
        lower_is_better: Flag for metrics which should be minimized.
    """

    def __init__(self, name: str, lower_is_better: Optional[bool] = None) -> None:
        """Inits Metric.

        Args:
            name: Name of metric.
            lower_is_better: Flag for metrics which should be minimized.
        """
        self._name = name
        self.lower_is_better = lower_is_better

    @property
    def name(self) -> str:
        """Get name of metric."""
        return self._name

    def fetch_trial_data(self, trial: core.base_trial.BaseTrial, **kwargs: Any) -> Data:
        """Fetch data for one trial."""
        raise NotImplementedError(
            f"Metric {self.name} does not implement data-fetching logic."
        )  # pragma: no cover

    def fetch_experiment_data(
        self, experiment: core.experiment.Experiment, **kwargs: Any
    ) -> Data:
        """Fetch this metric's data for an experiment.

        Default behavior is to fetch data from all trials expecting data
        and concatenate the results.
        """
        return Data.from_multiple_data(
            [
                self.fetch_trial_data(trial, **kwargs)
                if trial.status.expecting_data
                else Data()
                for trial in experiment.trials.values()
            ]
        )

    @classmethod
    def fetch_trial_data_multi(
        cls, trial: core.base_trial.BaseTrial, metrics: Iterable[Metric], **kwargs: Any
    ) -> Data:
        """Fetch multiple metrics data for one trial.

        Default behavior calls `fetch_trial_data` for each metric.
        Subclasses should override this to trial data computation for multiple metrics.
        """
        return Data.from_multiple_data(
            [metric.fetch_trial_data(trial, **kwargs) for metric in metrics]
        )

    @classmethod
    def fetch_experiment_data_multi(
        cls,
        experiment: core.experiment.Experiment,
        metrics: Iterable[Metric],
        trials: Optional[Iterable[core.base_trial.BaseTrial]] = None,
        **kwargs: Any,
    ) -> Data:
        """Fetch multiple metrics data for an experiment.

        Default behavior calls `fetch_trial_data_multi` for each trial.
        Subclasses should override to batch data computation across trials + metrics.
        """
        return Data.from_multiple_data(
            [
                cls.fetch_trial_data_multi(trial, metrics, **kwargs)
                if trial.status.expecting_data
                else Data()
                for trial in (experiment.trials.values() if trials is None else trials)
            ]
        )

    def clone(self) -> "Metric":
        """Create a copy of this Metric."""
        return Metric(name=self.name, lower_is_better=self.lower_is_better)

    def __repr__(self) -> str:
        return "{class_name}('{metric_name}')".format(
            class_name=self.__class__.__name__, metric_name=self.name
        )
