#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Type

from ax.core.data import Data
from ax.utils.common.equality import Base
from ax.utils.common.serialization import extract_init_args, serialize_init_args


if TYPE_CHECKING:  # pragma: no cover
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401


class Metric(Base):
    """Base class for representing metrics.

    The `fetch_trial_data` method is the essential method to override when
    subclassing, which specifies how to retrieve a Metric, for a given trial.

    A Metric must return a Data object, which requires (at minimum) the following:
        https://ax.dev/api/_modules/ax/core/data.html#Data.required_columns

    Attributes:
        lower_is_better: Flag for metrics which should be minimized.
        properties: Properties specific to a particular metric.
    """

    def __init__(
        self,
        name: str,
        lower_is_better: Optional[bool] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Inits Metric.

        Args:
            name: Name of metric.
            lower_is_better: Flag for metrics which should be minimized.
            properties: Dictionary of this metric's properties
        """
        self._name = name
        self.lower_is_better = lower_is_better
        self.properties = properties or {}

    @property
    def name(self) -> str:
        """Get name of metric."""
        return self._name

    @property
    def fetch_multi_group_by_metric(self) -> Type[Metric]:
        """Metric class, with which to group this metric in `Experiment._metrics_by_class`,
        which is used to combine metrics on experiment into groups and then fetch their
        data via `Metric.fetch_trial_data_multi` for each group.

        NOTE: By default, this property will just return the class on which it is
        defined; however, in some cases it is useful to group metrics by their
        superclass, in which case this property should return that superclass.
        """
        return self.__class__

    @classmethod
    def serialize_init_args(cls, metric: "Metric") -> Dict[str, Any]:
        """Serialize the properties needed to initialize the metric.
        Used for storage.
        """
        return serialize_init_args(
            object=metric, exclude_fields=["name", "lower_is_better", "precomp_config"]
        )

    @classmethod
    def deserialize_init_args(cls, args: Dict[str, Any]) -> Dict[str, Any]:
        """Given a dictionary, extract the properties needed to initialize the metric.
        Used for storage.
        """
        return extract_init_args(args=args, class_=cls)

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
