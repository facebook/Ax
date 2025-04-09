#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from dataclasses import dataclass

from ax.core.base_trial import TrialStatus
from ax.core.trial_status import NON_ABANDONED_STATUSES
from ax.exceptions.core import UnsupportedError


@dataclass(frozen=True)
class DataLoaderConfig:
    """This dataclass contains parameters that control the behavior
    of `Adapter._set_training_data`.

    Args:
        fit_out_of_design: If specified, all training data are used.
            Otherwise, only in design points are used. Note that in-design-ness is
            determined after expanding the modeling space, if applicable.
        fit_abandoned: Whether data for abandoned arms or trials should be included in
            model training data. If `False`, only non-abandoned points are returned.
        fit_only_completed_map_metrics: Whether to fit a model to map metrics only when
            the trial is completed. This is useful for applications like modeling
            partially completed learning curves in AutoML.
        latest_rows_per_group: If specified and data is an instance of MapData, uses
            MapData.latest() with `latest_rows_per_group` to retrieve the most recent
            rows for each group. Useful in cases where learning curves are frequently
            updated, preventing an excessive number of Observation objects.
        limit_rows_per_metric: Subsample the map data so that the total number of
            rows per metric is limited by this value.
        limit_rows_per_group: Subsample the map data so that the number of rows
            in the `map_key` column for each (arm, metric) is limited by this value.
    """

    fit_out_of_design: bool = False
    fit_abandoned: bool = False
    fit_only_completed_map_metrics: bool = True
    latest_rows_per_group: int | None = 1
    limit_rows_per_metric: int | None = None
    limit_rows_per_group: int | None = None

    def __post_init__(self) -> None:
        if self.latest_rows_per_group is not None and (
            self.limit_rows_per_metric is not None
            or self.limit_rows_per_group is not None
        ):
            raise UnsupportedError(
                "`latest_rows_per_group` must be None if either of "
                "`limit_rows_per_metric` or `limit_rows_per_group` is specified."
            )

    @property
    def statuses_to_fit(self) -> set[TrialStatus]:
        """The data from trials in these statuses will be used to fit the model
        for non map metrics. Defaults to all trial statuses if
        `fit_abandoned is True` and all statuses except ABANDONED, otherwise.
        """
        if self.fit_abandoned:
            return set(TrialStatus)
        return NON_ABANDONED_STATUSES

    @property
    def statuses_to_fit_map_metric(self) -> set[TrialStatus]:
        """The data from trials in these statuses will be used to fit the model
        for map metrics. Defaults to only COMPLETED trials if
        `fit_only_completed_map_metrics is True` and to `statuses_to_fit`, otherwise.
        """
        if self.fit_only_completed_map_metrics:
            return {TrialStatus.COMPLETED}
        return self.statuses_to_fit
