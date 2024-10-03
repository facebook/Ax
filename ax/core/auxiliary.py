# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from enum import Enum, unique
from typing import TYPE_CHECKING

from ax.core.data import Data
from ax.utils.common.base import SortableBase


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401


class AuxiliaryExperiment(SortableBase):
    """Class for defining an auxiliary experiment."""

    def __init__(
        self,
        experiment: core.experiment.Experiment,
        data: Data | None = None,
    ) -> None:
        """
        Lightweight container of an experiment, and its data,
        that will be used as auxiliary information for another experiment.
        """
        self.experiment = experiment
        self.data: Data = data or experiment.lookup_data()

    def _unique_id(self) -> str:
        # While there can be multiple `AuxiliarySource`-s made from the same
        # experiment (and thus sharing the experiment name), the uniqueness
        # here is only needed w.r.t. parent object ("main experiment", for which
        # this will be an auxiliary source for).
        return self.experiment.name


@unique
class AuxiliaryExperimentPurpose(Enum):
    pass
