# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from ax.core.data import Data
from ax.utils.common.base import SortableBase
from ax.utils.common.typeutils import not_none


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import core  # noqa F401


class AuxiliaryExperiment(SortableBase):
    def __init__(
        self,
        experiment: core.experiment.Experiment,
        data: Optional[Data] = None,
    ) -> None:
        """
        Lightweight container of an experiment, and its data,
        that will be used as auxiliary information for another experiment.
        """
        self.experiment = experiment
        if data is None:
            data = experiment.lookup_data()
        self.data: Data = not_none(data)

    def _unique_id(self) -> str:
        return self.experiment.name
