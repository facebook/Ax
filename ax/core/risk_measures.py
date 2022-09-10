#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Union

from ax.utils.common.base import SortableBase
from ax.utils.common.equality import equality_typechecker


class RiskMeasure(SortableBase):
    """A class for defining risk measures.

    This can be used with a `RobustSearchSpace`, to convert the predictions over
    `ParameterDistribution`s to robust metrics, which then get used in candidate
    generation to recommend robust candidates.

    See `ax/modelbridge/modelbridge_utils.py` for `RISK_MEASURE_NAME_TO_CLASS`,
    which lists the supported risk measures, and for `extract_risk_measure`
    helper, which extracts the BoTorch risk measure.
    """

    def __init__(
        self,
        risk_measure: str,
        options: Dict[str, Union[int, float, bool, List[float]]],
    ) -> None:
        """Initialize a risk measure.

        Args:
            risk_measure: The name of the risk measure to use. This should have a
                corresponding entry in `RISK_MEASURE_NAME_TO_CLASS`.
            options: A dictionary of keyword arguments for initializing the risk
                measure. Except for MARS, the risk measure will be initialized as
                `RISK_MEASURE_NAME_TO_CLASS[risk_measure](**options)`. For MARS,
                additional attributes are needed to inform the scalarization.
        """
        super().__init__()
        self.risk_measure = risk_measure
        self.options = options

    def clone(self) -> RiskMeasure:
        """Clone."""
        return RiskMeasure(
            risk_measure=self.risk_measure,
            options=deepcopy(self.options),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            "risk_measure=" + self.risk_measure + ", "
            "options=" + repr(self.options) + ")"
        )

    @property
    def _unique_id(self) -> str:
        return str(self)

    @equality_typechecker
    def __eq__(self, other: RiskMeasure) -> bool:
        return str(self) == str(other)
