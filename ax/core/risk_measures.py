#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
from copy import deepcopy
from typing import Dict, List, Type, Union

from ax.exceptions.core import UserInputError
from ax.utils.common.base import SortableBase
from botorch.acquisition.multi_objective.multi_output_risk_measures import (
    IndependentCVaR,
    IndependentVaR,
    MultiOutputExpectation,
    MultiOutputRiskMeasureMCObjective,
    MVaR,
)
from botorch.acquisition.risk_measures import (
    CVaR,
    Expectation,
    RiskMeasureMCObjective,
    VaR,
    WorstCase,
)


"""A mapping of risk measure names to the corresponding classes.

NOTE: This can be extended with user-defined risk measure classes by
importing the dictionary and adding the new risk measure class as
`RISK_MEASURE_NAME_TO_CLASS["my_risk_measure"] = MyRiskMeasure`.
An example of this is found in `tests/test_risk_measure`.
"""
RISK_MEASURE_NAME_TO_CLASS: Dict[str, Type[RiskMeasureMCObjective]] = {
    "Expectation": Expectation,
    "CVaR": CVaR,
    "MVaR": MVaR,
    "IndependentCVaR": IndependentCVaR,
    "IndependentVaR": IndependentVaR,
    "MultiOutputExpectation": MultiOutputExpectation,
    "VaR": VaR,
    "WorstCase": WorstCase,
}


class RiskMeasure(SortableBase):
    """A class for defining risk measures.

    This can be used with a `RobustSearchSpace`, to convert the predictions over
    `ParameterDistribution`s to robust metrics, which then get used in candidate
    generation to recommend robust candidates.
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
                measure. The risk measure will be initialized as
                `RISK_MEASURE_NAME_TO_CLASS[risk_measure](**options)`.
        """
        super().__init__()
        self.risk_measure = risk_measure
        self.options = options
        # Check that the risk measure is valid.
        self.module

    @property
    def is_multi_output(self) -> bool:
        return isinstance(self.module, MultiOutputRiskMeasureMCObjective)

    @property
    @functools.lru_cache()
    def module(self) -> RiskMeasureMCObjective:
        """Get the risk measure objective."""
        try:
            # pyre-ignore Incompatible parameter type [6]
            return RISK_MEASURE_NAME_TO_CLASS[self.risk_measure](**self.options)
        except (KeyError, RuntimeError, ValueError):
            raise UserInputError(
                "Got an error while constructing the risk measure. "
                f"Make sure that {self.risk_measure} exists in  "
                f"`RISK_MEASURE_NAME_TO_CLASS` and accepts arguments {self.options}."
            )

    def clone(self) -> RiskMeasure:
        """Clone."""
        return RiskMeasure(
            risk_measure=self.risk_measure,
            options=deepcopy(self.options),
        )

    def __hash__(self) -> int:
        """Make the class hashable to support the use of `lru_cache` above.

        NOTE: The hash of two `RiskMeasures`s with identical attributes
        will be the same. This is compatible with the use in `lru_cache` above,
        since the resulting risk measures will be the same.
        """
        return hash(repr(self))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            "risk_measure=" + self.risk_measure + ", "
            "options=" + repr(self.options) + ")"
        )

    @property
    def _unique_id(self) -> str:
        return str(self)
