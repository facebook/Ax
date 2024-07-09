#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import partial
from typing import Callable, List, Union

import numpy as np
from ax.exceptions.core import UserInputError
from ax.modelbridge.model_spec import ModelSpec
from ax.utils.common.base import Base
from ax.utils.common.typeutils import not_none

ARRAYLIKE = Union[np.ndarray, List[float], List[np.ndarray]]


class BestModelSelector(ABC, Base):
    @abstractmethod
    def best_model(self, model_specs: List[ModelSpec]) -> int:
        """
        Return the index of the best ``ModelSpec``.
        """


class ReductionCriterion(Enum):
    """An enum for callables that are used for aggregating diagnostics over metrics
    and selecting the best diagnostic in ``SingleDiagnosticBestModelSelector``.

    NOTE: This is used to ensure serializability of the callables.
    """

    # NOTE: Callables need to be wrapped in `partial` to be registered as members.
    MEAN: Callable[[ARRAYLIKE], np.ndarray] = partial(np.mean)
    MIN: Callable[[ARRAYLIKE], np.ndarray] = partial(np.min)
    MAX: Callable[[ARRAYLIKE], np.ndarray] = partial(np.max)

    def __call__(self, array_like: ARRAYLIKE) -> np.ndarray:
        return self.value(array_like)


class SingleDiagnosticBestModelSelector(BestModelSelector):
    """Choose the best model using a single cross-validation diagnostic.

    The input is a list of ``ModelSpec``, each corresponding to one model.
    The specified diagnostic is extracted from each of the models,
    its values (each of which corresponds to a separate metric) are
    aggregated with the aggregation function, the best one is determined
    with the criterion, and the index of the best diagnostic result is returned.

    Example:
     ::
        s = SingleDiagnosticBestModelSelector(
            diagnostic = 'Fisher exact test p',
            metric_aggregation = ReductionCriterion.MEAN,
            criterion = ReductionCriterion.MIN,
        )
        best_diagnostic_index = s.best_diagnostic(diagnostics)

    Args:
        diagnostic: The name of the diagnostic to use, which should be
            a key in ``CVDiagnostic``.
        metric_aggregation: ``ReductionCriterion`` applied to the values of the
            diagnostic for a single model to produce a single number.
        criterion: ``ReductionCriterion`` used to determine which of the
            (aggregated) diagnostics is the best.

    Returns:
        int: index of the selected best diagnostic.
    """

    def __init__(
        self,
        diagnostic: str,
        metric_aggregation: ReductionCriterion,
        criterion: ReductionCriterion,
    ) -> None:
        self.diagnostic = diagnostic
        if not isinstance(metric_aggregation, ReductionCriterion) or not isinstance(
            criterion, ReductionCriterion
        ):
            raise UserInputError(
                "Both `metric_aggregation` and `criterion` must be "
                f"`ReductionCriterion`. Got {metric_aggregation=}, {criterion=}."
            )
        if criterion == ReductionCriterion.MEAN:
            raise UserInputError(
                f"{criterion=} is not supported. Please use MIN or MAX."
            )
        self.metric_aggregation = metric_aggregation
        self.criterion = criterion

    def best_model(self, model_specs: List[ModelSpec]) -> int:
        for model_spec in model_specs:
            model_spec.cross_validate()
        aggregated_diagnostic_values = [
            self.metric_aggregation(
                list(not_none(model_spec.diagnostics)[self.diagnostic].values())
            )
            for model_spec in model_specs
        ]
        best_diagnostic = self.criterion(aggregated_diagnostic_values).item()
        return aggregated_diagnostic_values.index(best_diagnostic)
