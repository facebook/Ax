#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Number
from typing import Callable, Iterable, List

from ax.modelbridge.model_spec import ModelSpec
from ax.utils.common.typeutils import not_none


class BestModelSelector(ABC):
    @abstractmethod
    def best_model(self, model_specs: List[ModelSpec]) -> int:
        """
        Return the index of the best ``ModelSpec``.
        """


class SingleDiagnosticBestModelSelector(BestModelSelector):
    """Choose the best model using a single cross-validation diagnostic.

    The input is a list of CVDiagnostics, each corresponding to one model.
    The specified diagnostic is extracted from each of the CVDiagnostics,
    its values (each of which corresponds to a separate metric) are
    aggregated with the aggregation function, the best one is determined
    with the criterion, and the index of the best diagnostic result is returned.

    Example:
     ::
        s = SingleDiagnosticBestModelSelector(
            diagnostic = 'Fisher exact test p',
            criterion = np.min,
            metric_aggregation = min.mean,
        )
        best_diagnostic_index = s.best_diagnostic(diagnostics)

    Args:
        diagnostic: The name of the diagnostic to use, which should be
            a key in CVDiagnostic.
        metric_aggregation: Callable applied to the values of the diagnostic
            for a single model to produce a single number.
        criterion: Callable used to determine which of the (aggregated)
            diagnostics is the best.

    Returns:
        int: index of the selected best diagnostic.
    """

    def __init__(
        self,
        diagnostic: str,
        metric_aggregation: Callable[[Iterable[Number]], Number],
        criterion: Callable[[Iterable[Number]], Number],
    ) -> None:
        self.diagnostic = diagnostic
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
        best_diagnostic = self.criterion(aggregated_diagnostic_values)
        return aggregated_diagnostic_values.index(best_diagnostic)
