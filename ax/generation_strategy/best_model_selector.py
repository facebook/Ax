#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.utils.common.base import Base
from ax.utils.common.func_enum import FuncEnum
from pyre_extensions import none_throws

# pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
ARRAYLIKE = np.ndarray | list[float] | list[np.ndarray]


class BestModelSelector(ABC, Base):
    @abstractmethod
    def best_model(self, generator_specs: list[GeneratorSpec]) -> GeneratorSpec:
        """Return the best ``GeneratorSpec`` based on some criteria.

        NOTE: The returned ``GeneratorSpec`` may be a different object than
        what was provided in the original list. It may be possible to
        clone and modify the original ``GeneratorSpec`` to produce one that
        performs better.
        """


# Functions used in ReductionCriterion. They're looked up from the same module.
mean: Callable[[ARRAYLIKE], npt.NDArray] = np.mean
min: Callable[[ARRAYLIKE], npt.NDArray] = np.min
max: Callable[[ARRAYLIKE], npt.NDArray] = np.max


class ReductionCriterion(FuncEnum):
    """An enum for callables that are used for aggregating diagnostics over metrics
    and selecting the best diagnostic in ``SingleDiagnosticBestModelSelector``.

    NOTE: This is used to ensure serializability of the callables.
    """

    MEAN = "mean"
    MIN = "min"
    MAX = "max"

    def __call__(self, array_like: ARRAYLIKE) -> npt.NDArray:
        """This will look up the corresponding function in the same module and
        call it with the given arguments.
        """
        return self._get_function_for_value()(array_like)


class SingleDiagnosticBestModelSelector(BestModelSelector):
    """Choose the best model using a single cross-validation diagnostic.

    The input is a list of ``GeneratorSpec``, each corresponding to one model.
    The specified diagnostic is extracted from each of the models,
    its values (each of which corresponds to a separate metric) are
    aggregated with the aggregation function, the best one is determined
    with the criterion, and the index of the best diagnostic result is returned.

    Example:
     ::
        s = SingleDiagnosticBestModelSelector(
            diagnostic='Fisher exact test p',
            metric_aggregation=ReductionCriterion.MEAN,
            criterion=ReductionCriterion.MIN,
            cv_kwargs={"untransform": False},
        )
        best_model = s.best_model(generator_specs=generator_specs)

    Args:
        diagnostic: The name of the diagnostic to use, which should be
            a key in ``CVDiagnostic``.
        metric_aggregation: ``ReductionCriterion`` applied to the values of the
            diagnostic for a single model to produce a single number.
        criterion: ``ReductionCriterion`` used to determine which of the
            (aggregated) diagnostics is the best.
        cv_kwargs: Optional dictionary of kwargs to pass in while computing
            the cross validation diagnostics.
    """

    def __init__(
        self,
        diagnostic: str,
        metric_aggregation: ReductionCriterion,
        criterion: ReductionCriterion,
        cv_kwargs: dict[str, Any] | None = None,
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
        self.cv_kwargs = cv_kwargs

    def best_model(self, generator_specs: list[GeneratorSpec]) -> GeneratorSpec:
        """Return the best ``GeneratorSpec`` based on the specified diagnostic.

        Args:
            generator_specs: List of ``GeneratorSpec`` to choose from.

        Returns:
            The best ``GeneratorSpec`` based on the specified diagnostic.
        """
        for generator_spec in generator_specs:
            generator_spec.cross_validate(cv_kwargs=self.cv_kwargs)
        aggregated_diagnostic_values = [
            self.metric_aggregation(
                list(none_throws(generator_spec.diagnostics)[self.diagnostic].values())
            )
            for generator_spec in generator_specs
        ]
        best_diagnostic = self.criterion(aggregated_diagnostic_values).item()
        best_index = aggregated_diagnostic_values.index(best_diagnostic)
        return generator_specs[best_index]
