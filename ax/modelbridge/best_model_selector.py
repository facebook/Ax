#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from functools import partial
from typing import Any, Union

import numpy as np
import numpy.typing as npt
from ax.exceptions.core import UserInputError
from ax.modelbridge.model_spec import ModelSpec
from ax.utils.common.base import Base
from pyre_extensions import none_throws

# pyre-fixme[24]: Generic type `np.ndarray` expects 2 type parameters.
ARRAYLIKE = Union[np.ndarray, list[float], list[np.ndarray]]


class BestModelSelector(ABC, Base):
    @abstractmethod
    def best_model(self, model_specs: list[ModelSpec]) -> ModelSpec:
        """Return the best ``ModelSpec`` based on some criteria.

        NOTE: The returned ``ModelSpec`` may be a different object than
        what was provided in the original list. It may be possible to
        clone and modify the original ``ModelSpec`` to produce one that
        performs better.
        """


class ReductionCriterion(Enum):
    """An enum for callables that are used for aggregating diagnostics over metrics
    and selecting the best diagnostic in ``SingleDiagnosticBestModelSelector``.

    NOTE: This is used to ensure serializability of the callables.
    """

    # NOTE: Callables need to be wrapped in `partial` to be registered as members.
    # pyre-fixme[35]: Target cannot be annotated.
    MEAN: Callable[[ARRAYLIKE], npt.NDArray] = partial(np.mean)
    # pyre-fixme[35]: Target cannot be annotated.
    MIN: Callable[[ARRAYLIKE], npt.NDArray] = partial(np.min)
    # pyre-fixme[35]: Target cannot be annotated.
    MAX: Callable[[ARRAYLIKE], npt.NDArray] = partial(np.max)

    def __call__(self, array_like: ARRAYLIKE) -> npt.NDArray:
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
            diagnostic='Fisher exact test p',
            metric_aggregation=ReductionCriterion.MEAN,
            criterion=ReductionCriterion.MIN,
            model_cv_kwargs={"untransform": False},
        )
        best_model = s.best_model(model_specs=model_specs)

    Args:
        diagnostic: The name of the diagnostic to use, which should be
            a key in ``CVDiagnostic``.
        metric_aggregation: ``ReductionCriterion`` applied to the values of the
            diagnostic for a single model to produce a single number.
        criterion: ``ReductionCriterion`` used to determine which of the
            (aggregated) diagnostics is the best.
        model_cv_kwargs: Optional dictionary of kwargs to pass in while computing
            the cross validation diagnostics.
    """

    def __init__(
        self,
        diagnostic: str,
        metric_aggregation: ReductionCriterion,
        criterion: ReductionCriterion,
        model_cv_kwargs: dict[str, Any] | None = None,
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
        self.model_cv_kwargs = model_cv_kwargs

    def best_model(self, model_specs: list[ModelSpec]) -> ModelSpec:
        """Return the best ``ModelSpec`` based on the specified diagnostic.

        Args:
            model_specs: List of ``ModelSpec`` to choose from.

        Returns:
            The best ``ModelSpec`` based on the specified diagnostic.
        """
        for model_spec in model_specs:
            model_spec.cross_validate(model_cv_kwargs=self.model_cv_kwargs)
        aggregated_diagnostic_values = [
            self.metric_aggregation(
                list(none_throws(model_spec.diagnostics)[self.diagnostic].values())
            )
            for model_spec in model_specs
        ]
        best_diagnostic = self.criterion(aggregated_diagnostic_values).item()
        best_index = aggregated_diagnostic_values.index(best_diagnostic)
        return model_specs[best_index]
