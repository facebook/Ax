#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from logging import Logger
from typing import Optional, TYPE_CHECKING

from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.modelbridge.transforms.base import Transform
from ax.utils.common.logger import get_logger
from pyre_extensions import assert_is_instance


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401


logger: Logger = get_logger(__name__)


class Derelativize(Transform):
    """Changes relative constraints to not-relative constraints using a plug-in
    estimate of the status quo value.

    If status quo is in-design, uses model estimate at status quo. If not, uses
    raw observation at status quo.

    Will raise an error if status quo is in-design and model fails to predict
    for it, unless the flag "use_raw_status_quo" is set to True in the
    transform config, in which case it will fall back to using the observed
    value in the training data.

    Transform is done in-place.
    """

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        modelbridge: Optional["modelbridge_module.base.Adapter"] = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> OptimizationConfig:
        use_raw_sq = self.config.get("use_raw_status_quo", False)
        has_relative_constraint = any(
            c.relative for c in optimization_config.all_constraints
        )
        if not has_relative_constraint:
            return optimization_config
        if modelbridge is None:
            raise ValueError("Adapter not supplied to transform.")
        return modelbridge._derelativize_optimization_config(
            optimization_config=optimization_config,
            with_raw_status_quo=assert_is_instance(use_raw_sq, bool),
        )

    def untransform_outcome_constraints(
        self,
        outcome_constraints: list[OutcomeConstraint],
        fixed_features: ObservationFeatures | None = None,
    ) -> list[OutcomeConstraint]:
        # We intentionally leave outcome constraints derelativized when
        # untransforming.
        return outcome_constraints
