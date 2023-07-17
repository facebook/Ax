#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, TYPE_CHECKING

import numpy as np
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint, ScalarizedOutcomeConstraint
from ax.exceptions.core import DataRequiredError
from ax.modelbridge.base import unwrap_observation_data
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.ivw import ivw_metric_merge
from ax.utils.common.typeutils import not_none


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401


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
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> OptimizationConfig:
        use_raw_sq = self.config.get("use_raw_status_quo", False)
        has_relative_constraint = any(
            c.relative for c in optimization_config.all_constraints
        )
        if not has_relative_constraint:
            return optimization_config
        # Else, we have at least one relative constraint.
        # Estimate the value at the status quo.
        if modelbridge is None:
            raise ValueError("ModelBridge not supplied to transform.")
        # Unobserved status quo corresponds to a modelbridge.status_quo of None.
        if modelbridge.status_quo is None:
            raise DataRequiredError(
                "Optimization config has relative constraint, but model was "
                "not fit with status quo."
            )
        try:
            f, _ = modelbridge.predict([modelbridge.status_quo.features])
        except Exception:
            # Check if it is out-of-design.
            if use_raw_sq or not modelbridge.model_space.check_membership(
                modelbridge.status_quo.features.parameters
            ):
                # Out-of-design: use the raw observation
                sq_data = ivw_metric_merge(
                    obsd=not_none(modelbridge.status_quo).data,
                    conflicting_noiseless="raise",
                )
                f, _ = unwrap_observation_data([sq_data])
            else:
                # Should have worked.
                raise

        # Plug in the status quo value to each relative constraint.
        for c in optimization_config.all_constraints:
            if c.relative:
                if isinstance(c, ScalarizedOutcomeConstraint):
                    missing_metrics = {
                        metric.name for metric in c.metrics if metric.name not in f
                    }
                    if len(missing_metrics) > 0:
                        raise DataRequiredError(
                            f"Status-quo metric value not yet available for metric(s) "
                            f"{missing_metrics}."
                        )
                    # The sq_val of scalarized outcome is the weighted
                    # sum of its component metrics
                    sq_val = np.sum(
                        [
                            c.weights[i] * f[metric.name][0]
                            for i, metric in enumerate(c.metrics)
                        ]
                    )
                elif c.metric.name in f:
                    sq_val = f[c.metric.name][0]
                else:
                    raise DataRequiredError(
                        f"Status-quo metric value not yet available for metric "
                        f"{c.metric.name}."
                    )
                c.bound = (1 + c.bound / 100.0) * sq_val
                c.relative = False
        return optimization_config

    def untransform_outcome_constraints(
        self,
        outcome_constraints: List[OutcomeConstraint],
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> List[OutcomeConstraint]:
        # We intentionally leave outcome constraints derelativized when
        # untransforming.
        return outcome_constraints
