#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Optional

import numpy as np
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import ScalarizedOutcomeConstraint
from ax.modelbridge.base import unwrap_observation_data
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.ivw import ivw_metric_merge


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401  # pragma: no cover


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
        modelbridge: Optional["modelbridge_module.base.ModelBridge"],
        fixed_features: ObservationFeatures,
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
        if modelbridge.status_quo is None:
            raise ValueError(
                "Optimization config has relative constraint, but model was "
                "not fit with status quo."
            )
        try:
            # pyre-fixme[16]: `Optional` has no attribute `features`.
            f, _ = modelbridge.predict([modelbridge.status_quo.features])
        except Exception:
            # Check if it is out-of-design.
            if use_raw_sq or not modelbridge.model_space.check_membership(
                modelbridge.status_quo.features.parameters
            ):
                # Out-of-design: use the raw observation
                sq_data = ivw_metric_merge(
                    # pyre-fixme[16]: `Optional` has no attribute `data`.
                    obsd=modelbridge.status_quo.data,
                    conflicting_noiseless="raise",
                )
                f, _ = unwrap_observation_data([sq_data])
            else:
                # Should have worked.
                raise  # pragma: no cover

        # Plug in the status quo value to each relative constraint.
        for c in optimization_config.all_constraints:
            if c.relative:
                if isinstance(c, ScalarizedOutcomeConstraint):
                    # The sq_val of scalarized outcome is the weighted
                    # sum of its component metrics
                    sq_val = np.sum(
                        [
                            c.weights[i] * f[metric.name][0]
                            for i, metric in enumerate(c.metrics)
                        ]
                    )
                else:
                    sq_val = f[c.metric.name][0]
                c.bound = (1 + c.bound / 100.0) * sq_val
                c.relative = False
        return optimization_config
