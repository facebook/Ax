#!/usr/bin/env python3

from typing import Optional

from ae.lazarus.ae.core.observation import ObservationFeatures
from ae.lazarus.ae.core.optimization_config import OptimizationConfig
from ae.lazarus.ae.generator.base import unwrap_observation_data
from ae.lazarus.ae.generator.transforms.base import Transform
from ae.lazarus.ae.generator.transforms.ivw import ivw_metric_merge


class Derelativize(Transform):
    """Changes relative constraints to not-relative constraints using a plug-in
    estimate of the status quo value.

    If status quo is in-design, uses model estimate at status quo. If not, uses
    raw observation at status quo.

    Transform is done in-place.
    """

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        generator: Optional["Generator"],
        fixed_features: ObservationFeatures,
    ) -> OptimizationConfig:
        has_relative_constraint = any(
            c.relative for c in optimization_config.outcome_constraints
        )
        if not has_relative_constraint:
            return optimization_config
        # Else, we have at least one relative constraint.
        # Estimate the value at the status quo.
        if generator is None:
            raise ValueError("Generator not supplied to transform.")
        if generator.status_quo is None:
            raise ValueError(
                "Optimization config has relative constraint, but model was "
                "not fit with status quo."
            )
        try:
            f, _ = generator.predict([generator.status_quo.features])
        except Exception:
            # Check if it is out-of-design.
            if not generator.model_space.validate(
                generator.status_quo.features.parameters
            ):
                # Out-of-design: use the raw observation
                sq_data = ivw_metric_merge(
                    obsd=generator.status_quo.data, conflicting_noiseless="raise"
                )
                f, _ = unwrap_observation_data([sq_data])
            else:
                # Should have worked.
                raise  # pragma: no cover

        # Plug in the status quo value to each relative constraint.
        for c in optimization_config.outcome_constraints:
            if c.relative:
                # Compute new bound.
                c.bound = (1 + c.bound / 100.0) * f[c.metric.name][0]
                c.relative = False
        return optimization_config
