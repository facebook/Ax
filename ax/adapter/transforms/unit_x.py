#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import TYPE_CHECKING

from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.core.observation import ObservationFeatures
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.generators.types import TConfig

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


class UnitX(Transform):
    """Map X to [0, 1]^d for RangeParameter of type float and not log scale.

    Uses bounds l <= x <= u, sets x_tilde_i = (x_i - l_i) / (u_i - l_i).
    Constraints wTx <= b are converted to gTx_tilde <= h, where
    g_i = w_i (u_i - l_i) and h = b - wTl.

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: adapter_module.base.Adapter | None = None,
        config: TConfig | None = None,
    ) -> None:
        assert search_space is not None, "UnitX requires search space"
        super().__init__(
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        # Identify parameters that should be transformed
        self.bounds: dict[str, tuple[float, float]] = {}
        for p_name, p in search_space.parameters.items():
            if (
                isinstance(p, RangeParameter)
                and p.parameter_type == ParameterType.FLOAT
                and not p.log_scale
            ):
                self.bounds[p_name] = (p.lower, p.upper)

    def transform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, (l, u) in self.bounds.items():
                if p_name in obsf.parameters:
                    # pyre: param is declared to have type `float` but is used
                    # pyre-fixme[9]: as type `Optional[typing.Union[bool, float, str]]`.
                    param: float = obsf.parameters[p_name]
                    obsf.parameters[p_name] = self._normalize_value(param, (l, u))
        return observation_features

    def transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        for p_name, p in search_space.parameters.items():
            if (p_bounds := self.bounds.get(p_name)) is not None and isinstance(
                p, RangeParameter
            ):
                p.update_range(
                    lower=self._normalize_value(value=p.lower, bounds=p_bounds),
                    upper=self._normalize_value(value=p.upper, bounds=p_bounds),
                )
                if p.target_value is not None:
                    p._target_value = self._normalize_value(
                        # pyre-fixme[6]: For 1st argument expected `float` but got
                        #  `Union[bool, float, int, str]`.
                        value=p.target_value,
                        bounds=p_bounds,
                    )
        new_constraints: list[ParameterConstraint] = []
        for c in search_space.parameter_constraints:
            constraint_dict: dict[str, float] = {}
            bound = float(c.bound)
            for p_name, w in c.constraint_dict.items():
                # p is RangeParameter, but may not be transformed (Int or log)
                if p_name in self.bounds:
                    l, u = self.bounds[p_name]
                    new_w = w * (u - l)
                    constraint_dict[p_name] = new_w
                    bound -= w * l
                else:
                    constraint_dict[p_name] = w

            expr = " + ".join(
                f"{coeff} * {param}" for param, coeff in constraint_dict.items()
            )
            new_constraints.append(
                ParameterConstraint(
                    inequality=f"{expr} <= {bound}",
                )
            )
        search_space.set_parameter_constraints(new_constraints)
        return search_space

    def untransform_observation_features(
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, (l, u) in self.bounds.items():
                if p_name in obsf.parameters:
                    # pyre: param is declared to have type `float` but is used as
                    # pyre-fixme[9]: type `Optional[typing.Union[bool, float, str]]`.
                    param: float = obsf.parameters[p_name]
                    obsf.parameters[p_name] = param * (u - l) + l
        return observation_features

    def _normalize_value(self, value: float, bounds: tuple[float, float]) -> float:
        """Normalize the given value - bounds pair to [0.0, 1.0]."""
        lower, upper = bounds
        return (value - lower) / (upper - lower)

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        arm_data = experiment_data.arm_data
        if arm_data.empty:
            return experiment_data
        for p_name, (l, u) in self.bounds.items():
            if p_name in arm_data.columns:
                arm_data[p_name] = (arm_data[p_name] - l) / (u - l)
        return ExperimentData(
            arm_data=arm_data, observation_data=experiment_data.observation_data
        )
