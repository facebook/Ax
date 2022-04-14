#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.transforms.base import Transform
from ax.models.types import TConfig

if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401  # pragma: no cover


class UnitX(Transform):
    """Map X to [0, 1]^d for RangeParameter of type float and not log scale.

    Uses bounds l <= x <= u, sets x_tilde_i = (x_i - l_i) / (u_i - l_i).
    Constraints wTx <= b are converted to gTx_tilde <= h, where
    g_i = w_i (u_i - l_i) and h = b - wTl.

    Transform is done in-place.
    """

    target_lb: float = 0.0
    target_range: float = 1.0

    def __init__(
        self,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        # Identify parameters that should be transformed
        self.bounds: Dict[str, Tuple[float, float]] = {}
        for p_name, p in search_space.parameters.items():
            if (
                isinstance(p, RangeParameter)
                and p.parameter_type == ParameterType.FLOAT
                and not p.log_scale
            ):
                self.bounds[p_name] = (p.lower, p.upper)

    def transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, (l, u) in self.bounds.items():
                if p_name in obsf.parameters:
                    # pyre: param is declared to have type `float` but is used
                    # pyre-fixme[9]: as type `Optional[typing.Union[bool, float, str]]`.
                    param: float = obsf.parameters[p_name]
                    obsf.parameters[p_name] = self._normalize_value(param, (l, u))
        return observation_features

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
        for p_name, p in search_space.parameters.items():
            if p_name in self.bounds and isinstance(p, RangeParameter):
                p.update_range(
                    lower=self.target_lb,
                    upper=self.target_lb + self.target_range,
                )
            if p.target_value is not None:
                p._target_value = self._normalize_value(
                    p.target_value, self.bounds[p_name]  # pyre-ignore[6]
                )
        new_constraints: List[ParameterConstraint] = []
        for c in search_space.parameter_constraints:
            constraint_dict: Dict[str, float] = {}
            bound = float(c.bound)
            for p_name, w in c.constraint_dict.items():
                # p is RangeParameter, but may not be transformed (Int or log)
                if p_name in self.bounds:
                    l, u = self.bounds[p_name]
                    new_w = w * (u - l) / self.target_range
                    constraint_dict[p_name] = new_w
                    bound += self.target_lb * new_w - w * l
                else:
                    constraint_dict[p_name] = w
            new_constraints.append(
                ParameterConstraint(constraint_dict=constraint_dict, bound=bound)
            )
        search_space.set_parameter_constraints(new_constraints)
        return search_space

    def untransform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationFeatures]:
        for obsf in observation_features:
            for p_name, (l, u) in self.bounds.items():
                # pyre: param is declared to have type `float` but is used as
                # pyre-fixme[9]: type `Optional[typing.Union[bool, float, str]]`.
                param: float = obsf.parameters[p_name]
                obsf.parameters[p_name] = (
                    param - self.target_lb
                ) / self.target_range * (u - l) + l
        return observation_features

    def _transform_parameter_distributions(self, search_space: SearchSpace) -> None:
        """Transform the parameter distributions of the given search space, in-place.

        This method should be called in transform_search_space before parameters
        are transformed.
        """
        if not isinstance(search_space, RobustSearchSpace):
            return
        distributions = search_space.parameter_distributions
        for dist in distributions:
            if dist.multiplicative:
                # TODO: Transforming multiplicative distributions is a bit more
                # complicated. Will investigate further and implement as needed.
                raise NotImplementedError(
                    f"{self.__class__.__name__} transform of multiplicative "
                    "distributions is not yet implemented."
                )
            if len(dist.parameters) != 1:
                # Ignore if the ranges of all parameters are same as the target range.
                if (
                    all(
                        self.bounds[p_name][1] - self.bounds[p_name][0]
                        == self.target_range
                        for p_name in dist.parameters
                    )
                    and not search_space.is_environmental
                ):
                    continue
                # TODO: Support transforming multivariate distributions.
                raise UnsupportedError(
                    f"{self.__class__.__name__} transform of multivariate "
                    "distributions is not supported. Consider manually normalizing "
                    "the parameter and the corresponding distribution."
                )
            bounds = self.bounds[dist.parameters[0]]
            p_range = bounds[1] - bounds[0]
            if p_range == self.target_range and (
                not search_space.is_environmental or bounds[0] == self.target_lb
            ):
                # NOTE: This helps avoid raising the error below if using a discrete
                # distribution in cases where we do not need to transform.
                continue
            loc = dist.distribution_parameters.get("loc", 0.0)
            if search_space.is_environmental:
                loc = self._normalize_value(loc, bounds)
            else:
                loc = loc / p_range * self.target_range
            dist.distribution_parameters["loc"] = loc
            dist.distribution_parameters["scale"] = (
                dist.distribution_parameters.get("scale", 1.0)
                / p_range
                * self.target_range
            )
            # Check that the distribution is valid after the transform.
            try:
                dist.distribution
            except TypeError:
                raise UnsupportedError(
                    f"The distribution {str(dist)} does not support transforming via "
                    "`loc` and `scale` arguments. Consider manually normalizing the "
                    "parameter and the corresponding distribution."
                )

    def _normalize_value(self, value: float, bounds: Tuple[float, float]) -> float:
        """Normalize the given value - bounds pair to
        [self.target_lb, self.target_lb + self.target_range].
        """
        lower, upper = bounds
        return (value - lower) / (upper - lower) * self.target_range + self.target_lb
