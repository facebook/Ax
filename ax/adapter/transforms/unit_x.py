#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional, TYPE_CHECKING

import numpy as np
from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.core.observation import Observation, ObservationFeatures
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.parameter_distribution import ParameterDistribution
from ax.core.search_space import RobustSearchSpace, SearchSpace
from ax.exceptions.core import UnsupportedError, UserInputError
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
        observations: list[Observation] | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: Optional["adapter_module.base.Adapter"] = None,
        config: TConfig | None = None,
    ) -> None:
        assert search_space is not None, "UnitX requires search space"
        super().__init__(
            search_space=search_space,
            observations=observations,
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

    def _transform_search_space(self, search_space: SearchSpace) -> SearchSpace:
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
            new_constraints.append(
                ParameterConstraint(constraint_dict=constraint_dict, bound=bound)
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

    def _transform_parameter_distributions(self, search_space: SearchSpace) -> None:
        """Transform the parameter distributions of the given search space, in-place.

        This method should be called in `transform_search_space` before parameters
        are transformed.
        """
        if not isinstance(search_space, RobustSearchSpace):
            return
        for distribution in search_space.parameter_distributions:
            is_environmental = distribution.is_environmental(search_space=search_space)
            if distribution.multiplicative:
                # TODO: Transforming multiplicative distributions is a bit more
                # complicated. Will investigate further and implement as needed.
                raise NotImplementedError(
                    f"{self.__class__.__name__} transform of multiplicative "
                    "distributions is not yet implemented."
                )
            if len(distribution.parameters) != 1:
                self._transform_multivariate_distribution(
                    distribution=distribution, is_environmental=is_environmental
                )
            else:
                self._transform_univariate_distribution(
                    distribution=distribution, is_environmental=is_environmental
                )

    def _transform_univariate_distribution(
        self, distribution: ParameterDistribution, is_environmental: bool
    ) -> None:
        r"""Transform a univariate distribution in-place."""
        bounds = self.bounds[distribution.parameters[0]]
        p_range = bounds[1] - bounds[0]
        if p_range == 1.0 and (not is_environmental or bounds[0] == 0.0):
            # NOTE: This helps avoid raising the error below if using a discrete
            # distribution in cases where we do not need to transform.
            return
        loc = distribution.distribution_parameters.get("loc", 0.0)
        if is_environmental:
            loc = self._normalize_value(loc, bounds)
        else:
            loc = loc / p_range
        distribution.distribution_parameters["loc"] = loc
        distribution.distribution_parameters["scale"] = (
            distribution.distribution_parameters.get("scale", 1.0) / p_range
        )
        # Check that the distribution is valid after the transform.
        try:
            distribution.distribution
        except TypeError:
            raise UnsupportedError(
                f"The distribution {str(distribution)} does not support transforming "
                "via `loc` and `scale` arguments. Consider manually normalizing the "
                "parameter and the corresponding distribution."
            )

    def _transform_multivariate_distribution(
        self, distribution: ParameterDistribution, is_environmental: bool
    ) -> None:
        r"""Transform a multivariate distribution in-place."""
        # Ignore if the ranges of all parameters are same as the target range.
        if (
            all(
                self.bounds[p_name][1] - self.bounds[p_name][0] == 1.0
                for p_name in distribution.parameters
            )
            and not is_environmental
        ):
            return
        if distribution.distribution_class == "multivariate_normal":
            # If S is cov and A is the diagonal scale matrix,
            # the new cov will be ASA.
            n_dist_params = len(distribution.parameters)
            scale_vec = np.zeros(len(distribution.parameters))
            for i, p in enumerate(distribution.parameters):
                bounds = self.bounds[p]
                p_range = bounds[1] - bounds[0]
                scale_vec[i] = 1.0 / p_range
            cov = np.asarray(distribution.distribution_parameters.get("cov"))
            if cov.shape != (n_dist_params, n_dist_params):
                raise UserInputError(
                    "Expected `cov` to be a square matrix of size equal to "
                    "number of parameters. Received `cov` with shape "
                    f"{cov.shape} for the distribution of {n_dist_params} "
                    "parameters."
                )
            # Same as np.diag(scale_vec) @ cov @ np.diag(scale_vec) but faster.
            distribution.distribution_parameters["cov"] = (
                scale_vec[..., None] * cov * scale_vec[..., None, :]
            )
            mean = np.asarray(distribution.distribution_parameters.get("mean", 0.0))
            if not np.all(mean == 0):
                if mean.shape != (n_dist_params,):
                    raise UserInputError(
                        "Expected `mean` to be an array of shape "
                        f"{(n_dist_params,)}, but received {mean}."
                    )
                # Mean would simply be AM as long as it is not environmental.
                # If environmental, we need to first subtract the lower bounds
                # then add the target lb.
                if is_environmental:
                    lbs = np.array([self.bounds[p][0] for p in distribution.parameters])
                    new_mean = scale_vec * (mean - lbs)
                    distribution.distribution_parameters["mean"] = new_mean
                else:
                    distribution.distribution_parameters["mean"] = scale_vec * mean
        else:
            raise UnsupportedError(
                f"{self.__class__.__name__} transform of multivariate "
                "distributions, other than `multivariate_normal`, is not "
                "supported. Consider manually normalizing the parameter "
                "and the corresponding distribution."
            )

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
            arm_data[p_name] = (arm_data[p_name] - l) / (u - l)
        return ExperimentData(
            arm_data=arm_data, observation_data=experiment_data.observation_data
        )
