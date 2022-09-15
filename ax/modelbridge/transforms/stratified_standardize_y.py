#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from logging import Logger
from typing import DefaultDict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from ax.core.observation import Observation, ObservationFeatures, separate_observations
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import ChoiceParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TParamValue
from ax.modelbridge.transforms.base import Transform
from ax.modelbridge.transforms.standardize_y import compute_standardization_parameters
from ax.models.types import TConfig
from ax.utils.common.logger import get_logger


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import modelbridge as modelbridge_module  # noqa F401  # pragma: no cover

logger: Logger = get_logger(__name__)


class StratifiedStandardizeY(Transform):
    """Standardize Y, separately for each metric and for each value of a
    ChoiceParameter.

    The name of the parameter by which to stratify the standardization can be
    specified in config["parameter_name"]. If not specified, will use a task
    parameter if search space contains exactly 1 task parameter, and will raise
    an exception otherwise.

    The stratification parameter must be fixed during generation if there are
    outcome constraints, in order to apply the standardization to the
    constraints.

    Transform is done in-place.
    """

    def __init__(
        self,
        search_space: Optional[SearchSpace] = None,
        observations: Optional[List[Observation]] = None,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        config: Optional[TConfig] = None,
    ) -> None:
        assert search_space is not None, "StratifiedStandardizeY requires search space"
        assert observations is not None, "StratifiedStandardizeY requires observations"
        # Get parameter name for standardization.
        if config is not None and "parameter_name" in config:
            # pyre: Attribute `p_name` declared in class `ax.modelbridge.
            # pyre: transforms.stratified_standardize_y.
            # pyre: StratifiedStandardizeY` has type `str` but is used as type
            # pyre-fixme[8]: `typing.Union[float, int, str]`.
            self.p_name: str = config["parameter_name"]
            strat_p = search_space.parameters[self.p_name]
            if not isinstance(strat_p, ChoiceParameter):
                raise ValueError(f"{self.p_name} not a ChoiceParameter")
        else:
            # See if there is a task parameter
            task_parameters = [
                p.name
                for p in search_space.parameters.values()
                if isinstance(p, ChoiceParameter) and p.is_task
            ]
            if len(task_parameters) == 0:
                raise ValueError(
                    "Must specify parameter for stratified standardization"
                )
            elif len(task_parameters) != 1:
                raise ValueError(
                    "Must specify which task parameter to use for stratified "
                    "standardization"
                )
            self.p_name = task_parameters[0]
        # Compute means and SDs
        observation_features, observation_data = separate_observations(observations)
        Ys: DefaultDict[Tuple[str, TParamValue], List[float]] = defaultdict(list)
        for j, obsd in enumerate(observation_data):
            v = observation_features[j].parameters[self.p_name]
            for i, m in enumerate(obsd.metric_names):
                Ys[(m, v)].append(obsd.means[i])
        # Expected `DefaultDict[typing.Union[str, typing.Tuple[str,
        # Optional[typing.Union[bool, float, str]]]], List[float]]` for 1st anonymous
        # parameter to call
        # `ax.modelbridge.transforms.standardize_y.compute_standardization_parameters`
        # but got `DefaultDict[typing.Tuple[str, Optional[typing.Union[bool, float,
        # str]]], List[float]]`.
        # pyre-fixme[6]: Expected `DefaultDict[Union[str, Tuple[str, Optional[Union[b...
        # pyre-fixme[4]: Attribute must be annotated.
        self.Ymean, self.Ystd = compute_standardization_parameters(Ys)

    def transform_observations(
        self,
        observations: List[Observation],
    ) -> List[Observation]:
        # Transform observations
        for obs in observations:
            v = obs.features.parameters[self.p_name]
            means = np.array([self.Ymean[(m, v)] for m in obs.data.metric_names])
            stds = np.array([self.Ystd[(m, v)] for m in obs.data.metric_names])
            obs.data.means = (obs.data.means - means) / stds
            obs.data.covariance /= np.dot(stds[:, None], stds[:, None].transpose())
        return observations

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        modelbridge: Optional["modelbridge_module.base.ModelBridge"] = None,
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> OptimizationConfig:
        if len(optimization_config.all_constraints) == 0:
            return optimization_config
        if fixed_features is None or self.p_name not in fixed_features.parameters:
            raise ValueError(
                f"StratifiedStandardizeY transform requires {self.p_name} to be fixed "
                "during generation."
            )
        v = fixed_features.parameters[self.p_name]
        for c in optimization_config.all_constraints:
            if c.relative:
                raise ValueError(
                    "StratifiedStandardizeY transform does not support relative "
                    f"constraint {c}"
                )
            c.bound = (c.bound - self.Ymean[(c.metric.name, v)]) / self.Ystd[
                (c.metric.name, v)
            ]
        return optimization_config

    def untransform_observations(
        self,
        observations: List[Observation],
    ) -> List[Observation]:
        for obs in observations:
            v = obs.features.parameters[self.p_name]
            means = np.array([self.Ymean[(m, v)] for m in obs.data.metric_names])
            stds = np.array([self.Ystd[(m, v)] for m in obs.data.metric_names])
            obs.data.means = obs.data.means * stds + means
            obs.data.covariance *= np.dot(stds[:, None], stds[:, None].transpose())
        return observations

    def untransform_outcome_constraints(
        self,
        outcome_constraints: List[OutcomeConstraint],
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> List[OutcomeConstraint]:
        if fixed_features is None or self.p_name not in fixed_features.parameters:
            raise ValueError(  # pragma: no cover
                f"StratifiedStandardizeY requires {self.p_name} to be fixed here"
            )
        v = fixed_features.parameters[self.p_name]
        for c in outcome_constraints:
            if c.relative:
                raise ValueError(  # pragma: no cover
                    "StratifiedStandardizeY does not support relative constraints"
                )
            c.bound = float(
                c.bound * self.Ystd[(c.metric.name, v)] + self.Ymean[(c.metric.name, v)]
            )
        return outcome_constraints
