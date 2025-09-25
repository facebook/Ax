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
from ax.adapter.transforms.standardize_y import compute_standardization_parameters
from ax.core.observation import Observation, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import ChoiceParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TParamValue
from ax.generators.types import TConfig
from pyre_extensions import assert_is_instance, none_throws


if TYPE_CHECKING:
    # import as module to make sphinx-autodoc-typehints happy
    from ax import adapter as adapter_module  # noqa F401


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

    requires_data_for_initialization: bool = True

    def __init__(
        self,
        search_space: SearchSpace | None = None,
        experiment_data: ExperimentData | None = None,
        adapter: Optional["adapter_module.base.Adapter"] = None,
        config: TConfig | None = None,
    ) -> None:
        """Initialize StratifiedStandardizeY.

        Args:
            search_space: The search space of the experiment.
            experiment_data: A container for the parameterizations, metadata and
                observations for the trials in the experiment.
                Constructed using ``extract_experiment_data``.
            adapter: Adapter for referencing experiment, status quo, etc.
            config: A dictionary of options that may contain a "parameter_name" key
                specifying the name of the parameter to use for stratification and a
                "strata_mapping" key that corresponds to a dictionary that maps
                parameter values to strata for standardization. The strata can be
                of type bool, int, str, or float.

        """
        assert search_space is not None, "StratifiedStandardizeY requires search space"
        super().__init__(
            search_space=search_space,
            experiment_data=experiment_data,
            adapter=adapter,
            config=config,
        )
        # Get parameter name for standardization.
        self.strata_mapping = None  # pyre-ignore [8]
        if config is not None and "parameter_name" in config:
            # pyre: Attribute `p_name` declared in class `ax.adapter.
            # pyre: transforms.stratified_standardize_y.
            # pyre: StratifiedStandardizeY` has type `str` but is used as type
            # pyre-fixme[8]: `typing.Union[float, int, str]`.
            self.p_name: str = config["parameter_name"]
            strat_p = search_space.parameters[self.p_name]
            if not isinstance(strat_p, ChoiceParameter):
                raise ValueError(f"{self.p_name} not a ChoiceParameter")
            if "strata_mapping" in config:
                # pyre-ignore [8]
                self.strata_mapping: dict[
                    bool | float | int | str, bool | float | int | str
                ] = config["strata_mapping"]
                if set(strat_p.values) != set(self.strata_mapping.keys()):
                    raise ValueError(
                        f"{self.p_name} values {strat_p.values} do not match "
                        f"strata_mapping keys {self.strata_mapping.keys()}."
                    )
        else:
            # See if there is a task parameter
            task_parameters = [
                p.name
                for p in search_space.parameters.values()
                if isinstance(p, ChoiceParameter) and p.is_task
            ]
            if len(task_parameters) == 0:
                raise ValueError(
                    "Must specify parameter for stratified standardization. This can "
                    "happen if TrialAsTask is a no-op, due to there only being a single"
                    " task level."
                )
            elif len(task_parameters) != 1:
                raise ValueError(
                    "Must specify which task parameter to use for stratified "
                    "standardization"
                )
            self.p_name = task_parameters[0]
        if self.strata_mapping is None:
            strat_p = assert_is_instance(
                search_space.parameters[self.p_name], ChoiceParameter
            )
            # pyre-ignore [8]
            self.strata_mapping = {v: v for v in strat_p.values}
        # Compute means and SDs
        experiment_data = none_throws(experiment_data)
        if len(experiment_data.observation_data.index.names) > 2:
            raise NotImplementedError(
                "StratifiedStandardizeY does not support experiment data with "
                "map keys."
            )
        strata = (
            experiment_data.arm_data[self.p_name]
            .map(self.strata_mapping)
            .rename("strata")
        )
        means = experiment_data.observation_data["mean"]
        Ys: dict[tuple[str, TParamValue], list[float]] = {}
        # Group means by strata values and extract corresponding Ys.
        for strata_value, group in means.groupby(strata.loc[means.index]):
            for m in means.columns:
                Ys[(m, strata_value)] = group[m].dropna().values.tolist()

        # Expected `DefaultDict[typing.Union[str, typing.Tuple[str,
        # Optional[typing.Union[bool, float, str]]]], List[float]]` for 1st anonymous
        # parameter to call
        # `ax.adapter.transforms.standardize_y.compute_standardization_parameters`
        # but got `DefaultDict[typing.Tuple[str, Optional[typing.Union[bool, float,
        # str]]], List[float]]`.
        # pyre-fixme[6]: Expected `DefaultDict[Union[str, Tuple[str, Optional[Union[b...
        # pyre-fixme[4]: Attribute must be annotated.
        self.Ymean, self.Ystd = compute_standardization_parameters(Ys)

    def transform_observations(
        self,
        observations: list[Observation],
    ) -> list[Observation]:
        # Transform observations
        for obs in observations:
            v = none_throws(obs.features.parameters[self.p_name])
            strata = self.strata_mapping[v]
            means = np.array(
                [self.Ymean[(m, strata)] for m in obs.data.metric_signatures]
            )
            stds = np.array(
                [self.Ystd[(m, strata)] for m in obs.data.metric_signatures]
            )
            obs.data.means = (obs.data.means - means) / stds
            obs.data.covariance /= np.dot(stds[:, None], stds[:, None].transpose())
        return observations

    def transform_optimization_config(
        self,
        optimization_config: OptimizationConfig,
        adapter: Optional["adapter_module.base.Adapter"] = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> OptimizationConfig:
        if len(optimization_config.all_constraints) == 0:
            return optimization_config
        if fixed_features is None or self.p_name not in fixed_features.parameters:
            raise ValueError(
                f"StratifiedStandardizeY transform requires {self.p_name} to be fixed "
                "during generation."
            )
        v = none_throws(fixed_features.parameters[self.p_name])
        strata = self.strata_mapping[v]
        for c in optimization_config.all_constraints:
            if c.relative:
                raise ValueError(
                    "StratifiedStandardizeY transform does not support relative "
                    f"constraint {c}"
                )
            c.bound = (c.bound - self.Ymean[(c.metric.signature, strata)]) / self.Ystd[
                (c.metric.signature, strata)
            ]
        return optimization_config

    def untransform_observations(
        self,
        observations: list[Observation],
    ) -> list[Observation]:
        for obs in observations:
            v = none_throws(obs.features.parameters[self.p_name])
            strata = self.strata_mapping[v]
            means = np.array(
                [self.Ymean[(m, strata)] for m in obs.data.metric_signatures]
            )
            stds = np.array(
                [self.Ystd[(m, strata)] for m in obs.data.metric_signatures]
            )
            obs.data.means = obs.data.means * stds + means
            obs.data.covariance *= np.dot(stds[:, None], stds[:, None].transpose())
        return observations

    def untransform_outcome_constraints(
        self,
        outcome_constraints: list[OutcomeConstraint],
        fixed_features: ObservationFeatures | None = None,
    ) -> list[OutcomeConstraint]:
        if fixed_features is None or self.p_name not in fixed_features.parameters:
            raise ValueError(
                f"StratifiedStandardizeY requires {self.p_name} to be fixed here"
            )
        v = none_throws(fixed_features.parameters[self.p_name])
        strata = self.strata_mapping[v]
        for c in outcome_constraints:
            if c.relative:
                raise ValueError(
                    "StratifiedStandardizeY does not support relative constraints"
                )
            c.bound = float(
                c.bound * self.Ystd[(c.metric.signature, strata)]
                + self.Ymean[(c.metric.signature, strata)]
            )
        return outcome_constraints

    def transform_experiment_data(
        self, experiment_data: ExperimentData
    ) -> ExperimentData:
        strata = (
            experiment_data.arm_data[self.p_name]
            .map(self.strata_mapping)
            .rename("strata")
        )
        obs_data = experiment_data.observation_data
        mean_data = obs_data["mean"]
        # Get strata values for each row by matching (trial_index, arm_name).
        strata = strata.loc[[v[:2] for v in mean_data.index.values]]
        for metric in mean_data:
            means = strata.apply(lambda x: self.Ymean[(metric, x)])  # noqa B023
            stds = strata.apply(lambda x: self.Ystd[(metric, x)])  # noqa B023
            obs_data[("mean", metric)] = (obs_data[("mean", metric)] - means) / stds
            if obs_data[("sem", metric)].isnull().all():
                # If SEM is NaN, we don't need to transform it.
                continue
            obs_data[("sem", metric)] = obs_data[("sem", metric)] / stds
        return ExperimentData(
            arm_data=experiment_data.arm_data, observation_data=obs_data
        )
