#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ax.adapter.data_utils import ExperimentData
from ax.adapter.transforms.base import Transform
from ax.adapter.transforms.standardize_y import (
    _build_objective_from_metric_weights,
    compute_standardization_parameters,
)
from ax.core.observation import Observation, ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import ChoiceParameter
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp, TParamValue
from ax.generators.types import TConfig
from ax.utils.common.sympy import build_constraint_expression_str
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
        adapter: adapter_module.base.Adapter | None = None,
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
        strata_mapping: dict[TParamValue, TParamValue] | None = None
        if "parameter_name" in self.config:
            self.p_name: str = assert_is_instance(self.config["parameter_name"], str)
            strat_p = search_space.parameters[self.p_name]
            if not isinstance(strat_p, ChoiceParameter):
                raise ValueError(f"{self.p_name} not a ChoiceParameter")
            if "strata_mapping" in self.config:
                strata_map = assert_is_instance(self.config["strata_mapping"], dict)
                strata_mapping = strata_map
                if set(strat_p.values) != set(strata_map.keys()):
                    raise ValueError(
                        f"{self.p_name} values {strat_p.values} do not match "
                        f"strata_mapping keys {strata_map.keys()}."
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
        if strata_mapping is None:
            strat_p = assert_is_instance(
                search_space.parameters[self.p_name], ChoiceParameter
            )
            strata_mapping = {v: v for v in strat_p.values}
        self.strata_mapping: dict[TParamValue, TParamValue] = strata_mapping
        # Compute means and SDs
        experiment_data = none_throws(experiment_data)
        if len(experiment_data.observation_data.index.names) > 2:
            raise NotImplementedError(
                "StratifiedStandardizeY does not support experiment data with map keys."
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

        self.Ymean: dict[tuple[str, TParamValue], float]
        self.Ystd: dict[tuple[str, TParamValue], float]
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
        adapter: adapter_module.base.Adapter | None = None,
        fixed_features: ObservationFeatures | None = None,
    ) -> OptimizationConfig:
        if len(optimization_config.all_constraints) == 0 and not (
            optimization_config.objective.is_scalarized_objective
        ):
            return optimization_config
        if fixed_features is None or self.p_name not in fixed_features.parameters:
            raise ValueError(
                f"StratifiedStandardizeY transform requires {self.p_name} to be fixed "
                "during generation."
            )
        v = none_throws(fixed_features.parameters[self.p_name])
        strata = self.strata_mapping[v]

        # Handle scalarized objective: update weights by multiplying with std.
        # Transform \sum (wi * yi) to \sum (wi * si * zi) where zi = (yi - mu_i) / si
        # The constant term \sum (wi * mu_i) doesn't affect optimization.
        if optimization_config.objective.is_scalarized_objective:
            objective = optimization_config.objective
            obj_sigs = [
                self._get_metric_signature(n, adapter) for n in objective.metric_names
            ]
            old_weights = [w for _, w in objective.metric_weights]
            new_weights = [
                old_weights[i] * float(self.Ystd[(sig, strata)])
                for i, sig in enumerate(obj_sigs)
            ]
            new_metric_weights = [
                (name, new_w)
                for (name, _), new_w in zip(objective.metric_weights, new_weights)
            ]
            optimization_config.objective = _build_objective_from_metric_weights(
                new_metric_weights
            )

        optimization_config.outcome_constraints = self._transform_constraints(
            optimization_config.outcome_constraints, strata, adapter
        )

        if isinstance(optimization_config, MultiObjectiveOptimizationConfig):
            optimization_config.objective_thresholds = self._transform_constraints(
                optimization_config.objective_thresholds, strata, adapter
            )

        return optimization_config

    def _transform_constraints(
        self,
        constraints: list[OutcomeConstraint],
        strata: TParamValue,
        adapter: adapter_module.base.Adapter | None = None,
    ) -> list[OutcomeConstraint]:
        """Transform a list of constraints by standardizing bounds."""
        new_constraints = []
        for c in constraints:
            if c.relative:
                raise ValueError(
                    "StratifiedStandardizeY transform does not support relative "
                    f"constraint {c}"
                )
            if len(c.metric_names) > 1:
                c_sigs = [
                    self._get_metric_signature(n, adapter) for n in c.metric_names
                ]
                # Transform \sum (wi * yi) <= C to
                # \sum (wi * si * zi) <= C - \sum (wi * mu_i)
                # Update bound and weights.
                old_weights = [w for _, w in c.metric_weights]
                new_bound = float(
                    c.bound
                    - sum(
                        old_weights[i] * self.Ymean[(sig, strata)]
                        for i, sig in enumerate(c_sigs)
                    )
                )
                new_weights = [
                    old_weights[i] * self.Ystd[(sig, strata)]
                    for i, sig in enumerate(c_sigs)
                ]
                new_metric_weights = [
                    (name, new_w)
                    for (name, _), new_w in zip(c.metric_weights, new_weights)
                ]
                op_str = ">=" if c.op == ComparisonOp.GEQ else "<="
                new_constraints.append(
                    OutcomeConstraint(
                        expression=build_constraint_expression_str(
                            metric_weights=new_metric_weights,
                            op=op_str,
                            bound=new_bound,
                            relative=False,
                        )
                    )
                )
            else:
                c_sig = self._get_metric_signature(c.metric_names[0], adapter)
                new_bound = float(
                    (c.bound - self.Ymean[(c_sig, strata)]) / self.Ystd[(c_sig, strata)]
                )
                op_str = ">=" if c.op == ComparisonOp.GEQ else "<="
                new_constraints.append(
                    OutcomeConstraint(
                        expression=build_constraint_expression_str(
                            metric_weights=c.metric_weights,
                            op=op_str,
                            bound=new_bound,
                            relative=c.relative,
                        )
                    )
                )
        return new_constraints

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
        new_constraints = []
        for c in outcome_constraints:
            if c.relative:
                raise ValueError(
                    "StratifiedStandardizeY does not support relative constraints"
                )
            if len(c.metric_names) > 1:
                c_sigs = [self._get_metric_signature(n) for n in c.metric_names]
                # Untransform \sum (wi * si * zi) <= C' back to \sum (wi * yi) <= C
                # where C' = C - \sum (wi * mu_i) and weights were multiplied by si.
                # First untransform weights, then untransform bound.
                old_weights = [w for _, w in c.metric_weights]
                new_weights = [
                    old_weights[i] / self.Ystd[(sig, strata)]
                    for i, sig in enumerate(c_sigs)
                ]
                new_bound = float(
                    c.bound
                    + sum(
                        new_weights[i] * self.Ymean[(sig, strata)]
                        for i, sig in enumerate(c_sigs)
                    )
                )
                new_metric_weights = [
                    (name, new_w)
                    for (name, _), new_w in zip(c.metric_weights, new_weights)
                ]
                op_str = ">=" if c.op == ComparisonOp.GEQ else "<="
                new_constraints.append(
                    OutcomeConstraint(
                        expression=build_constraint_expression_str(
                            metric_weights=new_metric_weights,
                            op=op_str,
                            bound=new_bound,
                            relative=False,
                        )
                    )
                )
            else:
                c_sig = self._get_metric_signature(c.metric_names[0])
                new_bound = float(
                    c.bound * self.Ystd[(c_sig, strata)] + self.Ymean[(c_sig, strata)]
                )
                op_str = ">=" if c.op == ComparisonOp.GEQ else "<="
                new_constraints.append(
                    OutcomeConstraint(
                        expression=build_constraint_expression_str(
                            metric_weights=c.metric_weights,
                            op=op_str,
                            bound=new_bound,
                            relative=c.relative,
                        )
                    )
                )
        return new_constraints

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
