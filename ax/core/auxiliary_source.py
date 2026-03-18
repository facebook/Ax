# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from logging import Logger
from typing import Any

from ax.adapter.data_utils import ExperimentData
from ax.core.auxiliary import AuxiliaryExperiment
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.observation import Observation
from ax.core.parameter import (
    ChoiceParameter,
    DerivedParameter,
    FixedParameter,
    Parameter,
    RangeParameter,
)
from ax.core.search_space import SearchSpace
from ax.core.types import TParameterization
from ax.exceptions.core import UnsupportedError, UserInputError
from ax.utils.common.constants import Keys
from ax.utils.common.logger import get_logger
from pyre_extensions import assert_is_instance, none_throws

logger: Logger = get_logger(__name__)


class AuxiliarySource(AuxiliaryExperiment):
    def __init__(
        self,
        experiment: Experiment,
        is_active: bool = True,
        transfer_metric_config: dict[str, set[str]] | None = None,
        transfer_param_config: dict[str, str] | None = None,
        update_fixed_params: bool = True,
        strict_param_equality: bool = False,
        trial_indices: list[int] | None = None,
        metric_names: list[str] | None = None,
    ) -> None:
        """
        Specifies an experiment to be used as an auxiliary source for another
        experiment. Primarily used for transfer learning and multi-source
        experiments.

        NOTE: We clone the experiment before extracting any data from it. In
        addition, we remove some kwargs on the ``GeneratorRun``s attached to the
        trials to reduce the size of the ``AuxiliarySources`` in the DB. For
        additional context, see
        https://fb.workplace.com/groups/aeeng/permalink/1612979666127390/.

        Args:
            experiment: The Ax experiment with the auxiliary data.
            is_active: Whether the auxiliary experiment relation to the target
                experiment is currently active.
            transfer_metric_config: A map from target metric name to a set of auxiliary
                metric names, such that the specified auxiliary metrics will be
                included for that target metric. Otherwise, only metrics with the same
                name as the target metric will be included as sources.
            transfer_param_config: A map from target parameter name to the
                corresponding parameter name on the source experiment, for parameters
                that use different names on the source and target experiments. If not
                provided, will expect parameter to have the same name in both
                experiments.
            update_fixed_params: If True, will replace all fixed params on auxiliary
                arms with the fixed params on the target experiment. This does not
                require fixed params to match between the auxiliary and target
                experiments. Setting this to True is generally recommended as it will
                improve compatibility between the auxiliary and target search spaces,
                eliminating the need to utilize heteregeneous search space methods.
                In most GenerationStrategies, the fixed parameters are removed using
                ``RemoveFixed`` transform, so this should not have an effect on
                modeling and optimization.
            strict_param_equality: If True, will require parameters to match exactly
                from the transfer to the target experiment. If False, for choice
                parameters requires transfer choices to be a subset of target choices,
                for range parameters just requires matching type. This is only used in
                `check_search_space_compatibility`.
            trial_indices: Indices of trials to use from the experiment.
            metric_names: Names of metrics to use from the experiment. If None,
                use all metrics.
        """
        self.experiment: Experiment = experiment.clone_with(
            name=experiment.name,
            trial_indices=trial_indices,
            clear_trial_type=True,
        )
        data = none_throws(experiment.lookup_data())
        self.is_active = is_active
        self.metric_names = metric_names
        self.trial_indices = trial_indices
        if not data.empty:
            if metric_names is not None:
                data = data.filter(metric_names=metric_names)
            if trial_indices is not None:
                data = data.filter(trial_indices=trial_indices)
        self.data: Data = data
        self.transfer_metric_config: dict[str, set[str]] = transfer_metric_config or {}
        self.transfer_param_config: dict[str, str] = transfer_param_config or {}
        self.update_fixed_params = update_fixed_params
        if update_fixed_params and experiment.search_space.is_hierarchical:
            raise UnsupportedError(
                "Cannot update fixed params when using hierarchical search spaces."
            )
        self.strict_param_equality = strict_param_equality
        self.validate_metrics()

    def check_search_space_compatibility(
        self, target_search_space: SearchSpace, filled_params: list[str] | None = None
    ) -> None:
        """
        Check that the search space of this auxiliary experiment is compatible with the
        target search space.

        NOTE: This comparison ignores fixed parameters. We do not include fixed
        parameters in modeling. We remove them using ``RemoveFixed`` transform,
        and add back the fixed parameters from the target space. If the value
        of the fixed parameter changes between the search spaces, we can account
        for its effect indirectly through the task correlations. This is suboptimal,
        but it unblocks some use cases.

        Args:
            target_search_space: The search space for the target experiment.
            filled_params: Names of parameters that will be filled and so can be
                missing from the source search space.

        Raises:
            UserInputError: If the search spaces are not compatible, either because
                a parameter in the target search space is not present in the auxiliary
                experiment or because the auxiliary experiment has extra parameters.
            ValueError: If the search spaces contain parameters with the same name
                but with different domains.
        """
        # Check that target params are present and compatible in transfer space
        for target_p_name, target_p in target_search_space.parameters.items():
            if isinstance(target_p, (FixedParameter, DerivedParameter)):
                # No need to validate FixedParameters or DerivedParameters.
                # See the docstring.
                continue
            transfer_p_name = self.transfer_param_config.get(
                target_p_name, target_p_name
            )
            try:
                parameter = self.experiment.search_space.parameters[transfer_p_name]
            except KeyError:
                if filled_params is not None and target_p_name in filled_params:
                    # This parameter will be filled, so it is not an issue.
                    continue
                raise UserInputError(
                    f"Source experiment is missing parameter {transfer_p_name}."
                )
            _check_parameter_compatibility(
                parameter,
                target_p,
                strict=self.strict_param_equality,
            )

        # Check for extraneous parameters in the transfer search space
        # A map from transfer param name to target param name
        inv_p_map = {v: k for k, v in self.transfer_param_config.items()}
        for (
            transfer_p_name,
            transfer_p,
        ) in self.experiment.search_space.parameters.items():
            if self.update_fixed_params and isinstance(
                transfer_p, (FixedParameter, DerivedParameter)
            ):
                # FixedParameters will be replaced and DerivedParameters are
                # nontunable, so extraneous ones are not an issue.
                continue
            target_p_name = inv_p_map.get(transfer_p_name, transfer_p_name)
            if target_p_name not in target_search_space.parameters:
                raise UserInputError(
                    f"Parameter {transfer_p_name} is not in target search space."
                )

    def _map_params(
        self, old_params: TParameterization, target_search_space: SearchSpace
    ) -> TParameterization:
        """Applies the mapping described in `map_observations` to the given
        parameterization dictionary.
        """
        new_params = old_params.copy()
        if self.update_fixed_params:
            # Remove the fixed params from the auxiliary search space.
            for p_name in old_params:
                if isinstance(
                    self.experiment.search_space.parameters[p_name], FixedParameter
                ):
                    new_params.pop(p_name)
            # Add the fixed params from the target search space.
            for p_name, p in target_search_space.parameters.items():
                if isinstance(p, FixedParameter):
                    new_params[p_name] = p.value

        # Update the parameter names according to the mapping.
        for p_name in target_search_space.parameters:
            if p_name in self.transfer_param_config:
                try:
                    new_params[p_name] = new_params.pop(
                        self.transfer_param_config[p_name]
                    )
                except KeyError:
                    if self.experiment.search_space.is_hierarchical:
                        # For hierarchical search spaces, we may not always have
                        # the full parameterization available. In this case, we
                        # will skip mapping for the missing parameter.
                        continue
                    else:
                        raise
        return new_params

    def map_observations(
        self,
        observations: list[Observation],
        target_search_space: SearchSpace,
    ) -> list[Observation]:
        """
        Map observation parameters from the source problem to the target problem.

        Applies the parameter name map specified in `self.transfer_param_config`. If
        `self.update_fixed_params`, then replaces all fixed params with those from the
        target search space.

        Observations are dropped if they are out-of-design for the source search
        space.

        Args:
            observations: Observations to be mapped to the target search space.
            target_search_space: The target search space.

        Returns:
            A list of observations mapped to the target search space.

        Raises:
            UserInputError: If all observations are out-of-design for the
                source search space.
        """
        mapped_observations = []
        for obs in observations:
            old_params = obs.features.parameters
            # Check that arm is in-design for the source search space.
            if not self.experiment.search_space.check_membership(old_params):
                logger.debug(f"Dropping auxiliary arm {obs.arm_name}, out of design.")
                continue
            new_params = self._map_params(
                old_params=old_params, target_search_space=target_search_space
            )
            new_features = obs.features.clone(replace_parameters=new_params)
            # Update the full parameterization if necessary (for HierarchicalSS).
            if (
                new_features.metadata is not None
                and Keys.FULL_PARAMETERIZATION in new_features.metadata
            ):
                new_full_params = self._map_params(
                    old_params=new_features.metadata[Keys.FULL_PARAMETERIZATION],
                    target_search_space=target_search_space,
                )
                none_throws(new_features.metadata)[Keys.FULL_PARAMETERIZATION] = (
                    new_full_params
                )
            new_obs = Observation(
                features=new_features,
                data=obs.data,
                arm_name=obs.arm_name,
            )
            mapped_observations.append(new_obs)
        if not mapped_observations and observations:
            raise UserInputError(
                "No observations were mapped due to all provided observations being "
                f"out-of-design for the auxiliary experiment {self.experiment.name}."
            )
        return mapped_observations

    def map_experiment_data(
        self,
        experiment_data: ExperimentData,
        target_search_space: SearchSpace,
    ) -> ExperimentData:
        """
        Map experiment data from the source problem to the target problem.

        Applies the parameter name map specified in `self.transfer_param_config`. If
        `self.update_fixed_params`, then replaces all fixed params with those from the
        target search space.

        Data is dropped if it is out-of-design for the source search space.

        Args:
            experiment_data: Experiment data to be mapped to the target search space.
            target_search_space: The target search space.

        Returns:
            An ExperimentData object with its data mapped to the target search space.
        """
        # Filter out-of-design data using vectorized membership check.
        in_design = self.experiment.search_space.check_membership_df(
            arm_data=experiment_data.arm_data,
        )
        arm_data = experiment_data.arm_data.loc[in_design]
        obs_data = experiment_data.observation_data
        obs_data = obs_data[
            obs_data.index.get_level_values("arm_name").isin(
                arm_data.index.get_level_values("arm_name")
            )
        ]

        # Update fixed parameters.
        if self.update_fixed_params:
            # Remove the fixed params from the auxiliary search space.
            # See docstring for `AuxiliarySource.update_fixed_params` for more context.
            existing_fixed = [
                name
                for name, p in self.experiment.search_space.parameters.items()
                if isinstance(p, FixedParameter)
                and (
                    (target_p := target_search_space.parameters.get(name)) is None
                    or isinstance(target_p, FixedParameter)
                )
            ]
            arm_data = arm_data.drop(columns=existing_fixed)
            # Add the fixed params from the target search space.
            # In most cases, the parameter will be dropped using `RemoveFixed`,
            # transform, so the parameter value will not be used for modeling.
            new_fixed = {
                name: p.value
                for name, p in target_search_space.parameters.items()
                if isinstance(p, FixedParameter)
            }
            arm_data = arm_data.assign(**new_fixed)

        # Rename any columns that are in the transfer param config.
        if self.transfer_param_config:
            key_map: dict[str, str] = {
                old: new for new, old in self.transfer_param_config.items()
            }
            arm_data = arm_data.rename(columns=key_map)

            # We also need to rename the full parameterization if it is in the metadata.
            # This is only applicable for the hierarchical search spaces.

            def update_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
                if Keys.FULL_PARAMETERIZATION in metadata:
                    metadata[Keys.FULL_PARAMETERIZATION] = {
                        key_map.get(k, k): v
                        for k, v in metadata[Keys.FULL_PARAMETERIZATION].items()
                    }
                return metadata

            if self.experiment.search_space.is_hierarchical:
                arm_data["metadata"] = arm_data["metadata"].apply(update_metadata)

        return ExperimentData(arm_data=arm_data, observation_data=obs_data)

    def get_metrics_to_transfer_from(self, target_metric: str) -> list[str]:
        """
        Returns list of metrics to be transferred from auxiliary for target metric.

        Will lookup metrics to return for target_metric in self.transfer_metric_config.
        If it isn't there, will expect an exact match by name in the auxiliary dataset.

        Args:
            target_metric: Name of target metric.
        """
        metric_set = self.transfer_metric_config.get(target_metric, {target_metric})
        for metric_name in metric_set:
            if metric_name not in self.data.metric_names:
                raise ValueError(f"Could not find {metric_name} in source data.")
        return list(metric_set)

    def get_data_to_transfer_from(self, target_metric: str) -> Data | None:
        """
        Returns Data to be used for transfer learning. Returns None if
        the `transfer_metric_config` maps the target metric to an empty set.

        Args:
            target_metric: target metric to return data for.
        """
        metric_list = self.get_metrics_to_transfer_from(target_metric=target_metric)
        if metric_list:
            return self.data.filter(metric_names=metric_list)
        else:
            return None

    def validate_metrics(self) -> None:
        """
        Warns if auxiliary metrics from self.transfer_metric_config
        are not present in the underlying dataset.
        """
        metrics_not_in_data = set()

        for transfer_metrics in self.transfer_metric_config.values():
            metrics_not_in_data.update(transfer_metrics - self.data.metric_names)

        if metrics_not_in_data:
            metric_str = ", ".join(metrics_not_in_data)
            logger.warning(
                f"Metrics not in data: {metric_str}."
                " Please validate the metrics match "
                "between your experiments and retry."
            )


def _check_parameter_compatibility(
    param1: Parameter,
    param2: Parameter,
    strict: bool = False,
) -> None:
    """
    Check if param1 is compatible with param2. Does not require names to match.

    If strict, then requires parameters (all properties except
    the names) to be exactly equal. Otherwise,
    - Two RangeParameters or ChoiceParameters: Requires the same parameter type
    - One Fixed and one Range Parameter: Requires the same parameter type
    """
    p1type = param1.parameter_type
    p2type = param2.parameter_type
    if p1type != p2type:
        raise ValueError(f"{param1.name}: {p1type} does not match {p2type}")
    if isinstance(param1, FixedParameter) and isinstance(param2, FixedParameter):
        if param1.value != param2.value:
            raise ValueError(
                f"{param1.name}: Value mismatch from {param1.value} to {param2.value}"
            )
    if strict:
        if (t1 := type(param1)) != (t2 := type(param2)):
            raise ValueError(
                f"{param1.name}: {t1.__name__} does not match {t2.__name__}"
            )
        if isinstance(param1, RangeParameter):
            p1_bounds = (param1.lower, param1.upper)
            p2 = assert_is_instance(param2, RangeParameter)
            p2_bounds = (p2.lower, p2.upper)
            if p1_bounds != p2_bounds:
                raise ValueError(
                    f"{param1.name}: Range mismatch from {p1_bounds} to {p2_bounds}"
                )

        elif isinstance(param1, ChoiceParameter):
            p1vals = set(param1.values)
            p2vals = set(assert_is_instance(param2, ChoiceParameter).values)
            if strict and p1vals != p2vals:
                raise ValueError(
                    f"{param1.name}: Values mismatch from {p1vals} to {p2vals}"
                )
    else:
        # Check that the Parameter types match.
        # TODO: Support choice and derived parameters with other parameter types
        # and support using two fixed parameters
        if (type(param1) is not type(param2)) and (
            {type(param1), type(param2)} != {RangeParameter, FixedParameter}
        ):
            raise ValueError(
                f"{param2.name} ({type(param2)}) is not compatible with "
                f"{param1.name} ({type(param1)})."
            )
