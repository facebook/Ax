#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Set, Tuple

from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ChoiceParameter, FixedParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig, TGenMetadata, TParamValueList
from ax.modelbridge.array import (
    array_to_observation_data,
    extract_objective_weights,
    extract_outcome_constraints,
    validate_optimization_config,
)
from ax.modelbridge.base import ModelBridge
from ax.models.discrete_base import DiscreteModel


FIT_MODEL_ERROR = "Model must be fit before {action}."


# pyre-fixme[13]: Attribute `model` is never initialized.
# pyre-fixme[13]: Attribute `outcomes` is never initialized.
# pyre-fixme[13]: Attribute `parameters` is never initialized.
# pyre-fixme[13]: Attribute `search_space` is never initialized.
class DiscreteModelBridge(ModelBridge):
    """A model bridge for using models based on discrete parameters.

    Requires that all parameters have been transformed to ChoiceParameters.
    """

    model: DiscreteModel
    outcomes: List[str]
    parameters: List[str]
    search_space: Optional[SearchSpace]

    def _fit(
        self,
        model: DiscreteModel,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
    ) -> None:
        self.model = model
        # Convert observations to arrays
        self.parameters = list(search_space.parameters.keys())
        all_metric_names: Set[str] = set()
        for od in observation_data:
            all_metric_names.update(od.metric_names)
        self.outcomes = list(all_metric_names)
        # Convert observations to arrays
        Xs_array, Ys_array, Yvars_array = _convert_observations(
            observation_data=observation_data,
            observation_features=observation_features,
            outcomes=self.outcomes,
            parameters=self.parameters,
        )
        # Extract parameter values
        parameter_values = _get_parameter_values(search_space, self.parameters)
        self.model.fit(
            Xs=Xs_array,
            Ys=Ys_array,
            Yvars=Yvars_array,
            parameter_values=parameter_values,
            outcome_names=self.outcomes,
        )

    def _predict(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationData]:
        # Convert observations to array
        X = [
            [of.parameters[param] for param in self.parameters]
            for of in observation_features
        ]
        f, cov = self.model.predict(X=X)
        # Convert arrays to observations
        return array_to_observation_data(f=f, cov=cov, outcomes=self.outcomes)

    def _gen(
        self,
        n: int,
        search_space: SearchSpace,
        pending_observations: Dict[str, List[ObservationFeatures]],
        fixed_features: ObservationFeatures,
        model_gen_options: Optional[TConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ) -> Tuple[
        List[ObservationFeatures],
        List[float],
        Optional[ObservationFeatures],
        TGenMetadata,
    ]:
        """Generate new candidates according to search_space and
        optimization_config.

        The outcome constraints should be transformed to no longer be relative.
        """
        # Validation
        if not self.parameters:  # pragma: no cover
            raise ValueError(FIT_MODEL_ERROR.format(action="_gen"))
        # Extract parameter values
        parameter_values = _get_parameter_values(search_space, self.parameters)
        # Extract objective and outcome constraints
        if len(self.outcomes) == 0 or optimization_config is None:  # pragma: no cover
            objective_weights = None
            outcome_constraints = None
        else:
            validate_optimization_config(optimization_config, self.outcomes)
            objective_weights = extract_objective_weights(
                objective=optimization_config.objective, outcomes=self.outcomes
            )
            outcome_constraints = extract_outcome_constraints(
                outcome_constraints=optimization_config.outcome_constraints,
                outcomes=self.outcomes,
            )

        # Get fixed features
        fixed_features_dict = {
            self.parameters.index(p_name): val
            for p_name, val in fixed_features.parameters.items()
        }
        fixed_features_dict = (
            fixed_features_dict if len(fixed_features_dict) > 0 else None
        )

        # Pending observations
        if len(pending_observations) == 0:
            pending_array: Optional[List[List[TParamValueList]]] = None
        else:
            pending_array = [[] for _ in self.outcomes]
            for metric_name, po_list in pending_observations.items():
                pending_array[self.outcomes.index(metric_name)] = [
                    [po.parameters[p] for p in self.parameters] for po in po_list
                ]

        # Generate the candidates
        X, w, gen_metadata = self.model.gen(
            n=n,
            parameter_values=parameter_values,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            fixed_features=fixed_features_dict,
            pending_observations=pending_array,
            model_gen_options=model_gen_options,
        )
        observation_features = []
        for x in X:
            observation_features.append(
                ObservationFeatures(
                    parameters={p: x[i] for i, p in enumerate(self.parameters)}
                )
            )
        # TODO[drfreund, bletham]: implement best_point identification and
        # return best_point instead of None
        return observation_features, w, None, gen_metadata

    def _cross_validate(
        self,
        obs_feats: List[ObservationFeatures],
        obs_data: List[ObservationData],
        cv_test_points: List[ObservationFeatures],
    ) -> List[ObservationData]:
        """Make predictions at cv_test_points using only the data in obs_feats
        and obs_data.
        """
        Xs_train, Ys_train, Yvars_train = _convert_observations(
            observation_data=obs_data,
            observation_features=obs_feats,
            outcomes=self.outcomes,
            parameters=self.parameters,
        )
        X_test = [
            [obsf.parameters[param] for param in self.parameters]
            for obsf in cv_test_points
        ]
        # Use the model to do the cross validation
        f_test, cov_test = self.model.cross_validate(
            Xs_train=Xs_train, Ys_train=Ys_train, Yvars_train=Yvars_train, X_test=X_test
        )
        # Convert array back to ObservationData
        return array_to_observation_data(f=f_test, cov=cov_test, outcomes=self.outcomes)


def _convert_observations(
    observation_data: List[ObservationData],
    observation_features: List[ObservationFeatures],
    outcomes: List[str],
    parameters: List[str],
) -> Tuple[List[List[TParamValueList]], List[List[float]], List[List[float]]]:
    Xs: List[List[TParamValueList]] = [[] for _ in outcomes]
    Ys: List[List[float]] = [[] for _ in outcomes]
    Yvars: List[List[float]] = [[] for _ in outcomes]
    for i, obsf in enumerate(observation_features):
        try:
            x = [obsf.parameters[param] for param in parameters]
        except (KeyError, TypeError):
            # Out of design point
            raise ValueError("Out of design points cannot be converted.")
        for j, m in enumerate(observation_data[i].metric_names):
            k = outcomes.index(m)
            Xs[k].append(x)
            Ys[k].append(observation_data[i].means[j])
            Yvars[k].append(observation_data[i].covariance[j, j])
    return Xs, Ys, Yvars


def _get_parameter_values(
    search_space: SearchSpace, param_names: List[str]
) -> List[TParamValueList]:
    """Extract parameter values from a search space of discrete parameters.
    """
    parameter_values: List[TParamValueList] = []
    for p_name in param_names:
        p = search_space.parameters[p_name]
        # Validation
        if isinstance(p, ChoiceParameter):
            # Set values
            parameter_values.append(p.values)
        elif isinstance(p, FixedParameter):
            parameter_values.append([p.value])
        else:
            raise ValueError(f"{p} not ChoiceParameter or FixedParameter")
    return parameter_values
