#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


from ax.core.observation import (
    Observation,
    ObservationData,
    ObservationFeatures,
    separate_observations,
)
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ChoiceParameter, FixedParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TParamValueList
from ax.exceptions.core import UserInputError
from ax.modelbridge.base import GenResults, ModelBridge
from ax.modelbridge.modelbridge_utils import (
    array_to_observation_data,
    get_fixed_features,
)
from ax.modelbridge.torch import (
    extract_objective_weights,
    extract_outcome_constraints,
    validate_optimization_config,
)
from ax.models.discrete_base import DiscreteModel
from ax.models.types import TConfig


FIT_MODEL_ERROR = "Model must be fit before {action}."


class DiscreteModelBridge(ModelBridge):
    """A model bridge for using models based on discrete parameters.

    Requires that all parameters have been transformed to ChoiceParameters.
    """

    # pyre-fixme[13]: Attribute `model` is never initialized.
    model: DiscreteModel
    # pyre-fixme[13]: Attribute `outcomes` is never initialized.
    outcomes: list[str]
    # pyre-fixme[13]: Attribute `parameters` is never initialized.
    parameters: list[str]
    # pyre-fixme[13]: Attribute `search_space` is never initialized.
    search_space: SearchSpace | None

    def _fit(
        self,
        model: DiscreteModel,
        search_space: SearchSpace,
        observations: list[Observation],
    ) -> None:
        self.model = model
        # Convert observations to arrays
        self.parameters = list(search_space.parameters.keys())
        all_metric_names: set[str] = set()
        observation_features, observation_data = separate_observations(observations)
        for od in observation_data:
            all_metric_names.update(od.metric_names)
        self.outcomes = list(all_metric_names)
        # Convert observations to arrays
        Xs_array, Ys_array, Yvars_array = self._convert_observations(
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
        self, observation_features: list[ObservationFeatures]
    ) -> list[ObservationData]:
        # Convert observations to array
        X = [
            [of.parameters[param] for param in self.parameters]
            for of in observation_features
        ]
        f, cov = self.model.predict(X=X)
        # Convert arrays to observations
        return array_to_observation_data(f=f, cov=cov, outcomes=self.outcomes)

    def _validate_gen_inputs(
        self,
        n: int,
        search_space: SearchSpace | None = None,
        optimization_config: OptimizationConfig | None = None,
        pending_observations: dict[str, list[ObservationFeatures]] | None = None,
        fixed_features: ObservationFeatures | None = None,
        model_gen_options: TConfig | None = None,
    ) -> None:
        """Validate inputs to `ModelBridge.gen`.

        Currently, this is only used to ensure that `n` is a positive integer or -1.
        """
        if n < 1 and n != -1:
            raise UserInputError(
                f"Attempted to generate n={n} points. Number of points to generate "
                "must be either a positive integer or -1."
            )

    def _gen(
        self,
        n: int,
        search_space: SearchSpace,
        pending_observations: dict[str, list[ObservationFeatures]],
        fixed_features: ObservationFeatures | None,
        model_gen_options: TConfig | None = None,
        optimization_config: OptimizationConfig | None = None,
    ) -> GenResults:
        """Generate new candidates according to search_space and
        optimization_config.

        The outcome constraints should be transformed to no longer be relative.
        """
        # Validation
        if not self.parameters:
            raise ValueError(FIT_MODEL_ERROR.format(action="_gen"))
        # Extract parameter values
        parameter_values = _get_parameter_values(search_space, self.parameters)
        # Extract objective and outcome constraints
        if len(self.outcomes) == 0 or optimization_config is None:
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
        fixed_features_dict = get_fixed_features(
            fixed_features=fixed_features, param_names=self.parameters
        )

        # Pending observations
        if len(pending_observations) == 0:
            pending_array: list[list[TParamValueList]] | None = None
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
            fixed_features=fixed_features_dict,  # pyre-ignore
            pending_observations=pending_array,
            model_gen_options=model_gen_options,
        )
        observation_features = [
            ObservationFeatures(parameters=dict(zip(self.parameters, x))) for x in X
        ]

        if "best_x" in gen_metadata:
            best_observation_features = ObservationFeatures(
                parameters=dict(zip(self.parameters, gen_metadata["best_x"]))
            )
        else:
            best_observation_features = None

        return GenResults(
            observation_features=observation_features,
            weights=w,
            gen_metadata=gen_metadata,
            best_observation_features=best_observation_features,
        )

    def _cross_validate(
        self,
        search_space: SearchSpace,
        cv_training_data: list[Observation],
        cv_test_points: list[ObservationFeatures],
        use_posterior_predictive: bool = False,
    ) -> list[ObservationData]:
        """Make predictions at cv_test_points using only the data in obs_feats
        and obs_data.
        """
        observation_features, observation_data = separate_observations(cv_training_data)
        Xs_train, Ys_train, Yvars_train = self._convert_observations(
            observation_data=observation_data,
            observation_features=observation_features,
            outcomes=self.outcomes,
            parameters=self.parameters,
        )
        X_test = [
            [obsf.parameters[param] for param in self.parameters]
            for obsf in cv_test_points
        ]
        # Use the model to do the cross validation
        f_test, cov_test = self.model.cross_validate(
            Xs_train=Xs_train,
            Ys_train=Ys_train,
            Yvars_train=Yvars_train,
            X_test=X_test,
            use_posterior_predictive=use_posterior_predictive,
        )
        # Convert array back to ObservationData
        return array_to_observation_data(f=f_test, cov=cov_test, outcomes=self.outcomes)

    @classmethod
    def _convert_observations(
        cls,
        observation_data: list[ObservationData],
        observation_features: list[ObservationFeatures],
        outcomes: list[str],
        parameters: list[str],
    ) -> tuple[list[list[TParamValueList]], list[list[float]], list[list[float]]]:
        Xs: list[list[TParamValueList]] = [[] for _ in outcomes]
        Ys: list[list[float]] = [[] for _ in outcomes]
        Yvars: list[list[float]] = [[] for _ in outcomes]
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
    search_space: SearchSpace, param_names: list[str]
) -> list[TParamValueList]:
    """Extract parameter values from a search space of discrete parameters."""
    parameter_values: list[TParamValueList] = []
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
