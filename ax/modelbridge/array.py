#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from ax.core.objective import MultiObjective, Objective, ScalarizedObjective
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.core.search_space import SearchSpace
from ax.core.types import TBounds, TCandidateMetadata, TConfig, TGenMetadata
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.modelbridge_utils import (
    extract_parameter_constraints,
    get_bounds_and_task,
    get_fixed_features,
    parse_observation_features,
    pending_observations_as_array,
    transform_callback,
)
from ax.utils.common.typeutils import not_none


FIT_MODEL_ERROR = "Model must be fit before {action}."


# pyre-fixme[13]: Attribute `model` is never initialized.
# pyre-fixme[13]: Attribute `outcomes` is never initialized.
# pyre-fixme[13]: Attribute `parameters` is never initialized.
class ArrayModelBridge(ModelBridge):
    """A model bridge for using array-based models.

    Requires that all non-task parameters have been transformed to
    RangeParameters with float type and no log scale. Task parameters must be
    transformed to RangeParameters with int type.

    This will convert all parameter types to float and put data into arrays.
    """

    model: Any
    outcomes: List[str]
    parameters: List[str]

    def _fit(
        self,
        model: Any,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
    ) -> None:
        # Convert observations to arrays
        self.parameters = list(search_space.parameters.keys())
        all_metric_names: Set[str] = set()
        for od in observation_data:
            all_metric_names.update(od.metric_names)
        self.outcomes = sorted(all_metric_names)  # Deterministic order
        # Convert observations to arrays
        Xs_array, Ys_array, Yvars_array, candidate_metadata = _convert_observations(
            observation_data=observation_data,
            observation_features=observation_features,
            outcomes=self.outcomes,
            parameters=self.parameters,
        )
        # Extract bounds and task features
        bounds, task_features, target_fidelities = get_bounds_and_task(
            search_space=search_space, param_names=self.parameters
        )

        # Fit
        self._model_fit(
            model=model,
            Xs=Xs_array,
            Ys=Ys_array,
            Yvars=Yvars_array,
            bounds=bounds,
            task_features=task_features,
            feature_names=self.parameters,
            metric_names=self.outcomes,
            fidelity_features=list(target_fidelities.keys()),
            candidate_metadata=candidate_metadata,
        )

    def _model_fit(
        self,
        model: Any,
        Xs: List[np.ndarray],
        Ys: List[np.ndarray],
        Yvars: List[np.ndarray],
        bounds: List[Tuple[float, float]],
        task_features: List[int],
        feature_names: List[str],
        metric_names: List[str],
        fidelity_features: List[int],
        candidate_metadata: Optional[List[List[TCandidateMetadata]]],
    ) -> None:
        """Fit the model, given numpy types.
        """
        self.model = model
        self.model.fit(
            Xs=Xs,
            Ys=Ys,
            Yvars=Yvars,
            bounds=bounds,
            task_features=task_features,
            feature_names=feature_names,
            metric_names=metric_names,
            fidelity_features=fidelity_features,
            candidate_metadata=candidate_metadata,
        )

    def _update(
        self,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
    ) -> None:
        """Apply terminal transform for update data, and pass along to model."""
        Xs_array, Ys_array, Yvars_array, candidate_metadata = _convert_observations(
            observation_data=observation_data,
            observation_features=observation_features,
            outcomes=self.outcomes,
            parameters=self.parameters,
        )
        # Update in-design status for these new points.
        self._model_update(
            Xs=Xs_array,
            Ys=Ys_array,
            Yvars=Yvars_array,
            candidate_metadata=candidate_metadata,
        )

    def _model_update(
        self,
        Xs: List[np.ndarray],
        Ys: List[np.ndarray],
        Yvars: List[np.ndarray],
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
    ) -> None:
        self.model.update(
            Xs=Xs, Ys=Ys, Yvars=Yvars, candidate_metadata=candidate_metadata
        )

    def _predict(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationData]:
        # Convert observations to array
        X = np.array(
            [[of.parameters[p] for p in self.parameters] for of in observation_features]
        )
        f, cov = self._model_predict(X=X)
        # Convert arrays to observations
        return array_to_observation_data(f=f, cov=cov, outcomes=self.outcomes)

    def _model_predict(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        return self.model.predict(X=X)

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
        # Extract bounds
        bounds, _, target_fidelities = get_bounds_and_task(
            search_space=search_space, param_names=self.parameters
        )
        target_fidelities = {
            i: float(v) for i, v in target_fidelities.items()  # pyre-ignore [6]
        }

        if optimization_config is None:
            raise ValueError(
                "ArrayModelBridge requires an OptimizationConfig to be specified"
            )
        if self.outcomes is None or len(self.outcomes) == 0:  # pragma: no cover
            raise ValueError("No outcomes found during model fit--data are missing.")

        validate_optimization_config(optimization_config, self.outcomes)
        objective_weights = extract_objective_weights(
            objective=optimization_config.objective, outcomes=self.outcomes
        )
        outcome_constraints = extract_outcome_constraints(
            outcome_constraints=optimization_config.outcome_constraints,
            outcomes=self.outcomes,
        )
        linear_constraints = extract_parameter_constraints(
            search_space.parameter_constraints, self.parameters
        )
        fixed_features_dict = get_fixed_features(fixed_features, self.parameters)
        pending_array = pending_observations_as_array(
            pending_observations, self.outcomes, self.parameters
        )
        # Generate the candidates
        X, w, gen_metadata, candidate_metadata = self._model_gen(
            n=n,
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features_dict,
            pending_observations=pending_array,
            model_gen_options=model_gen_options,
            rounding_func=transform_callback(self.parameters, self.transforms),
            target_fidelities=target_fidelities,
        )
        # Transform array to observations
        observation_features = parse_observation_features(
            X=X, param_names=self.parameters, candidate_metadata=candidate_metadata
        )
        xbest = self._model_best_point(
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features_dict,
            model_gen_options=model_gen_options,
            target_fidelities=target_fidelities,
        )
        best_obsf = (
            None
            if xbest is None
            else ObservationFeatures(
                parameters={p: float(xbest[i]) for i, p in enumerate(self.parameters)}
            )
        )
        return observation_features, w.tolist(), best_obsf, gen_metadata

    def _model_gen(
        self,
        n: int,
        bounds: List[Tuple[float, float]],
        objective_weights: np.ndarray,
        outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]],
        linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]],
        fixed_features: Optional[Dict[int, float]],
        pending_observations: Optional[List[np.ndarray]],
        model_gen_options: Optional[TConfig],
        rounding_func: Callable[[np.ndarray], np.ndarray],
        target_fidelities: Optional[Dict[int, float]] = None,
    ) -> Tuple[
        np.ndarray, np.ndarray, TGenMetadata, List[TCandidateMetadata]
    ]:  # pragma: no cover
        if target_fidelities:
            raise NotImplementedError(
                "target_fidelities not supported by ArrayModelBridge"
            )
        return self.model.gen(
            n=n,
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            pending_observations=pending_observations,
            model_gen_options=model_gen_options,
            rounding_func=rounding_func,
        )

    def _model_best_point(
        self,
        bounds: List[Tuple[float, float]],
        objective_weights: np.ndarray,
        outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]],
        linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]],
        fixed_features: Optional[Dict[int, float]],
        model_gen_options: Optional[TConfig],
        target_fidelities: Optional[Dict[int, float]] = None,
    ) -> Optional[np.ndarray]:  # pragma: no cover
        if target_fidelities:
            raise NotImplementedError(
                "target_fidelities not supported by ArrayModelBridge"
            )
        try:
            return self.model.best_point(
                bounds=bounds,
                objective_weights=objective_weights,
                outcome_constraints=outcome_constraints,
                linear_constraints=linear_constraints,
                fixed_features=fixed_features,
                model_gen_options=model_gen_options,
            )
        except NotImplementedError:
            return None

    def _cross_validate(
        self,
        obs_feats: List[ObservationFeatures],
        obs_data: List[ObservationData],
        cv_test_points: List[ObservationFeatures],
    ) -> List[ObservationData]:
        """Make predictions at cv_test_points using only the data in obs_feats
        and obs_data.
        """
        Xs_train, Ys_train, Yvars_train, candidate_metadata = _convert_observations(
            observation_data=obs_data,
            observation_features=obs_feats,
            outcomes=self.outcomes,
            parameters=self.parameters,
        )
        X_test = np.array(
            [[obsf.parameters[p] for p in self.parameters] for obsf in cv_test_points]
        )
        # Use the model to do the cross validation
        f_test, cov_test = self._model_cross_validate(
            Xs_train=Xs_train, Ys_train=Ys_train, Yvars_train=Yvars_train, X_test=X_test
        )
        # Convert array back to ObservationData
        return array_to_observation_data(f=f_test, cov=cov_test, outcomes=self.outcomes)

    def _model_cross_validate(
        self,
        Xs_train: List[np.ndarray],
        Ys_train: List[np.ndarray],
        Yvars_train: List[np.ndarray],
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        return self.model.cross_validate(
            Xs_train=Xs_train, Ys_train=Ys_train, Yvars_train=Yvars_train, X_test=X_test
        )

    def _transform_callback(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        """A function that performs the `round trip` transformations.
        This function is passed to _model_gen.
        """
        # apply reverse terminal transform to turn array to ObservationFeatures
        observation_features = [
            ObservationFeatures(
                parameters={p: float(x[i]) for i, p in enumerate(self.parameters)}
            )
        ]
        # reverse loop through the transforms and do untransform
        for t in reversed(self.transforms.values()):
            observation_features = t.untransform_observation_features(
                observation_features
            )
        # forward loop through the transforms and do transform
        for t in self.transforms.values():
            observation_features = t.transform_observation_features(
                observation_features
            )
        new_x: List[float] = [
            # pyre-fixme[6]: Expected `Union[_SupportsIndex, bytearray, bytes, str,
            #  typing.SupportsFloat]` for 1st param but got `Union[None, bool, float,
            #  int, str]`.
            float(observation_features[0].parameters[p])
            for p in self.parameters
        ]
        # turn it back into an array
        return np.array(new_x)

    def feature_importances(self, metric_name: str) -> Dict[str, float]:
        importances_tensor = not_none(self.model).feature_importances()
        importances_dict = dict(zip(self.outcomes, importances_tensor))
        importances_arr = importances_dict[metric_name].flatten()
        return dict(zip(self.parameters, importances_arr))

    def _transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> Any:
        """Apply terminal transform to given observation features and return result.
        """
        return np.array(
            [[of.parameters[p] for p in self.parameters] for of in observation_features]
        )


def array_to_observation_data(
    f: np.ndarray, cov: np.ndarray, outcomes: List[str]
) -> List[ObservationData]:
    """Convert arrays of model predictions to a list of ObservationData.

    Args:
        f: An (n x d) array
        cov: An (n x d x d) array
        outcomes: A list of d outcome names

    Returns: A list of n ObservationData
    """
    observation_data = []
    for i in range(f.shape[0]):
        observation_data.append(
            ObservationData(
                metric_names=list(outcomes),
                means=f[i, :].copy(),
                covariance=cov[i, :, :].copy(),
            )
        )
    return observation_data


def _convert_observations(
    observation_data: List[ObservationData],
    observation_features: List[ObservationFeatures],
    outcomes: List[str],
    parameters: List[str],
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    Optional[List[List[TCandidateMetadata]]],
]:
    """Converts observations to model's `fit` or `update` inputs: Xs, Ys, Yvars, and
    candidate metadata.

    NOTE: All four outputs are organized as lists over outcomes. E.g. if there are two
    outcomes, 'x' and 'y', the Xs are formatted like so: `[Xs_x_ndarray, Xs_y_ndarray]`.
    We specifically do not assume that every point is observed for every outcome.
    This means that the array for each of those outcomes may be different, and in
    particular could have a different length (e.g. if a particular arm was observed
    only for half of the outcomes, it would be present in half of the arrays in the
    list but not the other half.)
    """
    Xs: List[List[List[float]]] = [[] for _ in outcomes]
    Ys: List[List[float]] = [[] for _ in outcomes]
    Yvars: List[List[float]] = [[] for _ in outcomes]
    candidate_metadata: List[List[TCandidateMetadata]] = [[] for _ in outcomes]
    any_candidate_metadata_is_not_none = False
    for i, of in enumerate(observation_features):
        try:
            x: List[float] = [
                float(of.parameters[p]) for p in parameters  # pyre-ignore
            ]
        except (KeyError, TypeError):
            raise ValueError("Out of design points cannot be converted.")
        for j, m in enumerate(observation_data[i].metric_names):
            k = outcomes.index(m)
            Xs[k].append(x)
            Ys[k].append(observation_data[i].means[j])
            Yvars[k].append(observation_data[i].covariance[j, j])
            if of.metadata is not None:
                any_candidate_metadata_is_not_none = True
            candidate_metadata[k].append(of.metadata)

    Xs_array = [np.array(x_) for x_ in Xs]
    Ys_array = [np.array(y_)[:, None] for y_ in Ys]
    Yvars_array = [np.array(var)[:, None] for var in Yvars]
    if not any_candidate_metadata_is_not_none:
        candidate_metadata = None  # pyre-ignore[9]: Change of variable type.
    return Xs_array, Ys_array, Yvars_array, candidate_metadata


def extract_objective_weights(objective: Objective, outcomes: List[str]) -> np.ndarray:
    """Extract a weights for objectives.

    Weights are for a maximization problem.

    Give an objective weight to each modeled outcome. Outcomes that are modeled
    but not part of the objective get weight 0.

    In the single metric case, the objective is given either +/- 1, depending
    on the minimize flag.

    In the multiple metric case, each objective is given the input weight,
    multiplied by the minimize flag.

    Args:
        objective: Objective to extract weights from.
        outcomes: n-length list of names of metrics.

    Returns:
        (n,) array of weights.

    """
    s = -1.0 if objective.minimize else 1.0
    objective_weights = np.zeros(len(outcomes))
    if isinstance(objective, ScalarizedObjective):
        for obj_metric, obj_weight in objective.metric_weights:
            objective_weights[outcomes.index(obj_metric.name)] = obj_weight * s
    elif isinstance(objective, MultiObjective):
        for obj_metric, obj_weight in objective.metric_weights:
            # Rely on previously extracted lower_is_better weights not objective.
            objective_weights[outcomes.index(obj_metric.name)] = obj_weight or s
    else:
        objective_weights[outcomes.index(objective.metric.name)] = s
    return objective_weights


def extract_outcome_constraints(
    outcome_constraints: List[OutcomeConstraint], outcomes: List[str]
) -> TBounds:
    # Extract outcome constraints
    if len(outcome_constraints) > 0:
        A = np.zeros((len(outcome_constraints), len(outcomes)))
        b = np.zeros((len(outcome_constraints), 1))
        for i, c in enumerate(outcome_constraints):
            s = 1 if c.op == ComparisonOp.LEQ else -1
            j = outcomes.index(c.metric.name)
            A[i, j] = s
            b[i, 0] = s * c.bound
        outcome_constraint_bounds: TBounds = (A, b)
    else:
        outcome_constraint_bounds = None
    return outcome_constraint_bounds


def validate_optimization_config(
    optimization_config: OptimizationConfig, outcomes: List[str]
) -> None:
    """Validate optimization config against model fitted outcomes.

    Args:
        optimization_config: Config to validate.
        outcomes: List of metric names w/ valid model fits.

    Raises:
        ValueError if:
            1. Relative constraints are found
            2. Optimization metrics are not present in model fitted outcomes.
    """
    for c in optimization_config.outcome_constraints:
        if c.relative:
            raise ValueError(f"{c} is a relative constraint.")
        if c.metric.name not in outcomes:  # pragma: no cover
            raise ValueError(
                f"Outcome constraint metric {c.metric.name} not found in fitted data."
            )
    obj_metric_names = [m.name for m in optimization_config.objective.metrics]
    for obj_metric_name in obj_metric_names:
        if obj_metric_name not in outcomes:  # pragma: no cover
            raise ValueError(
                f"Objective metric {obj_metric_name} not found in fitted data."
            )
