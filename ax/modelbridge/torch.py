#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import torch
from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import extract_arm_predictions
from ax.core.metric import Metric
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
    TRefPoint,
)
from ax.core.outcome_constraint import (
    ComparisonOp,
    ObjectiveThreshold,
    ScalarizedOutcomeConstraint,
)
from ax.core.search_space import SearchSpace
from ax.core.types import TCandidateMetadata, TModelPredictArm
from ax.exceptions.core import UnsupportedError
from ax.modelbridge.base import gen_arms, GenResults, ModelBridge
from ax.modelbridge.modelbridge_utils import (
    array_to_observation_data,
    extract_objective_thresholds,
    extract_objective_weights,
    extract_outcome_constraints,
    extract_parameter_constraints,
    extract_search_space_digest,
    get_fixed_features,
    observation_data_to_array,
    observation_features_to_array,
    parse_observation_features,
    pending_observations_as_array,
    SearchSpaceDigest,
    transform_callback,
    validate_and_apply_final_transform,
)
from ax.modelbridge.transforms.base import Transform
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.botorch_moo_defaults import infer_objective_thresholds
from ax.models.torch_base import TorchModel
from ax.models.types import TConfig
from ax.utils.common.typeutils import checked_cast, not_none
from botorch.utils.datasets import FixedNoiseDataset, SupervisedDataset
from torch import Tensor


FIT_MODEL_ERROR = "Model must be fit before {action}."


@dataclass(frozen=True)
class TorchModelGenArgs:
    search_space_digest: SearchSpaceDigest
    objective_weights: Tensor
    outcome_constraints: Optional[Tuple[Tensor, Tensor]]
    linear_constraints: Optional[Tuple[Tensor, Tensor]]
    fixed_features: Optional[Dict[int, float]]
    pending_observations: Optional[List[Tensor]]
    rounding_func: Callable[[Tensor], Tensor]
    extra_model_gen_kwargs: Dict[str, Any]


# pyre-fixme [13]: Attributes are never initialized.
class TorchModelBridge(ModelBridge):
    """A model bridge for using torch-based models.

    Specifies an interface that is implemented by TorchModel. In particular,
    model should have methods fit, predict, and gen. See TorchModel for the
    API for each of these methods.

    Requires that all parameters have been transformed to RangeParameters
    or FixedParameters with float type and no log scale.

    This class converts Ax parameter types to torch tensors before passing
    them to the model.
    """

    model: Optional[TorchModel] = None
    outcomes: List[str]
    parameters: List[str]
    _default_model_gen_options: TConfig

    def __init__(
        self,
        experiment: Experiment,
        search_space: SearchSpace,
        data: Data,
        model: TorchModel,
        transforms: List[Type[Transform]],
        transform_configs: Optional[Dict[str, TConfig]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        torch_device: Optional[torch.device] = None,
        status_quo_name: Optional[str] = None,
        status_quo_features: Optional[ObservationFeatures] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        fit_out_of_design: bool = False,
        objective_thresholds: Optional[TRefPoint] = None,
        default_model_gen_options: Optional[TConfig] = None,
    ) -> None:
        self.dtype = torch_dtype
        self.device = torch_device
        self._default_model_gen_options = default_model_gen_options or {}

        # Handle init for multi-objective optimization.
        self.is_moo_problem = False
        if optimization_config or (experiment and experiment.optimization_config):
            optimization_config = not_none(
                optimization_config or experiment.optimization_config
            )
            self.is_moo_problem = optimization_config.is_moo_problem
        if objective_thresholds:
            if not self.is_moo_problem:
                raise ValueError(
                    "objective_thresholds are only supported for multi objective "
                    "optimization."
                )
            optimization_config = checked_cast(
                MultiObjectiveOptimizationConfig, optimization_config
            )
            optimization_config = optimization_config.clone_with_args(
                objective_thresholds=objective_thresholds
            )

        super().__init__(
            experiment=experiment,
            search_space=search_space,
            data=data,
            model=model,
            transforms=transforms,
            transform_configs=transform_configs,
            status_quo_name=status_quo_name,
            status_quo_features=status_quo_features,
            optimization_config=optimization_config,
            fit_out_of_design=fit_out_of_design,
        )

    def feature_importances(self, metric_name: str) -> Dict[str, float]:
        importances_tensor = not_none(self.model).feature_importances()
        importances_dict = dict(zip(self.outcomes, importances_tensor))
        importances_arr = importances_dict[metric_name].flatten()
        return dict(zip(self.parameters, importances_arr))

    def infer_objective_thresholds(
        self,
        search_space: Optional[SearchSpace] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> List[ObjectiveThreshold]:
        """Infer objective thresholds.

        This method is only applicable for Multi-Objective optimization problems.

        This method uses the model-estimated Pareto frontier over the in-sample points
        to infer absolute (not relativized) objective thresholds.

        This uses a heuristic that sets the objective threshold to be a scaled nadir
        point, where the nadir point is scaled back based on the range of each
        objective across the current in-sample Pareto frontier.
        """
        if not self.is_moo_problem:
            raise UnsupportedError(
                "Objective thresholds are only supported for multi-objective "
                "optimization."
            )

        search_space = (search_space or self._model_space).clone()
        base_gen_args = self._get_transformed_gen_args(
            search_space=search_space,
            optimization_config=optimization_config,
            fixed_features=fixed_features,
        )
        # Get transformed args from TorchModelbridge.
        mgen_args = self._get_transformed_model_gen_args(
            search_space=base_gen_args.search_space,
            fixed_features=base_gen_args.fixed_features,
            pending_observations={},
            optimization_config=base_gen_args.optimization_config,
        )
        # Infer objective thresholds.
        model = checked_cast(MultiObjectiveBotorchModel, self.model)
        obj_thresholds = infer_objective_thresholds(
            model=not_none(model.model),
            objective_weights=mgen_args.objective_weights,
            bounds=mgen_args.search_space_digest.bounds,
            outcome_constraints=mgen_args.outcome_constraints,
            linear_constraints=mgen_args.linear_constraints,
            fixed_features=mgen_args.fixed_features,
            Xs=model.Xs,
        )

        return self._untransform_objective_thresholds(
            objective_thresholds=obj_thresholds,
            objective_weights=mgen_args.objective_weights,
            bounds=mgen_args.search_space_digest.bounds,
            # we should never be in a situation where we call this without there
            # being an optimization config involved.
            opt_config_metrics=not_none(base_gen_args.optimization_config).metrics,
            fixed_features=mgen_args.fixed_features,
        )

    def model_best_point(
        self,
        search_space: Optional[SearchSpace] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        pending_observations: Optional[Dict[str, List[ObservationFeatures]]] = None,
        fixed_features: Optional[ObservationFeatures] = None,
        model_gen_options: Optional[TConfig] = None,
    ) -> Optional[Tuple[Arm, Optional[TModelPredictArm]]]:
        # Get modifiable versions
        if search_space is None:
            search_space = self._model_space
        search_space = search_space.clone()

        base_gen_args = self._get_transformed_gen_args(
            search_space=search_space,
            optimization_config=optimization_config,
            pending_observations=pending_observations,
            fixed_features=fixed_features,
        )
        model_gen_args = self._get_transformed_model_gen_args(
            search_space=base_gen_args.search_space,
            pending_observations=base_gen_args.pending_observations,
            fixed_features=base_gen_args.fixed_features,
            model_gen_options=None,
            optimization_config=base_gen_args.optimization_config,
        )
        search_space_digest = model_gen_args.search_space_digest

        try:
            xbest = not_none(self.model).best_point(
                bounds=search_space_digest.bounds,
                objective_weights=model_gen_args.objective_weights,
                outcome_constraints=model_gen_args.outcome_constraints,
                linear_constraints=model_gen_args.linear_constraints,
                fixed_features=model_gen_args.fixed_features,
                model_gen_options=model_gen_options,
                target_fidelities=search_space_digest.target_fidelities,
            )
        except NotImplementedError:
            xbest = None

        if xbest is None:
            return None

        best_obsf = ObservationFeatures(
            parameters={p: float(xbest[i]) for i, p in enumerate(self.parameters)}
        )

        for t in reversed(list(self.transforms.values())):
            best_obsf = t.untransform_observation_features([best_obsf])[0]

        best_point_predictions = extract_arm_predictions(
            model_predictions=self.predict([best_obsf]), arm_idx=0
        )

        best_arms, _ = gen_arms(
            observation_features=[best_obsf],
            arms_by_signature=self._arms_by_signature,
        )
        best_arm = best_arms[0]

        return best_arm, best_point_predictions

    def _array_callable_to_tensor_callable(
        self, array_func: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[Tensor], Tensor]:
        tensor_func: Callable[[Tensor], Tensor] = lambda x: (
            self._array_to_tensor(array_func(x.detach().cpu().clone().numpy()))
        )
        return tensor_func

    def _array_list_to_tensors(self, arrays: List[np.ndarray]) -> List[Tensor]:
        return [self._array_to_tensor(x) for x in arrays]

    def _array_to_tensor(self, array: Union[np.ndarray, List[float]]) -> Tensor:
        return torch.as_tensor(array, dtype=self.dtype, device=self.device)

    @classmethod
    def _convert_observations(
        cls,
        observation_data: List[ObservationData],
        observation_features: List[ObservationFeatures],
        outcomes: List[str],
        parameters: List[str],
    ) -> Tuple[
        List[Optional[SupervisedDataset]], Optional[List[List[TCandidateMetadata]]]
    ]:
        """Converts observations to a dictionary of `Dataset` containers and (optional)
        candidate metadata.
        """
        Xs: Dict[str, List[Tensor]] = defaultdict(list)
        Ys: Dict[str, List[Tensor]] = defaultdict(list)
        Yvars: Dict[str, List[Tensor]] = defaultdict(list)
        datasets: List[Optional[SupervisedDataset]] = []
        candidate_metadata_dict: Dict[str, List[TCandidateMetadata]] = defaultdict(list)
        any_candidate_metadata_is_not_none = False

        for obsd, obsf in zip(observation_data, observation_features):
            try:
                x = torch.tensor(
                    [obsf.parameters[p] for p in parameters], dtype=torch.double
                )
            except (KeyError, TypeError):
                raise ValueError("Out of design points cannot be converted.")
            for metric_name, mean, var in zip(
                obsd.metric_names, obsd.means, obsd.covariance.diagonal()
            ):
                Xs[metric_name].append(x)
                Ys[metric_name].append(mean)
                Yvars[metric_name].append(var)
                if obsf.metadata is not None:
                    any_candidate_metadata_is_not_none = True
                candidate_metadata_dict[metric_name].append(obsf.metadata)

        candidate_metadata = []
        for outcome in outcomes:
            if outcome not in Xs:
                # This may happen when we update the data of only some metrics
                datasets.append(None)
                candidate_metadata.append(None)
                continue
            X = torch.stack(Xs[outcome], dim=0)
            Y = torch.tensor(Ys[outcome], dtype=torch.double).unsqueeze(-1)
            Yvar = torch.tensor(Yvars[outcome], dtype=torch.double).unsqueeze(-1)
            if Yvar.isnan().all():
                dataset = SupervisedDataset(X=X, Y=Y)
            else:
                dataset = FixedNoiseDataset(X=X, Y=Y, Yvar=Yvar)
            datasets.append(dataset)
            candidate_metadata.append(candidate_metadata_dict[outcome])

        if not any_candidate_metadata_is_not_none:
            return datasets, None

        return datasets, candidate_metadata

    def _cross_validate(
        self,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
        cv_test_points: List[ObservationFeatures],
    ) -> List[ObservationData]:
        """Make predictions at cv_test_points using only the data in obs_feats
        and obs_data.
        """
        if self.model is None:
            raise ValueError(FIT_MODEL_ERROR.format(action="_cross_validate"))
        datasets, candidate_metadata = self._convert_observations(
            observation_data=observation_data,
            observation_features=observation_features,
            outcomes=self.outcomes,
            parameters=self.parameters,
        )
        for outcome, dataset in zip(self.outcomes, datasets):
            if dataset is None:
                raise UnsupportedError(
                    f"{self.__class__._cross_validate} requires observations "
                    f"for all outcomes, but no observations for {outcome}"
                )
        search_space_digest = extract_search_space_digest(
            search_space=search_space, param_names=self.parameters
        )
        X_test = torch.tensor(
            [[obsf.parameters[p] for p in self.parameters] for obsf in cv_test_points],
            dtype=self.dtype,
            device=self.device,
        )
        # Use the model to do the cross validation
        f_test, cov_test = not_none(self.model).cross_validate(
            datasets=[not_none(dataset) for dataset in datasets],
            metric_names=self.outcomes,
            X_test=torch.as_tensor(X_test, dtype=self.dtype, device=self.device),
            search_space_digest=search_space_digest,
        )
        # Convert array back to ObservationData
        return array_to_observation_data(
            f=f_test.detach().cpu().clone().numpy(),
            cov=cov_test.detach().cpu().clone().numpy(),
            outcomes=self.outcomes,
        )

    def _evaluate_acquisition_function(
        self,
        observation_features: List[List[ObservationFeatures]],
        search_space: SearchSpace,
        optimization_config: OptimizationConfig,
        objective_thresholds: Optional[np.ndarray] = None,
        outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        pending_observations: Optional[List[np.ndarray]] = None,
        acq_options: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        if self.model is None:
            raise RuntimeError(
                FIT_MODEL_ERROR.format(action="_evaluate_acquisition_function")
            )
        search_space_digest = extract_search_space_digest(
            search_space=search_space, param_names=self.parameters
        )
        objective_weights = extract_objective_weights(
            objective=optimization_config.objective, outcomes=self.outcomes
        )
        X = np.array(
            [
                observation_features_to_array(self.parameters, obsf)
                for obsf in observation_features
            ]
        )

        obj_w, oc_c, l_c, pend_obs, obj_thresh = validate_and_apply_final_transform(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            pending_observations=pending_observations,
            objective_thresholds=objective_thresholds,
            final_transform=self._array_to_tensor,
        )

        evals = not_none(self.model).evaluate_acquisition_function(
            X=self._array_to_tensor(X),
            search_space_digest=search_space_digest,
            objective_weights=obj_w,
            objective_thresholds=obj_thresh,
            outcome_constraints=oc_c,
            linear_constraints=l_c,
            fixed_features=fixed_features,
            pending_observations=pend_obs,
            acq_options=acq_options,
        )

        return evals.tolist()

    def _fit(
        self,
        model: TorchModel,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
    ) -> None:  # pragma: no cover
        self._validate_observation_data(observation_data)
        self.parameters = list(search_space.parameters.keys())
        all_metric_names: Set[str] = set()
        for od in observation_data:
            all_metric_names.update(od.metric_names)
        self.outcomes = sorted(all_metric_names)  # Deterministic order
        # Convert observations to datasets
        datasets, candidate_metadata = self._convert_observations(
            observation_data=observation_data,
            observation_features=observation_features,
            outcomes=self.outcomes,
            parameters=self.parameters,
        )
        # Get all relevant information on the parameters
        search_space_digest = extract_search_space_digest(
            search_space=search_space, param_names=self.parameters
        )
        # Fit
        self.model = model
        self.model.fit(
            # datasets are guaranteed to have all outcomes here by construction
            datasets=[not_none(dataset) for dataset in datasets],
            metric_names=self.outcomes,
            search_space_digest=search_space_digest,
            candidate_metadata=candidate_metadata,
        )

    def _gen(
        self,
        n: int,
        search_space: SearchSpace,
        pending_observations: Dict[str, List[ObservationFeatures]],
        fixed_features: ObservationFeatures,
        model_gen_options: Optional[TConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ) -> GenResults:
        """Generate new candidates according to search_space and
        optimization_config.

        The outcome constraints should be transformed to no longer be relative.
        """
        if self.model is None:
            raise ValueError(FIT_MODEL_ERROR.format(action="_gen"))

        mgen_args = self._get_transformed_model_gen_args(
            search_space=search_space,
            pending_observations=pending_observations,
            fixed_features=fixed_features,
            model_gen_options=model_gen_options,
            optimization_config=optimization_config,
        )

        # Generate the candidates
        search_space_digest = mgen_args.search_space_digest
        augmented_model_gen_options = {
            **self._default_model_gen_options,
            **(model_gen_options or {}),
        }
        # TODO(ehotaj): For some reason, we're getting models which do not support MOO
        # even when optimization_config has multiple objectives, so we can't use
        # self.is_moo_problem here.
        is_moo_problem = self.is_moo_problem and isinstance(
            self.model, (BoTorchModel, MultiObjectiveBotorchModel)
        )
        extra_kwargs = {}
        obj_t = mgen_args.extra_model_gen_kwargs.get("objective_thresholds")
        if is_moo_problem and obj_t is not None:
            extra_kwargs["objective_thresholds"] = self._array_to_tensor(obj_t)

        gen_results = not_none(self.model).gen(
            n=n,
            bounds=search_space_digest.bounds,
            objective_weights=mgen_args.objective_weights,
            outcome_constraints=mgen_args.outcome_constraints,
            linear_constraints=mgen_args.linear_constraints,
            fixed_features=mgen_args.fixed_features,
            pending_observations=mgen_args.pending_observations,
            model_gen_options=augmented_model_gen_options,
            rounding_func=mgen_args.rounding_func,
            target_fidelities=search_space_digest.target_fidelities,
            **extra_kwargs,
        )

        gen_metadata = gen_results.gen_metadata
        if is_moo_problem:
            # If objective_thresholds are supplied by the user, then the transformed
            # user-specified objective thresholds are in gen_metadata. Otherwise,
            # inferred objective thresholds are in gen_metadata.
            opt_config_metrics = mgen_args.extra_model_gen_kwargs.get(
                "opt_config_metrics", not_none(self._optimization_config).metrics
            )
            gen_metadata[
                "objective_thresholds"
            ] = self._untransform_objective_thresholds(
                objective_thresholds=gen_metadata["objective_thresholds"],
                objective_weights=mgen_args.objective_weights,
                bounds=search_space_digest.bounds,
                opt_config_metrics=opt_config_metrics,
                fixed_features=mgen_args.fixed_features,
            )

        # Transform array to observations
        observation_features = parse_observation_features(
            X=gen_results.points.detach().cpu().clone().numpy(),
            param_names=self.parameters,
            candidate_metadata=gen_results.candidate_metadata,
        )
        try:
            xbest = not_none(self.model).best_point(
                bounds=search_space_digest.bounds,
                objective_weights=mgen_args.objective_weights,
                outcome_constraints=mgen_args.outcome_constraints,
                linear_constraints=mgen_args.linear_constraints,
                fixed_features=mgen_args.fixed_features,
                model_gen_options=model_gen_options,
                target_fidelities=search_space_digest.target_fidelities,
            )
        except NotImplementedError:
            xbest = None

        best_obsf = None
        if xbest is not None:
            best_obsf = ObservationFeatures(
                parameters={p: float(xbest[i]) for i, p in enumerate(self.parameters)}
            )

        return GenResults(
            observation_features=observation_features,
            weights=gen_results.weights.tolist(),
            best_observation_features=best_obsf,
            gen_metadata=gen_metadata,
        )

    def _get_extra_model_gen_kwargs(
        self, optimization_config: OptimizationConfig
    ) -> Dict[str, Any]:
        extra_kwargs_dict = {}
        if optimization_config.is_moo_problem:
            optimization_config = checked_cast(
                MultiObjectiveOptimizationConfig, optimization_config
            )
            extra_kwargs_dict["objective_thresholds"] = extract_objective_thresholds(
                objective_thresholds=optimization_config.objective_thresholds,
                objective=optimization_config.objective,
                outcomes=self.outcomes,
            )
            extra_kwargs_dict["opt_config_metrics"] = optimization_config.metrics
        return extra_kwargs_dict

    def _predict(
        self, observation_features: List[ObservationFeatures]
    ) -> List[ObservationData]:
        if not self.model:  # pragma: no cover
            raise ValueError(FIT_MODEL_ERROR.format(action="_model_predict"))
        # Convert observation features to array
        X = observation_features_to_array(self.parameters, observation_features)
        f, cov = not_none(self.model).predict(X=self._array_to_tensor(X))
        f = f.detach().cpu().clone().numpy()
        cov = cov.detach().cpu().clone().numpy()
        # Convert resulting arrays to observations
        return array_to_observation_data(f=f, cov=cov, outcomes=self.outcomes)

    def _transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> Any:  # TODO(jej): Make return type parametric
        """Apply terminal transform to given observation features and return result
        as an N x D array of points.
        """
        try:
            tobfs = np.array(
                [
                    # pyre-ignore[6]: Except statement below should catch wrongly
                    # typed parameters.
                    [float(of.parameters[p]) for p in self.parameters]
                    for of in observation_features
                ]
            )
        except (KeyError, TypeError):  # pragma: no cover
            raise ValueError("Invalid formatting of observation features.")
        return self._array_to_tensor(tobfs)

    def _get_transformed_model_gen_args(
        self,
        search_space: SearchSpace,
        pending_observations: Dict[str, List[ObservationFeatures]],
        fixed_features: ObservationFeatures,
        model_gen_options: Optional[TConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ) -> TorchModelGenArgs:
        # Validation
        if not self.parameters:  # pragma: no cover
            raise ValueError(FIT_MODEL_ERROR.format(action="_gen"))
        # Extract search space info
        search_space_digest = extract_search_space_digest(
            search_space=search_space, param_names=self.parameters
        )
        if optimization_config is None:
            raise ValueError(
                f"{self.__class__.__name__} requires an OptimizationConfig "
                "to be specified"
            )
        if self.outcomes is None or len(self.outcomes) == 0:  # pragma: no cover
            raise ValueError("No outcomes found during model fit--data are missing.")

        validate_optimization_config(optimization_config, self.outcomes)
        objective_weights = self._array_to_tensor(
            extract_objective_weights(
                objective=optimization_config.objective, outcomes=self.outcomes
            )
        )

        # TODO: Clean up the following conversions
        outcome_constraints = extract_outcome_constraints(
            outcome_constraints=optimization_config.outcome_constraints,
            outcomes=self.outcomes,
        )
        if outcome_constraints is not None:
            outcome_constraints = tuple(
                self._array_to_tensor(t) for t in outcome_constraints
            )

        extra_model_gen_kwargs = self._get_extra_model_gen_kwargs(
            optimization_config=optimization_config
        )

        linear_constraints = extract_parameter_constraints(
            search_space.parameter_constraints, self.parameters
        )
        if linear_constraints is not None:
            linear_constraints = tuple(
                self._array_to_tensor(t) for t in linear_constraints
            )

        fixed_features_dict = get_fixed_features(fixed_features, self.parameters)
        pending_arrays = pending_observations_as_array(
            pending_observations, self.outcomes, self.parameters
        )
        if pending_arrays is None:
            pending_tensors = None
        else:
            pending_tensors = [torch.from_numpy(pa) for pa in pending_arrays]
        rounding_func = self._array_callable_to_tensor_callable(
            transform_callback(self.parameters, self.transforms)
        )
        return TorchModelGenArgs(
            search_space_digest=search_space_digest,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features_dict,
            pending_observations=pending_tensors,
            rounding_func=rounding_func,
            extra_model_gen_kwargs=extra_model_gen_kwargs,
        )

    def _transform_observation_data(
        self, observation_data: List[ObservationData]
    ) -> Any:  # TODO(jej): Make return type parametric
        """Apply terminal transform to given observation data and return result.

        Converts a set of observation data to a tuple of
            - an (n x m) array of means
            - an (n x m x m) array of covariances
        """
        try:
            mean, cov = observation_data_to_array(
                outcomes=self.outcomes, observation_data=observation_data
            )
        except (KeyError, TypeError):  # pragma: no cover
            raise ValueError("Invalid formatting of observation data.")
        return self._array_to_tensor(mean), self._array_to_tensor(cov)

    def _untransform_objective_thresholds(
        self,
        objective_thresholds: Tensor,
        objective_weights: Tensor,
        bounds: List[Tuple[Union[int, float], Union[int, float]]],
        opt_config_metrics: Dict[str, Metric],
        fixed_features: Optional[Dict[int, float]],
    ) -> List[ObjectiveThreshold]:
        thresholds_np = objective_thresholds.cpu().numpy()
        idxs = objective_weights.nonzero().view(-1).tolist()

        # Create transformed ObjectiveThresholds from numpy thresholds.
        thresholds = []
        for idx in idxs:
            sign = torch.sign(objective_weights[idx])
            thresholds.append(
                ObjectiveThreshold(
                    metric=opt_config_metrics[self.outcomes[idx]],
                    bound=thresholds_np[idx],
                    relative=False,
                    op=ComparisonOp.LEQ if sign < 0 else ComparisonOp.GEQ,
                )
            )

        # Create dummy ObservationFeatures from the fixed features.
        fixed = fixed_features or {}
        observation_features = [
            ObservationFeatures(
                parameters={
                    name: fixed.get(i, 0.0) for i, name in enumerate(self.parameters)
                }
            )
        ]

        # Untransform ObjectiveThresholds along with the dummy ObservationFeatures.
        for t in reversed(list(self.transforms.values())):
            thresholds = t.untransform_objective_thresholds(
                objective_thresholds=thresholds,
                observation_features=observation_features,
            )
            observation_features = t.untransform_observation_features(
                observation_features
            )

        return thresholds

    def _update(
        self,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
    ) -> None:
        """Apply terminal transform for update data, and pass along to model."""
        datasets, candidate_metadata = self._convert_observations(
            observation_data=observation_data,
            observation_features=observation_features,
            outcomes=self.outcomes,
            parameters=self.parameters,
        )
        search_space_digest = extract_search_space_digest(
            search_space=search_space, param_names=self.parameters
        )
        # Update in-design status for these new points.
        if self.model is None:
            raise ValueError(FIT_MODEL_ERROR.format(action="_update"))
        self.model.update(
            datasets=datasets,
            metric_names=self.outcomes,
            search_space_digest=search_space_digest,
            candidate_metadata=candidate_metadata,
        )

    def _validate_observation_data(
        self, observation_data: List[ObservationData]
    ) -> None:
        if len(observation_data) == 0:
            raise ValueError(
                "Torch models cannot be fit without observation data. Possible "
                "reasons include empty data being passed to the model's constructor "
                "or data being excluded because it is out-of-design. Try setting "
                "`fit_out_of_design`=True during construction to fix the latter."
            )


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
        if isinstance(c, ScalarizedOutcomeConstraint):
            for c_metric in c.metrics:
                if c_metric.name not in outcomes:  # pragma: no cover
                    raise ValueError(
                        f"Scalarized constraint metric component {c.metric.name} "
                        + "not found in fitted data."
                    )
        elif c.metric.name not in outcomes:  # pragma: no cover
            raise ValueError(
                f"Outcome constraint metric {c.metric.name} not found in fitted data."
            )
    obj_metric_names = [m.name for m in optimization_config.objective.metrics]
    for obj_metric_name in obj_metric_names:
        if obj_metric_name not in outcomes:  # pragma: no cover
            raise ValueError(
                f"Objective metric {obj_metric_name} not found in fitted data."
            )
