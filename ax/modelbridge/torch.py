#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
    TRefPoint,
)
from ax.core.outcome_constraint import ComparisonOp, ObjectiveThreshold
from ax.core.search_space import SearchSpace
from ax.core.types import TCandidateMetadata, TGenMetadata
from ax.modelbridge.array import ArrayModelBridge, FIT_MODEL_ERROR
from ax.modelbridge.modelbridge_utils import (
    extract_objective_thresholds,
    SearchSpaceDigest,
    validate_and_apply_final_transform,
)
from ax.modelbridge.transforms.base import Transform
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.botorch_moo_defaults import infer_objective_thresholds
from ax.models.torch_base import TorchModel
from ax.models.types import TConfig
from ax.utils.common.typeutils import checked_cast, not_none
from torch import Tensor


# pyre-fixme[13]: Attribute `model` is never initialized.
class TorchModelBridge(ArrayModelBridge):
    """A model bridge for using torch-based models.

    Specifies an interface that is implemented by TorchModel. In particular,
    model should have methods fit, predict, and gen. See TorchModel for the
    API for each of these methods.

    Requires that all parameters have been transformed to RangeParameters
    or FixedParameters with float type and no log scale.

    This class converts Ax parameter types to torch tensors before passing
    them to the model.
    """

    model: Optional[TorchModel]
    _default_model_gen_options: TConfig

    def __init__(
        self,
        experiment: Experiment,
        search_space: SearchSpace,
        data: Data,
        model: TorchModel,
        transforms: List[Type[Transform]],
        transform_configs: Optional[Dict[str, TConfig]] = None,
        torch_dtype: Optional[torch.dtype] = None,  # noqa T484
        torch_device: Optional[torch.device] = None,
        status_quo_name: Optional[str] = None,
        status_quo_features: Optional[ObservationFeatures] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        fit_out_of_design: bool = False,
        objective_thresholds: Optional[TRefPoint] = None,
        default_model_gen_options: Optional[TConfig] = None,
    ) -> None:
        if torch_dtype is None:  # pragma: no cover
            torch_dtype = torch.float  # noqa T484
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

    def _fit(
        self,
        model: TorchModel,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
    ) -> None:  # pragma: no cover
        self._validate_observation_data(observation_data)
        super()._fit(
            model=model,
            search_space=search_space,
            observation_features=observation_features,
            observation_data=observation_data,
        )

    def _model_evaluate_acquisition_function(
        self,
        X: np.ndarray,
        search_space_digest: SearchSpaceDigest,
        objective_weights: np.ndarray,
        objective_thresholds: Optional[np.ndarray] = None,
        outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fixed_features: Optional[Dict[int, float]] = None,
        pending_observations: Optional[List[np.ndarray]] = None,
        acq_options: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if not self.model:  # pragma: no cover
            raise ValueError(
                FIT_MODEL_ERROR.format(action="_model_evaluate_acquisition_function")
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

        return evals.detach().cpu().clone().numpy()

    def _model_fit(
        self,
        model: TorchModel,
        Xs: List[np.ndarray],
        Ys: List[np.ndarray],
        Yvars: List[np.ndarray],
        search_space_digest: SearchSpaceDigest,
        metric_names: List[str],
        candidate_metadata: Optional[List[List[TCandidateMetadata]]],
    ) -> None:
        self.model = model
        # Convert numpy arrays to torch tensors
        # pyre-fixme[35]: Target cannot be annotated.
        Xs: List[Tensor] = self._array_list_to_tensors(Xs)
        # pyre-fixme[35]: Target cannot be annotated.
        Ys: List[Tensor] = self._array_list_to_tensors(Ys)
        # pyre-fixme[35]: Target cannot be annotated.
        Yvars: List[Tensor] = self._array_list_to_tensors(Yvars)
        # pyre-fixme[16]: `Optional` has no attribute `fit`.
        self.model.fit(
            Xs=Xs,
            Ys=Ys,
            Yvars=Yvars,
            search_space_digest=search_space_digest,
            metric_names=metric_names,
            candidate_metadata=candidate_metadata,
        )

    def _model_update(
        self,
        Xs: List[np.ndarray],
        Ys: List[np.ndarray],
        Yvars: List[np.ndarray],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]],
        metric_names: List[str],
    ) -> None:
        if not self.model:  # pragma: no cover
            raise ValueError(FIT_MODEL_ERROR.format(action="_model_update"))
        # pyre-fixme[35]: Target cannot be annotated.
        Xs: List[Tensor] = self._array_list_to_tensors(Xs)
        # pyre-fixme[35]: Target cannot be annotated.
        Ys: List[Tensor] = self._array_list_to_tensors(Ys)
        # pyre-fixme[35]: Target cannot be annotated.
        Yvars: List[Tensor] = self._array_list_to_tensors(Yvars)
        # pyre-fixme[16]: `Optional` has no attribute `update`.
        self.model.update(
            Xs=Xs,
            Ys=Ys,
            Yvars=Yvars,
            search_space_digest=search_space_digest,
            metric_names=self.outcomes,
            candidate_metadata=candidate_metadata,
        )

    def _model_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.model:  # pragma: no cover
            raise ValueError(FIT_MODEL_ERROR.format(action="_model_predict"))
        f, var = not_none(self.model).predict(X=self._array_to_tensor(X))
        return f.detach().cpu().clone().numpy(), var.detach().cpu().clone().numpy()

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
        target_fidelities: Optional[Dict[int, float]],
        objective_thresholds: Optional[np.ndarray] = None,
        opt_config_metrics: Optional[Dict[str, Metric]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, TGenMetadata, List[TCandidateMetadata]]:
        if not self.model:  # pragma: no cover
            raise ValueError(FIT_MODEL_ERROR.format(action="_model_gen"))
        obj_w, oc_c, l_c, pend_obs, obj_t = validate_and_apply_final_transform(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            pending_observations=pending_observations,
            objective_thresholds=objective_thresholds,
            final_transform=self._array_to_tensor,
        )
        tensor_rounding_func = self._array_callable_to_tensor_callable(rounding_func)
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
        extra_kwargs = {"objective_thresholds": obj_t} if is_moo_problem else {}
        X, w, gen_metadata, candidate_metadata = not_none(self.model).gen(
            n=n,
            bounds=bounds,
            objective_weights=obj_w,
            outcome_constraints=oc_c,
            linear_constraints=l_c,
            fixed_features=fixed_features,
            pending_observations=pend_obs,
            model_gen_options=augmented_model_gen_options,
            rounding_func=tensor_rounding_func,
            target_fidelities=target_fidelities,
            **extra_kwargs
        )

        if is_moo_problem:
            # If objective_thresholds are supplied by the user, then the transformed
            # user-specified objective thresholds are in gen_metadata. Otherwise,
            # inferred objective thresholds are in gen_metadata.
            opt_config_metrics = (
                opt_config_metrics or not_none(self._optimization_config).metrics
            )
            gen_metadata[
                "objective_thresholds"
            ] = self._untransform_objective_thresholds(
                objective_thresholds=gen_metadata["objective_thresholds"],
                objective_weights=obj_w,
                bounds=bounds,
                opt_config_metrics=opt_config_metrics,
                fixed_features=fixed_features,
            )

        return (
            X.detach().cpu().clone().numpy(),
            w.detach().cpu().clone().numpy(),
            gen_metadata,
            candidate_metadata,
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

    def _model_best_point(
        self,
        bounds: List[Tuple[float, float]],
        objective_weights: np.ndarray,
        outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]],
        linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]],
        fixed_features: Optional[Dict[int, float]],
        model_gen_options: Optional[TConfig],
        target_fidelities: Optional[Dict[int, float]],
    ) -> Optional[np.ndarray]:  # pragma: no cover
        if not self.model:  # pragma: no cover
            raise ValueError(FIT_MODEL_ERROR.format(action="_model_gen"))
        obj_w, oc_c, l_c, _, _ = validate_and_apply_final_transform(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            pending_observations=None,
            final_transform=self._array_to_tensor,
        )
        try:
            # pyre-fixme[16]: `Optional` has no attribute `best_point`.
            X = self.model.best_point(
                bounds=bounds,
                objective_weights=obj_w,
                outcome_constraints=oc_c,
                linear_constraints=l_c,
                fixed_features=fixed_features,
                model_gen_options=model_gen_options,
                target_fidelities=target_fidelities,
            )
            return None if X is None else X.detach().cpu().clone().numpy()

        except NotImplementedError:
            return None

    def _model_cross_validate(
        self,
        Xs_train: List[np.ndarray],
        Ys_train: List[np.ndarray],
        Yvars_train: List[np.ndarray],
        X_test: np.ndarray,
        search_space_digest: SearchSpaceDigest,
        metric_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.model:  # pragma: no cover
            raise ValueError(FIT_MODEL_ERROR.format(action="_model_cross_validate"))
        # pyre-fixme[35]: Target cannot be annotated.
        Xs_train: List[Tensor] = self._array_list_to_tensors(Xs_train)
        # pyre-fixme[35]: Target cannot be annotated.
        Ys_train: List[Tensor] = self._array_list_to_tensors(Ys_train)
        # pyre-fixme[35]: Target cannot be annotated.
        Yvars_train: List[Tensor] = self._array_list_to_tensors(Yvars_train)
        # pyre-fixme[35]: Target cannot be annotated.
        X_test: Tensor = self._array_to_tensor(X_test)
        # pyre-fixme[16]: `Optional` has no attribute `cross_validate`.
        f_test, cov_test = self.model.cross_validate(
            Xs_train=Xs_train,
            Ys_train=Ys_train,
            Yvars_train=Yvars_train,
            X_test=X_test,
            search_space_digest=search_space_digest,
            metric_names=metric_names,
        )
        return (
            f_test.detach().cpu().clone().numpy(),
            cov_test.detach().cpu().clone().numpy(),
        )

    def _array_to_tensor(self, array: Union[np.ndarray, List[float]]) -> Tensor:
        return torch.as_tensor(array, dtype=self.dtype, device=self.device)

    def _array_list_to_tensors(self, arrays: List[np.ndarray]) -> List[Tensor]:
        return [self._array_to_tensor(x) for x in arrays]

    def _array_callable_to_tensor_callable(
        self, array_func: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[Tensor], Tensor]:
        tensor_func: Callable[[Tensor], Tensor] = lambda x: (
            self._array_to_tensor(array_func(x.detach().cpu().clone().numpy()))
        )
        return tensor_func

    def _transform_observation_features(
        self, observation_features: List[ObservationFeatures]
    ) -> Tensor:
        return self._array_to_tensor(
            super()._transform_observation_features(observation_features)
        )

    def _transform_observation_data(
        self, observation_data: List[ObservationData]
    ) -> Tuple[Tensor, Tensor]:
        mean, cov = super()._transform_observation_data(observation_data)
        return self._array_to_tensor(mean), self._array_to_tensor(cov)

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

        assert (
            self.is_moo_problem
        ), "Objective thresholds are only supported for multi-objective optimization."

        search_space = (search_space or self._model_space).clone()
        base_gen_args = self._get_transformed_gen_args(
            search_space=search_space,
            optimization_config=optimization_config,
            fixed_features=fixed_features,
        )
        # Get transformed args from ArrayModelbridge.
        array_model_gen_args = self._get_transformed_model_gen_args(
            search_space=base_gen_args.search_space,
            fixed_features=base_gen_args.fixed_features,
            pending_observations={},
            optimization_config=base_gen_args.optimization_config,
        )
        # Get transformed args from TorchModelbridge.
        obj_w, oc_c, l_c, pend_obs, _ = validate_and_apply_final_transform(
            objective_weights=array_model_gen_args.objective_weights,
            outcome_constraints=array_model_gen_args.outcome_constraints,
            pending_observations=None,
            linear_constraints=array_model_gen_args.linear_constraints,
            final_transform=self._array_to_tensor,
        )
        # Infer objective thresholds.
        model = checked_cast(MultiObjectiveBotorchModel, self.model)
        obj_thresholds_arr = infer_objective_thresholds(
            model=not_none(model.model),
            objective_weights=obj_w,
            bounds=array_model_gen_args.search_space_digest.bounds,
            outcome_constraints=oc_c,
            linear_constraints=l_c,
            fixed_features=array_model_gen_args.fixed_features,
            Xs=model.Xs,
        )
        return self._untransform_objective_thresholds(
            objective_thresholds=obj_thresholds_arr,
            objective_weights=obj_w,
            bounds=array_model_gen_args.search_space_digest.bounds,
            opt_config_metrics=base_gen_args.optimization_config.metrics,
            fixed_features=array_model_gen_args.fixed_features,
        )

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
        for t in reversed(self.transforms.values()):
            thresholds = t.untransform_objective_thresholds(
                objective_thresholds=thresholds,
                observation_features=observation_features,
            )
            observation_features = t.untransform_observation_features(
                observation_features
            )

        return thresholds
