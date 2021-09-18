#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.core.types import TCandidateMetadata, TConfig, TGenMetadata
from ax.modelbridge.array import FIT_MODEL_ERROR, ArrayModelBridge
from ax.modelbridge.modelbridge_utils import (
    validate_and_apply_final_transform,
    SearchSpaceDigest,
)
from ax.modelbridge.transforms.base import Transform
from ax.models.torch_base import TorchModel
from ax.utils.common.typeutils import not_none
from torch import Tensor


# pyre-fixme[13]: Attribute `model` is never initialized.
# pyre-fixme[13]: Attribute `outcomes` is never initialized.
# pyre-fixme[13]: Attribute `parameters` is never initialized.
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
        default_model_gen_options: Optional[TConfig] = None,
    ) -> None:
        if torch_dtype is None:  # pragma: no cover
            torch_dtype = torch.float  # noqa T484
        self.dtype = torch_dtype
        self.device = torch_device
        self._default_model_gen_options = default_model_gen_options or {}
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

    def _model_evaluate_acquisition_function(self, X: np.ndarray) -> np.ndarray:
        if not self.model:  # pragma: no cover
            raise ValueError(
                FIT_MODEL_ERROR.format(action="_model_evaluate_acquisition_function")
            )
        evals = not_none(self.model).evaluate_acquisition_function(
            X=self._array_to_tensor(X)
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
    ) -> Tuple[np.ndarray, np.ndarray, TGenMetadata, List[TCandidateMetadata]]:
        if not self.model:  # pragma: no cover
            raise ValueError(FIT_MODEL_ERROR.format(action="_model_gen"))
        obj_w, oc_c, l_c, pend_obs, _ = validate_and_apply_final_transform(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            pending_observations=pending_observations,
            final_transform=self._array_to_tensor,
        )
        tensor_rounding_func = self._array_callable_to_tensor_callable(rounding_func)
        augmented_model_gen_options = {
            **self._default_model_gen_options,
            **(model_gen_options or {}),
        }
        # pyre-fixme[16]: `Optional` has no attribute `gen`.
        X, w, gen_metadata, candidate_metadata = self.model.gen(
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
        )
        return (
            X.detach().cpu().clone().numpy(),
            w.detach().cpu().clone().numpy(),
            gen_metadata,
            candidate_metadata,
        )

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
