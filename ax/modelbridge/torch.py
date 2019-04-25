#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.search_space import SearchSpace
from ax.core.types import TConfig
from ax.modelbridge.array import FIT_MODEL_ERROR, ArrayModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.models.torch_base import TorchModel
from torch import Tensor


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

    # pyre-fixme[13]: Attribute `model` is never initialized.
    model: Optional[TorchModel]
    # pyre-fixme[13]: Attribute `outcomes` is never initialized.
    outcomes: Optional[List[str]]
    # pyre-fixme[13]: Attribute `parameters` is never initialized.
    parameters: Optional[List[str]]

    def __init__(
        self,
        experiment: Experiment,
        search_space: SearchSpace,
        data: Data,
        model: TorchModel,
        transforms: List[Type[Transform]],
        transform_configs: Optional[Dict[str, TConfig]] = None,
        # pyre-fixme[11]: Type `dtype` is not defined.
        torch_dtype: Optional[torch.dtype] = None,  # noqa T484
        torch_device: Optional[torch.device] = None,
        status_quo_name: Optional[str] = None,
        status_quo_features: Optional[ObservationFeatures] = None,
    ) -> None:
        if torch_dtype is None:  # pragma: no cover
            torch_dtype = torch.float  # noqa T484
        self.dtype = torch_dtype
        self.device = torch_device
        super().__init__(
            experiment=experiment,
            search_space=search_space,
            data=data,
            model=model,
            transforms=transforms,
            transform_configs=transform_configs,
            status_quo_name=status_quo_name,
            status_quo_features=status_quo_features,
        )

    def _fit(
        self,
        model: TorchModel,
        search_space: SearchSpace,
        observation_features: List[ObservationFeatures],
        observation_data: List[ObservationData],
    ) -> None:  # pragma: no cover
        super()._fit(
            model=model,
            search_space=search_space,
            observation_features=observation_features,
            observation_data=observation_data,
        )

    def _model_fit(
        self,
        model: TorchModel,
        Xs: List[np.ndarray],
        Ys: List[np.ndarray],
        Yvars: List[np.ndarray],
        bounds: List[Tuple[float, float]],
        task_features: List[int],
        feature_names: List[str],
    ) -> None:
        self.model = model
        # Convert numpy arrays to torch tensors
        Xs: List[Tensor] = self._array_list_to_tensors(Xs)
        Ys: List[Tensor] = self._array_list_to_tensors(Ys)
        Yvars: List[Tensor] = self._array_list_to_tensors(Yvars)
        self.model.fit(
            Xs=Xs,
            Ys=Ys,
            Yvars=Yvars,
            bounds=bounds,
            task_features=task_features,
            feature_names=feature_names,
        )

    def _model_update(
        self, Xs: List[np.ndarray], Ys: List[np.ndarray], Yvars: List[np.ndarray]
    ) -> None:
        if not self.model:  # pragma: no cover
            raise ValueError(FIT_MODEL_ERROR.format(action="_model_update"))
        Xs: List[Tensor] = self._array_list_to_tensors(Xs)
        Ys: List[Tensor] = self._array_list_to_tensors(Ys)
        Yvars: List[Tensor] = self._array_list_to_tensors(Yvars)
        self.model.update(Xs=Xs, Ys=Ys, Yvars=Yvars)

    def _model_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.model:  # pragma: no cover
            raise ValueError(FIT_MODEL_ERROR.format(action="_model_predict"))
        f, var = self.model.predict(X=self._array_to_tensor(X))
        return f.detach().cpu().clone().numpy(), var.detach().cpu().clone().numpy()

    def _validate_and_convert_to_tensors(
        self,
        objective_weights: np.ndarray,
        outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]],
        linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]],
        pending_observations: Optional[List[np.ndarray]],
    ) -> Tuple[
        Tensor,
        Optional[Tuple[Tensor, Tensor]],
        Optional[Tuple[Tensor, Tensor]],
        Optional[List[Tensor]],
    ]:
        objective_weights: Tensor = self._array_to_tensor(objective_weights)
        if outcome_constraints is not None:  # pragma: no cover
            outcome_constraints = (
                self._array_to_tensor(outcome_constraints[0]),
                self._array_to_tensor(outcome_constraints[1]),
            )
        if linear_constraints is not None:  # pragma: no cover
            linear_constraints = (
                self._array_to_tensor(linear_constraints[0]),
                self._array_to_tensor(linear_constraints[1]),
            )
        if pending_observations is not None:  # pragma: no cover
            pending_observations = self._array_list_to_tensors(pending_observations)
        return (
            objective_weights,
            outcome_constraints,
            linear_constraints,
            pending_observations,
        )

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.model:  # pragma: no cover
            raise ValueError(FIT_MODEL_ERROR.format(action="_model_gen"))
        obj_w, oc_c, l_c, pend_obs = self._validate_and_convert_to_tensors(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            pending_observations=pending_observations,
        )
        tensor_rounding_func = self._array_callable_to_tensor_callable(rounding_func)
        X, w = self.model.gen(
            n=n,
            bounds=bounds,
            objective_weights=obj_w,
            outcome_constraints=oc_c,
            linear_constraints=l_c,
            fixed_features=fixed_features,
            pending_observations=pend_obs,
            model_gen_options=model_gen_options,
            rounding_func=tensor_rounding_func,
        )
        return X.detach().cpu().clone().numpy(), w.detach().cpu().clone().numpy()

    def _model_best_point(
        self,
        bounds: List[Tuple[float, float]],
        objective_weights: np.ndarray,
        outcome_constraints: Optional[Tuple[np.ndarray, np.ndarray]],
        linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]],
        fixed_features: Optional[Dict[int, float]],
        model_gen_options: Optional[TConfig],
    ) -> Optional[np.ndarray]:  # pragma: no cover
        if not self.model:  # pragma: no cover
            raise ValueError(FIT_MODEL_ERROR.format(action="_model_gen"))
        obj_w, oc_c, l_c, _ = self._validate_and_convert_to_tensors(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            pending_observations=None,
        )
        try:
            X = self.model.best_point(
                bounds=bounds,
                objective_weights=obj_w,
                outcome_constraints=oc_c,
                linear_constraints=l_c,
                fixed_features=fixed_features,
                model_gen_options=model_gen_options,
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.model:  # pragma: no cover
            raise ValueError(FIT_MODEL_ERROR.format(action="_model_cross_validate"))
        Xs_train: List[Tensor] = self._array_list_to_tensors(Xs_train)
        Ys_train: List[Tensor] = self._array_list_to_tensors(Ys_train)
        Yvars_train: List[Tensor] = self._array_list_to_tensors(Yvars_train)
        X_test: Tensor = self._array_to_tensor(X_test)
        f_test, cov_test = self.model.cross_validate(
            Xs_train=Xs_train, Ys_train=Ys_train, Yvars_train=Yvars_train, X_test=X_test
        )
        return (
            f_test.detach().cpu().clone().numpy(),
            cov_test.detach().cpu().clone().numpy(),
        )

    def _array_to_tensor(self, array: np.ndarray) -> Tensor:
        return torch.tensor(array, dtype=self.dtype, device=self.device)

    def _array_list_to_tensors(self, arrays: List[np.ndarray]) -> List[Tensor]:
        return [self._array_to_tensor(x) for x in arrays]

    def _array_callable_to_tensor_callable(
        self, array_func: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[Tensor], Tensor]:
        tensor_func: Callable[[Tensor], Tensor] = lambda x: (
            self._array_to_tensor(array_func(x.detach().cpu().clone().numpy()))
        )
        return tensor_func
