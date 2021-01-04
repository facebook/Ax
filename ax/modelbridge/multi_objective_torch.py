#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
    TRefPoint,
)
from ax.core.search_space import SearchSpace
from ax.core.types import TCandidateMetadata, TConfig, TGenMetadata
from ax.modelbridge.array import FIT_MODEL_ERROR
from ax.modelbridge.modelbridge_utils import (
    extract_objective_thresholds,
    validate_and_apply_final_transform,
)
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.models.torch.frontier_utils import (
    TFrontierEvaluator,
    get_default_frontier_evaluator,
)
from ax.models.torch_base import TorchModel
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast_optional, not_none


logger = get_logger("MultiObjectiveTorchModelBridge")


class MultiObjectiveTorchModelBridge(TorchModelBridge):
    """A model bridge for using multi-objective torch-based models.

    Specifies an interface that is implemented by MultiObjectiveTorchModel. In
    particular, model should have methods fit, predict, and gen. See
    MultiObjectiveTorchModel for the API for each of these methods.

    Requires that all parameters have been transformed to RangeParameters
    or FixedParameters with float type and no log scale.

    This class converts Ax parameter types to torch tensors before passing
    them to the model.
    """

    _objective_metric_names: Optional[List[str]]

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
        optimization_config: Optional[MultiObjectiveOptimizationConfig] = None,
        fit_out_of_design: bool = False,
        objective_thresholds: Optional[TRefPoint] = None,
        default_model_gen_options: Optional[TConfig] = None,
    ) -> None:
        self._objective_metric_names = None
        # Optimization_config
        mooc = optimization_config or checked_cast_optional(
            MultiObjectiveOptimizationConfig, experiment.optimization_config
        )
        # Extract objective_thresholds from optimization_config, or inject it.
        if not mooc:
            raise ValueError(
                (
                    "experiment must have an existing optimization_config "
                    "of type MultiObjectiveOptimizationConfig "
                    "or `optimization_config` must be passed as an argument."
                )
            )
        if not isinstance(mooc, MultiObjectiveOptimizationConfig):
            mooc = not_none(MultiObjectiveOptimizationConfig.from_opt_conf(mooc))
        if objective_thresholds:
            mooc = mooc.clone_with_args(objective_thresholds=objective_thresholds)

        optimization_config = mooc

        super().__init__(
            experiment=experiment,
            search_space=search_space,
            data=data,
            model=model,
            transforms=transforms,
            transform_configs=transform_configs,
            torch_dtype=torch_dtype,
            torch_device=torch_device,
            status_quo_name=status_quo_name,
            status_quo_features=status_quo_features,
            optimization_config=optimization_config,
            fit_out_of_design=fit_out_of_design,
            default_model_gen_options=default_model_gen_options,
        )

    def _get_extra_model_gen_kwargs(
        self, optimization_config: OptimizationConfig
    ) -> Dict[str, Any]:
        extra_kwargs_dict = {}
        if isinstance(optimization_config, MultiObjectiveOptimizationConfig):
            objective_thresholds = extract_objective_thresholds(
                objective_thresholds=optimization_config.objective_thresholds,
                outcomes=self.outcomes,
            )
        else:
            objective_thresholds = np.array([])
        # pyre-fixme[35]: Target cannot be annotated.
        objective_thresholds: Optional[np.ndarray] = (
            objective_thresholds if len(objective_thresholds) else None
        )
        if objective_thresholds is not None:
            extra_kwargs_dict["objective_thresholds"] = objective_thresholds
        return extra_kwargs_dict

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
    ) -> Tuple[np.ndarray, np.ndarray, TGenMetadata, List[TCandidateMetadata]]:
        if not self.model:  # pragma: no cover
            raise ValueError(FIT_MODEL_ERROR.format(action="_model_gen"))
        obj_w, oc_c, l_c, pend_obs = validate_and_apply_final_transform(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            pending_observations=pending_observations,
            final_transform=self._array_to_tensor,
        )
        obj_t = (
            self._array_to_tensor(objective_thresholds)
            if objective_thresholds is not None
            else None
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
            objective_thresholds=obj_t,
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

    def _transform_data(
        self,
        obs_feats: List[ObservationFeatures],
        obs_data: List[ObservationData],
        search_space: SearchSpace,
        transforms: Optional[List[Type[Transform]]],
        transform_configs: Optional[Dict[str, TConfig]],
    ) -> Tuple[List[ObservationFeatures], List[ObservationData], SearchSpace]:
        """Initialize transforms and apply them to provided data."""
        # Run superclass version to fit transforms to observations
        obs_feats, obs_data, search_space = super()._transform_data(
            obs_feats=obs_feats,
            obs_data=obs_data,
            search_space=search_space,
            transforms=transforms,
            transform_configs=transform_configs,
        )
        return obs_feats, obs_data, search_space

    @copy_doc(TorchModelBridge.gen)
    def gen(
        self,
        n: int,
        search_space: Optional[SearchSpace] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        pending_observations: Optional[Dict[str, List[ObservationFeatures]]] = None,
        fixed_features: Optional[ObservationFeatures] = None,
        model_gen_options: Optional[TConfig] = None,
    ) -> GeneratorRun:
        if optimization_config:
            # Update objective metric names if new optimization config is present.
            self._objective_metric_names = [
                m.name for m in optimization_config.objective.metrics
            ]
        return super().gen(
            n=n,
            search_space=search_space,
            optimization_config=optimization_config,
            pending_observations=pending_observations,
            fixed_features=fixed_features,
            model_gen_options=model_gen_options,
        )

    def _get_frontier_evaluator(self) -> TFrontierEvaluator:
        return (
            # pyre-ignore [16]: `TorchModel has no attribute `frontier_evaluator`
            not_none(self.model).frontier_evaluator
            if hasattr(self.model, "frontier_evaluator")
            else get_default_frontier_evaluator()
        )
