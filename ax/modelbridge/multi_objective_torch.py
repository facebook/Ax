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
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
    TRefPoint,
)
from ax.core.outcome_constraint import ComparisonOp, ObjectiveThreshold
from ax.core.search_space import SearchSpace
from ax.core.types import TCandidateMetadata, TConfig, TGenMetadata
from ax.modelbridge.modelbridge_utils import (
    extract_objective_thresholds,
    parse_observation_features,
    validate_and_apply_final_transform,
)
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.models.torch.botorch_moo import MultiObjectiveBotorchModel
from ax.models.torch.botorch_moo_defaults import infer_objective_thresholds
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none
from torch import Tensor


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

    def __init__(
        self,
        experiment: Experiment,
        search_space: SearchSpace,
        data: Data,
        model: MultiObjectiveBotorchModel,
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
        moo_config = optimization_config or experiment.optimization_config
        if not moo_config:
            raise ValueError(
                (
                    "Experiment must have an existing `optimization_config` "
                    "of type `MultiObjectiveOptimizationConfig` "
                    "or non-null `optimization_config` must be passed as an argument."
                )
            )
        if not isinstance(moo_config, MultiObjectiveOptimizationConfig):
            raise TypeError(
                "`optimization_config` must be a `MultiObjectiveOptimizationConfig`;"
                f" received: {moo_config}."
            )
        if objective_thresholds:
            moo_config = moo_config.clone_with_args(
                objective_thresholds=objective_thresholds
            )

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
            optimization_config=moo_config,
            fit_out_of_design=fit_out_of_design,
            default_model_gen_options=default_model_gen_options,
        )

    def _get_extra_model_gen_kwargs(
        self, optimization_config: OptimizationConfig
    ) -> Dict[str, Any]:
        extra_kwargs_dict = {}
        if isinstance(optimization_config, MultiObjectiveOptimizationConfig):
            extra_kwargs_dict["objective_thresholds"] = extract_objective_thresholds(
                objective_thresholds=optimization_config.objective_thresholds,
                objective=optimization_config.objective,
                outcomes=self.outcomes,
            )
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
        X, w, gen_metadata, candidate_metadata = super()._model_gen(
            n=n,
            bounds=bounds,
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            fixed_features=fixed_features,
            pending_observations=pending_observations,
            model_gen_options=model_gen_options,
            rounding_func=rounding_func,
            target_fidelities=target_fidelities,
            objective_thresholds=objective_thresholds,
        )
        # If objective_thresholds are supplied by the user, then the transformed
        # user-specified objective thresholds are in gen_metadata. Otherwise,
        # inferred objective thresholds are in gen_metadata.
        gen_metadata["objective_thresholds"] = self._untransform_objective_thresholds(
            objective_thresholds=gen_metadata["objective_thresholds"],
            objective_weights=gen_metadata["objective_weights"],
            bounds=bounds,
            fixed_features=fixed_features,
        )
        return X, w, gen_metadata, candidate_metadata

    def infer_objective_thresholds(
        self,
        search_space: Optional[SearchSpace] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        fixed_features: Optional[ObservationFeatures] = None,
    ) -> List[ObjectiveThreshold]:
        """Infer objective thresholds.

        This method uses the model-estimated Pareto frontier over the in-sample points
        to infer absolute (not relativized) objective thresholds.

        This uses a heuristic that sets the objective threshold to be a scaled nadir
        point, where the nadir point is scaled back based on the range of each
        objective across the current in-sample Pareto frontier.
        """
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
            fixed_features=array_model_gen_args.fixed_features,
        )

    def _untransform_objective_thresholds(
        self,
        objective_thresholds: Tensor,
        objective_weights: Tensor,
        bounds: List[Tuple[Union[int, float], Union[int, float]]],
        fixed_features: Optional[Dict[int, float]],
    ) -> List[ObjectiveThreshold]:
        objective_thresholds_np = objective_thresholds.cpu().numpy()
        objective_indices = objective_weights.nonzero().view(-1).tolist()
        objective_names = [self.outcomes[i] for i in objective_indices]
        # Create an ObservationData object for untransforming the objective thresholds.
        observation_data = [
            ObservationData(
                metric_names=objective_names,
                means=objective_thresholds_np[objective_indices].copy(),
                covariance=np.zeros((len(objective_indices), len(objective_indices))),
            )
        ]
        # Untransform objective thresholds. Note: there is one objective threshold
        # for every outcome.
        # Construct dummy observation features.
        X = [bound[0] for bound in bounds]
        fixed_features = fixed_features or {}
        for i, val in fixed_features.items():
            X[i] = val
        observation_features = parse_observation_features(
            X=np.array([X]),
            param_names=self.parameters,
        )
        # Apply reverse transforms, in reverse order.
        for t in reversed(self.transforms.values()):
            observation_data = t.untransform_observation_data(
                observation_data=observation_data,
                observation_features=observation_features,
            )
            observation_features = t.untransform_observation_features(
                observation_features=observation_features,
            )
        observation_data = observation_data[0]
        oc = not_none(self._optimization_config)
        metrics_names_to_metric = oc.metrics
        obj_thresholds = []
        for idx, (name, bound) in enumerate(
            zip(observation_data.metric_names, observation_data.means)
        ):
            if not np.isnan(bound):
                obj_weight = objective_weights[objective_indices[idx]]
                op = (
                    ComparisonOp.LEQ
                    if torch.sign(obj_weight) == -1.0
                    else ComparisonOp.GEQ
                )
                obj_thresholds.append(
                    ObjectiveThreshold(
                        metric=metrics_names_to_metric[name],
                        bound=bound,
                        relative=False,
                        op=op,
                    )
                )
        return obj_thresholds
