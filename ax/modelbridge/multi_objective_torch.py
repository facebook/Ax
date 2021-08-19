#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

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
from ax.core.outcome_constraint import ComparisonOp, ObjectiveThreshold
from ax.core.search_space import SearchSpace
from ax.core.types import TCandidateMetadata, TConfig, TGenMetadata
from ax.exceptions.core import AxError
from ax.modelbridge.array import FIT_MODEL_ERROR
from ax.modelbridge.modelbridge_utils import (
    extract_objective_thresholds,
    parse_observation_features,
    validate_and_apply_final_transform,
)
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.models.torch.botorch_moo_defaults import infer_objective_thresholds
from ax.models.torch.frontier_utils import (
    TFrontierEvaluator,
    get_default_frontier_evaluator,
)
from ax.models.torch_base import TorchModel
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast_optional, not_none
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
                objective=optimization_config.objective,
                outcomes=self.outcomes,
            )
        else:
            objective_thresholds = None
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
        (obj_w, oc_c, l_c, pend_obs, obj_t,) = validate_and_apply_final_transform(
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
        # if objective_thresholds are supplied by the user, then the
        # transformed user-specified objective thresholds are in
        # gen_metadata. Otherwise, inferred objective thresholds are
        # in gen_metadata.
        objective_thresholds = gen_metadata["objective_thresholds"]
        obj_thlds = self.untransform_objective_thresholds(
            objective_thresholds=objective_thresholds,
            objective_weights=obj_w,
            bounds=bounds,
            fixed_features=fixed_features,
        )
        gen_metadata["objective_thresholds"] = obj_thlds
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
        if search_space is None:
            search_space = self._model_space
        search_space = search_space.clone()
        base_gen_args = self._get_transformed_gen_args(
            search_space=search_space,
            optimization_config=optimization_config,
            fixed_features=fixed_features,
        )
        # get transformed args from ArrayModelbridge
        array_model_gen_args = self._get_transformed_model_gen_args(
            search_space=base_gen_args.search_space,
            fixed_features=base_gen_args.fixed_features,
            pending_observations={},
            optimization_config=base_gen_args.optimization_config,
        )
        # get transformed args from TorchModelbridge
        obj_w, oc_c, l_c, pend_obs, _ = validate_and_apply_final_transform(
            objective_weights=array_model_gen_args.objective_weights,
            outcome_constraints=array_model_gen_args.outcome_constraints,
            pending_observations=None,
            linear_constraints=array_model_gen_args.linear_constraints,
            final_transform=self._array_to_tensor,
        )
        # infer objective thresholds
        model = not_none(self.model)
        try:
            torch_model = model.model  # pyre-ignore [16]
            Xs = model.Xs  # pyre-ignore [16]
        except AttributeError:
            raise AxError(
                "infer_objective_thresholds requires a TorchModel with model "
                "and Xs attributes."
            )
        obj_thresholds_arr = infer_objective_thresholds(
            model=torch_model,
            objective_weights=obj_w,
            bounds=array_model_gen_args.search_space_digest.bounds,
            outcome_constraints=oc_c,
            linear_constraints=l_c,
            fixed_features=array_model_gen_args.fixed_features,
            Xs=Xs,
        )
        return self.untransform_objective_thresholds(
            objective_thresholds=obj_thresholds_arr,
            objective_weights=obj_w,
            bounds=array_model_gen_args.search_space_digest.bounds,
            fixed_features=array_model_gen_args.fixed_features,
        )

    def untransform_objective_thresholds(
        self,
        objective_thresholds: Tensor,
        objective_weights: Tensor,
        bounds: List[Tuple[Union[int, float], Union[int, float]]],
        fixed_features: Optional[Dict[int, float]],
    ) -> List[ObjectiveThreshold]:
        objective_thresholds_np = objective_thresholds.cpu().numpy()
        # pyre-ignore [16]
        objective_indices = objective_weights.nonzero().view(-1).tolist()
        objective_names = [self.outcomes[i] for i in objective_indices]
        # create an ObservationData object for untransforming the objective thresholds
        observation_data = [
            ObservationData(
                metric_names=objective_names,
                means=objective_thresholds_np[objective_indices].copy(),
                covariance=np.zeros((len(objective_indices), len(objective_indices))),
            )
        ]
        # Untransform objective thresholds. Note: there is one objective threshold
        # for every outcome.
        # Construct dummy observation features
        X = [bound[0] for bound in bounds]
        fixed_features = fixed_features or {}
        for i, val in fixed_features.items():
            X[i] = val
        observation_features = parse_observation_features(
            X=np.array([X]),
            param_names=self.parameters,
        )
        # Apply reverse transforms, in reverse order
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
