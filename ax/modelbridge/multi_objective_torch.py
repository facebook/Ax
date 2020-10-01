#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig
from ax.core.search_space import SearchSpace
from ax.core.types import TCandidateMetadata, TConfig, TGenMetadata
from ax.modelbridge.array import (
    FIT_MODEL_ERROR,
    array_to_observation_data,
    extract_objective_weights,
    extract_outcome_constraints,
)
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.models.torch_base import TorchModel
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none
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

    _transformed_ref_point: Optional[Dict[str, float]]
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
        optimization_config: Optional[OptimizationConfig] = None,
        fit_out_of_design: bool = False,
        ref_point: Optional[Dict[str, float]] = None,
        default_model_gen_options: Optional[TConfig] = None,
    ) -> None:
        if isinstance(experiment, MultiTypeExperiment) and ref_point is not None:
            raise NotImplementedError(
                "Ref-point dependent multi-objective optimization algorithms "
                "like EHVI are not yet supported for MultiTypeExperiments. "
                "Remove the reference point arg and use a compatible algorithm "
                "like ParEGO."
            )
        self.ref_point = ref_point
        self._transformed_ref_point = None
        self._objective_metric_names = None
        oc = optimization_config or experiment.optimization_config
        if oc:
            self._objective_metric_names = [m.name for m in oc.objective.metrics]
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

    def _get_ref_point_list(self, ref_point: Dict[str, float]) -> List[float]:
        """Formats ref_point's values as a list, in the order of self.outcomes."""
        return [
            ref_point[name] for name in not_none(self.outcomes) if name in ref_point
        ]

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
        obj_w, oc_c, l_c, pend_obs = self._validate_and_convert_to_tensors(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            pending_observations=pending_observations,
        )
        ref_point = None
        if self._transformed_ref_point:
            ref_point = not_none(self._transformed_ref_point)
        elif self.ref_point:
            ref_point = self.ref_point
            logger.warning(
                "No attribute _transformed_ref_point. Using untransformed ref_point."
            )
        if ref_point is not None:
            ref_point_list = self._get_ref_point_list(ref_point)
        else:
            ref_point_list = None
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
            ref_point=ref_point_list,
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

        ref_point = self.ref_point
        if ref_point and obs_data:
            self._transformed_ref_point = self._transform_ref_point(
                ref_point=ref_point, padding_obs_data=obs_data[0]
            )
        return obs_feats, obs_data, search_space

    def _transform_ref_point(
        self, ref_point: Dict[str, float], padding_obs_data: ObservationData
    ) -> Dict[str, float]:
        """Transform ref_point using same transforms as those applied to data.

        Args:
            ref_point: Reference point to transform.
            padding_obs_data: Data used to add dummy outcomes that aren't part
                of the reference point. This is necessary to apply transforms.

        Return:
            A transformed reference point.
        """
        metric_names = list(self._metric_names or [])
        objective_metric_names = list(self._objective_metric_names or [])
        num_metrics = len(metric_names)
        # Create synthetic ObservationData representing the reference point.
        # Pad with non-objective outcomes from existing data.
        # Should always have existing data with BO.
        padding_obs_data
        padded_ref_dict: Dict[str, float] = dict(
            zip(padding_obs_data.metric_names, padding_obs_data.means)
        )
        padded_ref_dict.update(ref_point)
        ref_obs_data = [
            ObservationData(
                metric_names=list(padded_ref_dict.keys()),
                means=np.array(list(padded_ref_dict.values())),
                covariance=np.zeros((num_metrics, num_metrics)),
            )
        ]
        ref_obs_feats = []

        # Apply initialized transforms to reference point.
        for t in self.transforms.values():
            ref_obs_data = t.transform_observation_data(ref_obs_data, ref_obs_feats)
        transformed_ref_obsd = ref_obs_data.pop()
        transformed_ref_dict = dict(
            zip(transformed_ref_obsd.metric_names, transformed_ref_obsd.means)
        )
        transformed_ref_point = {
            objective_metric_name: transformed_ref_dict[objective_metric_name]
            for objective_metric_name in objective_metric_names
        }
        return transformed_ref_point

    # pyre-fixme[56]: While applying decorator
    #  `ax.utils.common.docutils.copy_doc(...)`: Call expects argument `n`.
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

    def _pareto_frontier(
        self,
        ref_point: Optional[Dict[str, float]] = None,
        observation_features: Optional[List[ObservationFeatures]] = None,
        observation_data: Optional[List[ObservationData]] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ) -> List[ObservationData]:
        """Helper that applies transforms and calls frontier_evaluator."""
        X = (
            self.transform_observation_features(observation_features)
            if observation_features
            else None
        )
        Y, Yvar = (
            self.transform_observation_data(observation_data)
            if observation_data
            else (None, None)
        )

        # Optimization_config
        if optimization_config is None:
            optimization_config = (
                # pyre-fixme[16]: `Optional` has no attribute `clone`.
                self._optimization_config.clone()
                if self._optimization_config is not None
                else None
            )
        else:
            optimization_config = optimization_config.clone()

        # Transform OptimizationConfig, ObservationFeatures and ref_point
        for t in self.transforms.values():
            optimization_config = t.transform_optimization_config(
                optimization_config=optimization_config,
                modelbridge=self,
                fixed_features=ObservationFeatures(parameters={}),
            )
        objective_weights = extract_objective_weights(
            objective=optimization_config.objective, outcomes=self.outcomes
        )
        outcome_constraints = extract_outcome_constraints(
            outcome_constraints=optimization_config.outcome_constraints,
            outcomes=self.outcomes,
        )
        obj_w, oc_c, _, _ = self._validate_and_convert_to_tensors(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=None,
            pending_observations=None,
        )
        training_observations = self.get_training_data()
        assert (
            training_observations
        ), "Cannot pad reference_point without training_observations"
        padding_obs_data = training_observations[0].data
        ref_point = self._transform_ref_point(
            ref_point=not_none(ref_point), padding_obs_data=padding_obs_data
        )
        rf_pt = self._array_to_tensor(np.array(self._get_ref_point_list(ref_point)))
        # Transform features, Evaluate a frontier, and untransform results
        # pyre-ignore [16]:  `TorchModel` has no attribute `frontier_evaluator`
        f, cov = not_none(self.model).frontier_evaluator(
            model=self.model,
            X=X,
            Y=Y,
            Yvar=Yvar,
            ref_point=rf_pt,
            objective_weights=obj_w,
            outcome_constraints=oc_c,
        )
        f, cov = f.detach().cpu().clone().numpy(), cov.detach().cpu().clone().numpy()
        frontier_observation_data = array_to_observation_data(
            f=f, cov=cov, outcomes=not_none(self.outcomes)
        )
        # Untransform observations
        for t in reversed(self.transforms.values()):  # noqa T484
            frontier_observation_data = t.untransform_observation_data(
                frontier_observation_data, []
            )
        return frontier_observation_data

    def predicted_pareto_frontier(
        self,
        ref_point: Optional[Dict[str, float]] = None,
        observation_features: Optional[List[ObservationFeatures]] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ) -> List[ObservationData]:
        """Generate a pareto frontier based on predictions of given observation features.

        Given a model and features to evaluate use the model to predict which points
        lie on the pareto frontier.

        Args:
            ref_point: metric values bounding the region of interest in the objective
                outcome space.
            observation_features: observation features to predict. Model's training
                data used by default if unspecified.
            optimization_config: Optimization config

        Returns:
            Data representing points on the pareto frontier.
        """
        # If observation_features is not provided, use model training features.
        observation_features = (
            observation_features
            if observation_features is not None
            else [obs.features for obs in self.get_training_data()]
        )
        if not observation_features:
            raise ValueError(
                "Must receive observation_features as input or the model must "
                "have training data."
            )

        return self._pareto_frontier(
            ref_point=ref_point,
            observation_features=observation_features,
            optimization_config=optimization_config,
        )

    def observed_pareto_frontier(
        self,
        ref_point: Optional[Dict[str, float]] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ) -> List[ObservationData]:
        """Generate a pareto frontier based on observed data.

        Given observed data, return those outcomes in the pareto frontier.

        Args:
            ref_point: point defing the origin of hyperrectangles that can contribute
                to hypervolume.
            optimization_config: Optimization config

        Returns:
            Data representing points on the pareto frontier.
        """
        # Get observation_data from current training data
        observation_data = [obs.data for obs in self.get_training_data()]

        return self._pareto_frontier(
            ref_point=ref_point,
            observation_data=observation_data,
            optimization_config=optimization_config,
        )

    def hypervolume(self, X: Tensor) -> Tensor:
        raise NotImplementedError()  # pragma: no cover

    def observed_hypervolume(self) -> Tensor:
        raise NotImplementedError()  # pragma: no cover
