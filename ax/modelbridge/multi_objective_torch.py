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
from ax.core.multi_type_experiment import MultiTypeExperiment
from ax.core.observation import ObservationData, ObservationFeatures
from ax.core.optimization_config import OptimizationConfig, TRefPoint
from ax.core.search_space import SearchSpace
from ax.core.types import TCandidateMetadata, TConfig, TGenMetadata
from ax.modelbridge.array import (
    FIT_MODEL_ERROR,
    array_to_observation_data,
    extract_objective_weights,
    extract_outcome_constraints,
    extract_ref_point,
)
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.base import Transform
from ax.models.torch.botorch_moo_defaults import (
    _get_weighted_mc_objective_and_ref_point,
)
from ax.models.torch_base import TorchModel
from ax.utils.common.docutils import copy_doc
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import not_none
from botorch.utils.multi_objective.hypervolume import Hypervolume


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
        optimization_config: Optional[OptimizationConfig] = None,
        fit_out_of_design: bool = False,
        ref_point: Optional[TRefPoint] = None,
        default_model_gen_options: Optional[TConfig] = None,
    ) -> None:
        if isinstance(experiment, MultiTypeExperiment) and ref_point is not None:
            raise NotImplementedError(
                "Ref-point dependent multi-objective optimization algorithms "
                "like EHVI are not yet supported for MultiTypeExperiments. "
                "Remove the reference point arg and use a compatible algorithm "
                "like ParEGO."
            )
        self._objective_metric_names = None
        optimization_config = optimization_config or experiment.optimization_config
        # TODO: Validate optimization config?
        # Extract ref_point from optimization_config, or inject it.
        if optimization_config:
            self._objective_metric_names = [
                m.name for m in optimization_config.objective.metrics
            ]
            # If ref_point was not passed as an arg, check optimization_config
            if optimization_config.ref_point:
                self.ref_point = ref_point or optimization_config.ref_point
            elif ref_point is not None:
                optimization_config = optimization_config.clone_with_args(
                    ref_point=ref_point
                )
                self.ref_point = ref_point
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
        ref_point = extract_ref_point(
            ref_point=optimization_config.ref_point, outcomes=self.outcomes
        )
        ref_point: Optional[np.ndarray] = ref_point if len(ref_point) else None
        return {"ref_point": ref_point}

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
        ref_point: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, TGenMetadata, List[TCandidateMetadata]]:
        if not self.model:  # pragma: no cover
            raise ValueError(FIT_MODEL_ERROR.format(action="_model_gen"))
        obj_w, oc_c, l_c, pend_obs = self._validate_and_convert_to_tensors(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=linear_constraints,
            pending_observations=pending_observations,
        )
        rf_pt = self._array_to_tensor(ref_point) if ref_point is not None else None
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
            ref_point=rf_pt,
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
        ref_point: Optional[TRefPoint] = None,
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

        if ref_point is not None:
            optimization_config = optimization_config.clone_with_args(
                ref_point=ref_point
            )

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
        ref_point_arr = extract_ref_point(
            ref_point=optimization_config.ref_point, outcomes=self.outcomes
        )
        rf_pt = self._array_to_tensor(ref_point_arr)
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
        ref_point: Optional[TRefPoint] = None,
        observation_features: Optional[List[ObservationFeatures]] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ) -> List[ObservationData]:
        """Generate a pareto frontier based on the posterior means of given
        observation features.

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
        ref_point: Optional[TRefPoint] = None,
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

    def _hypervolume(
        self,
        ref_point: Optional[TRefPoint] = None,
        observation_features: Optional[List[ObservationFeatures]] = None,
        observation_data: Optional[List[ObservationData]] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ) -> float:
        """Helper function that computes hypervolume of a given list of outcomes."""
        # Extract a tensor of outcome means from observation data.
        observation_data = self._pareto_frontier(
            ref_point=ref_point,
            observation_features=observation_features,
            observation_data=observation_data,
            optimization_config=optimization_config,
        )
        means, _ = self._transform_observation_data(observation_data)

        # Extract objective_weights and ref_points
        if optimization_config is None:
            optimization_config = (
                # pyre-fixme[16]: `Optional` has no attribute `clone`.
                self._optimization_config.clone()
                if self._optimization_config is not None
                else None
            )
        else:
            optimization_config = optimization_config.clone()

        if ref_point is not None:
            optimization_config = optimization_config.clone_with_args(
                ref_point=ref_point
            )

        objective_weights = extract_objective_weights(
            objective=optimization_config.objective, outcomes=self.outcomes
        )
        obj_w, _, _, _ = self._validate_and_convert_to_tensors(
            objective_weights=objective_weights,
            outcome_constraints=None,
            linear_constraints=None,
            pending_observations=None,
        )
        ref_point_arr = extract_ref_point(
            ref_point=optimization_config.ref_point, outcomes=self.outcomes
        )
        rf_pt = self._array_to_tensor(ref_point_arr)
        obj, rf_pt = _get_weighted_mc_objective_and_ref_point(
            objective_weights=obj_w, ref_point=rf_pt
        )
        means = obj(means)
        hv = Hypervolume(ref_point=rf_pt)
        return hv.compute(means)

    def predicted_hypervolume(
        self,
        ref_point: Optional[TRefPoint] = None,
        observation_features: Optional[List[ObservationFeatures]] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ) -> float:
        """Calculate hypervolume of a pareto frontier based on the posterior means of
        given observation features.

        Given a model and features to evaluate calculate the hypervolume of the pareto
        frontier formed from their predicted outcomes.

        Args:
            model: Model used to predict outcomes.
            ref_point: point defining the origin of hyperrectangles that can contribute
                to hypervolume.
            observation_features: observation features to predict. Model's training
                data used by default if unspecified.
            optimization_config: Optimization config

        Returns:
            (float) calculated hypervolume.
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

        return self._hypervolume(
            ref_point=ref_point,
            observation_features=observation_features,
            optimization_config=optimization_config,
        )

    def observed_hypervolume(
        self,
        ref_point: Optional[TRefPoint] = None,
        optimization_config: Optional[OptimizationConfig] = None,
    ) -> float:
        """Calculate hypervolume of a pareto frontier based on observed data.

        Given observed data, return the hypervolume of the pareto frontier formed from
        those outcomes.

        Args:
            model: Model used to predict outcomes.
            ref_point: point defining the origin of hyperrectangles that can contribute
                to hypervolume.
            observation_features: observation features to predict. Model's training
                data used by default if unspecified.
            optimization_config: Optimization config

        Returns:
            (float) calculated hypervolume.
        """
        # Get observation_data from current training data.
        observation_data = [obs.data for obs in self.get_training_data()]

        return self._hypervolume(
            ref_point=ref_point,
            observation_data=observation_data,
            optimization_config=optimization_config,
        )
