#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABCMeta, abstractmethod
from functools import partial
from typing import List, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from ax.core.experiment import Experiment
from ax.core.observation import ObservationFeatures
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.types import TModelPredictArm, TParameterization
from ax.modelbridge.array import ArrayModelBridge
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.modelbridge_utils import _get_modelbridge_training_data
from ax.modelbridge.modelbridge_utils import observed_hypervolume, predicted_hypervolume
from ax.modelbridge.modelbridge_utils import (
    validate_and_apply_final_transform,
    _array_to_tensor,
    observation_data_to_array,
    extract_objective_thresholds,
    extract_objective_weights,
    extract_outcome_constraints,
)
from ax.modelbridge.registry import get_model_from_generator_run, ModelRegistryBase
from ax.modelbridge.transforms.derelativize import Derelativize
from ax.models.torch.botorch_moo_defaults import (
    get_outcome_constraint_transforms,
    get_weighted_mc_objective_and_objective_thresholds,
)
from ax.plot.pareto_utils import get_tensor_converter_model
from ax.service.utils import best_point as best_point_utils
from ax.utils.common.typeutils import checked_cast, not_none
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning


class BestPointMixin(metaclass=ABCMeta):
    @abstractmethod
    def get_best_trial(
        self,
        optimization_config: Optional[OptimizationConfig] = None,
        trial_indices: Optional[Iterable[int]] = None,
        use_model_predictions: bool = True,
    ) -> Optional[Tuple[int, TParameterization, Optional[TModelPredictArm]]]:
        """Identifies the best parameterization tried in the experiment so far.

        First attempts to do so with the model used in optimization and
        its corresponding predictions if available. Falls back to the best raw
        objective based on the data fetched from the experiment.

        NOTE: ``TModelPredictArm`` is of the form:
            ({metric_name: mean}, {metric_name_1: {metric_name_2: cov_1_2}})

        Args:
            optimization_config: Optimization config to use in place of the one stored
                on the experiment.
            trial_indices: Indices of trials for which to retrieve data. If None will
                retrieve data from all available trials.
            use_model_predictions: Whether to extract the best point using
                model predictions or directly observed values. If ``True``,
                the metric means and covariances in this method's output will
                also be based on model predictions and may differ from the
                observed values.

        Returns:
            Tuple of trial index, parameterization and model predictions for it.
        """
        pass

    def get_best_parameters(
        self,
        optimization_config: Optional[OptimizationConfig] = None,
        trial_indices: Optional[Iterable[int]] = None,
        use_model_predictions: bool = True,
    ) -> Optional[Tuple[TParameterization, Optional[TModelPredictArm]]]:
        """Identifies the best parameterization tried in the experiment so far.

        First attempts to do so with the model used in optimization and
        its corresponding predictions if available. Falls back to the best raw
        objective based on the data fetched from the experiment.

        NOTE: ``TModelPredictArm`` is of the form:
            ({metric_name: mean}, {metric_name_1: {metric_name_2: cov_1_2}})

        Args:
            optimization_config: Optimization config to use in place of the one stored
                on the experiment.
            trial_indices: Indices of trials for which to retrieve data. If None will
                retrieve data from all available trials.
            use_model_predictions: Whether to extract the best point using
                model predictions or directly observed values. If ``True``,
                the metric means and covariances in this method's output will
                also be based on model predictions and may differ from the
                observed values.

        Returns:
            Tuple of parameterization and model predictions for it.
        """
        res = self.get_best_trial(
            optimization_config=optimization_config,
            trial_indices=trial_indices,
            use_model_predictions=use_model_predictions,
        )

        if res is None:
            return res  # pragma: no cover

        _, parameterization, vals = res
        return parameterization, vals

    @abstractmethod
    def get_pareto_optimal_parameters(
        self,
        optimization_config: Optional[OptimizationConfig] = None,
        trial_indices: Optional[Iterable[int]] = None,
        use_model_predictions: bool = True,
    ) -> Optional[Dict[int, Tuple[TParameterization, TModelPredictArm]]]:
        """Identifies the best parameterizations tried in the experiment so far,
        using model predictions if ``use_model_predictions`` is true and using
        observed values from the experiment otherwise. By default, uses model
        predictions to account for observation noise.

        NOTE: The format of this method's output is as follows:
        { trial_index --> (parameterization, (means, covariances) }, where means
        are a dictionary of form { metric_name --> metric_mean } and covariances
        are a nested dictionary of form
        { one_metric_name --> { another_metric_name: covariance } }.

        Args:
            optimization_config: Optimization config to use in place of the one stored
                on the experiment.
            trial_indices: Indices of trials for which to retrieve data. If None will
                retrieve data from all available trials.
            use_model_predictions: Whether to extract the Pareto frontier using
                model predictions or directly observed values. If ``True``,
                the metric means and covariances in this method's output will
                also be based on model predictions and may differ from the
                observed values.

        Returns:
            ``None`` if it was not possible to extract the Pareto frontier,
            otherwise a mapping from trial index to the tuple of:
            - the parameterization of the arm in that trial,
            - two-item tuple of metric means dictionary and covariance matrix
                (model-predicted if ``use_model_predictions=True`` and observed
                otherwise).
        """
        pass

    @abstractmethod
    def get_hypervolume(
        self,
        optimization_config: Optional[MultiObjectiveOptimizationConfig] = None,
        trial_indices: Optional[Iterable[int]] = None,
        use_model_predictions: bool = True,
    ) -> float:
        """Calculate hypervolume of a pareto frontier based on either the posterior
        means of given observation features or observed data.

        Args:
            optimization_config: Optimization config to use in place of the one stored
                on the experiment.
            trial_indices: Indices of trials for which to retrieve data. If None will
                retrieve data from all available trials.
            use_model_predictions: Whether to extract the Pareto frontier using
                model predictions or directly observed values. If ``True``,
                the metric means and covariances in this method's output will
                also be based on model predictions and may differ from the
                observed values.
        """
        pass

    @staticmethod
    def _get_best_trial(
        experiment: Experiment,
        generation_strategy: GenerationStrategy,
        optimization_config: Optional[OptimizationConfig] = None,
        trial_indices: Optional[Iterable[int]] = None,
        use_model_predictions: bool = True,
    ) -> Optional[Tuple[int, TParameterization, Optional[TModelPredictArm]]]:
        if not_none(experiment.optimization_config).is_moo_problem:
            raise NotImplementedError(  # pragma: no cover
                "Please use `get_pareto_optimal_parameters` for multi-objective "
                "problems."
            )
        # TODO[drfreund]: Find a way to include data for last trial in the
        # calculation of best parameters.
        if use_model_predictions:
            current_model = generation_strategy._curr.model
            # Cover for the case where source of `self._curr.model` was not a `Models`
            # enum but a factory function, in which case we cannot do
            # `get_model_from_generator_run` (since we don't have model type and inputs
            # recorded on the generator run.
            models_enum = (
                current_model.__class__
                if isinstance(current_model, ModelRegistryBase)
                else None
            )

            if models_enum is not None:
                res = best_point_utils.get_best_parameters_from_model_predictions_with_trial_index(  # noqa
                    experiment=experiment,
                    models_enum=models_enum,
                    optimization_config=optimization_config,
                    trial_indices=trial_indices,
                )

                if res is not None:
                    return res  # pragma: no cover

        return best_point_utils.get_best_by_raw_objective_with_trial_index(
            experiment=experiment,
            optimization_config=optimization_config,
            trial_indices=trial_indices,
        )

    @staticmethod
    def _get_pareto_optimal_parameters(
        experiment: Experiment,
        generation_strategy: GenerationStrategy,
        optimization_config: Optional[OptimizationConfig] = None,
        trial_indices: Optional[Iterable[int]] = None,
        use_model_predictions: bool = True,
    ) -> Optional[Dict[int, Tuple[TParameterization, TModelPredictArm]]]:
        if not not_none(experiment.optimization_config).is_moo_problem:
            raise NotImplementedError(  # pragma: no cover
                "Please use `get_best_parameters` for single-objective problems."
            )
        return best_point_utils.get_pareto_optimal_parameters(
            experiment=experiment,
            generation_strategy=generation_strategy,
            optimization_config=optimization_config,
            trial_indices=trial_indices,
            use_model_predictions=use_model_predictions,
        )

    @staticmethod
    def _get_hypervolume(
        experiment: Experiment,
        generation_strategy: GenerationStrategy,
        optimization_config: Optional[MultiObjectiveOptimizationConfig] = None,
        trial_indices: Optional[Iterable[int]] = None,
        use_model_predictions: bool = True,
    ) -> float:
        moo_optimization_config = checked_cast(
            MultiObjectiveOptimizationConfig,
            optimization_config or experiment.optimization_config,
        )

        if use_model_predictions:
            current_model = generation_strategy._curr.model
            # Cover for the case where source of `self._curr.model` was not a `Models`
            # enum but a factory function, in which case we cannot do
            # `get_model_from_generator_run` (since we don't have model type and inputs
            # recorded on the generator run.
            models_enum = (
                current_model.__class__
                if isinstance(current_model, ModelRegistryBase)
                else None
            )

            if models_enum is None:
                raise ValueError(
                    f"Model {current_model} is not in the ModelRegistry, cannot "
                    "calculate predicted hypervolume."
                )

            model = get_model_from_generator_run(
                generator_run=not_none(generation_strategy.last_generator_run),
                experiment=experiment,
                data=experiment.fetch_data(trial_indices=trial_indices),
                models_enum=models_enum,
            )
            if not isinstance(model, ArrayModelBridge):
                raise ValueError(
                    f"Model {current_model} is not of type ArrayModelBridge, cannot "
                    "calculate predicted hypervolume."
                )

            return predicted_hypervolume(
                modelbridge=model, optimization_config=optimization_config
            )

        minimal_model = get_tensor_converter_model(
            experiment=experiment,
            data=experiment.lookup_data(trial_indices=trial_indices),
        )

        return observed_hypervolume(
            modelbridge=minimal_model, optimization_config=moo_optimization_config
        )

    @staticmethod
    def get_trace(
        experiment: Experiment,
        optimization_config: Optional[OptimizationConfig] = None,
    ) -> List[float]:
        """Get the optimization trace of the given experiment.

        The output is equivalent to calling `_get_hypervolume` or `_get_best_trial`
        repeatedly, with an increasing sequence of `trial_indices` and with
        `use_model_predictions = False`, though this does it more efficiently.

        Args:
            experiment: The experiment to get the trace for.
            optimization_config: An optional optimization config to use for computing
                the trace. This allows computing the traces under different objectives
                or constraints without having to modify the experiment.

        Returns:
            A list of observed hypervolume or the best value.
        """
        # Use a minimal model to help parse the data.
        modelbridge = get_tensor_converter_model(
            experiment=experiment,
            data=experiment.fetch_data(),
        )
        obs_feats, obs_data, _ = _get_modelbridge_training_data(modelbridge=modelbridge)
        array_to_tensor = partial(_array_to_tensor, modelbridge=modelbridge)
        Y, _ = observation_data_to_array(
            outcomes=modelbridge.outcomes, observation_data=obs_data
        )
        Y = array_to_tensor(Y)

        tf = Derelativize(
            search_space=modelbridge.model_space.clone(),
            observation_data=obs_data,
            observation_features=obs_feats,
            config={"use_raw_status_quo": True},
        )
        optimization_config = optimization_config or not_none(
            experiment.optimization_config
        )
        optimization_config = tf.transform_optimization_config(
            optimization_config=optimization_config.clone(),
            modelbridge=modelbridge,
            fixed_features=ObservationFeatures({}),
        )

        # Extract weights, constraints, and objective_thresholds.
        objective_weights = extract_objective_weights(
            objective=optimization_config.objective, outcomes=modelbridge.outcomes
        )
        outcome_constraints = extract_outcome_constraints(
            outcome_constraints=optimization_config.outcome_constraints,
            outcomes=modelbridge.outcomes,
        )
        if optimization_config.is_moo_problem:
            objective_thresholds = extract_objective_thresholds(
                objective_thresholds=checked_cast(
                    MultiObjectiveOptimizationConfig, optimization_config
                ).objective_thresholds,
                objective=optimization_config.objective,
                outcomes=modelbridge.outcomes,
            )
            objective_thresholds = array_to_tensor(objective_thresholds)
        else:
            objective_thresholds = None
        (
            objective_weights,
            outcome_constraints,
            _,
            _,
            _,
        ) = validate_and_apply_final_transform(
            objective_weights=objective_weights,
            outcome_constraints=outcome_constraints,
            linear_constraints=None,
            pending_observations=None,
            final_transform=array_to_tensor,
        )
        # Get weighted tensor objectives.
        if optimization_config.is_moo_problem:
            (
                obj,
                weighted_objective_thresholds,
            ) = get_weighted_mc_objective_and_objective_thresholds(
                objective_weights=objective_weights,
                objective_thresholds=not_none(objective_thresholds),
            )
            Y_obj = obj(Y)
            infeas_value = weighted_objective_thresholds
        else:
            Y_obj = Y @ objective_weights
            infeas_value = Y_obj.min()
        # Account for feasibility.
        if outcome_constraints is not None:
            cons_tfs = not_none(get_outcome_constraint_transforms(outcome_constraints))
            feas = torch.all(torch.stack([c(Y) <= 0 for c in cons_tfs], dim=-1), dim=-1)
            # Set the infeasible points to reference point or the worst observed value.
            Y_obj[~feas] = infeas_value
        if optimization_config.is_moo_problem:
            # Compute the hypervolume trace.
            partitioning = DominatedPartitioning(
                ref_point=weighted_objective_thresholds.double()
            )
            # compute hv at each iteration
            hvs = []
            for Yi in Y_obj.split(1):
                # update with new point
                partitioning.update(Y=Yi)
                hv = partitioning.compute_hypervolume().item()
                hvs.append(hv)
            return hvs
        else:
            # Find the best observed value.
            raw_maximum = np.maximum.accumulate(Y_obj.cpu().numpy())
            if optimization_config.objective.minimize:
                # Negate the result if it is a minimization problem.
                raw_maximum = -raw_maximum
            return raw_maximum.tolist()
