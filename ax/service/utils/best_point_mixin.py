#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod, ABCMeta
from typing import Dict, Tuple, Optional

from ax.core.experiment import Experiment
from ax.core.optimization_config import OptimizationConfig
from ax.core.types import TModelPredictArm, TParameterization
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.registry import ModelRegistryBase
from ax.service.utils import best_point as best_point_utils
from ax.utils.common.typeutils import not_none


class BestPointMixin(metaclass=ABCMeta):
    @abstractmethod
    def get_best_trial(
        self,
        optimization_config: Optional[OptimizationConfig] = None,
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
            use_model_predictions: Whether to extract the best point using
                model predictions or directly observed values. If ``True``,
                the metric means and covariances in this method's output will
                also be based on model predictions and may differ from the
                observed values.

        Returns:
            Tuple of trial index, parameterization and model predictions for it.
        """
        pass

    @abstractmethod
    def get_best_parameters(
        self,
        optimization_config: Optional[OptimizationConfig] = None,
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
            use_model_predictions: Whether to extract the best point using
                model predictions or directly observed values. If ``True``,
                the metric means and covariances in this method's output will
                also be based on model predictions and may differ from the
                observed values.

        Returns:
            Tuple of parameterization and model predictions for it.
        """
        pass

    @abstractmethod
    def get_pareto_optimal_parameters(
        self,
        optimization_config: Optional[OptimizationConfig] = None,
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

    @staticmethod
    def _get_best_trial(
        experiment: Experiment,
        generation_strategy: GenerationStrategy,
        optimization_config: Optional[OptimizationConfig] = None,
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
                )

                if res is not None:
                    return res  # pragma: no cover

        return best_point_utils.get_best_by_raw_objective_with_trial_index(
            experiment=experiment,
            optimization_config=optimization_config,
        )

    @staticmethod
    def _get_best_parameters(
        experiment: Experiment,
        generation_strategy: GenerationStrategy,
        optimization_config: Optional[OptimizationConfig] = None,
        use_model_predictions: bool = True,
    ) -> Optional[Tuple[TParameterization, Optional[TModelPredictArm]]]:
        if not_none(experiment.optimization_config).is_moo_problem:
            raise NotImplementedError(  # pragma: no cover
                "Please use `get_pareto_optimal_parameters` for multi-objective "
                "problems."
            )

        res = BestPointMixin._get_best_trial(
            experiment=experiment,
            generation_strategy=generation_strategy,
            optimization_config=optimization_config,
            use_model_predictions=use_model_predictions,
        )

        if res is None:
            return res  # pragma: no cover

        _, parameterization, vals = res
        return parameterization, vals

    @staticmethod
    def _get_pareto_optimal_parameters(
        experiment: Experiment,
        generation_strategy: GenerationStrategy,
        optimization_config: Optional[OptimizationConfig] = None,
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
            use_model_predictions=use_model_predictions,
        )
