#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABCMeta, abstractmethod
from functools import partial
from logging import Logger
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
    OptimizationConfig,
)
from ax.core.types import ComparisonOp, TModelPredictArm, TParameterization
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.modelbridge_utils import (
    extract_objective_thresholds,
    extract_objective_weights,
    extract_outcome_constraints,
    observed_hypervolume,
    predicted_hypervolume,
    validate_and_apply_final_transform,
)
from ax.modelbridge.registry import get_model_from_generator_run, ModelRegistryBase
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.derelativize import Derelativize
from ax.models.torch.botorch_moo_defaults import (
    get_outcome_constraint_transforms,
    get_weighted_mc_objective_and_objective_thresholds,
)
from ax.plot.pareto_utils import get_tensor_converter_model
from ax.service.utils import best_point as best_point_utils
from ax.service.utils.best_point import extract_Y_from_data
from ax.utils.common.logger import get_logger
from ax.utils.common.typeutils import checked_cast, not_none
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning


logger: Logger = get_logger(__name__)

NUM_BINS_PER_TRIAL = 3


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

    @abstractmethod
    def get_trace(
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
            A list of observed hypervolumes or best values.
        """
        pass

    @abstractmethod
    def get_trace_by_progression(
        optimization_config: Optional[OptimizationConfig] = None,
        bins: Optional[List[float]] = None,
        final_progression_only: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """Get the optimization trace with respect to trial progressions instead of
        `trial_indices` (which is the behavior used in `get_trace`). Note that this
        method does not take into account the parallelism of trials and essentially
        assumes that trials are run one after another, in the sense that it considers
        the total number of progressions "used" at the end of trial k to be the
        cumulative progressions "used" in trials 0,...,k. This method assumes that the
        final value of a particular trial is used and does not take the best value
        of a trial over its progressions.

        The best observed value is computed at each value in `bins` (see below for
        details). If `bins` is not supplied, the method defaults to a heuristic of
        approximately `NUM_BINS_PER_TRIAL` per trial, where each trial is assumed to
        run until maximum progression (inferred from the data).

        Args:
            experiment: The experiment to get the trace for.
            optimization_config: An optional optimization config to use for computing
                the trace. This allows computing the traces under different objectives
                or constraints without having to modify the experiment.
            bins: A list progression values at which to calculate the best observed
                value. The best observed value at bins[i] is defined as the value
                observed in trials 0,...,j where j = largest trial such that the total
                progression in trials 0,...,j is less than bins[i].
            final_progression_only: If True, considers the value of the last step to be
                the value of the trial. If False, considers the best along the curve to
                be the value of the trial.

        Returns:
            A tuple containing (1) the list of observed hypervolumes or best values and
            (2) a list of associated x-values (i.e., progressions) useful for plotting.
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
    ) -> Dict[int, Tuple[TParameterization, TModelPredictArm]]:
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
        data = experiment.lookup_data()
        if len(data.df) == 0:
            return 0.0
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
            if not isinstance(model, TorchModelBridge):
                raise ValueError(
                    f"Model {current_model} is not of type TorchModelBridge, cannot "
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
    def _get_trace(
        experiment: Experiment,
        optimization_config: Optional[OptimizationConfig] = None,
    ) -> List[float]:
        optimization_config = optimization_config or not_none(
            experiment.optimization_config
        )
        # Get the names of the metrics in optimization config.
        metric_names = set(optimization_config.objective.metric_names)
        for cons in optimization_config.outcome_constraints:
            metric_names.update({cons.metric.name})
        metric_names = list(metric_names)
        # Convert data into a tensor.
        Y = extract_Y_from_data(experiment=experiment, metric_names=metric_names)
        if Y.numel() == 0:
            return []

        # Derelativize the optimization config.
        tf = Derelativize(
            search_space=None, observations=None, config={"use_raw_status_quo": True}
        )
        optimization_config = tf.transform_optimization_config(
            optimization_config=optimization_config.clone(),
            # pyre-ignore -- experiment works here since we only need the status quo.
            modelbridge=experiment,
            fixed_features=None,
        )

        # Extract weights, constraints, and objective_thresholds.
        objective_weights = extract_objective_weights(
            objective=optimization_config.objective, outcomes=metric_names
        )
        outcome_constraints = extract_outcome_constraints(
            outcome_constraints=optimization_config.outcome_constraints,
            outcomes=metric_names,
        )
        to_tensor = partial(
            torch.as_tensor, dtype=torch.double, device=torch.device("cpu")
        )
        if optimization_config.is_moo_problem:
            multiobjective_optimization_config = checked_cast(
                MultiObjectiveOptimizationConfig, optimization_config
            )

            provided_thresholds = (
                multiobjective_optimization_config.objective_thresholds
            )

            # For Objectives without thresholds provided, infer the threshold from
            # the nadir point.
            provided_threshold_names = [
                threshold.metric.name for threshold in provided_thresholds
            ]
            objectives_without_threshold = [
                objective
                for objective in checked_cast(
                    MultiObjective, optimization_config.objective
                ).objectives
                if objective.metric.name not in provided_threshold_names
            ]
            inferred_thresholds = [
                _objective_threshold_from_nadir(
                    experiment=experiment,
                    objective=objective,
                    optimization_config=multiobjective_optimization_config,
                )
                for objective in objectives_without_threshold
            ]

            objective_thresholds = extract_objective_thresholds(
                objective_thresholds=[*provided_thresholds, *inferred_thresholds],
                objective=optimization_config.objective,
                outcomes=metric_names,
            )
            objective_thresholds = to_tensor(not_none(objective_thresholds))
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
            final_transform=to_tensor,
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

    @staticmethod
    def _get_trace_by_progression(
        experiment: Experiment,
        optimization_config: Optional[OptimizationConfig] = None,
        bins: Optional[List[float]] = None,
        final_progression_only: bool = False,
    ) -> Tuple[List[float], List[float]]:
        optimization_config = optimization_config or not_none(
            experiment.optimization_config
        )
        objective = optimization_config.objective.metric.name
        minimize = optimization_config.objective.minimize
        map_data = experiment.lookup_data()
        if not isinstance(map_data, MapData):
            raise ValueError("`get_trace_by_progression` requires MapData.")
        map_df = map_data.map_df

        # assume the first map_key is progression
        map_key = map_data.map_keys[0]

        map_df = map_df[map_df["metric_name"] == objective]
        map_df = map_df.sort_values(by=["trial_index", map_key])
        df = (
            map_df.drop_duplicates(MapData.DEDUPLICATE_BY_COLUMNS, keep="last")
            if final_progression_only
            else map_df
        )

        # compute cumulative steps
        prev_steps_df = map_df.drop_duplicates(
            MapData.DEDUPLICATE_BY_COLUMNS, keep="last"
        )[["trial_index", map_key]].copy()

        # shift the cumsum by one so that we count cumulative steps not including
        # the current trial
        prev_steps_df[map_key] = (
            prev_steps_df[map_key].cumsum().shift(periods=1).fillna(0)
        )
        prev_steps_df = prev_steps_df.rename(columns={map_key: "prev_steps"})
        df = df.merge(prev_steps_df, on=["trial_index"])
        df["cumulative_steps"] = df[map_key] + df["prev_steps"]
        progressions = df["cumulative_steps"].to_numpy()

        if bins is None:
            # this assumes that there is at least one completed trial that
            # reached the maximum progression
            prog_per_trial = df[map_key].max()
            num_trials = len(experiment.trials)
            bins = np.linspace(
                0, prog_per_trial * num_trials, NUM_BINS_PER_TRIAL * num_trials
            )
        else:
            bins = np.array(bins)  # pyre-ignore[9]

        bins = np.expand_dims(bins, axis=0)

        # compute for each bin value the largest trial index finished by then
        # (interpreting the bin value as a cumulative progression)
        best_observed_idcs = np.maximum.accumulate(
            np.argmax(np.expand_dims(progressions, axis=1) >= bins, axis=0)
        )
        obj_vals = (df["mean"].cummin() if minimize else df["mean"].cummax()).to_numpy()
        best_observed = obj_vals[best_observed_idcs]
        return best_observed.tolist(), bins.squeeze(axis=0).tolist()


def _objective_threshold_from_nadir(
    experiment: Experiment,
    objective: Objective,
    optimization_config: Optional[MultiObjectiveOptimizationConfig] = None,
) -> ObjectiveThreshold:
    """
    Find the worst value observed for each objective and create an ObjectiveThreshold
    with this as the bound.
    """

    logger.info(f"Inferring ObjectiveThreshold for {objective} using nadir point.")

    optimization_config = optimization_config or checked_cast(
        MultiObjectiveOptimizationConfig, experiment.optimization_config
    )

    data_df = experiment.fetch_data().df

    mean = data_df[data_df["metric_name"] == objective.metric.name]["mean"]
    bound = max(mean) if objective.minimize else min(mean)
    op = ComparisonOp.LEQ if objective.minimize else ComparisonOp.GEQ

    return ObjectiveThreshold(
        metric=objective.metric, bound=bound, op=op, relative=False
    )
