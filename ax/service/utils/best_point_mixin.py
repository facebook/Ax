#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from functools import partial
from logging import Logger

import numpy as np
import torch
from ax.core.experiment import Experiment
from ax.core.map_data import MapData
from ax.core.objective import ScalarizedObjective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.trial import Trial
from ax.core.types import TModelPredictArm, TParameterization
from ax.exceptions.core import UserInputError
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.modelbridge_utils import (
    extract_objective_thresholds,
    extract_objective_weights,
    extract_outcome_constraints,
    observed_hypervolume,
    predicted_hypervolume,
    validate_and_apply_final_transform,
)
from ax.modelbridge.registry import ModelRegistryBase
from ax.modelbridge.torch import TorchModelBridge
from ax.modelbridge.transforms.derelativize import Derelativize
from ax.models.torch.botorch_moo_defaults import (
    get_outcome_constraint_transforms,
    get_weighted_mc_objective_and_objective_thresholds,
)
from ax.plot.pareto_utils import get_tensor_converter_model
from ax.service.utils import best_point as best_point_utils
from ax.service.utils.best_point import (
    extract_Y_from_data,
    fill_missing_thresholds_from_nadir,
)
from ax.service.utils.best_point_utils import select_baseline_name_default_first_trial
from ax.utils.common.logger import get_logger
from botorch.utils.multi_objective.box_decompositions import DominatedPartitioning
from pyre_extensions import assert_is_instance, none_throws


logger: Logger = get_logger(__name__)

NUM_BINS_PER_TRIAL = 3


class BestPointMixin(metaclass=ABCMeta):
    @abstractmethod
    def get_best_trial(
        self,
        optimization_config: OptimizationConfig | None = None,
        trial_indices: Iterable[int] | None = None,
        use_model_predictions: bool = True,
    ) -> tuple[int, TParameterization, TModelPredictArm | None] | None:
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
        optimization_config: OptimizationConfig | None = None,
        trial_indices: Iterable[int] | None = None,
        use_model_predictions: bool = True,
    ) -> tuple[TParameterization, TModelPredictArm | None] | None:
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
            return res

        _, parameterization, vals = res
        return parameterization, vals

    @abstractmethod
    def get_pareto_optimal_parameters(
        self,
        optimization_config: OptimizationConfig | None = None,
        trial_indices: Iterable[int] | None = None,
        use_model_predictions: bool = True,
    ) -> dict[int, tuple[TParameterization, TModelPredictArm]]:
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
            A mapping from trial index to the tuple of:
            - the parameterization of the arm in that trial,
            - two-item tuple of metric means dictionary and covariance matrix
                (model-predicted if ``use_model_predictions=True`` and observed
                otherwise).
            Raises a `NotImplementedError` if extracting the Pareto frontier is
            not possible. Note that the returned dict may be empty.
        """
        pass

    @abstractmethod
    def get_hypervolume(
        self,
        optimization_config: MultiObjectiveOptimizationConfig | None = None,
        trial_indices: Iterable[int] | None = None,
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
        optimization_config: OptimizationConfig | None = None,
    ) -> list[float]:
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
        optimization_config: OptimizationConfig | None = None,
        bins: list[float] | None = None,
        final_progression_only: bool = False,
    ) -> tuple[list[float], list[float]]:
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
        optimization_config: OptimizationConfig | None = None,
        trial_indices: Iterable[int] | None = None,
        use_model_predictions: bool = True,
    ) -> tuple[int, TParameterization, TModelPredictArm | None] | None:
        optimization_config = optimization_config or none_throws(
            experiment.optimization_config
        )
        if optimization_config.is_moo_problem:
            raise NotImplementedError(
                "Please use `get_pareto_optimal_parameters` for multi-objective "
                "problems."
            )
        # TODO[drfreund]: Find a way to include data for last trial in the
        # calculation of best parameters.
        if use_model_predictions:
            current_model = generation_strategy._curr.model_spec_to_gen_from.model_enum
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
                    return res

        return best_point_utils.get_best_by_raw_objective_with_trial_index(
            experiment=experiment,
            optimization_config=optimization_config,
            trial_indices=trial_indices,
        )

    @staticmethod
    def _get_best_observed_value(
        experiment: Experiment,
        optimization_config: OptimizationConfig | None = None,
        trial_indices: Iterable[int] | None = None,
    ) -> float | None:
        """Identifies the best objective value observed in the experiment
        among the trials indicated by `trial_indices`.

        Args:
            experiment: The experiment to get the best objective value for.
            optimization_config: Optimization config to use in place of the one stored
                on the experiment.
            trial_indices: Indices of trials for which to retrieve data. If None will
                retrieve data from all available trials.

        Returns:
            The best objective value so far.
        """
        if optimization_config is None:
            optimization_config = none_throws(experiment.optimization_config)
        if optimization_config.is_moo_problem:
            raise NotImplementedError(
                "Please use `get_hypervolume` for multi-objective problems."
            )

        res = best_point_utils.get_best_by_raw_objective_with_trial_index(
            experiment=experiment,
            optimization_config=optimization_config,
            trial_indices=trial_indices,
        )

        predictions = res[2] if res is not None else None
        if predictions is None:
            return None

        means = none_throws(predictions)[0]
        objective = optimization_config.objective
        if isinstance(objective, ScalarizedObjective):
            value = 0
            for metric, weight in objective.metric_weights:
                value += means[metric.name] * weight
            return value
        else:
            name = objective.metric_names[0]
            return means[name]

    @staticmethod
    def _get_pareto_optimal_parameters(
        experiment: Experiment,
        generation_strategy: GenerationStrategy,
        optimization_config: OptimizationConfig | None = None,
        trial_indices: Iterable[int] | None = None,
        use_model_predictions: bool = True,
    ) -> dict[int, tuple[TParameterization, TModelPredictArm]]:
        optimization_config = optimization_config or none_throws(
            experiment.optimization_config
        )
        if not optimization_config.is_moo_problem:
            raise NotImplementedError(
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
        optimization_config: MultiObjectiveOptimizationConfig | None = None,
        trial_indices: Iterable[int] | None = None,
        use_model_predictions: bool = True,
    ) -> float:
        data = experiment.lookup_data()
        if len(data.df) == 0:
            return 0.0
        moo_optimization_config = assert_is_instance(
            optimization_config or experiment.optimization_config,
            MultiObjectiveOptimizationConfig,
        )

        if use_model_predictions:
            # Make sure that the model is fitted. If model is fitted already,
            # this should be a no-op.
            generation_strategy._fit_current_model(data=None)
            model = generation_strategy.model
            if not isinstance(model, TorchModelBridge):
                raise ValueError(
                    f"Model {model} is not of type TorchModelBridge, cannot "
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
        optimization_config: OptimizationConfig | None = None,
    ) -> list[float]:
        """Compute the optimization trace at each iteration.

        Given an experiment and an optimization config, compute the performance
        at each iteration. For multi-objective, the performance is computed as
        the hypervolume. For single objective, the performance is computed as
        the best observed objective value.

        An iteration here refers to a completed or early-stopped (batch) trial.
        There will be one performance metric in the trace for each iteration.

        Args:
            experiment: The experiment to get the trace for.
            optimization_config: Optimization config to use in place of the one
                stored on the experiment.

        Returns:
            A list of performance values at each iteration.
        """
        optimization_config = optimization_config or none_throws(
            experiment.optimization_config
        )
        # Get the names of the metrics in optimization config.
        metric_names = set(optimization_config.objective.metric_names)
        for cons in optimization_config.outcome_constraints:
            metric_names.update({cons.metric.name})
        metric_names = list(metric_names)
        # Convert data into a tensor.
        Y, trial_indices = extract_Y_from_data(
            experiment=experiment, metric_names=metric_names
        )
        if Y.numel() == 0:
            return []

        # Derelativize the optimization config.
        tf = Derelativize(
            search_space=None, observations=None, config={"use_raw_status_quo": True}
        )
        optimization_config = tf.transform_optimization_config(
            optimization_config=optimization_config.clone(),
            modelbridge=get_tensor_converter_model(
                experiment=experiment, data=experiment.lookup_data()
            ),
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
            objective_thresholds = extract_objective_thresholds(
                objective_thresholds=fill_missing_thresholds_from_nadir(
                    experiment=experiment, optimization_config=optimization_config
                ),
                objective=optimization_config.objective,
                outcomes=metric_names,
            )
            objective_thresholds = to_tensor(none_throws(objective_thresholds))
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
                objective_thresholds=none_throws(objective_thresholds),
            )
            Y_obj = obj(Y)
            infeas_value = weighted_objective_thresholds
        else:
            Y_obj = Y @ objective_weights
            infeas_value = Y_obj.min()
        # Account for feasibility.
        if outcome_constraints is not None:
            cons_tfs = none_throws(
                get_outcome_constraint_transforms(outcome_constraints)
            )
            feas = torch.all(torch.stack([c(Y) <= 0 for c in cons_tfs], dim=-1), dim=-1)
            # Set the infeasible points to reference point or the worst observed value.
            Y_obj[~feas] = infeas_value
        # Get unique trial indices. Note: only completed/early-stopped
        # trials are present.
        unique_trial_indices = trial_indices.unique().sort().values.tolist()
        # compute the performance at each iteration (completed/early-stopped
        # trial).
        # For `BatchTrial`s, there is one performance value per iteration, even
        # if the iteration (`BatchTrial`) has multiple arms.
        if optimization_config.is_moo_problem:
            # Compute the hypervolume trace.
            partitioning = DominatedPartitioning(
                ref_point=weighted_objective_thresholds.double()
            )
            # compute hv for each iteration (trial_index)
            hvs = []
            for trial_index in unique_trial_indices:
                new_Y = Y_obj[trial_indices == trial_index]
                # update with new point
                partitioning.update(Y=new_Y)
                hv = partitioning.compute_hypervolume().item()
                hvs.append(hv)
            return hvs
        running_max = float("-inf")
        raw_maximum = np.zeros(len(unique_trial_indices))
        # Find the best observed value for each iterations.
        # Enumerate the unique trial indices because only indices
        # of completed/early-stopped trials are present.
        for i, trial_index in enumerate(unique_trial_indices):
            new_Y = Y_obj[trial_indices == trial_index]
            running_max = max(running_max, new_Y.max().item())
            raw_maximum[i] = running_max
        if optimization_config.objective.minimize:
            # Negate the result if it is a minimization problem.
            raw_maximum = -raw_maximum
        return raw_maximum.tolist()

    @staticmethod
    def _get_trace_by_progression(
        experiment: Experiment,
        optimization_config: OptimizationConfig | None = None,
        bins: list[float] | None = None,
        final_progression_only: bool = False,
    ) -> tuple[list[float], list[float]]:
        optimization_config = optimization_config or none_throws(
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

        # pyre-fixme[9]: bins has type `Optional[List[float]]`; used as
        #  `ndarray[typing.Any, dtype[typing.Any]]`.
        bins = np.expand_dims(bins, axis=0)

        # compute for each bin value the largest trial index finished by then
        # (interpreting the bin value as a cumulative progression)
        best_observed_idcs = np.maximum.accumulate(
            np.argmax(np.expand_dims(progressions, axis=1) >= bins, axis=0)
        )
        obj_vals = (df["mean"].cummin() if minimize else df["mean"].cummax()).to_numpy()
        best_observed = obj_vals[best_observed_idcs]
        # pyre-fixme[16]: Item `List` of `Union[List[float], ndarray[typing.Any,
        #  np.dtype[typing.Any]]]` has no attribute `squeeze`.
        return best_observed.tolist(), bins.squeeze(axis=0).tolist()

    def get_improvement_over_baseline(
        self,
        experiment: Experiment,
        generation_strategy: GenerationStrategy,
        baseline_arm_name: str | None = None,
    ) -> float:
        """Returns the scalarized improvement over baseline, if applicable.

        Returns:
            For Single Objective cases, returns % improvement of objective.
            Positive indicates improvement over baseline. Negative indicates regression.
            For Multi Objective cases, throws NotImplementedError
        """
        if experiment.is_moo_problem:
            raise NotImplementedError(
                "`get_improvement_over_baseline` not yet implemented"
                + " for multi-objective problems."
            )
        if not baseline_arm_name:
            baseline_arm_name, _ = select_baseline_name_default_first_trial(
                experiment=experiment,
                baseline_arm_name=baseline_arm_name,
            )

        optimization_config = experiment.optimization_config
        if not optimization_config:
            raise ValueError("No optimization config found.")

        objective_metric_name = optimization_config.objective.metric.name

        # get the baseline trial
        data = experiment.lookup_data().df
        data = data[data["arm_name"] == baseline_arm_name]
        if len(data) == 0:
            raise UserInputError(
                "`get_improvement_over_baseline`"
                " could not find baseline arm"
                f" `{baseline_arm_name}` in the experiment data."
            )
        data = data[data["metric_name"] == objective_metric_name]
        baseline_value = data.iloc[0]["mean"]

        # Find objective value of the best trial
        idx, param, best_arm = none_throws(
            self._get_best_trial(
                experiment=experiment,
                generation_strategy=generation_strategy,
                optimization_config=optimization_config,
                use_model_predictions=False,
            )
        )
        best_arm = none_throws(best_arm)
        best_obj_value = best_arm[0][objective_metric_name]

        def percent_change(x: float, y: float, minimize: bool) -> float:
            if x == 0:
                raise ZeroDivisionError(
                    "Cannot compute percent improvement when denom is zero"
                )
            percent_change = (y - x) / abs(x) * 100
            if minimize:
                percent_change = -percent_change
            return percent_change

        return percent_change(
            x=baseline_value,
            y=best_obj_value,
            minimize=optimization_config.objective.minimize,
        )

    @staticmethod
    def _to_best_point_tuple(
        experiment: Experiment,
        trial_index: int,
        parameterization: TParameterization,
        model_prediction: TModelPredictArm | None,
    ) -> tuple[TParameterization, dict[str, float | tuple[float, float]], int, str]:
        """
        Return the tuple expected by the return signature of get_best_parameterization
        and get_pareto_frontier in the Ax API.

        TODO: Remove this helper when we clean up BestPointMixin.

        Returns:
            - The parameters predicted to have the best optimization value without
                violating any outcome constraints.
            - The metric values for the best parameterization. Uses model prediction if
                use_model_predictions=True, otherwise returns observed data.
            - The trial which most recently ran the best parameterization
            - The name of the best arm (each trial has a unique name associated with
                each parameterization)
        """

        if model_prediction is not None:
            mean, covariance = model_prediction

            prediction: dict[str, float | tuple[float, float]] = {
                metric_name: (
                    mean[metric_name],
                    none_throws(covariance)[metric_name][metric_name],
                )
                for metric_name in mean.keys()
            }
        else:
            data_dict = experiment.lookup_data(trial_indices=[trial_index]).df.to_dict()

            prediction: dict[str, float | tuple[float, float]] = {
                data_dict["metric_name"][i]: (data_dict["mean"][i], data_dict["sem"][i])
                for i in range(len(data_dict["metric_name"]))
            }

        trial = assert_is_instance(experiment.trials[trial_index], Trial)
        arm = none_throws(trial.arm)

        return parameterization, prediction, trial_index, arm.name
