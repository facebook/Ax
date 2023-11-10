# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.generator_run import GeneratorRun
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.types import TModelPredictArm, TParameterization
from ax.utils.common.base import Base
from ax.utils.common.typeutils import not_none


class GenerationStrategyInterface(ABC, Base):
    _name: Optional[str]
    # All generator runs created through this generation strategy, in chronological
    # order.
    _generator_runs: List[GeneratorRun]
    # Experiment, for which this generation strategy has generated trials, if
    # it exists.
    _experiment: Optional[Experiment] = None

    @abstractmethod
    def gen_multiple_with_ensembling(
        self,
        experiment: Experiment,
        num_generator_runs: int,
        data: Optional[Data] = None,
        n: int = 1,
        extra_gen_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[List[GeneratorRun]]:
        """Produce GeneratorRuns for multiple trials at once with the possibility of
        ensembling, or using multiple models per trial, getting multiple
        GeneratorRuns per trial.

        Args:
            experiment: Experiment, for which the generation strategy is producing
                a new generator run in the course of `gen`, and to which that
                generator run will be added as trial(s). Information stored on the
                experiment (e.g., trial statuses) is used to determine which model
                will be used to produce the generator run returned from this method.
            data: Optional data to be passed to the underlying model's `gen`, which
                is called within this method and actually produces the resulting
                generator run. By default, data is all data on the `experiment`.
            n: Integer representing how many trials should be in the generator run
                produced by this method. NOTE: Some underlying models may ignore
                the ``n`` and produce a model-determined number of arms. In that
                case this method will also output a generator run with number of
                arms that can differ from ``n``.
            pending_observations: A map from metric name to pending
                observations for that metric, used by some models to avoid
                resuggesting points that are currently being evaluated.
            extra_gen_metadata: A dictionary containing any additional metadata
                to be attached to created GeneratorRuns.

        Returns:
            A list of lists of lists generator runs. Each outer list represents
            a trial being suggested and  each inner list represents a generator
            run for that trial.
        """
        pass

    @property
    def name(self) -> str:
        """Name of this generation strategy. Defaults to a combination of model
        names provided in generation steps.
        """
        if self._name is not None:
            return not_none(self._name)

        self._name = f"GenerationStrategy {self.db_id}"
        return not_none(self._name)

    @name.setter
    def name(self, name: str) -> None:
        """Set generation strategy name."""
        self._name = name

    @property
    def experiment(self) -> Experiment:
        """Experiment, currently set on this generation strategy."""
        if self._experiment is None:
            raise ValueError("No experiment set on generation strategy.")
        return not_none(self._experiment)

    @experiment.setter
    def experiment(self, experiment: Experiment) -> None:
        """If there is an experiment set on this generation strategy as the
        experiment it has been generating generator runs for, check if the
        experiment passed in is the same as the one saved and log an information
        statement if its not. Set the new experiment on this generation strategy.
        """
        if self._experiment is None or experiment._name == self.experiment._name:
            self._experiment = experiment
        else:
            raise ValueError(
                "This generation strategy has been used for experiment "
                f"{self.experiment._name} so far; cannot reset experiment"
                f" to {experiment._name}. If this is a new optimization, "
                "a new generation strategy should be created instead."
            )

    @property
    def last_generator_run(self) -> Optional[GeneratorRun]:
        """Latest generator run produced by this generation strategy.
        Returns None if no generator runs have been produced yet.
        """
        # Used to restore current model when decoding a serialized GS.
        return self._generator_runs[-1] if self._generator_runs else None

    @abstractmethod
    def get_pareto_optimal_parameters(
        self,
        experiment: Experiment,
        optimization_config: Optional[OptimizationConfig] = None,
        trial_indices: Optional[Iterable[int]] = None,
        use_model_predictions: bool = True,
    ) -> Dict[int, Tuple[TParameterization, TModelPredictArm]]:
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
            experiment: Experiment, from which to find Pareto-optimal arms.
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
        """
        pass

    @abstractmethod
    def get_best_trial(
        self,
        experiment: Experiment,
        optimization_config: Optional[OptimizationConfig] = None,
        trial_indices: Optional[Iterable[int]] = None,
    ) -> Optional[Tuple[int, TParameterization, Optional[TModelPredictArm]]]:
        """Given an experiment, returns the best predicted parameterization and
        corresponding prediction based on the most recent Trial with predictions.
        If no trials have predictions returns None.

        Only some models return predictions. For instance GPEI does
        while Sobol does not.

        TModelPredictArm is of the form:
            ({metric_name: mean}, {metric_name_1: {metric_name_2: cov_1_2}})

        Args:
            experiment: Experiment, on which to identify best raw objective arm.
            optimization_config: Optimization config to use in place of the one stored
                on the experiment.
            trial_indices: Indices of trials for which to retrieve data. If None will
                retrieve data from all available trials.

        Returns:
            Tuple of trial index, parameterization, and model predictions for it."""
        pass

    @abstractmethod
    def get_hypervolume(
        self,
        experiment: Experiment,
        optimization_config: Optional[MultiObjectiveOptimizationConfig] = None,
        trial_indices: Optional[Iterable[int]] = None,
    ) -> float:
        """Calculate hypervolume of a pareto frontier based on the posterior means of
        given observation features.

        Given a model and features to evaluate calculate the hypervolume of the pareto
        frontier formed from their predicted outcomes.

        Args:
            modelbridge: Modelbridge used to predict metrics outcomes.
            optimization_config: Optimization config
            selected_metrics: If specified, hypervolume will only be evaluated on
                the specified subset of metrics. Otherwise, all metrics will be used.
            trial_indices: Indices of trials for which to retrieve data. If None will
                retrieve data from all available trials.

        Returns:
            calculated hypervolume.
        """
        pass

    @abstractproperty
    def num_baseline_trials(self) -> int:
        """Number of trials generated by the first step in this generation strategy."""
        pass
