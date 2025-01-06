# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass

from ax.core.experiment import Experiment
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.types import TParameterization
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy

from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.service.utils.best_point_mixin import BestPointMixin
from ax.utils.common.base import Base
from pyre_extensions import none_throws


@dataclass(kw_only=True)
class BenchmarkMethod(Base):
    """Benchmark method, represented in terms of Ax generation strategy (which tells us
    which models to use when) and scheduler options (which tell us extra execution
    information like maximum parallelism, early stopping configuration, etc.).

    Args:
        name: String description.
        generation_strategy: The `GenerationStrategy` to use.
        timeout_hours: Number of hours after which to stop a benchmark
            replication.
        distribute_replications: Indicates whether the replications should be
            run in a distributed manner. Ax itself does not use this attribute.
        use_model_predictions_for_best_point: Whether to use model
            predictions with ``get_pareto_optimal_parameters`` (if
            multi-objective) or `BestPointMixin._get_best_trial` (if
            single-objective). However, note that if multi-objective,
            best-point selection is not currently supported and
            ``get_pareto_optimal_parameters`` will raise a
            ``NotImplementedError``.
        batch_size: Number of arms per trial. If greater than 1, trials are
            ``BatchTrial``s; otherwise, they are ``Trial``s. Defaults to 1. This
            and the following arguments are passed to ``SchedulerOptions``.
        run_trials_in_batches: Passed to ``SchedulerOptions``.
        max_pending_trials: Passed to ``SchedulerOptions``.
    """

    name: str = "DEFAULT"
    generation_strategy: GenerationStrategy

    timeout_hours: float = 4.0
    distribute_replications: bool = False
    use_model_predictions_for_best_point: bool = False

    batch_size: int = 1
    run_trials_in_batches: bool = False
    max_pending_trials: int = 1
    early_stopping_strategy: BaseEarlyStoppingStrategy | None = None

    def __post_init__(self) -> None:
        if self.name == "DEFAULT":
            self.name = self.generation_strategy.name

    def get_best_parameters(
        self,
        experiment: Experiment,
        optimization_config: OptimizationConfig,
        n_points: int,
    ) -> list[TParameterization]:
        """
        Get ``n_points`` promising points. NOTE: Only SOO with n_points = 1 is
        supported.

        The expected use case is that these points will be evaluated against an
        oracle for hypervolume (if multi-objective) or for the value of the best
        parameter (if single-objective).

        For multi-objective cases, ``n_points > 1`` is needed. For SOO, ``n_points > 1``
        reflects setups where we can choose some points which will then be
        evaluated noiselessly or at high fidelity and then use the best one.


        Args:
            experiment: The experiment to get the data from. This should contain
                values that would be observed in a realistic setting and not
                contain oracle values.
            optimization_config: The ``optimization_config`` for the corresponding
                ``BenchmarkProblem``.
            n_points: The number of points to return.
        """
        if isinstance(optimization_config, MultiObjectiveOptimizationConfig):
            raise NotImplementedError(
                "BenchmarkMethod.get_pareto_optimal_parameters is not currently "
                "supported for multi-objective problems."
            )

        if n_points != 1:
            raise NotImplementedError(
                f"Currently only n_points=1 is supported. Got {n_points=}."
            )

        # SOO, n=1 case.
        # Note: This has the same effect as Scheduler.get_best_parameters
        result = BestPointMixin._get_best_trial(
            experiment=experiment,
            generation_strategy=self.generation_strategy,
            optimization_config=optimization_config,
            use_model_predictions=self.use_model_predictions_for_best_point,
        )
        if result is None:
            # This can happen if no points are predicted to satisfy all outcome
            # constraints.
            return []

        i, params, prediction = none_throws(result)
        return [params]
