# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass

from ax.core.experiment import Experiment
from ax.core.optimization_config import OptimizationConfig
from ax.core.types import TParameterization
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy

from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.utils.best_point_mixin import BestPointMixin
from ax.utils.common.base import Base
from pyre_extensions import none_throws


@dataclass(kw_only=True)
class BenchmarkMethod(Base):
    """Benchmark method, represented in terms of Ax generation strategy (which tells us
    which models to use when) and Orchestrator options (which tell us extra execution
    information like maximum parallelism, early stopping configuration, etc.).

    Args:
        name: String description.
        generation_strategy: The `GenerationStrategy` to use.
        timeout_hours: Number of hours after which to stop a benchmark
            replication.
        distribute_replications: Indicates whether the replications should be
            run in a distributed manner. Ax itself does not use this attribute.
        batch_size: Number of arms per trial. If greater than 1, trials are
            ``BatchTrial``s; otherwise, they are ``Trial``s. Defaults to 1. This
            and the following arguments are passed to ``OrchestratorOptions``.
        run_trials_in_batches: Passed to ``OrchestratorOptions``.
        max_pending_trials: Passed to ``OrchestratorOptions``.
    """

    name: str = "DEFAULT"
    generation_strategy: GenerationStrategy

    timeout_hours: float = 4.0
    distribute_replications: bool = False

    batch_size: int | None = 1
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
    ) -> TParameterization | None:
        """
        Get the most promising point.

        Only SOO is supported. It will return None if no best point can be found.

        Args:
            experiment: The experiment to get the data from. This should contain
                values that would be observed in a realistic setting and not
                contain oracle values.
            optimization_config: The ``optimization_config`` for the corresponding
                ``BenchmarkProblem``.
        """
        result = BestPointMixin._get_best_trial(
            experiment=experiment,
            generation_strategy=self.generation_strategy,
            optimization_config=optimization_config,
        )
        if result is None:
            # This can happen if no points are predicted to satisfy all outcome
            # constraints.
            return None
        _, params, _ = none_throws(result)
        return params
