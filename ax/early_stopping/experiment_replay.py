#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
from logging import Logger
from time import perf_counter

from ax.adapter.registry import Generators
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.early_stopping.dispatch import get_default_ess_or_none
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.early_stopping.utils import estimate_early_stopping_savings
from ax.exceptions.core import UnsupportedError
from ax.generation_strategy.generation_strategy import (
    GenerationStep,
    GenerationStrategy,
)
from ax.metrics.map_replay import MapDataReplayMetric
from ax.orchestration.orchestrator_options import OrchestratorOptions
from ax.runners.map_replay import MapDataReplayRunner
from ax.utils.common.logger import get_logger

logger: Logger = get_logger(__name__)

# Constants for experiment replay
MAX_REPLAY_TRIALS: int = 50
REPLAY_NUM_POINTS_PER_CURVE: int = 20
MAX_PENDING_TRIALS: int = 5
MIN_SAVINGS_THRESHOLD: float = 0.1  # 10% threshold


def replay_experiment(
    historical_experiment: Experiment,
    num_samples_per_curve: int,
    max_replay_trials: int,
    metric: Metric,
    max_pending_trials: int,
    early_stopping_strategy: BaseEarlyStoppingStrategy | None,
    logging_level: int = logging.ERROR,
) -> Experiment | None:
    """A utility function for replaying a historical experiment's data
    by initializing a Orchestrator that quickly steps through the existing data.
    The main purpose of this function is to compute an hypothetical capacity
    savings for a given `early_stopping_strategy`.
    """
    historical_map_data = historical_experiment.lookup_data()
    if not historical_map_data.has_step_column:
        logger.warning(
            "Replaying an experiment requires the data to have a 'step' column."
        )
        return None
    historical_map_data = historical_map_data.subsample(
        limit_rows_per_group=num_samples_per_curve, include_first_last=True
    )
    replay_metric = MapDataReplayMetric(
        name=f"replay_{historical_experiment.name}",
        map_data=historical_map_data,
        metric_name=metric.name,
        lower_is_better=metric.lower_is_better,
    )
    optimization_config = OptimizationConfig(
        objective=Objective(metric=replay_metric),
    )
    runner = MapDataReplayRunner(replay_metric=replay_metric)

    # Setup a new experiment with a dummy search space
    dummy_search_space = SearchSpace(
        parameters=[
            RangeParameter(
                name="dummy_param",
                lower=0.0,
                upper=1.0,
                parameter_type=ParameterType.FLOAT,
            )
        ]
    )
    experiment = Experiment(
        name=f"replay_{historical_experiment.name}",
        optimization_config=optimization_config,
        search_space=dummy_search_space,
        runner=runner,
    )

    # Setup a Orchestrator with a dummy gs to replay the historical experiment
    # Lazy import to avoid sqlalchemy dependency at module load time
    from ax.orchestration.orchestrator import Orchestrator

    dummy_sobol_gs = GenerationStrategy(
        name="sobol",
        steps=[
            GenerationStep(generator=Generators.SOBOL, num_trials=-1),
        ],
    )
    options = OrchestratorOptions(
        max_pending_trials=max_pending_trials,
        total_trials=min(len(historical_experiment.trials), max_replay_trials),
        seconds_between_polls_backoff_factor=1.0,
        min_seconds_before_poll=0.0,
        init_seconds_between_polls=0,
        early_stopping_strategy=early_stopping_strategy,
        logging_level=logging_level,
    )
    orchestrator = Orchestrator(
        experiment=experiment, generation_strategy=dummy_sobol_gs, options=options
    )
    start_time = perf_counter()
    orchestrator.run_all_trials()
    logger.info(f"Replayed the experiment in {perf_counter() - start_time} seconds.")
    return experiment


def estimate_hypothetical_early_stopping_savings(
    experiment: Experiment,
    metric: Metric,
    max_pending_trials: int = MAX_PENDING_TRIALS,
) -> float:
    """Estimate hypothetical early stopping savings using experiment replay.

    This function replays the experiment with a default early stopping strategy
    to calculate what savings would have been achieved if early stopping were
    enabled.

    Args:
        experiment: The experiment to analyze.
        metric: The metric to use for early stopping replay.
        max_pending_trials: Maximum number of pending trials for the replay
            orchestrator. Defaults to 5.

    Returns:
        Estimated savings as a fraction (0.0 to 1.0).

    Raises:
        UnsupportedError: If early stopping savings cannot be estimated.
            This can happen when:
            - No default early stopping strategy is available for this experiment
              (e.g., multi-objective, constrained, or non-MapMetric experiments)
            - The experiment data does not have progression data for replay
            - The experiment replay fails due to invalid experiment state
    """
    default_ess = get_default_ess_or_none(experiment=experiment)
    if default_ess is None:
        raise UnsupportedError(
            "No default early stopping strategy available (multi-objective, "
            "constrained, or non-MapMetric experiment)."
        )

    replayed_experiment = replay_experiment(
        historical_experiment=experiment,
        num_samples_per_curve=REPLAY_NUM_POINTS_PER_CURVE,
        max_replay_trials=MAX_REPLAY_TRIALS,
        metric=metric,
        max_pending_trials=max_pending_trials,
        early_stopping_strategy=default_ess,
    )

    if replayed_experiment is None:
        raise UnsupportedError(
            "Experiment data does not have progression data for replay."
        )

    return estimate_early_stopping_savings(experiment=replayed_experiment)
