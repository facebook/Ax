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
from ax.core.map_data import MapData
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.early_stopping.strategies.base import BaseEarlyStoppingStrategy
from ax.generation_strategy.generation_strategy import (
    GenerationStep,
    GenerationStrategy,
)
from ax.metrics.map_replay import MapDataReplayMetric
from ax.runners.map_replay import MapDataReplayRunner
from ax.service.orchestrator import Orchestrator, OrchestratorOptions
from ax.utils.common.logger import get_logger

logger: Logger = get_logger(__name__)


def replay_experiment(
    historical_experiment: Experiment,
    num_samples_per_curve: int,
    max_replay_trials: int,
    metric: Metric,
    max_pending_trials: int,
    early_stopping_strategy: BaseEarlyStoppingStrategy | None,
    logging_level: int = logging.ERROR,
) -> Experiment | None:
    """A utility function for replaying a historical experiment's MapData
    by initializing a Orchestrator that quickly steps through the existing data.
    The main purpose of this function is to compute an hypothetical capacity
    savings for a given `early_stopping_strategy`.
    """
    historical_map_data = historical_experiment.lookup_data()
    if not isinstance(historical_map_data, MapData):
        logger.warning("Replaying an experiment requires MapData.")
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
