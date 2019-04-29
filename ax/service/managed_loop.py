#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
from typing import List, Optional

from ax.core.experiment import Experiment
from ax.core.search_space import SearchSpace
from ax.core.simple_experiment import SimpleExperiment, TEvaluationFunction
from ax.core.types import TParameterization
from ax.service.utils.best_point import (
    get_best_from_model_predictions,
    get_best_raw_objective_point,
)
from ax.service.utils.dispatch import choose_generation_strategy
from ax.service.utils.instantiation import (
    TParameterRepresentation,
    constraint_from_str,
    outcome_constraint_from_str,
    parameter_from_json,
)
from ax.utils.common.logger import get_logger


logger: logging.Logger = get_logger(__name__)


class OptimizationLoop:
    """Managed optimization loop, in which Ax oversees deployment of trials and
    gathering data."""

    def __init__(
        self,
        experiment: Experiment,
        total_trials: int = 20,
        arms_per_trial: int = 1,
        wait_time: int = 0,
        run_async: bool = False,  # TODO[Lena],
    ) -> None:
        assert not run_async, "OptimizationLoop does not yet support async."
        self.wait_time = wait_time
        self.total_trials = total_trials
        self.arms_per_trial = arms_per_trial
        assert len(experiment.trials) == 0, (
            "Optimization Loop should not be initialized with an experiment "
            "that has trials already."
        )
        self.experiment = experiment
        self.generation_strategy = choose_generation_strategy(
            search_space=experiment.search_space, arms_per_trial=self.arms_per_trial
        )
        self.current_trial = 0

    @staticmethod
    def with_evaluation_function(
        parameters: List[TParameterRepresentation],
        evaluation_function: TEvaluationFunction,
        experiment_name: Optional[str] = None,
        objective_name: Optional[str] = None,
        minimize: bool = False,
        parameter_constraints: Optional[List[str]] = None,
        outcome_constraints: Optional[List[str]] = None,
        total_trials: int = 20,
        arms_per_trial: int = 1,
        wait_time: int = 0,
    ) -> "OptimizationLoop":
        """Constructs a synchronous `OptimizationLoop` using an evaluation
        function."""
        exp_parameters = [parameter_from_json(p) for p in parameters]
        parameter_map = {p.name: p for p in exp_parameters}
        experiment = SimpleExperiment(
            name=experiment_name,
            search_space=SearchSpace(
                parameters=exp_parameters,
                parameter_constraints=None
                if parameter_constraints is None
                else [
                    constraint_from_str(c, parameter_map) for c in parameter_constraints
                ],
            ),
            objective_name=objective_name,
            evaluation_function=evaluation_function,
            minimize=minimize,
            outcome_constraints=[
                outcome_constraint_from_str(c) for c in (outcome_constraints or [])
            ],
        )
        return OptimizationLoop(
            experiment=experiment,
            total_trials=total_trials,
            arms_per_trial=arms_per_trial,
            wait_time=wait_time,
        )

    @classmethod
    def with_runners_and_metrics(
        cls,
        parameters: List[TParameterRepresentation],
        path_to_runner: str,
        paths_to_metrics: List[str],
        experiment_name: Optional[str] = None,
        objective_name: Optional[str] = None,
        minimize: bool = False,
        parameter_constraints: Optional[List[str]] = None,
        outcome_constraints: Optional[List[str]] = None,
        total_trials: int = 20,
        arms_per_trial: int = 1,
        wait_time: int = 0,
    ) -> "OptimizationLoop":
        """Constructs an asynchronous `OptimizationLoop` using Ax runners and
        metrics."""
        # TODO[drfreund], T42401002
        raise NotImplementedError  # pragma: no cover

    def run_trial(self) -> None:
        """Run a single step of the optimization plan."""
        if self.current_trial >= self.total_trials:
            raise ValueError(f"Optimization is complete, cannot run another trial.")
        logger.info(f"Running optimization trial {self.current_trial + 1}...")
        arms_per_trial = self.arms_per_trial
        dat = (
            self.experiment._fetch_trial_data(self.current_trial - 1)
            if self.current_trial > 0
            else None
        )
        if arms_per_trial == 1:
            trial = self.experiment.new_trial(
                generator_run=self.generation_strategy.gen(
                    experiment=self.experiment, new_data=dat
                )
            )
        elif arms_per_trial > 1:
            trial = self.experiment.new_batch_trial(
                generator_run=self.generation_strategy.gen(
                    experiment=self.experiment, new_data=dat, n=arms_per_trial
                )
            )
        else:
            raise ValueError(f"Invalid number of arms per trial: {arms_per_trial}")
        trial.fetch_data()
        self.current_trial += 1

    def full_run(self) -> "OptimizationLoop":
        """Runs full optimization loop as defined in the provided optimization
        plan."""
        num_steps = self.total_trials
        logger.info(f"Started full optimization with {num_steps} steps.")
        for _ in range(num_steps):
            self.run_trial()
        return self

    def get_best_point(self) -> TParameterization:
        """Obtains the best point encountered in the course
        of this optimization."""
        # Find latest trial which has a generator_run attached and get its predictions
        model_predictions = get_best_from_model_predictions(experiment=self.experiment)
        if model_predictions is not None:
            return model_predictions[0]

        # Could not find through model, default to using raw objective.
        parameterization, _ = get_best_raw_objective_point(experiment=self.experiment)
        return parameterization


def optimize(
    parameters: List[TParameterRepresentation],
    evaluation_function: TEvaluationFunction,
    experiment_name: Optional[str] = None,
    objective_name: Optional[str] = None,
    minimize: bool = False,
    parameter_constraints: Optional[List[str]] = None,
    outcome_constraints: Optional[List[str]] = None,
    total_trials: int = 20,
    arms_per_trial: int = 1,
    wait_time: int = 0,
) -> TParameterization:
    """Construct and run a full optimization loop."""
    loop = OptimizationLoop.with_evaluation_function(
        parameters=parameters,
        objective_name=objective_name,
        evaluation_function=evaluation_function,
        experiment_name=experiment_name,
        minimize=minimize,
        parameter_constraints=parameter_constraints,
        outcome_constraints=outcome_constraints,
        total_trials=total_trials,
        arms_per_trial=arms_per_trial,
        wait_time=wait_time,
    )
    loop.full_run()
    return loop.get_best_point()
