#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
from typing import Dict, List, NamedTuple, Optional, Union

from ax.core.experiment import Experiment
from ax.core.search_space import SearchSpace
from ax.core.simple_experiment import SimpleExperiment, TEvaluationFunction
from ax.service.utils.dispatch import choose_generation_strategy
from ax.service.utils.instantiation import (
    TParameterRepresentation,
    constraint_from_str,
    outcome_constraint_from_str,
    parameter_from_json,
)
from ax.utils.common.logger import get_logger


logger: logging.Logger = get_logger(__name__)


class ScheduleConfig(NamedTuple):
    """[Work in progress] Scheduler configuration for managed loop."""

    wait_time: int = 0  # How many seconds to sleep for after deplying a trial.
    # Whether to maintain running process between iterations.
    run_async: bool = False


class OptimizationStep(NamedTuple):
    """[Work in progress] Configuration of one optimization step for managed
    loop."""

    num_trials: int = 1
    arms_per_trial: int = 1


class OptimizationPlan:
    """[Work in progress] Configuration of full optimization plan for managed
    loop."""

    total_iterations: int  # QUESTION: should this measure total arms or total trials?
    optimization_steps: List[OptimizationStep]

    def __init__(
        self,
        total_iterations: Optional[int] = None,
        optimization_steps: Optional[List[OptimizationStep]] = None,
    ) -> None:
        if (
            total_iterations
            and optimization_steps
            and sum(s.num_trials * s.arms_per_trial for s in optimization_steps)
            != total_iterations
        ):
            raise ValueError(
                "If both optimization steps and total iterations settings are "
                "provided, sum of number of trials in optimization steps must "
                "equal total iterations."
            )

        self.total_iterations = total_iterations or 20
        self.optimization_steps = (
            optimization_steps or [OptimizationStep()] * self.total_iterations
        )


class OptimizationLoop:
    """[Work in progress] Managed optimization loop, in which Ax oversees
    deployment of trials and gathering data."""

    def __init__(
        self,
        experiment: Experiment,
        optimization_plan: Optional[OptimizationPlan] = None,
        schedule_config: Optional[ScheduleConfig] = None,
    ) -> None:
        self.schedule_config = schedule_config or ScheduleConfig()
        assert (  # TODO[drfreund]: implement run_async
            not self.schedule_config.run_async
        ), "OptimizationLoop does not yet support async."
        self.schedule_config = schedule_config
        self.optimization_plan = optimization_plan or OptimizationPlan()
        self.experiment = experiment
        assert len(experiment.trials) == 0, (
            "Optimization Loop should not be initialized with an experiment "
            "that has trials already."
        )
        self.generation_strategy = choose_generation_strategy(
            search_space=experiment.search_space
        )
        self.current_step = 0

    @staticmethod
    def with_evaluation_function(
        parameters: List[TParameterRepresentation],
        evaluation_function: TEvaluationFunction,
        experiment_name: Optional[str] = None,
        objective_name: Optional[str] = None,
        minimize: bool = False,
        parameter_constraints: Optional[List[str]] = None,
        outcome_constraints: Optional[List[str]] = None,
        optimization_plan: Optional[OptimizationPlan] = None,
        schedule_config: Optional[ScheduleConfig] = None,
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
            optimization_plan=optimization_plan,
            schedule_config=schedule_config,
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
        optimization_plan: Optional[OptimizationPlan] = None,
        schedule_config: Optional[ScheduleConfig] = None,
    ) -> "OptimizationLoop":
        """Constructs an asynchronous `OptimizationLoop` using Ax runners and
        metrics."""
        # TODO[drfreund], T42401002
        raise NotImplementedError  # pragma: no cover

    def run_step(self) -> None:
        """Run a single step of the optimization plan."""
        step = self.optimization_plan.optimization_steps[self.current_step]
        logger.info(f"Running optimization step {self.current_step + 1}...")
        for _ in range(step.num_trials):
            if step.arms_per_trial == 1:
                trial = self.experiment.new_trial(
                    generator_run=self.generation_strategy.gen(
                        experiment=self.experiment,
                        new_data=self.experiment.fetch_data(),
                    )
                )
            elif step.arms_per_trial > 1:  # TODO[drfreund] T41922457
                raise NotImplementedError(
                    "OptimizationLoop does not yet support batches."
                )
            else:
                # TODO[drfreund]: handle -1?  T41922457
                raise ValueError(
                    f"Invalid number of arms per trial: {step.arms_per_trial}"
                )
            trial.fetch_data()
            self.current_step += 1

    def run(self, iterations: Optional[int]) -> None:
        """Runs a given number of iterations of this optimization loop."""
        # QUESTION: do we need this functionality? Running a given number of iterations?
        # If so, should iterations correspond to arms or to optimization steps?
        pass

    def full_run(self) -> "OptimizationLoop":
        """Runs full optimization loop as defined in the provided optimization
        plan."""
        num_steps = len(self.optimization_plan.optimization_steps)
        logger.info(f"Started full optimization with {num_steps} steps.")
        for _ in range(num_steps):
            self.run_step()
        return self

    def get_best_point(self) -> Dict[str, Union[str, float, bool, int]]:
        """[Work in progress] Obtains the best point encountered in the course
        of this optimization."""
        # TODO[drfreund]: Use models' best_point function. T42389552
        dat = self.experiment.fetch_data()
        objective_rows = dat.df.loc[
            dat.df["metric_name"]
            == self.experiment.optimization_config.objective.metric.name
        ]
        best_arm = (
            objective_rows.loc[objective_rows["mean"].idxmin()]
            if self.experiment.optimization_config.objective.minimize
            else objective_rows.loc[objective_rows["mean"].idxmax()]
        )["arm_name"]
        return self.experiment.arms_by_name.get(best_arm).parameters


def optimize(
    parameters: List[TParameterRepresentation],
    evaluation_function: TEvaluationFunction,
    experiment_name: Optional[str] = None,
    objective_name: Optional[str] = None,
    minimize: bool = False,
    parameter_constraints: Optional[List[str]] = None,
    outcome_constraints: Optional[List[str]] = None,
    optimization_plan: Optional[OptimizationPlan] = None,
    schedule_config: Optional[ScheduleConfig] = None,
) -> Dict[str, Union[str, float, bool, int]]:
    """Construct and run a full optimization loop."""
    loop = OptimizationLoop.with_evaluation_function(
        parameters=parameters,
        objective_name=objective_name,
        evaluation_function=evaluation_function,
        experiment_name=experiment_name,
        minimize=minimize,
        parameter_constraints=parameter_constraints,
        outcome_constraints=outcome_constraints,
        optimization_plan=optimization_plan,
        schedule_config=schedule_config,
    )
    loop.full_run()
    return loop.get_best_point()
