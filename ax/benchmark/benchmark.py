# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Module for benchmarking Ax algorithms.

Key terms used:

* Replication: 1 run of an optimization loop; (BenchmarkProblem, BenchmarkMethod) pair.
* Test: multiple replications, ran for statistical significance.
* Full run: multiple tests on many (BenchmarkProblem, BenchmarkMethod) pairs.
* Method: (one of) the algorithm(s) being benchmarked.
* Problem: a synthetic function, a surrogate surface, or an ML model, on which
  to assess the performance of algorithms.

"""

from itertools import product
from logging import Logger
from time import time
from typing import Dict, Iterable, List

import numpy as np

from ax.benchmark.benchmark_method import BenchmarkMethod
from ax.benchmark.benchmark_problem import (
    BenchmarkProblemProtocol,
    BenchmarkProblemWithKnownOptimum,
)
from ax.benchmark.benchmark_result import AggregatedBenchmarkResult, BenchmarkResult
from ax.benchmark.metrics.base import BenchmarkMetricBase, GroundTruthMetricMixin
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    OptimizationConfig,
)
from ax.core.outcome_constraint import ObjectiveThreshold, OutcomeConstraint
from ax.core.utils import get_model_times
from ax.service.scheduler import Scheduler
from ax.utils.common.logger import get_logger
from ax.utils.common.random import with_rng_seed
from ax.utils.common.typeutils import checked_cast, not_none

logger: Logger = get_logger(__name__)


def compute_score_trace(
    optimization_trace: np.ndarray,
    num_baseline_trials: int,
    problem: BenchmarkProblemProtocol,
) -> np.ndarray:
    """Computes a score trace from the optimization trace."""

    # Use the first GenerationStep's best found point as baseline. Sometimes (ex. in
    # a timeout) the first GenerationStep will not have not completed and we will not
    # have enough trials; in this case we do not score.
    if (len(optimization_trace) <= num_baseline_trials) or not isinstance(
        problem, BenchmarkProblemWithKnownOptimum
    ):
        return np.full(len(optimization_trace), np.nan)
    optimum = problem.optimal_value
    baseline = optimization_trace[num_baseline_trials - 1]

    score_trace = 100 * (1 - (optimization_trace - optimum) / (baseline - optimum))
    if score_trace.max() > 100:
        logger.info(
            "Observed scores above 100. This indicates that we found a trial that is "
            "better than the optimum. Clipping scores to 100 for now."
        )
    return score_trace.clip(min=0, max=100)


def _create_benchmark_experiment(
    problem: BenchmarkProblemProtocol, method_name: str
) -> Experiment:
    """Creates an empty experiment for the given problem and method.

    If the problem is "noiseless" (i.e. evaluations prior to adding artificial
    noise to the outcomes are determinstic), also adds the respective ground
    truth metrics as tracking metrics for each metric defined on the problem
    (including tracking metrics and the metrics in the OptimizationConfig).

    Args:
        problem: The BenchmarkProblem to test against (can be synthetic or real).
        method_name: Name of the method being tested.

    Returns:
        The Experiment object to be used for benchmarking.
    """
    tracking_metrics = problem.tracking_metrics
    if not problem.is_noiseless and problem.has_ground_truth:
        # Make the ground truth counterparts for each metric defined on the problem,
        # which will be added as tracking metrics on the Experiment object below.
        # In the analysis, a modified OptimziationConfig referencing those metrics
        # will be passed to the `Scheduler.get_trace()` method, which allows to extract
        # the optimziation trace based on the ground truth outcomes (without noise).
        # If the problem is known to be noiseless, this is unneccesary and we can just
        # use the observations made during the optimization loop directly.
        gt_metric_dict = make_ground_truth_metrics(problem=problem)
        tracking_metrics = tracking_metrics + list(gt_metric_dict.values())
    return Experiment(
        name=f"{problem.name}|{method_name}_{int(time())}",
        search_space=problem.search_space,
        optimization_config=problem.optimization_config,
        tracking_metrics=tracking_metrics,  # pyre-ignore [6]: Incompatible
        # parameter type: In call `Experiment.__init__`, for argument
        # `tracking_metrics`, expected `Optional[List[Metric]]` but got
        # `Union[List[Union[BenchmarkMetricBase, Metric]], List[BenchmarkMetricBase]]`.
        runner=problem.runner,
    )


def benchmark_replication(
    problem: BenchmarkProblemProtocol,
    method: BenchmarkMethod,
    seed: int,
) -> BenchmarkResult:
    """Runs one benchmarking replication (equivalent to one optimization loop).

    Args:
        problem: The BenchmarkProblem to test against (can be synthetic or real)
        method: The BenchmarkMethod to test
        seed: The seed to use for this replication.
    """

    experiment = _create_benchmark_experiment(problem=problem, method_name=method.name)

    scheduler = Scheduler(
        experiment=experiment,
        generation_strategy=method.generation_strategy.clone_reset(),
        options=method.scheduler_options,
    )

    with with_rng_seed(seed=seed):
        scheduler.run_n_trials(max_trials=problem.num_trials)

    if not problem.is_noiseless and problem.has_ground_truth:
        # We modify the optimization config so we can use `Scheduler.get_trace()`
        # to use the true (not corrupted by noise) observations that were logged
        # as tracking metrics on the Experiment object. If the problem is known to
        # be noiseless, this is unnecssary and we can just use the observations
        # made during the optimization loop directly.
        analysis_opt_config = make_ground_truth_optimization_config(
            experiment=experiment
        )
    else:
        analysis_opt_config = experiment.optimization_config

    optimization_trace = np.asarray(
        scheduler.get_trace(optimization_config=analysis_opt_config)
    )

    try:
        # Catch any errors that may occur during score computation, such as errors
        # while accessing "steps" in node based generation strategies. The error
        # handling here is intentionally broad. The score computations is not
        # an essential step of the benchmark runs. We do not want to throw away
        # valuable results after the benchmark run completes just because a
        # non-essential step failed.
        num_baseline_trials = scheduler.standard_generation_strategy._steps[
            0
        ].num_trials

        score_trace = compute_score_trace(
            optimization_trace=optimization_trace,
            num_baseline_trials=num_baseline_trials,
            problem=problem,
        )
    except Exception as e:
        logger.warning(
            f"Failed to compute score trace. Returning NaN. Original error message: {e}"
        )
        score_trace = np.full(len(optimization_trace), np.nan)

    fit_time, gen_time = get_model_times(experiment=experiment)

    return BenchmarkResult(
        name=scheduler.experiment.name,
        seed=seed,
        experiment=scheduler.experiment,
        optimization_trace=optimization_trace,
        score_trace=score_trace,
        fit_time=fit_time,
        gen_time=gen_time,
    )


def benchmark_one_method_problem(
    problem: BenchmarkProblemProtocol,
    method: BenchmarkMethod,
    seeds: Iterable[int],
) -> AggregatedBenchmarkResult:
    return AggregatedBenchmarkResult.from_benchmark_results(
        results=[
            benchmark_replication(problem=problem, method=method, seed=seed)
            for seed in seeds
        ]
    )


def benchmark_multiple_problems_methods(
    problems: Iterable[BenchmarkProblemProtocol],
    methods: Iterable[BenchmarkMethod],
    seeds: Iterable[int],
) -> List[AggregatedBenchmarkResult]:
    """
    For each `problem` and `method` in the Cartesian product of `problems` and
    `methods`, run the replication on each seed in `seeds` and get the results
    as an `AggregatedBenchmarkResult`, then return a list of each
    `AggregatedBenchmarkResult`.
    """
    return [
        benchmark_one_method_problem(problem=p, method=m, seeds=seeds)
        for p, m in product(problems, methods)
    ]


def make_ground_truth_metrics(
    problem: BenchmarkProblemProtocol,
    include_tracking_metrics: bool = True,
) -> Dict[str, Metric]:
    """Makes a ground truth version for each metric defined on the problem.

    Args:
        problem: The BenchmarkProblem to test against (can be synthetic or real).
        include_tracking_metrics: Whether or not to include tracking metrics.

    Returns:
        A dict mapping (original) metric names to their respective ground truth metric.
    """
    if not problem.has_ground_truth:
        raise ValueError(
            "Cannot create ground truth metrics for problems that "
            "do not have a ground truth."
        )
    metrics: List[BenchmarkMetricBase] = [
        checked_cast(BenchmarkMetricBase, metric)
        for metric in problem.optimization_config.metrics.values()
    ]
    if include_tracking_metrics:
        metrics = metrics + problem.tracking_metrics
    return {metric.name: metric.make_ground_truth_metric() for metric in metrics}


def make_ground_truth_optimization_config(
    experiment: Experiment,
) -> OptimizationConfig:
    """Makes a clone of the OptimizationConfig on the experiment in which each metric
    is replaced by its respective "ground truth" counterpart, which has been added to
    the experiment's tracking metrics in `_create_benchmark_experiment` and which
    returns the ground truth (i.e., uncorrupted by noise) observations.
    """
    optimization_config = not_none(experiment.optimization_config)

    if optimization_config.risk_measure is not None:
        raise NotImplementedError("Support for risk measures is not yet implemented.")

    # dict for caching metric lookup
    gt_metric_dict: Dict[str, BenchmarkMetricBase] = {}

    def get_gt_metric(metric: Metric) -> BenchmarkMetricBase:
        """Look up corresponding ground truth metric of the experiment. Will error
        out if no corresponding ground truth metric exists."""
        if not isinstance(metric, BenchmarkMetricBase):
            raise ValueError(
                "Only BenchmarkMetricBase metrics are supported for ground truth "
                f"metrics. Got {type(metric)}."
            )

        if metric.name in gt_metric_dict:
            return gt_metric_dict[metric.name]

        for tracking_metric in experiment.tracking_metrics:
            if getattr(tracking_metric, "is_ground_truth", False):
                # TODO: Figure out if there is a better way to match the ground truth
                # metric and the original metric.
                ground_truth_name = tracking_metric.name
                orig_name = checked_cast(
                    GroundTruthMetricMixin, tracking_metric
                ).get_original_name(ground_truth_name)
                if orig_name == metric.name:
                    tracking_metric = checked_cast(BenchmarkMetricBase, tracking_metric)
                    gt_metric_dict[metric.name] = tracking_metric
                    return tracking_metric
        raise ValueError(f"Ground truth metric for metric {metric.name} not found!")

    # convert outcome constraints
    if optimization_config.outcome_constraints is not None:
        gt_outcome_constraints = [
            OutcomeConstraint(
                metric=get_gt_metric(oc.metric),
                op=oc.op,
                bound=oc.bound,
                relative=oc.relative,
            )
            for oc in optimization_config.outcome_constraints
        ]
    else:
        gt_outcome_constraints = None

    # we need to distinguish MOO and non-MOO problems
    if not optimization_config.is_moo_problem:
        gt_objective = Objective(
            metric=get_gt_metric(optimization_config.objective.metric)
        )

        return OptimizationConfig(
            objective=gt_objective, outcome_constraints=gt_outcome_constraints
        )

    gt_objective = MultiObjective(
        metrics=[
            get_gt_metric(metric) for metric in optimization_config.objective.metrics
        ]
    )
    # there may be objective thresholds to also convert
    objective_thresholds = checked_cast(
        MultiObjectiveOptimizationConfig, optimization_config
    ).objective_thresholds
    if objective_thresholds is not None:
        gt_objective_thresholds = [
            ObjectiveThreshold(
                metric=get_gt_metric(ot.metric),
                bound=ot.bound,
                relative=ot.relative,
                op=ot.op,
            )
            for ot in objective_thresholds
        ]
    else:
        gt_objective_thresholds = None

    return MultiObjectiveOptimizationConfig(
        objective=gt_objective,
        outcome_constraints=gt_outcome_constraints,
        objective_thresholds=gt_objective_thresholds,
    )
