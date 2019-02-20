#!/usr/bin/env python3

from typing import NamedTuple

from ae.lazarus.ae.core.objective import Objective
from ae.lazarus.ae.core.optimization_config import OptimizationConfig
from ae.lazarus.ae.core.outcome_constraint import OutcomeConstraint
from ae.lazarus.ae.core.parameter import ParameterType, RangeParameter
from ae.lazarus.ae.core.search_space import SearchSpace
from ae.lazarus.ae.core.types.types import ComparisonOp
from ae.lazarus.ae.metrics.branin import (
    BraninConstraintMetric,
    BraninMetric,
    NegativeBraninMetric,
)
from ae.lazarus.ae.metrics.hartmann6 import Hartmann6Metric
from ae.lazarus.ae.tests.fake import get_branin_search_space


class BenchmarkProblem(NamedTuple):
    """Contains features that describe a benchmarking problem: its name, its
    global optimum (maximum for maximization problems, minimum for minimization),
    its optimization configuration, and its search space.

    Args:
        name (str): name of this problem
        fbest (float): global optimum
        optimization_config (OptimizationConfig): optimization configuration
        search_space (SearchSpace): search space, on which this problem is defined
    """

    name: str
    fbest: float
    optimization_config: OptimizationConfig
    search_space: SearchSpace


# Branin problems
branin = BenchmarkProblem(
    name="branin",
    fbest=0.397_887,
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=BraninMetric(
                name="branin_objective", param_names=["x1", "x2"], noise_sd=5.0
            ),
            minimize=True,
        )
    ),
    search_space=get_branin_search_space(),
)


branin_max = BenchmarkProblem(
    name="branin_max",
    fbest=0.397_887,
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=NegativeBraninMetric(
                name="neg_branin", param_names=["x1", "x2"], noise_sd=5.0
            ),
            minimize=False,
        )
    ),
    search_space=get_branin_search_space(),
)


constrained_branin = BenchmarkProblem(
    name="constrained_branin",
    fbest=0.397_887,
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=BraninMetric(
                name="branin_objective", param_names=["x1", "x2"], noise_sd=5.0
            ),
            minimize=True,
        ),
        outcome_constraints=[
            OutcomeConstraint(
                metric=BraninConstraintMetric(
                    name="branin_constraint", param_names=["x1", "x2"], noise_sd=5.0
                ),
                op=ComparisonOp.LEQ,
                bound=0.0,
                relative=False,
            )
        ],
    ),
    search_space=get_branin_search_space(),
)


# Hartmann 6 problems

hartmann6 = BenchmarkProblem(
    name="hartmann6",
    fbest=-3.32237,
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=Hartmann6Metric(
                name="hartmann6", param_names=[f"x{i}" for i in range(6)], noise_sd=0.2
            ),
            minimize=True,
        )
    ),
    search_space=SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i}", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
            )
            for i in range(6)
        ]
    ),
)
