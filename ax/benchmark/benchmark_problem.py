#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from typing import NamedTuple

from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import OutcomeConstraint
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import ComparisonOp
from ax.metrics.branin import BraninMetric, NegativeBraninMetric
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.metrics.l2norm import L2NormMetric
from ax.utils.measurement.synthetic_functions import (
    branin as branin_function,
    hartmann6 as hartmann6_function,
)
from ax.utils.testing.fake import get_branin_search_space


class BenchmarkProblem(NamedTuple):
    """Contains features that describe a benchmarking problem: its name, its
    global optimum (maximum for maximization problems, minimum for minimization),
    its optimization configuration, and its search space.

    Args:
        name: name of this problem
        fbest: global optimum
        optimization_config: optimization configuration
        search_space: search space, on which this problem is defined
    """

    name: str
    fbest: float
    optimization_config: OptimizationConfig
    search_space: SearchSpace


# Branin problems
branin = BenchmarkProblem(
    name=branin_function.name,
    fbest=branin_function.fmin,
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
    name=branin_function.name,
    fbest=branin_function.fmax,
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


# Hartmann 6 problems

hartmann6 = BenchmarkProblem(
    name=hartmann6_function.name,
    fbest=hartmann6_function.fmin,
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=Hartmann6Metric(
                name=hartmann6_function.name,
                param_names=[f"x{i}" for i in range(6)],
                noise_sd=0.01,
            ),
            minimize=True,
        )
    ),
    search_space=SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i}",
                parameter_type=ParameterType.FLOAT,
                lower=param_domain[0],
                upper=param_domain[1],
            )
            for i, param_domain in enumerate(hartmann6_function.domain)
        ]
    ),
)


hartmann6_constrained = BenchmarkProblem(
    name=hartmann6_function.name,
    fbest=hartmann6_function.fmin,
    optimization_config=OptimizationConfig(
        objective=Objective(
            metric=Hartmann6Metric(
                name="hartmann6", param_names=[f"x{i}" for i in range(6)], noise_sd=0.2
            ),
            minimize=True,
        ),
        outcome_constraints=[
            OutcomeConstraint(
                metric=L2NormMetric(
                    name="l2norm", param_names=[f"x{i}" for i in range(6)], noise_sd=0.2
                ),
                op=ComparisonOp.LEQ,
                bound=1.25,
                relative=False,
            )
        ],
    ),
    search_space=SearchSpace(
        parameters=[
            RangeParameter(
                name=f"x{i}",
                parameter_type=ParameterType.FLOAT,
                lower=param_domain[0],
                upper=param_domain[1],
            )
            for i, param_domain in enumerate(hartmann6_function.domain)
        ]
    ),
)
