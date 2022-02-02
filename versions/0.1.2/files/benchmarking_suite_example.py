
from ax.benchmark.benchmark_problem import hartmann6

import numpy as np

from ax.benchmark.benchmark_problem import BenchmarkProblem
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core.outcome_constraint import ComparisonOp, OutcomeConstraint
from ax.core.parameter import ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.metrics.noisy_function import NoisyFunctionMetric

# Create a Metric object for each function used in the problem
class GramacyObjective(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return x.sum()

class GramacyConstraint1(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return 1.5 - x[0] - 2 * x[1] - 0.5 * np.sin(2 * np.pi * (x[0] ** 2 - 2 * x[1]))

class GramacyConstraint2(NoisyFunctionMetric):
    def f(self, x: np.ndarray) -> float:
        return x[0] ** 2 + x[1] ** 2 - 1.5

# Create the search space and optimization config
search_space = SearchSpace(
    parameters=[
        RangeParameter(name="x1", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0),
        RangeParameter(name="x2", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0),
    ]
)

# When we create the OptimizationConfig, we can define the noise level for each metric.
optimization_config=OptimizationConfig(
    objective=Objective(
        metric=GramacyObjective(
            name="objective", param_names=["x1", "x2"], noise_sd=0.05
        ),
        minimize=True,
    ),
    outcome_constraints=[
        OutcomeConstraint(
            metric=GramacyConstraint1(name="constraint_1", param_names=["x1", "x2"], noise_sd=0.05),
            op=ComparisonOp.LEQ,
            bound=0,
            relative=False,
        ),
        OutcomeConstraint(
            metric=GramacyConstraint2(name="constraint_2", param_names=["x1", "x2"], noise_sd=0.2),
            op=ComparisonOp.LEQ,
            bound=0,
            relative=False,
        ),
    ],
)

# Create a BenchmarkProblem object
gramacy_problem = BenchmarkProblem(
    name="Gramacy",
    fbest=0.5998,
    optimization_config=optimization_config,
    search_space=search_space,
)

from ax.modelbridge.torch import TorchModelBridge
from ax.models.torch.botorch import BotorchModel
from ax.modelbridge.transforms.unit_x import UnitX
from ax.modelbridge.transforms.standardize_y import StandardizeY

def get_botorch_model(experiment, data, search_space):
    m = BotorchModel()  # This can be any implementation of TorchModel
    return TorchModelBridge(
        experiment=experiment,
        search_space=search_space,
        data=data,
        model=m,
        transforms=[UnitX, StandardizeY],
    )

from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep

def unscrambled_sobol(search_space):
    return get_sobol(search_space, scramble=False)

strategy1 = GenerationStrategy(
    name='GP+NEI',
    steps=[
        GenerationStep(model=unscrambled_sobol, num_arms=10),
        GenerationStep(model=get_botorch_model, num_arms=10),
    ],
)

from ax.modelbridge.factory import get_sobol

strategy2 = GenerationStrategy(
    name='Quasirandom',
    steps=[
        GenerationStep(model=unscrambled_sobol, num_arms=10),
        GenerationStep(model=get_sobol, num_arms=10),
    ],
)

from ax.benchmark.benchmark_suite import BOBenchmarkingSuite

b = BOBenchmarkingSuite()

b.run(
    num_runs=5,  # Each benchmark task is repeated this many times
    total_iterations=20,  # The total number of iterations in each optimization
    batch_size=2,  # Number of synchronous parallel evaluations
    bo_strategies=[strategy1, strategy2],
    bo_problems=[hartmann6, gramacy_problem],
)

from IPython.core.display import HTML

report = b.generate_report(include_individual=False)
HTML(report)
